# POINTS v1.5 聊天模型实现
# 本文件实现了 POINTS v1.5 多模态聊天模型，该模型结合了 Qwen2 语言模型和视觉编码器，
# 支持图像输入的视觉问答场景。视觉编码器基于 Navit（Naive Vision Transformer）架构，
# 并使用 Qwen2VisionPatchMerger 将视觉特征投影到语言模型的嵌入空间。
import copy  # 导入深拷贝模块
from typing import Iterable, List, Optional, Set, Tuple  # 导入类型提示

import torch  # 导入 PyTorch 框架
import torch.nn.functional as F  # 导入 PyTorch 函数式模块
from torch import nn  # 导入神经网络模块

from sglang.srt.configs.points_v15_chat import POINTSV15ChatConfig  # 导入 POINTS v1.5 聊天配置
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.managers.mm_utils import (  # 导入多模态工具
    MultiModalityDataPaddingPatternMultimodalTokens,  # 多模态数据填充模式
    general_mm_embed_routine,  # 通用多模态嵌入例程
)
from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类
    Modality,  # 模态类型
    MultimodalDataItem,  # 多模态数据项
    MultimodalInputs,  # 多模态输入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批次信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.qwen2 import Qwen2ForCausalLM  # 导入 Qwen2 因果语言模型
from sglang.srt.models.qwen2_vl import Qwen2VisionPatchMerger, Qwen2VisionTransformer  # 导入 Qwen2 视觉组件
from sglang.srt.utils import add_prefix  # 导入前缀添加工具


class Qwen2VisionTransformerForNavitPOINTS(Qwen2VisionTransformer):
    """适配 Navit POINTS 的 Qwen2 视觉 Transformer 编码器"""

    def __init__(
        self,
        vision_config: POINTSV15ChatConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        """初始化视觉 Transformer 编码器"""
        super().__init__(  # 调用父类初始化
            vision_config,  # 视觉配置
            norm_eps=norm_eps,  # 归一化 epsilon
            quant_config=quant_config,  # 量化配置
            prefix=prefix,  # 参数前缀
        )

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """视觉 Transformer 前向传播，将图像 patch 编码为视觉特征"""
        # patchify  # 图像分块
        x = x.to(device=self.device, dtype=self.dtype)  # 将输入转到编码器设备和数据类型
        x = self.patch_embed(x)  # 通过 patch 嵌入层

        # compute position embedding  # 计算位置嵌入
        rotary_pos_emb = self.rot_pos_emb(grid_thw)  # 根据网格维度计算旋转位置嵌入
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)  # 拼接得到完整旋转嵌入
        position_embeddings = (emb.cos(), emb.sin())  # 分解为余弦和正弦分量

        # compute cu_seqlens  # 计算变长序列的累积长度
        cu_seqlens = torch.repeat_interleave(  # 根据时间维度重复空间维度乘积
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]  # 每个图像的 patch 数 * 帧数
        ).cumsum(dim=0, dtype=torch.int32)  # 累积求和得到序列边界
        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)  # 在前面补零作为起始位置

        # transformers  # Transformer 块
        x = x.unsqueeze(1)  # 增加批次维度
        for blk in self.blocks:  # 遍历所有 Transformer 块
            x = blk(x, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)  # 通过每个块

        return x  # 返回编码后的视觉特征


class POINTSV15ChatModel(nn.Module):
    """POINTS v1.5 聊天模型，整合了语言模型、视觉编码器和视觉投影器"""

    def __init__(
        self,
        config: POINTSV15ChatConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        """初始化 POINTS v1.5 聊天模型的所有组件"""
        super().__init__()  # 调用父类初始化
        config.llm_config._attn_implementation = "flash_attention_2"  # 设置语言模型使用 Flash Attention 2
        config._attn_implementation_autoset = False  # 标记注意力实现非自动设置
        self.config = config  # 保存配置
        self.quant_config = quant_config  # 保存量化配置

        llm_config = copy.deepcopy(config.llm_config)  # 深拷贝语言模型配置
        llm_config.architectures = ["Qwen2ForCausalLM"]  # 设置架构为 Qwen2 因果语言模型
        self.llm = Qwen2ForCausalLM(  # 创建 Qwen2 语言模型
            config=llm_config,  # 语言模型配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("llm", prefix),  # 参数前缀
        )

        self.vision_encoder = Qwen2VisionTransformerForNavitPOINTS(  # 创建视觉编码器
            config.vision_config,  # 视觉配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("vision_encoder", prefix),  # 参数前缀
        )

        self.vision_projector = Qwen2VisionPatchMerger(  # 创建视觉投影器
            d_model=config.llm_config.hidden_size,  # 语言模型隐藏维度
            context_dim=1280,  # 视觉编码器上下文维度
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("vision_projector", prefix),  # 参数前缀
        )

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        """使用多模态 token 填充模式对输入 ID 进行填充"""
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()  # 创建填充模式
        return pattern.pad_input_tokens(input_ids, mm_inputs)  # 返回填充后的输入 token

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """从多模态数据项中提取图像特征，经过视觉编码器和投影器"""
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(  # 拼接所有像素值并转换类型
            self.vision_encoder.dtype  # 转换为视觉编码器数据类型
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)  # 拼接所有图像网格维度

        assert pixel_values.dim() == 2, pixel_values.dim()  # 断言像素值为 2 维
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()  # 断言网格维度为 2 维

        image_features = self.vision_encoder(pixel_values, grid_thw=image_grid_thw)  # 通过视觉编码器
        image_features = self.vision_projector(image_features)  # 通过视觉投影器
        return image_features  # 返回图像特征

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ):
        """POINTS v1.5 聊天模型前向传播，处理多模态输入并生成隐藏状态"""
        hidden_states = general_mm_embed_routine(  # 调用通用多模态嵌入例程
            input_ids=input_ids,  # 输入 ID
            forward_batch=forward_batch,  # 前向批次
            language_model=self.llm,  # 语言模型
            data_embedding_funcs={  # 数据嵌入函数映射
                Modality.IMAGE: self.get_image_feature,  # 图像模态对应的特征提取函数
            },
            positions=positions,  # 位置信息
        )

        return hidden_states  # 返回隐藏状态

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载模型权重，处理堆叠参数和视觉编码器的特殊映射"""
        stacked_params_mapping = [  # 堆叠参数映射表
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            ("qkv_proj", "q_proj", "q"),  # QKV 投影中的 Q
            ("qkv_proj", "k_proj", "k"),  # QKV 投影中的 K
            ("qkv_proj", "v_proj", "v"),  # QKV 投影中的 V
            ("gate_up_proj", "gate_proj", 0),  # 门控上投影中的 gate
            ("gate_up_proj", "up_proj", 1),  # 门控上投影中的 up
        ]
        params_dict = dict(self.named_parameters())  # 获取参数字典
        loaded_params: Set[str] = set()  # 已加载参数集合

        for name, loaded_weight in weights:  # 遍历所有权重
            if "rotary_emb.inv_freq" in name:  # 跳过旋转嵌入的逆频率
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不在参数名中
                    continue
                name = name.replace(weight_name, param_name)  # 替换为堆叠参数名

                if name.endswith(".bias") and name not in params_dict:  # 跳过模型中不存在的偏置
                    continue

                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载分片权重
                break
            else:  # 非堆叠参数的处理
                if "vision_encoder" in name:  # 视觉编码器参数
                    # adapt to VisionAttention  # 适配 VisionAttention
                    name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")  # 替换注意力参数名

                try:  # 尝试加载参数
                    # Skip loading extra bias for GPTQ models.  # 跳过 GPTQ 模型的额外偏置
                    if name.endswith(".bias") and name not in params_dict:  # 如果偏置不在参数字典中
                        continue
                    param = params_dict[name]  # 获取参数
                except KeyError:  # 参数未找到
                    print(params_dict.keys())  # 打印可用参数名
                    raise

                weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 获取权重加载器
                weight_loader(param, loaded_weight)  # 加载权重


EntryClass = [POINTSV15ChatModel]  # 模型入口类列表
