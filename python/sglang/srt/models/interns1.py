# InternS1 多模态条件生成模型实现
# 本文件实现了 InternS1 视觉语言模型，将 InternVision 视觉编码器与 Qwen2/Qwen3 系列
# 语言模型结合，支持图像特征提取、像素重排和权重映射等功能。

from typing import Iterable, List, Optional, Tuple  # 导入类型注解工具

import torch  # 导入 PyTorch 深度学习框架
from torch import nn  # 导入神经网络模块
from transformers import PretrainedConfig  # 导入预训练配置类

from sglang.srt.layers.attention import vision_utils  # 导入视觉注意力工具
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # 导入融合 MoE 层
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.managers.mm_utils import (  # 导入多模态工具
    MultiModalityDataPaddingPatternTokenPairs,  # 多模态数据填充模式（token 对）
    general_mm_embed_routine,  # 通用多模态嵌入处理流程
)
from sglang.srt.managers.schedule_batch import (  # 导入调度批处理相关类
    Modality,  # 模态类型枚举
    MultimodalDataItem,  # 多模态数据项
    MultimodalInputs,  # 多模态输入
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 导入前向批处理信息
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.internvl import InternVisionModel  # 导入 InternVision 视觉模型
from sglang.srt.models.qwen2 import Qwen2ForCausalLM  # 导入 Qwen2 因果语言模型
from sglang.srt.models.qwen3 import Qwen3ForCausalLM  # 导入 Qwen3 因果语言模型
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM  # 导入 Qwen3 MoE 因果语言模型
from sglang.utils import logger  # 导入日志工具


class InternS1ForConditionalGeneration(nn.Module):  # InternS1 条件生成模型类，结合视觉和语言模型
    def __init__(  # 初始化方法
        self,
        config: PretrainedConfig,  # 模型配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        use_flash_attn=True,  # 是否使用 Flash Attention，默认 True
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.config = config  # 保存模型配置
        self.quant_config = quant_config  # 保存量化配置
        vision_utils.update_vit_attn_dummy_heads_config(self.config)  # 更新 ViT 注意力虚拟头配置
        image_size = (  # 获取图像尺寸
            getattr(config, "force_image_size", None) or config.vision_config.image_size  # 优先使用强制图像尺寸
        )
        patch_size = config.vision_config.patch_size  # 获取补丁尺寸
        if isinstance(image_size, list):  # 如果图像尺寸是列表
            image_size = image_size[0]  # 取第一个值
        if isinstance(patch_size, list):  # 如果补丁尺寸是列表
            patch_size = patch_size[0]  # 取第一个值
        self.patch_size = patch_size  # 保存补丁尺寸
        self.select_layer = config.vision_feature_layer  # 视觉特征层选择
        self.num_image_token = int(  # 计算每个图像的 token 数量
            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)  # 补丁数的平方乘以下采样比的平方
        )
        self.downsample_ratio = config.downsample_ratio  # 下采样比率

        config.vision_config.use_flash_attn = True if use_flash_attn else False  # 设置视觉模型是否使用 Flash Attention
        config.text_config._attn_implementation = (  # 设置文本模型的注意力实现方式
            "flash_attention_2" if use_flash_attn else "eager"  # Flash Attention 或普通实现
        )

        logger.info(f"num_image_token: {self.num_image_token}")  # 记录图像 token 数量

        self.vision_model = InternVisionModel(config.vision_config)  # 创建 InternVision 视觉模型
        if config.text_config.architectures[0] == "Qwen2ForCausalLM":  # 如果文本模型架构是 Qwen2
            self.language_model = Qwen2ForCausalLM(  # 创建 Qwen2 语言模型
                config=config.text_config, quant_config=quant_config  # 传入配置和量化配置
            )
        elif config.text_config.architectures[0] == "Qwen3MoeForCausalLM":  # 如果文本模型架构是 Qwen3 MoE
            self.language_model = Qwen3MoeForCausalLM(  # 创建 Qwen3 MoE 语言模型
                config=config.text_config, quant_config=quant_config  # 传入配置和量化配置
            )
        elif config.text_config.architectures[0] == "Qwen3ForCausalLM":  # 如果文本模型架构是 Qwen3
            self.language_model = Qwen3ForCausalLM(  # 创建 Qwen3 语言模型
                config=config.text_config, quant_config=quant_config  # 传入配置和量化配置
            )
        else:  # 其他架构
            raise NotImplementedError(  # 抛出未实现异常
                f"{config.text_config.architectures[0]} is not implemented."  # 架构未实现
            )

        vit_hidden_size = config.vision_config.hidden_size  # 视觉模型隐藏层维度
        llm_hidden_size = config.text_config.hidden_size  # 语言模型隐藏层维度

        self.mlp1 = nn.Sequential(  # 多层感知机，将视觉特征投影到语言模型空间
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),  # 层归一化
            nn.Linear(  # 线性层，升维到语言模型空间
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size  # 输入维度为视觉维度乘以下采样因子平方，输出为语言模型维度
            ),
            nn.GELU(),  # GELU 激活函数
            nn.Linear(llm_hidden_size, llm_hidden_size),  # 线性层，保持语言模型维度
        )

    def pixel_shuffle(self, x, scale_factor=0.5):  # 像素重排操作，用于空间降采样
        n, w, h, c = x.size()  # 获取输入尺寸：批次、宽、高、通道
        # N, W, H, C --> N, W, H * scale, C // scale
        # N, W, H, C --> N, W, H * scale, C // scale  将高度扩大、通道缩小
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))  # 重塑张量
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale  交换宽高维度
        x = x.permute(0, 2, 1, 3).contiguous()  # 置换维度
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)  将宽度扩大、通道进一步缩小
        x = x.view(  # 重塑张量
            n,  # 批次维度
            int(h * scale_factor),  # 高度维度
            int(w * scale_factor),  # 宽度维度
            int(c / (scale_factor * scale_factor)),  # 通道维度
        )
        x = x.permute(0, 2, 1, 3).contiguous()  # 再次交换宽高维度，恢复原始顺序
        return x  # 返回像素重排后的张量

    def extract_feature(self, pixel_values):  # 从像素值中提取视觉特征
        if self.select_layer == -1:  # 如果选择最后一层
            vit_embeds = self.vision_model(  # 通过视觉模型
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True  # 不输出中间隐藏状态
            ).last_hidden_state  # 获取最后一层隐藏状态
        else:  # 否则
            vit_embeds = self.vision_model(  # 通过视觉模型
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True  # 输出所有隐藏状态
            ).hidden_states[self.select_layer]  # 获取指定层的隐藏状态
        vit_embeds = vit_embeds[:, 1:, :]  # 去掉 CLS token

        h = w = int(vit_embeds.shape[1] ** 0.5)  # 计算特征图的高宽
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)  # 重塑为 2D 特征图
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)  # 应用像素重排
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])  # 展平空间维度
        vit_embeds = self.mlp1(vit_embeds)  # 通过 MLP 投影到语言模型空间
        return vit_embeds  # 返回视觉特征

    def get_image_feature(self, items: List[MultimodalDataItem]):  # 获取图像特征
        """
        Projects the last hidden state from the vision model into language model space.
        # 将视觉模型的最后隐藏状态投影到语言模型空间。

        Returns:  # 返回值说明
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
            # image_features: 图像特征张量，形状为 (num_images, image_length, embed_dim)
        """
        pixel_values = torch.cat([item.feature for item in items])  # 拼接所有图像的像素值
        image_features = self.extract_feature(pixel_values)  # 提取图像特征
        return image_features  # 返回图像特征

    @torch.no_grad()  # 禁用梯度计算，用于推理
    def forward(  # 前向传播方法
        self,
        input_ids: torch.Tensor,  # 输入 token ID
        positions: torch.Tensor,  # 位置编码
        forward_batch: ForwardBatch,  # 前向批处理信息
        input_embeds: torch.Tensor = None,  # 输入嵌入，可选
    ) -> torch.Tensor:  # 返回语言模型输出

        hs = general_mm_embed_routine(  # 调用通用多模态嵌入处理流程
            input_ids=input_ids,  # 输入 token ID
            forward_batch=forward_batch,  # 前向批处理信息
            language_model=self.language_model,  # 语言模型
            data_embedding_funcs={  # 数据嵌入函数映射
                Modality.IMAGE: self.get_image_feature,  # 图像模态使用 get_image_feature
            },
            positions=positions,  # 位置编码
        )

        return hs  # 返回语言模型输出

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):  # 填充输入 token ID，插入多模态占位符
        # Get all special token IDs  # 获取所有特殊 token ID
        im_start_id: int = mm_inputs.im_start_id  # 图像起始 token ID
        im_end_id: int = mm_inputs.im_end_id  # 图像结束 token ID

        media_token_pairs = [(im_start_id, im_end_id)]  # 媒体 token 对（起始和结束）
        helper = MultiModalityDataPaddingPatternTokenPairs(media_token_pairs)  # 创建填充辅助器

        return helper.pad_input_tokens(input_ids, mm_inputs)  # 使用辅助器填充输入 token

    def _mapping_interns1_name(self, name):  # InternS1 权重名称映射方法，将检查点名称映射到模型内部名称
        names_map = {  # 名称映射字典
            "lm_head.weight": "language_model.lm_head.weight",  # 语言模型头权重
            "model.multi_modal_projector.layer_norm.bias": "mlp1.0.bias",  # 多模态投影器层归一化偏置
            "model.multi_modal_projector.layer_norm.weight": "mlp1.0.weight",  # 多模态投影器层归一化权重
            "model.multi_modal_projector.linear_1.bias": "mlp1.1.bias",  # 多模态投影器第一个线性层偏置
            "model.multi_modal_projector.linear_1.weight": "mlp1.1.weight",  # 多模态投影器第一个线性层权重
            "model.multi_modal_projector.linear_2.bias": "mlp1.3.bias",  # 多模态投影器第二个线性层偏置
            "model.multi_modal_projector.linear_2.weight": "mlp1.3.weight",  # 多模态投影器第二个线性层权重
            "model.vision_tower.embeddings.cls_token": "vision_model.embeddings.class_embedding",  # 视觉塔 CLS token
            "model.vision_tower.embeddings.patch_embeddings.projection.bias": "vision_model.embeddings.patch_embedding.bias",  # 视觉塔补丁嵌入投影偏置
            "model.vision_tower.embeddings.patch_embeddings.projection.weight": "vision_model.embeddings.patch_embedding.weight",  # 视觉塔补丁嵌入投影权重
            "model.vision_tower.embeddings.position_embeddings": "vision_model.embeddings.position_embedding",  # 视觉塔位置嵌入
        }
        if name in names_map:  # 如果名称在映射字典中
            name = names_map[name]  # 使用映射后的名称
        elif name.startswith("model.language_model."):  # 如果名称以语言模型前缀开头
            name = "language_model.model." + name[len("model.language_model.") :]  # 重新映射语言模型前缀
        elif name.startswith("model.vision_tower."):  # 如果名称以视觉塔前缀开头
            name = "vision_model." + name[len("model.vision_tower.") :]  # 重新映射视觉塔前缀

        if name.startswith("vision_model.encoder.layer"):  # 如果名称属于视觉编码器层

            name = name.replace(r".layer.", r".layers.")  # 将 "layer" 替换为 "layers"
            name = name.replace(r".attention.", r".attn.attn.")  # 将 "attention" 替换为 "attn.attn"
            name = name.replace(r".projection_layer.", r".proj.")  # 将 "projection_layer" 替换为 "proj"
            name = name.replace(r".lambda_1", r".ls1")  # 将 "lambda_1" 替换为 "ls1"
            name = name.replace(r".lambda_2", r".ls2")  # 将 "lambda_2" 替换为 "ls2"
            name = name.replace(r".layernorm_before.", r".norm1.")  # 将 "layernorm_before" 替换为 "norm1"
            name = name.replace(r".layernorm_after.", r".norm2.")  # 将 "layernorm_after" 替换为 "norm2"
        return name  # 返回映射后的名称

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):  # 加载模型权重
        stacked_params_mapping = [  # 堆叠参数映射表
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片 ID)
            ("qkv_proj", "q_proj", "q"),  # QKV 投影中的 Q 部分
            ("qkv_proj", "k_proj", "k"),  # QKV 投影中的 K 部分
            ("qkv_proj", "v_proj", "v"),  # QKV 投影中的 V 部分
            ("gate_up_proj", "gate_proj", 0),  # gate_up 投影中的 gate 部分
            ("gate_up_proj", "up_proj", 1),  # gate_up 投影中的 up 部分
        ]
        expert_params_mapping = []  # 专家参数映射表，初始化为空
        if "Qwen3MoeForCausalLM" in self.config.text_config.architectures:  # 如果使用 Qwen3 MoE 架构
            expert_params_mapping = FusedMoE.make_expert_params_mapping(  # 创建专家参数映射
                ckpt_gate_proj_name="gate_proj",  # 检查点中 gate 投影的名称
                ckpt_down_proj_name="down_proj",  # 检查点中 down 投影的名称
                ckpt_up_proj_name="up_proj",  # 检查点中 up 投影的名称
                num_experts=self.config.num_experts,  # 专家数量
            )

        params_dict = dict(self.named_parameters())  # 将模型参数转为字典

        for name, loaded_weight in weights:  # 遍历所有权重
            if "rotary_emb.inv_freq" in name:  # 如果是旋转嵌入的逆频率
                continue  # 跳过，不需要加载
            name = self._mapping_interns1_name(name)  # 通过名称映射方法转换权重名称
            if "vision_model" in name:  # 如果是视觉模型权重
                loaded_weight = vision_utils.pad_vit_attn_dummy_heads(  # 对 ViT 注意力虚拟头进行填充
                    self.config, name, loaded_weight  # 传入配置、名称和权重
                )

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                if weight_name not in name:  # 如果权重名不匹配
                    continue  # 跳过
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                # 检查点中有 mlp.experts[0].gate_proj。
                # 由于我们在下面的 expert_params_mapping 中处理专家权重，
                # 需要在更新名称之前跳过，否则名称会被更新为 mlp.experts[0].gate_up_proj，
                # 然后会在 expert_params_mapping 中被再次更新为 mlp.experts[0].gate_gate_up_proj，导致加载失败。
                if "mlp.experts" in name:  # 如果是 MoE 专家权重
                    continue  # 跳过，在专家映射中处理
                name = name.replace(weight_name, param_name)  # 替换为堆叠参数名
                # Skip loading extra bias for GPTQ models.
                # 跳过 GPTQ 模型中的额外偏置加载
                if name.endswith(".bias") and name not in params_dict:  # 如果是额外偏置
                    continue  # 跳过
                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载分片权重
                break  # 跳出内层循环
            else:  # 如果没有匹配堆叠参数映射
                for mapping in expert_params_mapping:  # 遍历专家参数映射
                    param_name, weight_name, expert_id, shard_id = mapping  # 解包映射信息
                    if weight_name not in name:  # 如果权重名不匹配
                        continue  # 跳过
                    name = name.replace(weight_name, param_name)  # 替换为专家参数名
                    param = params_dict[name]  # 获取参数
                    weight_loader = param.weight_loader  # 获取权重加载器
                    weight_loader(  # 加载专家权重
                        param,  # 目标参数
                        loaded_weight,  # 加载的权重
                        name,  # 映射后的名称
                        shard_id=shard_id,  # 分片 ID
                        expert_id=expert_id,  # 专家 ID
                    )
                    break  # 跳出内层循环
                else:  # 如果没有匹配专家参数映射
                    # Skip loading extra bias for GPTQ models.
                    # 跳过 GPTQ 模型中的额外偏置加载
                    if name.endswith(".bias") and name not in params_dict:  # 如果是额外偏置
                        continue  # 跳过
                    param = params_dict[name]  # 获取参数
                    weight_loader = getattr(  # 获取权重加载器
                        param, "weight_loader", default_weight_loader  # 默认使用 default_weight_loader
                    )
                    weight_loader(param, loaded_weight)  # 加载权重


EntryClass = InternS1ForConditionalGeneration  # 模型入口类，用于框架自动发现和注册模型
