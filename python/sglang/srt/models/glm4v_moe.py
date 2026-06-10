# GLM-4V MoE 视觉语言模型文件
# 本文件实现了仅推理模式的 GLM-4V MoE（混合专家）多模态视觉语言模型，
# 继承自 Glm4vForConditionalGeneration，结合了 MoE 语言模型和视觉编码器，
# 包含共享专家融合、权重加载等功能。

import logging  # 导入日志模块
from functools import lru_cache  # 导入 LRU 缓存装饰器
from typing import Iterable, Optional, Tuple  # 导入类型注解

import torch  # 导入 PyTorch
import torch.nn as nn  # 导入神经网络模块
from transformers.models.glm4v_moe.configuration_glm4v_moe import Glm4vMoeConfig  # 导入 GLM4V MoE 配置类

from sglang.srt.distributed import (  # 导入分布式相关函数
    get_moe_expert_parallel_world_size,  # 获取 MoE 专家并行世界大小
    get_tensor_model_parallel_world_size,  # 获取张量并行世界大小
)
from sglang.srt.distributed.parallel_state import get_pp_group  # 导入获取流水线并行组的函数
from sglang.srt.layers.attention import vision_utils  # 导入视觉注意力工具
from sglang.srt.layers.logits_processor import LogitsProcessor  # 导入 logits 处理器
from sglang.srt.layers.moe import get_moe_a2a_backend  # 导入获取 MoE all-to-all 后端的函数
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # 导入融合 MoE Triton 层
from sglang.srt.layers.pooler import Pooler, PoolingType  # 导入池化层和池化类型
from sglang.srt.layers.quantization.base_config import QuantizationConfig  # 导入量化配置基类
from sglang.srt.layers.utils import PPMissingLayer  # 导入流水线并行缺失层
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead  # 导入并行语言模型头
from sglang.srt.model_loader.weight_utils import default_weight_loader  # 导入默认权重加载器
from sglang.srt.models.glm4_moe import Glm4MoeModel  # 从 glm4_moe 模型导入 MoE 模型
from sglang.srt.models.glm4v import Glm4vForConditionalGeneration, Glm4vVisionModel  # 从 glm4v 模型导入条件生成模型和视觉模型
from sglang.srt.server_args import get_global_server_args  # 导入获取全局服务器参数的函数
from sglang.srt.utils import add_prefix, get_device_sm, is_cuda, log_info_on_rank0  # 导入前缀添加、设备SM获取、CUDA判断和rank0日志工具
from sglang.srt.utils.hf_transformers_utils import get_processor  # 导入 HuggingFace 处理器获取函数

_is_cuda = is_cuda()  # 判断当前是否为 CUDA 设备
_device_sm = get_device_sm()  # 获取当前设备的 SM 版本

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

cached_get_processor = lru_cache(get_processor)  # 缓存处理器获取函数


class Glm4vMoeForConditionalGeneration(Glm4vForConditionalGeneration):  # GLM-4V MoE 条件生成模型类，继承自 Glm4vForConditionalGeneration
    def __init__(  # 初始化方法
        self,
        config: Glm4vMoeConfig,  # GLM4V MoE 配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置，可选
        prefix: str = "",  # 参数前缀，默认为空字符串
    ) -> None:
        nn.Module.__init__(self)  # 调用 nn.Module 的初始化

        self.pp_group = get_pp_group()  # 获取流水线并行组
        self.config = config  # 保存配置
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder  # 是否启用多模态数据并行编码器
        vision_utils.update_vit_attn_dummy_heads_config(self.config)  # 更新 ViT 注意力虚拟头配置
        self.tp_size = get_tensor_model_parallel_world_size()  # 获取张量并行世界大小
        self.quant_config = quant_config  # 保存量化配置
        self.num_fused_shared_experts = 0  # 初始化融合共享专家数量为0
        self.determine_num_fused_shared_experts()  # 确定融合共享专家数量

        self.model = Glm4MoeModel(  # 创建 MoE 语言模型
            config,  # 模型配置
            quant_config,  # 量化配置
            prefix=add_prefix("language_model", prefix),  # 添加语言模型前缀
        )
        self.visual = Glm4vVisionModel(  # 创建视觉模型
            config.vision_config,  # 视觉配置
            quant_config=quant_config,  # 量化配置
            prefix=add_prefix("visual", prefix),  # 添加前缀
            use_data_parallel=self.use_data_parallel,  # 是否使用数据并行
        )

        if self.pp_group.is_last_rank:  # 如果是流水线并行的最后一个秩
            if self.pp_group.world_size == 1 and self.config.tie_word_embeddings:  # 如果世界大小为1且绑定词嵌入
                self.lm_head = self.model.embed_tokens  # 语言模型头与嵌入层共享
            else:  # 否则
                self.lm_head = ParallelLMHead(  # 创建并行语言模型头
                    config.vocab_size,  # 词表大小
                    config.hidden_size,  # 隐藏层维度
                    quant_config=quant_config,  # 量化配置
                    prefix=add_prefix("lm_head", prefix),  # 添加前缀
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,  # 是否使用注意力张量并行组
                )
        else:  # 否则
            # ranks other than the last rank will have a placeholder layer  # 非最后一个秩将有一个占位层
            self.lm_head = PPMissingLayer()  # 流水线并行缺失层

        self.logits_processor = LogitsProcessor(config)  # 创建 logits 处理器
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)  # 创建池化层，使用最后token池化和归一化
        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling  # 是否启用多节旋转位置编码

        # For EAGLE3 support  # 用于 EAGLE3 支持
        self.capture_aux_hidden_states = False  # 是否捕获辅助隐藏状态，默认为 False

    def determine_num_fused_shared_experts(self):  # 确定融合共享专家数量的方法
        if get_global_server_args().disable_shared_experts_fusion:  # 如果禁用共享专家融合
            return  # 直接返回

        disable_reason = None  # 初始化禁用原因为 None
        if not getattr(self.config, "n_shared_experts", None):  # 如果配置中没有定义共享专家
            disable_reason = "No shared experts are defined in the config."  # 禁用原因：配置中没有定义共享专家
        elif not _is_cuda:  # 如果不是 CUDA 设备
            disable_reason = "Shared experts fusion currently requires CUDA devices."  # 禁用原因：共享专家融合目前需要 CUDA 设备
        elif _is_cuda and (_device_sm is not None) and (_device_sm < 80):  # 如果是 CUDA 但 SM 版本低于80
            disable_reason = "Shared experts fusion requires SM80 or newer GPUs."  # 禁用原因：共享专家融合需要 SM80 或更新的 GPU
        elif get_moe_expert_parallel_world_size() > 1:  # 如果启用了专家并行且世界大小大于1
            disable_reason = "Shared experts fusion is not supported together with expert parallelism yet."  # 禁用原因：共享专家融合尚不支持专家并行
        elif get_moe_a2a_backend().is_deepep():  # 如果使用 DeepEP MoE 后端
            disable_reason = "Shared experts fusion is not supported when Deepep MoE backend is enabled."  # 禁用原因：启用 DeepEP MoE 后端时不支持共享专家融合

        if disable_reason is not None:  # 如果有禁用原因
            get_global_server_args().disable_shared_experts_fusion = True  # 全局禁用共享专家融合
            log_info_on_rank0(  # 在 rank0 上记录信息
                logger,  # 日志记录器
                f"{disable_reason} Shared experts fusion optimization is disabled.",  # 禁用原因及共享专家融合优化已禁用
            )
            return  # 返回

        self.num_fused_shared_experts = self.config.n_shared_experts  # 设置融合共享专家数量为配置中的共享专家数
        assert (  # 断言
            self.num_fused_shared_experts == 1  # 融合共享专家数量为1
        ), "Only 1 fused shared expert is supported for Glm4vMoeForConditionalGeneration"  # Glm4vMoeForConditionalGeneration 仅支持1个融合共享专家
        log_info_on_rank0(logger, "Shared experts fusion optimization enabled.")  # 在 rank0 上记录共享专家融合优化已启用

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):  # 加载权重方法，支持 NextN 模式
        if is_nextn:  # 如果是 NextN 模式
            if hasattr(self.config, "num_nextn_predict_layers"):  # 如果配置中有 NextN 预测层数
                num_nextn_layers = self.config.num_nextn_predict_layers  # 获取 NextN 预测层数
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"  # 断言只支持1个 NextN 层
                # compatible with old design  # 兼容旧设计
                nextn_layer_id = (  # NextN 层ID
                    0  # 如果只有1个隐藏层则为0
                    if self.config.num_hidden_layers == 1  # 如果隐藏层数为1
                    else self.config.num_hidden_layers  # 否则为隐藏层数（最后一层之后）
                )
            else:  # 否则
                raise ValueError("num_nextn_predict_layers is not in the config")  # 抛出配置中缺少 NextN 预测层数的错误

        stacked_params_mapping = [  # 堆叠参数映射
            # (param_name, shard_name, shard_id)  # (参数名, 分片名, 分片ID)
            ("qkv_proj", "q_proj", "q"),  # QKV 投影中的 Q
            ("qkv_proj", "k_proj", "k"),  # QKV 投影中的 K
            ("qkv_proj", "v_proj", "v"),  # QKV 投影中的 V
            ("gate_up_proj", "gate_proj", 0),  # 门控上投影中的门控投影
            ("gate_up_proj", "up_proj", 1),  # 门控上投影中的上投影
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(  # 创建专家参数映射
            ckpt_gate_proj_name="gate_proj",  # 检查点门控投影名
            ckpt_down_proj_name="down_proj",  # 检查点下投影名
            ckpt_up_proj_name="up_proj",  # 检查点上投影名
            num_experts=self.config.n_routed_experts + self.num_fused_shared_experts,  # 专家数量为路由专家数加融合共享专家数
        )

        if is_nextn:  # 如果是 NextN 模式
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"  # NextN 层前缀
            nextn_spec_weight_names = [  # NextN 特有权重名称列表
                "shared_head.norm",  # 共享头归一化
                "eh_proj",  # 嵌入隐藏投影
                "enorm",  # 嵌入归一化
                "hnorm",  # 隐藏状态归一化
            ]

        params_dict = dict(self.named_parameters())  # 获取模型参数字典
        weight_names = []  # 初始化权重名称列表
        for name, loaded_weight in weights:  # 遍历所有权重
            if "language_model." in name:  # 如果名称包含 language_model.
                name = name.replace("language_model.", "")  # 移除 language_model. 前缀
            if "model.visual." in name:  # 如果名称包含 model.visual.
                name = name.replace("model.visual.", "visual.")  # 替换为 visual. 前缀
            if "rotary_emb.inv_freq" in name:  # 如果是旋转嵌入的逆频率
                continue  # 跳过

            weight_names.append(name)  # 记录权重名称

            if self.num_fused_shared_experts > 0 and "mlp.shared_experts" in name:  # 如果融合共享专家数大于0且是共享专家权重
                # Shared expert becomes expert ID = n_routed_experts  # 共享专家变为专家ID = 路由专家数
                name = name.replace(  # 替换名称
                    "mlp.shared_experts",  # 原始名称
                    f"mlp.experts.{self.config.n_routed_experts}",  # 替换为专家ID
                )

            if not is_nextn:  # 如果不是 NextN 模式
                if hasattr(self.config, "num_nextn_predict_layers"):  # 如果配置中有 NextN 预测层数
                    num_nextn_layers = self.config.num_nextn_predict_layers  # 获取 NextN 预测层数
                    if num_nextn_layers > 0 and name.startswith("model.layers"):  # 如果 NextN 层数大于0且是模型层权重
                        name_list = name.split(".")  # 按点分割名称
                        if (  # 如果
                            len(name_list) >= 3  # 名称部分数大于等于3
                            and int(name_list[2]) >= self.config.num_hidden_layers  # 层索引大于等于隐藏层数
                        ):
                            continue  # 跳过 NextN 层权重
            else:  # 如果是 NextN 模式
                if not name.startswith(nextn_layer_prefix):  # 如果不是 NextN 层的权重
                    continue  # 跳过

                # Use shared head and embed weights from target model  # 使用目标模型的共享头和嵌入权重
                if "shared_head.head" in name or "embed_tokens" in name:  # 如果是共享头或嵌入层权重
                    continue  # 跳过

                is_decoder = True  # 标记为解码器权重
                # For nextn specific weights  # 对于 NextN 特有权重
                for weight_name in nextn_spec_weight_names:  # 遍历 NextN 特有权重名称
                    if weight_name in name:  # 如果名称中包含该权重名
                        name = name.replace(nextn_layer_prefix, "model")  # 替换前缀
                        is_decoder = False  # 标记不是解码器权重
                        break  # 跳出循环
                # For decoder layer weights  # 对于解码器层权重
                if is_decoder:  # 如果是解码器权重
                    name = name.replace(nextn_layer_prefix, "model.decoder")  # 替换为解码器前缀

            for param_name, weight_name, shard_id in stacked_params_mapping:  # 遍历堆叠参数映射
                # Skip non-stacked layers and experts (experts handled below).  # 跳过非堆叠层和专家（专家在下面处理）。
                if weight_name not in name:  # 如果分片名不在名称中
                    continue  # 跳过
                # We have mlp.experts[0].gate_proj in the checkpoint.  # 检查点中有 mlp.experts[0].gate_proj。
                # Since we handle the experts below in expert_params_mapping,  # 由于我们在下面的 expert_params_mapping 中处理专家，
                # we need to skip here BEFORE we update the name, otherwise  # 我们需要在更新名称之前跳过，否则
                # name will be updated to mlp.experts[0].gate_up_proj, which  # 名称将被更新为 mlp.experts[0].gate_up_proj，这将
                # will then be updated below in expert_params_mapping  # 然后在下面的 expert_params_mapping 中被更新
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.  # 为 mlp.experts[0].gate_gate_up_proj，导致加载失败。
                if "mlp.experts" in name:  # 如果是专家权重
                    continue  # 跳过，交由专家参数映射处理
                name = name.replace(weight_name, param_name)  # 替换分片名为参数名
                # Skip loading extra bias for GPTQ models.  # 跳过 GPTQ 模型的额外偏置加载。
                if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在参数字典中
                    continue  # 跳过
                if name not in params_dict:  # 如果参数名不在参数字典中
                    continue  # 跳过

                param = params_dict[name]  # 获取参数
                weight_loader = param.weight_loader  # 获取权重加载器
                weight_loader(param, loaded_weight, shard_id)  # 加载权重
                break  # 跳出内层循环
            else:  # 如果没有匹配的堆叠参数
                # Track if this is an expert weight to enable early skipping  # 跟踪是否为专家权重以实现早期跳过
                is_expert_weight = False  # 初始化专家权重标志为 False

                for mapping in expert_params_mapping:  # 遍历专家参数映射
                    param_name, weight_name, expert_id, shard_id = mapping  # 解包映射
                    if weight_name not in name:  # 如果分片名不在名称中
                        continue  # 跳过

                    # Mark as expert weight regardless of whether we can process it  # 无论是否能处理，都标记为专家权重
                    is_expert_weight = True  # 设置专家权重标志为 True

                    name = name.replace(weight_name, param_name)  # 替换分片名为参数名
                    if name not in params_dict:  # 如果参数名不在参数字典中
                        # Expert weight not on this rank, will be skipped below  # 专家权重不在当前秩上，将在下面跳过
                        continue  # 跳过

                    param = params_dict[name]  # 获取参数
                    weight_loader = param.weight_loader  # 获取权重加载器
                    weight_loader(  # 加载权重
                        param,  # 参数
                        loaded_weight,  # 加载的权重
                        name,  # 参数名称
                        shard_id=shard_id,  # 分片ID
                        expert_id=expert_id,  # 专家ID
                    )
                    break  # 跳出内层循环
                else:  # 如果没有匹配的专家参数映射
                    if is_expert_weight:  # 如果是专家权重
                        # This is an expert weight but not mapped to this rank, skip all remaining processing  # 这是专家权重但未映射到当前秩，跳过所有后续处理
                        continue  # 跳过

                    if "visual" in name:  # 如果是视觉模型参数
                        # adapt to VisionAttention for GLM-V  # 适配 GLM-V 的 VisionAttention
                        name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")  # 替换注意力 QKV 名称

                    # Skip loading extra bias for GPTQ models.  # 跳过 GPTQ 模型的额外偏置加载。
                    if name.endswith(".bias") and name not in params_dict:  # 如果是偏置且不在参数字典中
                        continue  # 跳过
                    if name not in params_dict:  # 如果参数名不在参数字典中
                        continue  # 跳过

                    if name in params_dict.keys():  # 如果参数名在参数字典中
                        param = params_dict[name]  # 获取参数
                        weight_loader = getattr(  # 获取权重加载器
                            param, "weight_loader", default_weight_loader  # 默认使用 default_weight_loader
                        )
                        if "visual" in name:  # 如果是视觉模型参数
                            loaded_weight = vision_utils.pad_vit_attn_dummy_heads(  # 填充 ViT 注意力虚拟头
                                self.config, name, loaded_weight  # 传入配置、名称和权重
                            )
                        weight_loader(param, loaded_weight)  # 加载权重
                    else:  # 否则
                        logger.warning(f"Parameter {name} not found in params_dict")  # 记录参数未找到的警告


EntryClass = [Glm4vMoeForConditionalGeneration]  # 入口类列表，用于模型注册
