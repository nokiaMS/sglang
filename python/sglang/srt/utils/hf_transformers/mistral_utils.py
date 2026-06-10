# Mistral模型配置解析工具，将Mistral原生参数格式转换为HuggingFace兼容格式
# 支持MoE、视觉、音频、EAGLE推测解码等Mistral模型变体的配置适配
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/configs/mistral.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json  # 导入JSON解析模块
import tempfile  # 导入临时文件模块
from functools import lru_cache  # 导入LRU缓存装饰器
from pathlib import Path  # 导入路径处理模块
from typing import Any, Optional  # 导入类型注解

from transformers import AutoConfig, PretrainedConfig, WhisperConfig  # 导入transformers配置类

from sglang.srt.utils import logger  # 导入日志记录器

from .common import _ensure_sub_configs, download_from_hf  # 导入子配置确保和HF下载工具


def adapt_config_dict(  # 适配Mistral配置字典，转换为HF兼容格式
    config_dict: dict[str, Any], model: str, **kwargs  # 配置字典、模型名称、其他参数
) -> tuple[dict, PretrainedConfig]:  # 返回适配后的字典和配置对象
    config_dict.update(kwargs)  # 更新额外参数
    config_dict = _remap_general_mistral_args(config_dict)  # 重映射通用Mistral参数

    if bool(config_dict.get("quantization")):  # 如果有量化配置
        config_dict = _remap_mistral_quantization_args(config_dict)  # 重映射量化参数

    is_moe = bool(config_dict.get("moe"))  # 是否为MoE模型
    is_mistral_large_3 = (  # 是否为Mistral Large 3模型
        is_moe and (config_dict["moe"].get("num_shared_experts") or 0) > 0  # MoE且有共享专家
    )
    is_eagle = "eagle" in model.lower()  # 是否为EAGLE推测解码模型
    is_mla_eagle = is_eagle and any(  # 是否为MLA EAGLE模型
        config_dict.get(k) is not None
        for k in ("kv_lora_rank", "q_lora_rank", "v_head_dim")  # 检查MLA相关字段
    )
    if is_eagle and not is_moe and is_mla_eagle:  # 密集MLA EAGLE草稿模型
        # Dense MLA EAGLE draft model (e.g. Mistral Small 4 EAGLE).  # 密集MLA EAGLE草稿模型（如Mistral Small 4 EAGLE）
        # Uses MLA attention like MistralLarge3 but has no MoE layers.  # 使用类似MistralLarge3的MLA注意力但没有MoE层
        # Set model_type to deepseek_v3 for MLA support, and override  # 设置model_type为deepseek_v3以支持MLA，并覆盖
        # MoE fields so all layers are dense.  # MoE字段使所有层为密集层
        config_dict["model_type"] = "deepseek_v3"  # 设置模型类型
        config_dict["architectures"] = ["MistralLarge3ForCausalLMEagle"]  # 设置架构
        num_layers = config_dict.get("num_hidden_layers", 0)  # 获取层数
        config_dict["n_routed_experts"] = 1  # 设置路由专家数为1
        config_dict["first_k_dense_replace"] = num_layers  # 所有层都使用密集层
        config_dict["moe_layer_freq"] = 1  # MoE层频率
        config_dict["n_shared_experts"] = 0  # 无共享专家
        config_dict["n_group"] = 1  # 专家组数
        config_dict["topk_group"] = 1  # TopK组
        config_dict["num_experts_per_tok"] = 1  # 每token专家数
        config_dict["moe_intermediate_size"] = 1  # MoE中间层大小
        config_dict["routed_scaling_factor"] = 1.0  # 路由缩放因子
        config_dict["topk_method"] = None  # TopK方法
        config_dict["scoring_func"] = "softmax"  # 评分函数
        config_dict["routing_method_type"] = 1  # 路由方法类型
    elif is_eagle and not is_moe:  # 密集GQA EAGLE草稿模型
        # Dense GQA EAGLE draft model (e.g. Mistral Medium 3.5 EAGLE).  # 密集GQA EAGLE草稿模型（如Mistral Medium 3.5 EAGLE）
        # Routes to a Llama-backbone draft body — no MoE shimming required.  # 使用Llama骨干草稿体——无需MoE填充
        config_dict["architectures"] = ["MistralForCausalLMEagle"]  # 设置架构
        config_dict["model_type"] = "mistral"  # 设置模型类型
        config_dict["rope_is_neox_style"] = False  # Mistral使用非Neox风格RoPE
        for mla_key in (  # 清除MLA相关字段
            "q_lora_rank",
            "qk_rope_head_dim",
            "qk_nope_head_dim",
            "kv_lora_rank",
            "v_head_dim",
        ):
            if config_dict.get(mla_key) is None:  # 如果字段值为None
                config_dict.pop(mla_key, None)  # 移除字段（避免覆盖默认值）
    elif is_moe:  # MoE模型
        if is_mistral_large_3:  # Mistral Large 3 MoE模型
            config_dict = _remap_moe_args(config_dict)  # 重映射MoE参数
            config_dict["model_type"] = "deepseek_v3"  # 设置为deepseek_v3类型
            if is_eagle:  # 如果是EAGLE变体
                config_dict["architectures"] = ["MistralLarge3ForCausalLMEagle"]  # 设置EAGLE架构
            else:  # 非EAGLE变体
                config_dict["architectures"] = ["MistralLarge3ForCausalLM"]  # 设置架构

            assert (  # 断言llama_4_scaling配置存在
                "llama_4_scaling" in config_dict
            ), "MistralLarge3 expect llama4 scaling config."  # MistralLarge3需要llama4缩放配置
            llama_4_scaling_config_keys = ["original_max_position_embeddings", "beta"]  # 必需的配置键
            assert all(  # 断言所有必需键都存在
                [
                    key in config_dict["llama_4_scaling"]
                    for key in llama_4_scaling_config_keys
                ]
            ), (
                "llama_4_scaling config should define the keys: "
                f"{','.join(llama_4_scaling_config_keys)}"  # 显示缺少的键
            )
        else:  # 非Mistral Large 3的MoE模型
            config_dict["architectures"] = ["MixtralForCausalLM"]  # 使用Mixtral架构
    else:  # 密集模型
        config_dict["architectures"] = ["MistralForCausalLM"]  # 使用Mistral架构
        config_dict["model_type"] = "mistral"  # 设置模型类型
        # Mistral models use non-interleaved RoPE (is_neox_style=False),  # Mistral模型使用非交错RoPE（is_neox_style=False）
        # unlike Llama which defaults to True.  # 与Llama默认为True不同
        config_dict["rope_is_neox_style"] = False  # 设置RoPE风格
        # Remove None-valued MLA fields that would shadow defaults in  # 移除值为None的MLA字段，避免覆盖
        # model_config._derive_model_shapes (getattr returns None instead  # model_config._derive_model_shapes中的默认值
        # of the fallback when the attribute exists but is None).  # （当属性存在但为None时getattr返回None而非回退值）
        for mla_key in (  # 遍历MLA相关字段
            "q_lora_rank",
            "qk_rope_head_dim",
            "qk_nope_head_dim",
            "kv_lora_rank",
            "v_head_dim",
        ):
            if config_dict.get(mla_key) is None:  # 如果字段值为None
                config_dict.pop(mla_key, None)  # 移除字段

    if bool(config_dict.get("yarn")):  # 如果有YaRN配置
        config_dict = _remap_mistral_yarn_args(config_dict)  # 重映射YaRN参数

    is_vision = bool(  # 是否为视觉模型
        (config_dict.get("multimodal") or {}).get("vision_encoder_args")  # 多模态配置中有视觉编码器
        or config_dict.get("vision_encoder")  # 或独立的视觉编码器
    )
    is_audio = bool(  # 是否为音频模型
        ((config_dict.get("multimodal") or {}).get("whisper_model_args") or {}).get(
            "encoder_args"  # 多模态配置中有Whisper编码器
        )
    )

    assert not (is_vision and is_audio), "Vision and audio are mutually exclusive"  # 视觉和音频互斥

    if is_vision:  # 如果是视觉模型
        config_dict = _remap_mistral_vision_args(config_dict)  # 重映射视觉参数
    if is_audio:  # 如果是音频模型
        config_dict = _remap_mistral_audio_args(config_dict)  # 重映射音频参数

    config = PretrainedConfig.from_dict(config_dict)  # 从字典创建配置对象

    logger.debug("Initialized config %s", config)  # 记录配置初始化

    return config_dict, config  # 返回适配后的字典和配置


def _remap_mistral_vision_args(config: dict) -> dict:  # 重映射Mistral视觉模型参数为Pixtral格式
    if config.get("multimodal"):  # 如果有多模态配置
        vision_config = config.pop("multimodal")  # 弹出多模态配置
    else:  # 否则
        vision_config = config.pop("vision_encoder")  # 弹出视觉编码器配置

    quant_config = config.get("quantization_config")  # 获取量化配置

    config = {  # 构造Pixtral格式配置
        "model_type": "pixtral",  # 模型类型为pixtral
        "architectures": ["PixtralForConditionalGeneration"],  # Pixtral架构
        "text_config": config,  # 文本配置
        "vision_config": {"model_type": "pixtral", **vision_config},  # 视觉配置
    }
    if quant_config:  # 如果有量化配置
        config["quantization_config"] = quant_config  # 保留量化配置
    return config  # 返回重映射后的配置


def _remap_mistral_yarn_args(config: dict) -> dict:  # 重映射Mistral YaRN参数为deepseek_yarn格式
    yarn_config_map = {  # YaRN配置字段映射
        "factor": "factor",  # 因子
        "original_max_position_embeddings": "original_max_position_embeddings",  # 原始最大位置嵌入
        "beta": "beta_fast",  # beta映射为beta_fast
        "alpha": "beta_slow",  # alpha映射为beta_slow
        "apply_scale": "apply_yarn_scaling",  # apply_scale映射为apply_yarn_scaling
    }
    yarn_config = config.get("yarn") or {}  # 获取YaRN配置
    config["rope_scaling"] = {  # 创建rope_scaling配置
        "rope_type": "deepseek_yarn",  # 使用deepseek_yarn类型
        "mscale_all_dim": 1,  # mscale维度设置
    }
    # Include rope_theta in rope_scaling if present at the top level,  # 如果顶层有rope_theta则包含在rope_scaling中
    # as transformers yarn validation requires it.  # 因为transformers的yarn验证需要它
    if "rope_theta" in config:  # 如果有rope_theta
        config["rope_scaling"]["rope_theta"] = config["rope_theta"]  # 复制到rope_scaling
    for old_name, new_name in yarn_config_map.items():  # 遍历映射
        if old_name in yarn_config:  # 如果旧名称存在于配置中
            value = yarn_config.pop(old_name)  # 弹出旧值
            if new_name is not None:  # 如果有对应的新名称
                config["rope_scaling"][new_name] = value  # 设置新名称的值

    assert len(yarn_config) == 0, f"Unparsed yarn config: {yarn_config}"  # 确保所有配置都已解析

    return config  # 返回重映射后的配置


def _remap_general_mistral_args(config: dict) -> dict:  # 重映射通用Mistral参数为HF格式
    # Mistral key -> HF key  # Mistral键到HF键的映射
    config_mapping = {  # 配置键映射
        "dim": "hidden_size",  # 维度映射为隐藏层大小
        "norm_eps": "rms_norm_eps",  # 归一化epsilon映射
        "n_kv_heads": "num_key_value_heads",  # KV头数映射
        "n_layers": "num_hidden_layers",  # 层数映射
        "n_heads": "num_attention_heads",  # 注意力头数映射
        "hidden_dim": "intermediate_size",  # 隐藏维度映射为中间层大小
    }
    # HF key -> (Mistral key, default value)  # HF键到(Mistral键, 默认值)的映射
    top_level_mapping_with_default = {  # 带默认值的顶层映射
        "model_type": ("model_type", "transformer"),  # 模型类型默认为transformer
        "hidden_act": ("activation", "silu"),  # 激活函数默认为silu
        "tie_word_embeddings": ("tied_embeddings", False),  # 词嵌入共享默认为False
        "max_seq_len": ("max_seq_len", 128_000),  # 最大序列长度默认128K
        "max_position_embeddings": ("max_position_embeddings", 128_000),  # 最大位置嵌入默认128K
    }

    for key, new_key in config_mapping.items():  # 遍历键映射
        if key in config:  # 如果旧键存在
            config[new_key] = config.pop(key)  # 重命名键

    for new_key, (key, default_value) in top_level_mapping_with_default.items():  # 遍历带默认值的映射
        config[new_key] = config.pop(key, default_value)  # 使用默认值填充缺失的键

    return config  # 返回重映射后的配置


def _remap_mistral_quantization_args(config: dict) -> dict:  # 重映射Mistral量化参数为HF格式
    if config.get("quantization"):  # 如果有量化配置
        quantization = config.pop("quantization", {})  # 弹出量化配置
        if quantization.get("qformat_weight") == "fp8_e4m3":  # 如果是FP8 E4M3量化
            qscheme_act = quantization.get("qscheme_act")  # 获取激活量化方案
            assert qscheme_act in (  # 仅支持NO_SCALES和TENSOR
                "NO_SCALES",
                "TENSOR",
                None,
            ), "Only NO_SCALES and TENSOR (default) are supported for qscheme_act"  # 仅支持NO_SCALES和TENSOR
            is_dynamic = qscheme_act == "NO_SCALES"  # 是否为动态量化
            config["quantization_config"] = {  # 设置量化配置
                "quant_method": "fp8",  # 量化方法为FP8
                "activation_scheme": "dynamic" if is_dynamic else "static",  # 激活方案
            }
        else:  # 不支持的量化格式
            raise ValueError(f"Found unknown quantization='{quantization}' in config")  # 抛出异常

    return config  # 返回重映射后的配置


def _remap_mistral_audio_args(config: dict) -> dict:  # 重映射Mistral音频模型参数为Voxtral/Whixtral格式
    whisper_args = config["multimodal"].pop("whisper_model_args")  # 弹出Whisper模型参数
    encoder_args = whisper_args["encoder_args"]  # 编码器参数
    downsample_args = whisper_args["downsample_args"]  # 下采样参数

    quant_config = config.get("quantization_config")  # 获取量化配置
    config = {  # 构造Voxtral格式配置
        "model_type": "whixtral",  # 模型类型为whixtral
        "architectures": ["VoxtralForConditionalGeneration"],  # Voxtral架构
        "text_config": PretrainedConfig.from_dict(config),  # 文本配置
        "audio_config": WhisperConfig(  # 音频配置
            num_mel_bins=encoder_args["audio_encoding_args"]["num_mel_bins"],  # 梅尔频率箱数
            window_size=encoder_args["audio_encoding_args"]["window_size"],  # 窗口大小
            sampling_rate=encoder_args["audio_encoding_args"]["sampling_rate"],  # 采样率
            hop_length=encoder_args["audio_encoding_args"]["hop_length"],  # 跳跃长度
            downsample_factor=downsample_args["downsample_factor"],  # 下采样因子
            d_model=encoder_args["dim"],  # 模型维度
            encoder_layers=encoder_args["n_layers"],  # 编码器层数
            encoder_ffn_dim=encoder_args["hidden_dim"],  # 编码器FFN维度
            encoder_attention_heads=encoder_args["n_heads"],  # 编码器注意力头数
            vocab_size=encoder_args["vocab_size"],  # 词汇表大小
            max_source_positions=encoder_args["max_source_positions"],  # 最大源位置
            is_encoder_decoder=False,  # Override WhisperConfig default  # 覆盖WhisperConfig默认值
        ),
    }
    if quant_config:  # 如果有量化配置
        config["quantization_config"] = quant_config  # 保留量化配置
    return config  # 返回重映射后的配置


def _remap_moe_args(config: dict) -> dict:  # 重映射MoE参数为DeepSeek格式
    moe_config_map = {  # MoE配置字段映射
        "route_every_n": "moe_layer_freq",  # 路由频率映射
        "first_k_dense_replace": "first_k_dense_replace",  # 首k密集替换
        "num_experts_per_tok": "num_experts_per_tok",  # 每token专家数
        "num_experts": "n_routed_experts",  # 路由专家数
        "expert_hidden_dim": "moe_intermediate_size",  # 专家中间层大小
        "routed_scale": "routed_scaling_factor",  # 路由缩放因子
        "num_shared_experts": "n_shared_experts",  # 共享专家数
        "num_expert_groups": "n_group",  # 专家组数
        "num_expert_groups_per_tok": "topk_group",  # 每token专家组数
    }
    moe_config = config.get("moe", {})  # 获取MoE配置
    for old_name, new_name in moe_config_map.items():  # 遍历映射
        if old_name in moe_config:  # 如果旧名称存在
            value = moe_config.pop(old_name)  # 弹出旧值
            config[new_name] = value  # 设置新名称的值

    config["topk_method"] = None  # TopK方法设为None
    config["scoring_func"] = "softmax"  # 评分函数设为softmax
    config["routing_method_type"] = 1  # RoutingMethodType.Renormalize  # 路由方法类型设为1（重归一化）

    return config  # 返回重映射后的配置


class MistralConfigParser:  # Mistral配置解析器类
    def get_hf_file_to_dict(  # 读取HF文件并解析为字典
        self, file_name: str, model: str | Path, revision: str | None = "main"  # 文件名、模型路径、版本
    ):
        file_path = Path(model) / file_name  # 构造文件路径
        if not file_path.is_file():  # 如果文件不存在
            raise FileNotFoundError(f"File not found {model}, {file_name}")  # 抛出文件未找到异常

        with open(file_path) as file:  # 打开文件
            return json.load(file)  # 解析并返回JSON字典

    def _download_mistral_config_file(self, model, revision) -> dict:  # 下载Mistral配置文件
        config_file_name = "params.json"  # Mistral配置文件名
        config_dict = self.get_hf_file_to_dict(config_file_name, model, revision)  # 读取配置文件
        if config_dict is None:  # 如果读取失败
            raise ValueError(  # 抛出异常
                f"Failed to load mistral '{config_file_name}' config for model "
                f"{model}. Please check if the model is a mistral-format model "
                f"and if the config file exists."  # 请检查模型格式和配置文件
            )
        assert isinstance(config_dict, dict)  # 确保配置是字典类型
        return config_dict  # 返回配置字典

    def parse(  # 解析Mistral模型配置
        self,
        model: str | Path,  # 模型路径
        revision: str | None = None,  # 模型版本
        **kwargs,  # 其他参数
    ) -> tuple[dict, PretrainedConfig]:  # 返回配置字典和配置对象
        config_dict = self._download_mistral_config_file(model, revision)  # 下载配置文件
        if config_dict.get("max_position_embeddings") is None:  # 如果缺少max_position_embeddings
            logger.warning(  # 记录警告
                "The params.json file is missing 'max_position_embeddings'"
                " and could not get a value from the HF config."
                " Defaulting to 128000"  # 默认使用128000
            )
            config_dict["max_position_embeddings"] = 128_000  # 设置默认值

        config_dict, config = adapt_config_dict(config_dict, model)  # 适配配置字典

        # Mistral configs may define sliding_window as list[int]. Convert it  # Mistral配置可能将sliding_window定义为int列表
        # to int and add the layer_types list[str] to make it HF compatible  # 转换为int并添加layer_types列表以兼容HF
        if (sliding_window := getattr(config, "sliding_window", None)) and isinstance(  # 如果有sliding_window且为列表
            sliding_window, list
        ):
            pattern_repeats = config.num_hidden_layers // len(sliding_window)  # 计算模式重复次数
            layer_types = sliding_window * pattern_repeats  # 扩展到所有层
            config.layer_types = [  # 创建层类型列表
                "full_attention" if layer_type is None else "sliding_attention"  # None为全注意力，否则为滑动注意力
                for layer_type in layer_types  # 遍历每层的滑动窗口值
            ]
            config.sliding_window = next(filter(None, sliding_window), None)  # 取第一个非None值

        return config_dict, config  # 返回配置字典和配置对象


def is_mistral_model(name) -> bool:  # 判断模型名称是否为需要自定义解析器的Mistral模型
    """Return True if *name* refers to a Mistral model needing the custom parser."""  # 如果名称指代需要自定义解析器的Mistral模型则返回True
    lower = str(name).lower()  # 转为小写
    if "mistral-large-3" in lower or "mistral-small-4" in lower or "leanstral" in lower:  # 已知的Mistral模型
        return True  # 返回True
    # EAGLE drafts for Mistral targets ship native-format only (params.json +  # Mistral目标的EAGLE草稿仅发布原生格式（params.json +
    # consolidated.safetensors, no config.json), so route them through the  # consolidated.safetensors，没有config.json），因此通过
    # custom parser regardless of the base model name.  # 自定义解析器路由，无论基础模型名称
    if "eagle" in lower and "mistral" in lower:  # Mistral的EAGLE变体
        return True  # 返回True
    return False  # 不是Mistral模型


@lru_cache(maxsize=2)  # LRU缓存，最多2个条目
def load_mistral_config(  # 加载并解析Mistral模型配置
    model_path: str,  # 模型路径
    trust_remote_code: bool = False,  # 是否信任远程代码
    revision: Optional[str] = None,  # 模型版本
):
    """Load and parse a Mistral model config via the custom params.json format.  # 通过自定义params.json格式加载并解析Mistral模型配置

    Returns a ``PretrainedConfig`` with dict sub-configs (text_config,  # 返回PretrainedConfig，其中字典子配置（text_config、
    vision_config) converted to proper AutoConfig objects.  # vision_config）已转换为正确的AutoConfig对象
    """
    local_path = download_from_hf(model_path)  # 下载模型文件
    parser = MistralConfigParser()  # 创建配置解析器
    config_dict, _ = parser.parse(local_path)  # 解析配置

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as f:  # 创建临时JSON文件
        json.dump(config_dict, f)  # 写入配置字典
        f.flush()  # 刷新缓冲区
        loaded_config = AutoConfig.from_pretrained(  # 通过AutoConfig加载
            f.name, trust_remote_code=trust_remote_code, revision=revision  # 使用临时文件路径
        )
    _ensure_sub_configs(loaded_config, "text_config", "vision_config")  # 确保子配置为AutoConfig对象

    return loaded_config  # 返回加载的配置


def wrap_as_pixtral(processor, config):  # 将分词器包装为PixtralProcessor以支持Mistral视觉模型
    """Wrap a tokenizer as a PixtralProcessor for Mistral vision models."""  # 将分词器包装为PixtralProcessor以支持Mistral视觉模型
    from transformers.models.pixtral.image_processing_pixtral import (  # 导入Pixtral图像处理器
        PixtralImageProcessor,
    )
    from transformers.models.pixtral.processing_pixtral import (  # 导入Pixtral处理器
        PixtralProcessor as HFPixtralProcessor,
    )

    vision_config = config.vision_config  # 获取视觉配置
    patch_size = vision_config.patch_size  # 获取补丁大小
    image_size = vision_config.image_size  # 获取图像大小
    spatial_merge_size = getattr(vision_config, "spatial_merge_size", 1)  # 获取空间合并大小

    effective_patch = patch_size * spatial_merge_size  # 计算有效补丁大小
    image_processor = PixtralImageProcessor(  # 创建图像处理器
        do_resize=True,  # 启用调整大小
        size={"longest_edge": image_size},  # 最长边尺寸
        patch_size={"height": effective_patch, "width": effective_patch},  # 补丁尺寸
    )
    return HFPixtralProcessor(  # 返回Pixtral处理器
        image_processor=image_processor,  # 图像处理器
        tokenizer=processor,  # 分词器
        patch_size=patch_size,  # 补丁大小
        spatial_merge_size=spatial_merge_size,  # 空间合并大小
    )


# kwargs that MistralCommon tokenizers reject.  # MistralCommon分词器拒绝的kwargs
_MISTRAL_COMMON_REJECTED_KWARGS = frozenset(  # MistralCommon拒绝的参数集合
    {
        "trust_remote_code",  # 信任远程代码
        "tokenizer_revision",  # 分词器版本
        "use_fast",  # 使用快速分词器
        "_from_auto",  # 自动类标志
        "clean_up_tokenization_spaces",  # 清理分词空格
    }
)

# Models whose tokenizer should be loaded from a different checkpoint.  # 需要从不同检查点加载分词器的模型
_MISTRAL_TOKENIZER_REDIRECTS = {  # 分词器重定向映射
    # TODO(Xinyuan): Remove this once we have a proper tokenizer for Devstral  # 待办：Devstral有正式分词器后移除
    "mistralai/Devstral-Small-2505": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",  # Devstral重定向到Mistral Small
}


def retry_without_mistral_common_kwargs(tokenizer_name, *args, **common_kwargs):  # 去除MistralCommon拒绝的参数后重试加载分词器
    """Retry ``AutoTokenizer.from_pretrained`` without kwargs that MistralCommon rejects.  # 去除MistralCommon拒绝的参数后重试AutoTokenizer.from_pretrained

    Returns the loaded tokenizer, or *None* if the error is not a  # 返回加载的分词器，如果错误不是
    MistralCommon kwargs rejection.  # MistralCommon参数拒绝则返回None
    """
    from transformers import AutoTokenizer  # 导入AutoTokenizer

    stripped = {  # 去除被拒绝的参数
        k: v
        for k, v in common_kwargs.items()
        if k not in _MISTRAL_COMMON_REJECTED_KWARGS  # 过滤掉被拒绝的参数
    }
    return AutoTokenizer.from_pretrained(tokenizer_name, *args, **stripped)  # 使用过滤后的参数重试


def patch_mistral_common_tokenizer(tokenizer):  # 修补MistralCommonTokenizer以兼容HF分词器API
    """Patch MistralCommonTokenizer/Backend to be compatible with HF tokenizer API.  # 修补MistralCommonTokenizer/Backend以兼容HF分词器API

    MistralCommon tokenizers (used by Voxtral, Pixtral, etc.) reject several  # MistralCommon分词器（Voxtral、Pixtral等使用）拒绝多个
    standard kwargs and lack some attributes that sglang expects.  We wrap the  # 标准参数且缺少sglang期望的一些属性。我们包装
    offending methods once at load time so that the rest of the codebase does  # 有问题的方法，在加载时执行一次，使代码库其余部分
    not need any special-casing.  # 不需要任何特殊处理
    """
    cls_name = type(tokenizer).__name__  # 获取分词器类名
    if "MistralCommon" not in cls_name:  # 如果不是MistralCommon分词器
        return tokenizer  # 直接返回
    if getattr(tokenizer, "_mistral_common_patched", False):  # 如果已经修补过
        return tokenizer  # 直接返回
    tokenizer._mistral_common_patched = True  # 标记为已修补

    if not hasattr(tokenizer, "get_added_vocab"):  # 如果缺少get_added_vocab方法
        tokenizer.get_added_vocab = lambda: {}  # 添加空实现

    # Set a chat_template containing "audio" so that sglang's content format  # 设置包含"audio"的chat_template，使sglang的内容格式
    # detector returns "openai" (which preserves audio_url extraction).  # 检测器返回"openai"（保留audio_url提取）
    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:  # 如果缺少chat_template
        tokenizer.chat_template = "<!-- audio/image multimodal -->"  # 设置多模态模板

    _orig_convert = tokenizer.convert_tokens_to_ids  # 保存原始的convert_tokens_to_ids方法

    def _safe_convert(val):  # 安全的token到ID转换，捕获AssertionError
        try:
            return _orig_convert(val)  # 调用原始方法
        except AssertionError:  # 捕获断言错误
            logger.debug(  # 记录调试信息
                "convert_tokens_to_ids failed for %r, returning unk_token_id", val  # 显示失败的token
            )
            return getattr(tokenizer, "unk_token_id", None)  # 返回未知token ID

    tokenizer.convert_tokens_to_ids = _safe_convert  # 替换为安全版本

    def _drop_kwargs(fn, keys):  # 创建去除指定参数的包装器
        def wrapper(*args, **kwargs):  # 包装函数
            for k in keys:  # 遍历要去除的参数
                kwargs.pop(k, None)  # 移除参数
            return fn(*args, **kwargs)  # 调用原始函数

        return wrapper  # 返回包装器

    tokenizer.decode = _drop_kwargs(tokenizer.decode, ["spaces_between_special_tokens"])  # 去除decode的无效参数
    tokenizer.batch_decode = _drop_kwargs(  # 去除batch_decode的无效参数
        tokenizer.batch_decode, ["spaces_between_special_tokens"]
    )

    tokenizer._orig_apply_chat_template = tokenizer.apply_chat_template  # 保存原始的apply_chat_template

    def _safe_apply_chat_template(messages, **kwargs):  # 安全的apply_chat_template实现
        kwargs.pop("add_generation_prompt", None)  # 移除不支持的参数
        cleaned = []  # 清理后的消息列表
        for msg in messages:  # 遍历消息
            if isinstance(msg, dict):  # 如果消息是字典
                content = msg.get("content", "")  # 获取内容
                if isinstance(content, list):  # 如果内容是列表（多模态）
                    text_parts = [  # 提取文本部分
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"  # 仅取文本类型
                    ]
                    msg = {**msg, "content": " ".join(text_parts) if text_parts else ""}  # 替换为纯文本
                cleaned.append(msg)  # 添加清理后的消息
            else:  # 非字典消息
                cleaned.append(msg)  # 直接添加
        return tokenizer._orig_apply_chat_template(cleaned, **kwargs)  # 调用原始方法

    tokenizer.apply_chat_template = _safe_apply_chat_template  # 替换为安全版本
