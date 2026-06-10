# BitsAndBytes 量化模块
# 本文件实现了 BitsAndBytes 量化方法，支持 4 位和 8 位量化。
# 包含配置类 BitsAndBytesConfig、线性层量化方法 BitsAndBytesLinearMethod
# 和 MoE 层量化方法 BitsAndBytesMoEMethod。
# 参考: https://arxiv.org/abs/2305.14314

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from: https://github.com/vllm-project/vllm/blob/d4d2751732c3ccae162a5a0160c7d4fe05d2779a/vllm/model_executor/layers/quantization/bitsandbytes.py
from __future__ import annotations  # 启用延迟注解评估

from typing import TYPE_CHECKING, Any, Optional  # 导入类型注解工具

import torch  # 导入 PyTorch 深度学习框架
from packaging import version  # 导入版本号解析工具

from sglang.srt.layers.linear import LinearBase  # 导入线性层基类
from sglang.srt.layers.quantization.base_config import (  # 导入量化配置基类
    FusedMoEMethodBase,  # 融合 MoE 方法基类
    LinearMethodBase,  # 线性方法基类
    QuantizationConfig,  # 量化配置基类
    QuantizeMethodBase,  # 量化方法基类
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod  # 导入未量化线性方法
from sglang.srt.utils import set_weight_attrs  # 导入权重属性设置工具
from sglang.srt.utils.custom_op import register_custom_op  # 导入自定义算子注册工具

if TYPE_CHECKING:  # 仅在类型检查时导入
    from sglang.srt.layers.moe.token_dispatcher import (  # MoE 令牌分发器类型
        CombineInput,  # 合并输入
        StandardDispatchOutput,  # 标准分发输出
    )


class BitsAndBytesConfig(QuantizationConfig):  # BitsAndBytes 量化配置类
    """Config class for BitsAndBytes Quantization.  # BitsAndBytes 量化配置类

    Reference: https://arxiv.org/abs/2305.14314  # 参考论文链接
    """

    def __init__(  # 初始化方法
        self,
        load_in_8bit: bool = False,  # 是否加载为 8 位量化，默认 False
        load_in_4bit: bool = True,  # 是否加载为 4 位量化，默认 True
        bnb_4bit_compute_dtype: str = "float32",  # 4 位量化的计算数据类型，默认 float32
        bnb_4bit_quant_storage: str = "uint8",  # 4 位量化的存储数据类型，默认 uint8
        bnb_4bit_quant_type: str = "fp4",  # 4 位量化类型，默认 fp4
        bnb_4bit_use_double_quant: bool = False,  # 是否使用双量化，默认 False
        llm_int8_enable_fp32_cpu_offload: bool = False,  # 是否启用 FP32 CPU 卸载，默认 False
        llm_int8_has_fp16_weight: bool = False,  # 8 位量化是否有 FP16 权重，默认 False
        llm_int8_skip_modules: list[str] | None = None,  # 跳过量化的模块列表
        llm_int8_threshold: float = 6.0,  # 8 位量化的阈值，默认 6.0
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.load_in_8bit = load_in_8bit  # 保存 8 位加载标志
        self.load_in_4bit = load_in_4bit  # 保存 4 位加载标志
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype  # 保存计算数据类型
        self.bnb_4bit_quant_storage = bnb_4bit_quant_storage  # 保存存储数据类型
        self.bnb_4bit_quant_type = bnb_4bit_quant_type  # 保存量化类型
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant  # 保存双量化标志
        self.llm_int8_enable_fp32_cpu_offload = llm_int8_enable_fp32_cpu_offload  # 保存 CPU 卸载标志
        self.llm_int8_has_fp16_weight = llm_int8_has_fp16_weight  # 保存 FP16 权重标志
        self.llm_int8_skip_modules = llm_int8_skip_modules or []  # 保存跳过模块列表
        self.llm_int8_threshold = llm_int8_threshold  # 保存量化阈值

        if self.bnb_4bit_quant_storage not in ["uint8"]:  # 检查存储类型是否支持
            raise ValueError(  # 抛出不支持的存储类型错误
                f"Unsupported bnb_4bit_quant_storage: {self.bnb_4bit_quant_storage}"  # 不支持的 4 位量化存储类型
            )

    def __repr__(self) -> str:  # 返回配置对象的字符串表示
        return (  # 返回格式化字符串
            f"BitsAndBytesConfig(load_in_8bit={self.load_in_8bit}, "  # 8 位加载标志
            f"load_in_4bit={self.load_in_4bit}, "  # 4 位加载标志
            f"bnb_4bit_compute_dtype={self.bnb_4bit_compute_dtype}, "  # 计算数据类型
            f"bnb_4bit_quant_storage={self.bnb_4bit_quant_storage}, "  # 存储数据类型
            f"bnb_4bit_quant_type={self.bnb_4bit_quant_type}, "  # 量化类型
            f"llm_int8_skip_modules={self.llm_int8_skip_modules})"  # 跳过模块列表
        )

    def get_name(self) -> str:  # 获取量化方法名称
        return "bitsandbytes"  # 返回名称

    def get_scaled_act_names(self) -> list[str]:  # 获取需要后缩放的激活函数名
        return []  # BitsAndBytes 不需要后缩放

    def get_supported_act_dtypes(self) -> list[torch.dtype]:  # 获取支持的激活数据类型
        return [torch.float32, torch.float16, torch.bfloat16]  # 支持 float32、float16 和 bfloat16

    @classmethod
    def get_min_capability(cls) -> int:  # 获取最低 GPU 计算能力要求
        return 70  # 最低需要计算能力 7.0

    @staticmethod
    def get_config_filenames() -> list[str]:  # 获取配置文件名列表
        return []  # 无配置文件

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BitsAndBytesConfig":  # 从配置字典创建配置对象
        def get_safe_value(config, keys, default_value=None):  # 安全获取配置值的内部函数
            try:  # 尝试获取值
                value = QuantizationConfig.get_from_keys(config, keys)  # 从配置中按键获取
                return value if value is not None else default_value  # 值非空则返回，否则返回默认值
            except ValueError:  # 键不存在时
                return default_value  # 返回默认值

        load_in_8bit = get_safe_value(config, ["load_in_8bit"], default_value=False)  # 获取 8 位加载标志
        load_in_4bit = get_safe_value(config, ["load_in_4bit"], default_value=True)  # 获取 4 位加载标志
        bnb_4bit_compute_dtype = get_safe_value(  # 获取计算数据类型
            config, ["bnb_4bit_compute_dtype"], default_value="float32"  # 默认 float32
        )
        bnb_4bit_quant_storage = get_safe_value(  # 获取存储数据类型
            config, ["bnb_4bit_quant_storage"], default_value="uint8"  # 默认 uint8
        )
        bnb_4bit_quant_type = get_safe_value(  # 获取量化类型
            config, ["bnb_4bit_quant_type"], default_value="fp4"  # 默认 fp4
        )
        bnb_4bit_use_double_quant = get_safe_value(  # 获取双量化标志
            config, ["bnb_4bit_use_double_quant"], default_value=False  # 默认 False
        )
        llm_int8_enable_fp32_cpu_offload = get_safe_value(  # 获取 CPU 卸载标志
            config, ["llm_int8_enable_fp32_cpu_offload"], default_value=False  # 默认 False
        )
        llm_int8_has_fp16_weight = get_safe_value(  # 获取 FP16 权重标志
            config, ["llm_int8_has_fp16_weight"], default_value=False  # 默认 False
        )
        llm_int8_skip_modules = get_safe_value(  # 获取跳过模块列表
            config, ["llm_int8_skip_modules"], default_value=[]  # 默认空列表
        )
        llm_int8_threshold = get_safe_value(  # 获取量化阈值
            config, ["llm_int8_threshold"], default_value=6.0  # 默认 6.0
        )

        return cls(  # 返回新创建的配置对象
            load_in_8bit=load_in_8bit,  # 8 位加载标志
            load_in_4bit=load_in_4bit,  # 4 位加载标志
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,  # 计算数据类型
            bnb_4bit_quant_storage=bnb_4bit_quant_storage,  # 存储数据类型
            bnb_4bit_quant_type=bnb_4bit_quant_type,  # 量化类型
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,  # 双量化标志
            llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload,  # CPU 卸载标志
            llm_int8_has_fp16_weight=llm_int8_has_fp16_weight,  # FP16 权重标志
            llm_int8_skip_modules=llm_int8_skip_modules,  # 跳过模块列表
            llm_int8_threshold=llm_int8_threshold,  # 量化阈值
        )

    def get_quant_method(  # 获取适用于指定层的量化方法
        self, layer: torch.nn.Module, prefix: str  # 目标层和层前缀
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # 导入融合 MoE 层

        if isinstance(layer, LinearBase):  # 如果是线性层
            if is_layer_skipped_bnb(prefix, self.llm_int8_skip_modules):  # 检查是否跳过该层
                return UnquantizedLinearMethod()  # 返回未量化的线性方法
            return BitsAndBytesLinearMethod(self)  # 返回 BitsAndBytes 线性方法
        elif isinstance(layer, FusedMoE):  # 如果是融合 MoE 层
            return BitsAndBytesMoEMethod(self)  # 返回 BitsAndBytes MoE 方法
        return None  # 不支持的层返回 None


def is_layer_skipped_bnb(prefix: str, llm_int8_skip_modules: list[str]):  # 检查层是否在跳过列表中
    # Split the prefix into its dot-separated components  # 将前缀按点号分隔为组件
    components = prefix.split(".")  # 分割前缀字符串

    # Check if any of the skip modules exactly matches any component  # 检查跳过模块是否与任何组件完全匹配
    substr_check = any(  # 子串检查
        module_name in components for module_name in llm_int8_skip_modules  # 遍历跳过模块
    )

    # Allow certain layers to not be quantized  # 允许某些层不被量化
    set_components = set(".".join(components[: i + 1]) for i in range(len(components)))  # 生成所有前缀路径集合
    set_llm_int8_skip_modules = set(llm_int8_skip_modules)  # 将跳过模块转为集合
    prefix_check = len(set_llm_int8_skip_modules & set_components) != 0  # 检查前缀是否与跳过模块有交集

    return substr_check or prefix_check  # 返回子串检查或前缀检查的结果


def calculate_quant_ratio(dtype):  # 计算量化比率
    if dtype.is_floating_point:  # 如果是浮点类型
        return torch.finfo(dtype).bits // torch.iinfo(torch.uint8).bits  # 浮点类型位数除以 uint8 位数
    else:  # 如果是整型
        return torch.iinfo(dtype).bits // torch.iinfo(torch.uint8).bits  # 整型位数除以 uint8 位数


class BitsAndBytesLinearMethod(LinearMethodBase):  # BitsAndBytes 线性层量化方法
    """Linear method for BitsAndBytes.  # BitsAndBytes 的线性方法

    Args:  # 参数
       quant_config: The BitsAndBytes quantization config.  # BitsAndBytes 量化配置
    """

    def __init__(self, quant_config: BitsAndBytesConfig):  # 初始化方法
        try:  # 尝试导入 bitsandbytes
            import bitsandbytes  # 导入 bitsandbytes 库

            if version.parse(bitsandbytes.__version__) < version.parse("0.46.1"):  # 检查版本号
                raise ImportError(  # 版本不满足时抛出错误
                    "bitsandbytes version is wrong. Please "  # bitsandbytes 版本不正确，请
                    "install bitsandbytes>=0.46.1."  # 安装 bitsandbytes>=0.46.1
                )
        except ImportError as err:  # 捕获导入错误
            raise ImportError(  # 抛出安装提示错误
                "Please install bitsandbytes>=0.46.1 via "  # 请通过以下方式安装 bitsandbytes>=0.46.1
                "`pip install bitsandbytes>=0.46.1` to use "  # pip install bitsandbytes>=0.46.1 以使用
                "bitsandbytes quantizer."  # bitsandbytes 量化器
            ) from err

        self.quant_config = quant_config  # 保存量化配置

    def create_weights(  # 创建线性层权重
        self,
        layer: torch.nn.Module,  # 目标层
        input_size_per_partition: int,  # 当前分区的输入维度大小
        output_partition_sizes: list[int],  # 各逻辑权重的输出维度大小列表
        input_size: int,  # 跨所有秩的输入维度总大小
        output_size: int,  # 跨所有秩的输出维度总大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        from bitsandbytes.nn import Int8Params  # 导入 Int8 参数类

        def create_qweight_for_8bit():  # 创建 8 位量化权重的内部函数
            qweight = Int8Params(  # 创建 Int8 参数
                data=torch.empty(  # 创建空张量
                    sum(output_partition_sizes),  # 输出维度总和
                    input_size_per_partition,  # 输入维度
                    dtype=torch.int8,  # 数据类型为 int8
                ),
                has_fp16_weights=self.quant_config.llm_int8_has_fp16_weight,  # 是否有 FP16 权重
                requires_grad=False,  # 不需要梯度
            )
            set_weight_attrs(  # 设置权重属性
                qweight,
                {  # 属性字典
                    "input_dim": 0,  # 输入维度索引
                    "output_dim": 0,  # 输出维度索引
                    "pack_factor": 1,  # 打包因子为 1
                    "use_bitsandbytes_8bit": True,  # 标记使用 bitsandbytes 8 位量化
                    "generation": 0,  # 生成计数器
                },
            )
            return qweight  # 返回 8 位量化权重

        def create_qweight_for_4bit():  # 创建 4 位量化权重的内部函数
            quant_ratio = calculate_quant_ratio(params_dtype)  # 计算量化比率

            total_size = input_size_per_partition * sum(output_partition_sizes)  # 计算总大小
            if total_size % quant_ratio != 0:  # 检查是否可整除
                raise ValueError(  # 不可整除时报错
                    "The input size is not aligned with the quantized weight shape."  # 输入大小与量化权重形状不对齐
                )

            qweight = torch.nn.Parameter(  # 创建参数张量
                torch.empty(total_size // quant_ratio, 1, dtype=torch.uint8),  # 按量化比率缩小的空张量
                requires_grad=False,  # 不需要梯度
            )
            set_weight_attrs(  # 设置权重属性
                qweight,
                {  # 属性字典
                    "input_dim": 0,  # 输入维度索引
                    "output_dim": 0,  # 输出维度索引
                    "pack_factor": quant_ratio,  # 打包因子
                    "use_bitsandbytes_4bit": True,  # 标记使用 bitsandbytes 4 位量化
                },
            )
            return qweight  # 返回 4 位量化权重

        if self.quant_config.load_in_8bit:  # 如果使用 8 位量化
            qweight = create_qweight_for_8bit()  # 创建 8 位量化权重
        else:  # 否则使用 4 位量化
            qweight = create_qweight_for_4bit()  # 创建 4 位量化权重
        # Enable parameters to have the same name as in the BNB  # 使参数名称与 BNB 检查点格式一致
        # checkpoint format.  # 检查点格式
        layer.register_parameter("weight", qweight)  # 在层上注册权重参数
        set_weight_attrs(qweight, extra_weight_attrs)  # 设置额外权重属性

    def apply(  # 应用量化方法进行前向计算
        self,
        layer: torch.nn.Module,  # 目标层
        x: torch.Tensor,  # 输入张量
        bias: torch.Tensor | None = None,  # 偏置张量，可选
    ) -> torch.Tensor:
        if self.quant_config.load_in_8bit:  # 如果使用 8 位量化
            return self._apply_8bit_weight(layer, x, bias)  # 调用 8 位权重应用方法
        else:  # 否则使用 4 位量化
            return self._apply_4bit_weight(layer, x, bias)  # 调用 4 位权重应用方法

    def _apply_8bit_weight(  # 应用 8 位量化权重
        self,
        layer: torch.nn.Module,  # 目标层
        x: torch.Tensor,  # 输入张量
        bias: torch.Tensor | None = None,  # 偏置张量，可选
    ) -> torch.Tensor:
        # only load the bitsandbytes module when needed  # 仅在需要时加载 bitsandbytes 模块
        from bitsandbytes import MatmulLtState, matmul  # 导入 bitsandbytes 矩阵乘法相关类

        original_type = x.dtype  # 保存原始数据类型
        original_shape = x.shape  # 保存原始形状
        reshape_after_matmul = False  # 是否需要在矩阵乘法后重塑
        if x.ndim > 2:  # 如果输入维度大于 2
            x = x.reshape(-1, x.size(-1))  # 重塑为二维
            reshape_after_matmul = True  # 标记需要重塑
        bf_x = x.to(torch.bfloat16)  # 转换为 bfloat16

        qweight = layer.weight  # 获取量化权重
        offsets = qweight.bnb_shard_offsets  # 获取分片偏移量
        quant_states = qweight.bnb_quant_state  # 获取量化状态
        matmul_states = qweight.matmul_state  # 获取矩阵乘法状态
        generation = qweight.generation  # 获取生成计数器

        out_dim_0 = x.shape[0]  # 输出维度 0
        out_dim_1 = sum(  # 输出维度 1，汇总所有分片的输出大小
            [quant_state[1].shape[0] for quant_state in quant_states.items()]  # 遍历量化状态获取形状
        )
        out = torch.empty(out_dim_0, out_dim_1, dtype=torch.float16, device=x.device)  # 创建输出张量

        current_index = 0  # 当前输出索引
        for i in range(len(quant_states)):  # 遍历所有量化状态
            output_size = quant_states[i].shape[0]  # 当前分片的输出大小

            # in profile_run or the first generation of inference,  # 在 profile_run 或首次推理时
            # create new matmul_states  # 创建新的矩阵乘法状态
            if generation == 0 or generation == 1:  # 首次或第二次生成
                matmul_states[i] = MatmulLtState()  # 创建新的 MatmulLtState
                matmul_states[i].CB = qweight[offsets[i] : offsets[i + 1]]  # 设置 CB 矩阵
                matmul_states[i].SCB = quant_states[i].to(x.device)  # 设置 SCB 缩放因子
                matmul_states[i].threshold = self.quant_config.llm_int8_threshold  # 设置阈值
                matmul_states[i].has_fp16_weights = (  # 设置是否有 FP16 权重
                    self.quant_config.llm_int8_has_fp16_weight  # 从配置中获取
                )
                matmul_states[i].is_training = False  # 设置为推理模式
                if (  # 如果满足条件
                    matmul_states[i].threshold > 0.0  # 阈值大于 0
                    and not matmul_states[i].has_fp16_weights  # 且没有 FP16 权重
                ):
                    matmul_states[i].use_pool = True  # 启用内存池

            new_x = bf_x.unsqueeze(0)  # 增加一个批次维度

            out[:, current_index : current_index + output_size] = matmul(  # 执行矩阵乘法
                new_x, qweight[offsets[i] : offsets[i + 1]], state=matmul_states[i]  # 传入输入、权重和状态
            )

            current_index += output_size  # 更新输出索引

            # only update the matmul_states if it is not profile_run  # 仅在非 profile_run 时更新矩阵乘法状态
            if (  # 如果满足条件
                generation > 0  # 非首次生成
                and not self.quant_config.llm_int8_has_fp16_weight  # 没有 FP16 权重
                and matmul_states[i].CB is not None  # CB 矩阵非空
                and matmul_states[i].CxB is not None  # CxB 矩阵非空
            ):
                del matmul_states[i].CB  # 删除 CB 矩阵以释放内存
                qweight[offsets[i] : offsets[i + 1]] = matmul_states[i].CxB  # 用 CxB 替换权重

        out = out.to(original_type)  # 转回原始数据类型

        if reshape_after_matmul:  # 如果需要重塑
            out = out.view(*original_shape[:-1], out.size(-1))  # 恢复原始形状

        if bias is not None:  # 如果有偏置
            out += bias  # 添加偏置

        qweight.generation += 1  # 增加生成计数器

        return out  # 返回输出

    def _apply_4bit_weight(  # 应用 4 位量化权重
        self,
        layer: torch.nn.Module,  # 目标层
        x: torch.Tensor,  # 输入张量
        bias: torch.Tensor | None = None,  # 偏置张量，可选
    ) -> torch.Tensor:
        original_type = x.dtype  # 保存原始数据类型
        original_shape = x.shape  # 保存原始形状
        reshape_after_matmul = False  # 是否需要在矩阵乘法后重塑
        if x.ndim > 2:  # 如果输入维度大于 2
            x = x.reshape(-1, x.size(-1))  # 重塑为二维
            reshape_after_matmul = True  # 标记需要重塑
        bf_x = x.to(torch.bfloat16)  # 转换为 bfloat16

        qweight = layer.weight  # 获取量化权重
        quant_states = qweight.bnb_quant_state  # 获取量化状态
        offsets = qweight.bnb_shard_offsets  # 获取分片偏移量

        out_dim_0 = x.shape[0]  # 输出维度 0
        out_dim_1 = sum(  # 输出维度 1，汇总所有分片的输出大小
            [quant_state[1].shape[0] for quant_state in quant_states.items()]  # 遍历量化状态获取形状
        )
        out = torch.empty(out_dim_0, out_dim_1, dtype=torch.bfloat16, device=x.device)  # 创建输出张量
        apply_bnb_4bit(bf_x, qweight, offsets, out)  # 调用 4 位量化矩阵乘法
        out = out.to(original_type)  # 转回原始数据类型

        if reshape_after_matmul:  # 如果需要重塑
            out = out.view(*original_shape[:-1], out.size(-1))  # 恢复原始形状

        if bias is not None:  # 如果有偏置
            out += bias  # 添加偏置

        return out  # 返回输出


@register_custom_op(mutates_args=["out"])  # 注册为自定义算子，声明 out 参数会被修改
def apply_bnb_4bit(  # 应用 4 位量化矩阵乘法
    x: torch.Tensor,  # 输入张量
    weight: torch.Tensor,  # 权重张量
    offsets: torch.Tensor,  # 分片偏移量
    out: torch.Tensor,  # 输出张量
) -> None:
    # only load the bitsandbytes module when needed  # 仅在需要时加载 bitsandbytes 模块
    from bitsandbytes import matmul_4bit  # 导入 4 位矩阵乘法函数

    quant_states = weight.bnb_quant_state  # 获取量化状态
    current_index = 0  # 当前输出索引
    for i in range(len(quant_states)):  # 遍历所有量化状态
        output_size = quant_states[i].shape[0]  # 当前分片的输出大小
        # It is more efficient to use out kwarg like  # 使用 out 关键字参数更高效，如
        # matmul_4bit(..., out = ...).  Infeasible now due to the bug  # matmul_4bit(..., out = ...)。但由于 bug 目前不可行
        # https://github.com/TimDettmers/bitsandbytes/issues/1235.  # bitsandbytes 仓库的 issue
        # Need to change  after the bug is fixed.  # bug 修复后需要更改
        out[:, current_index : current_index + output_size] = matmul_4bit(  # 执行 4 位矩阵乘法
            x, weight[offsets[i] : offsets[i + 1]].t(), quant_states[i]  # 输入、权重转置和量化状态
        )
        current_index += output_size  # 更新输出索引


class BitsAndBytesMoEMethod(FusedMoEMethodBase):  # BitsAndBytes MoE 层量化方法
    """MoE method for BitsAndBytes.  # BitsAndBytes 的 MoE 方法

    Args:  # 参数
       quant_config: The BitsAndBytes quantization config.  # BitsAndBytes 量化配置
    """

    def __init__(  # 初始化方法
        self,
        quant_config: BitsAndBytesConfig,  # 量化配置
    ):
        super().__init__()  # 调用父类初始化
        try:  # 尝试导入 bitsandbytes
            import bitsandbytes  # 导入 bitsandbytes 库

            if version.parse(bitsandbytes.__version__) < version.parse("0.46.1"):  # 检查版本号
                raise ImportError(  # 版本不满足时抛出错误
                    "bitsandbytes version is wrong. Please "  # bitsandbytes 版本不正确，请
                    "install bitsandbytes>=0.46.1."  # 安装 bitsandbytes>=0.46.1
                )
        except ImportError as err:  # 捕获导入错误
            raise ImportError(  # 抛出安装提示错误
                "Please install bitsandbytes>=0.46.1 via "  # 请通过以下方式安装 bitsandbytes>=0.46.1
                "`pip install bitsandbytes>=0.46.1` to use "  # pip install bitsandbytes>=0.46.1 以使用
                "bitsandbytes quantizer."  # bitsandbytes 量化器
            ) from err
        self.quant_config = quant_config  # 保存量化配置

    def create_weights(  # 创建 MoE 层权重
        self,
        layer: torch.nn.Module,  # 目标层
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size_per_partition: int,  # 当前分区的中间层大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        if self.quant_config.load_in_8bit:  # 如果使用 8 位量化
            call_fun = self._create_weights_8bit  # 使用 8 位权重创建函数
        else:  # 否则使用 4 位量化
            call_fun = self._create_weights_4bit  # 使用 4 位权重创建函数
        call_fun(  # 调用对应的权重创建函数
            layer,  # 目标层
            num_experts,  # 专家数量
            hidden_size,  # 隐藏层大小
            intermediate_size_per_partition,  # 中间层大小
            params_dtype,  # 参数数据类型
            **extra_weight_attrs,  # 额外权重属性
        )

    def create_moe_runner(self, layer: torch.nn.Module, moe_runner_config):  # 创建 MoE 运行器
        self.moe_runner_config = moe_runner_config  # 保存 MoE 运行器配置

    def apply(  # 应用 MoE 量化方法
        self,
        layer: torch.nn.Module,  # 目标层
        dispatch_output: StandardDispatchOutput,  # 标准分发输出
    ) -> CombineInput:
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe  # 导入融合 MoE 函数
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput  # 导入标准合并输入

        x = dispatch_output.hidden_states  # 获取隐藏状态
        topk_output = dispatch_output.topk_output  # 获取 top-k 输出

        # TODO(bnell): Do these need to be called on the hot path?  # TODO(bnell): 这些是否需要在热路径上调用？
        if self.quant_config.load_in_8bit:  # 如果使用 8 位量化
            w13, w2 = self._apply_8bit_dequant(layer)  # 8 位反量化
        else:  # 否则使用 4 位量化
            w13, w2 = self._apply_4bit_dequant(layer)  # 4 位反量化

        moe_runner_config = self.moe_runner_config  # 获取 MoE 运行器配置
        output = fused_moe(  # 执行融合 MoE 计算
            hidden_states=x,  # 隐藏状态
            w1=w13,  # gate_up 权重
            w2=w2,  # down 权重
            topk_output=topk_output,  # top-k 输出
            moe_runner_config=moe_runner_config,  # MoE 运行器配置
        )
        return StandardCombineInput(hidden_states=output)  # 返回标准合并输入

    def _create_weights_4bit(  # 创建 4 位量化权重
        self,
        layer: torch.nn.Module,  # 目标层
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size_per_partition: int,  # 当前分区的中间层大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        quant_ratio = calculate_quant_ratio(params_dtype)  # 计算量化比率
        # Fused gate_up_proj (column parallel)  # 融合 gate_up_proj（列并行）
        w13_total_size = (  # 计算 w13 总大小
            hidden_size * 2 * intermediate_size_per_partition  # 隐藏大小 * 2 * 中间大小
        ) // quant_ratio  # 除以量化比率
        w13_qweight = torch.nn.Parameter(  # 创建 w13 量化权重参数
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                w13_total_size,  # 总大小维度
                1,  # 第三维度
                dtype=torch.uint8,  # 数据类型为 uint8
            ),
            requires_grad=False,  # 不需要梯度
        )
        layer.register_parameter("w13_weight", w13_qweight)  # 注册 w13 权重参数
        set_weight_attrs(w13_qweight, extra_weight_attrs)  # 设置额外属性
        set_weight_attrs(  # 设置 w13 权重属性
            w13_qweight,
            {  # 属性字典
                "num_experts": num_experts,  # 专家数量
                "input_dim": hidden_size,  # 输入维度
                "output_dim": 2 * intermediate_size_per_partition,  # 输出维度
                "experts_shape": (  # 专家形状
                    num_experts,  # 专家数量
                    intermediate_size_per_partition * 2,  # 中间大小 * 2
                    hidden_size,  # 隐藏大小
                ),
                "pack_factor": quant_ratio,  # 打包因子
                "use_bitsandbytes_4bit": True,  # 标记使用 4 位量化
            },
        )
        # down_proj (row parallel)  # down_proj（行并行）
        w2_total_size = (hidden_size * intermediate_size_per_partition) // quant_ratio  # 计算 w2 总大小
        w2_qweight = torch.nn.Parameter(  # 创建 w2 量化权重参数
            torch.empty(  # 创建空张量
                num_experts,  # 专家数量维度
                w2_total_size,  # 总大小维度
                1,  # 第三维度
                dtype=torch.uint8,  # 数据类型为 uint8
            ),
            requires_grad=False,  # 不需要梯度
        )
        set_weight_attrs(  # 设置 w2 权重属性
            w2_qweight,
            {  # 属性字典
                "num_experts": num_experts,  # 专家数量
                "input_dim": intermediate_size_per_partition,  # 输入维度
                "output_dim": hidden_size,  # 输出维度
                "experts_shape": (  # 专家形状
                    num_experts,  # 专家数量
                    hidden_size,  # 隐藏大小
                    intermediate_size_per_partition,  # 中间大小
                ),
                "pack_factor": quant_ratio,  # 打包因子
                "use_bitsandbytes_4bit": True,  # 标记使用 4 位量化
            },
        )
        layer.register_parameter("w2_weight", w2_qweight)  # 注册 w2 权重参数
        set_weight_attrs(w2_qweight, extra_weight_attrs)  # 设置额外属性

    def _create_weights_8bit(  # 创建 8 位量化权重
        self,
        layer: torch.nn.Module,  # 目标层
        num_experts: int,  # 专家数量
        hidden_size: int,  # 隐藏层大小
        intermediate_size_per_partition: int,  # 当前分区的中间层大小
        params_dtype: torch.dtype,  # 参数数据类型
        **extra_weight_attrs,  # 额外权重属性
    ):
        raise NotImplementedError  # 8 位 MoE 权重创建尚未实现

    def _apply_4bit_dequant(  # 4 位量化权重反量化
        self, layer: torch.nn.Module  # 目标层
    ) -> tuple[torch.Tensor, torch.Tensor]:  # 返回 w13 和 w2 张量
        from bitsandbytes.functional import dequantize_4bit  # 导入 4 位反量化函数

        w13 = dequantize_4bit(  # 反量化 w13 权重
            layer.w13_weight.reshape(-1, 1),  # 重塑为一维
            layer.w13_weight.bnb_quant_state,  # 量化状态
        )
        w2 = dequantize_4bit(  # 反量化 w2 权重
            layer.w2_weight.reshape(-1, 1),  # 重塑为一维
            layer.w2_weight.bnb_quant_state,  # 量化状态
        )
        w13 = w13.reshape(layer.w13_weight.experts_shape)  # 恢复 w13 专家形状
        w2 = w2.reshape(layer.w2_weight.experts_shape)  # 恢复 w2 专家形状
        return w13, w2  # 返回反量化后的权重

    def _apply_8bit_dequant(  # 8 位量化权重反量化
        self, layer: torch.nn.Module  # 目标层
    ) -> tuple[torch.Tensor, torch.Tensor]:  # 返回 w13 和 w2 张量
        raise NotImplementedError  # 8 位 MoE 反量化尚未实现
