# DeepGEMM MoE Runner模块：实现基于DeepGEMM库的MoE计算核心，支持FP8/BF16的连续GEMM和掩码GEMM模式，
# 以及与standard/deepep_normal/deepep_ll调度后端之间的pre-permute和post-permute转换函数。
from __future__ import annotations  # 启用延迟类型注解求值

from dataclasses import dataclass  # 数据类装饰器
from typing import TYPE_CHECKING, Any, List, Optional, Tuple  # 类型提示工具

import einops  # 张量重排库
import torch  # PyTorch深度学习框架

from sglang.jit_kernel.dsv4 import silu_and_mul_masked_post_quant  # DSV4 SiLU+Mul+量化JIT内核
from sglang.srt.environ import envs  # 环境变量配置
from sglang.srt.layers import deep_gemm_wrapper  # DeepGEMM封装模块
from sglang.srt.layers.moe.moe_runner.base import (  # 从基模块导入MoE Runner基础类
    MoeQuantInfo,  # MoE量化信息基类
    MoeRunnerConfig,  # MoE Runner配置类
    MoeRunnerCore,  # MoE Runner核心基类
    RunnerInput,  # Runner输入基类
    RunnerOutput,  # Runner输出基类
    register_post_permute,  # 注册后置换装饰器
    register_pre_permute,  # 注册前置换装饰器
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend  # MoE Runner后端枚举
from sglang.srt.utils import (  # 工具函数
    ceil_div,  # 向上取整除法
    dispose_tensor,  # 释放张量
    get_bool_env_var,  # 获取布尔环境变量
    is_cuda,  # 判断是否为CUDA设备
    is_hip,  # 判断是否为HIP（AMD）设备
    is_musa,  # 判断是否为MUSA设备
    is_npu,  # 判断是否为NPU设备
)
from sglang.srt.utils.offloader import get_offloader  # 获取卸载器实例

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.layers.moe.token_dispatcher.deepep import (  # DeepEP调度器类型
        DeepEPLLCombineInput,  # DeepEP低延迟Combine输入
        DeepEPLLDispatchOutput,  # DeepEP低延迟调度输出
        DeepEPNormalCombineInput,  # DeepEP普通Combine输入
        DeepEPNormalDispatchOutput,  # DeepEP普通调度输出
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (  # 标准调度器类型
        StandardCombineInput,  # 标准Combine输入
        StandardDispatchOutput,  # 标准调度输出
    )

_is_hip = is_hip()  # 全局标记：当前是否为HIP环境
_is_npu = is_npu()  # 全局标记：当前是否为NPU环境
_is_cuda = is_cuda()  # 全局标记：当前是否为CUDA环境
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip  # 是否使用AITER（仅HIP）
_is_musa = is_musa()  # 全局标记：当前是否为MUSA环境

# Imported only for the SGLANG_OPT_FIX_MEGA_MOE_MEMORY=False fallback path.  # 仅在SGLANG_OPT_FIX_MEGA_MOE_MEMORY=False回退路径中导入。
if not (_is_npu or _is_hip) and _is_cuda:  # CUDA环境且非NPU/HIP
    from sglang.jit_kernel.activation import silu_and_mul as _legacy_silu_and_mul  # 旧版SiLU+Mul内核
elif _is_musa:  # MUSA环境
    _silu_and_mul_musa = torch.nn.SwishGLU()  # MUSA的SwishGLU实现
else:  # 其他环境
    _legacy_silu_and_mul = None  # 不导入


_MASKED_GEMM_FAST_ACT = get_bool_env_var("SGLANG_MASKED_GEMM_FAST_ACT")  # 是否使用掩码GEMM快速激活
_DEEPGEMM_ON_H20 = get_bool_env_var("SGLANG_DEEPGEMM_ON_H20")  # 是否在H20 GPU上使用DeepGEMM


# TODO(kaixih@nvidia): ideally we should merge this logic into  # TODO(kaixih@nvidia)：理想情况下应将此逻辑合并到
# `fill_gateup_input_triton_kernel` to directly generate e8m0 scale.  # `fill_gateup_input_triton_kernel`中，直接生成e8m0缩放。
@torch.compile(disable=_is_hip or _is_npu)  # torch.compile优化（HIP/NPU上禁用）
def _cast_to_e8m0_with_rounding_up(x: torch.Tensor) -> torch.Tensor:  # 将缩放因子转换为e8m0格式（带上取整）
    temp = x.to(torch.float32).view(torch.int32)  # 将输入视为float32再转为int32位模式
    exp = torch.bitwise_right_shift(temp, 23)  # 提取指数部分（右移23位）
    mant = torch.bitwise_and(temp, 0x7FFFFF)  # 提取尾数部分（低23位）
    is_ru = torch.logical_and(  # 判断是否需要上取整
        torch.logical_and((mant > 0), (exp != 0xFE)),  # 有尾数且非无穷
        ~torch.logical_and((exp == 0), (mant <= 0x400000)),  # 非次正规数或尾数足够大
    )
    exp = torch.where(is_ru, exp + 1, exp)  # 需要上取整时指数加1
    new_x = exp.to(torch.uint8).view(torch.int)  # 转为uint8再视为int类型
    return new_x.transpose(1, 2).contiguous().transpose(1, 2)  # 转置以确保内存连续


def copy_list_to_gpu_no_ce(arr: List[int]):  # 将整数列表复制到GPU（不使用拷贝引擎）
    from sgl_kernel.elementwise import copy_to_gpu_no_ce  # 导入无拷贝引擎的GPU复制函数

    tensor_cpu = torch.tensor(arr, dtype=torch.int32, device="cpu")  # 创建CPU张量
    tensor_gpu = torch.empty_like(tensor_cpu, device="cuda")  # 分配GPU内存
    copy_to_gpu_no_ce(tensor_cpu, tensor_gpu)  # 不使用拷贝引擎复制到GPU
    return tensor_gpu  # 返回GPU张量


@dataclass
class DeepGemmRunnerInput(RunnerInput):  # DeepGEMM Runner输入数据类
    hidden_states: torch.Tensor  # 隐藏状态张量
    hidden_states_scale: torch.Tensor  # 隐藏状态缩放因子
    use_masked_gemm: bool  # 是否使用掩码GEMM模式
    masked_m: Optional[torch.Tensor] = None  # 每组实际行数（掩码GEMM用）
    expected_m: Optional[int] = None  # 预期最大行数（掩码GEMM用）
    m_indices: Optional[torch.Tensor] = None  # 行索引（连续GEMM用）

    @property
    def runner_backend(self) -> MoeRunnerBackend:  # 返回DeepGEMM后端标识
        return MoeRunnerBackend.DEEP_GEMM


@dataclass
class DeepGemmRunnerOutput(RunnerOutput):  # DeepGEMM Runner输出数据类
    hidden_states: torch.Tensor  # 输出隐藏状态张量

    @property
    def runner_backend(self) -> MoeRunnerBackend:  # 返回DeepGEMM后端标识
        return MoeRunnerBackend.DEEP_GEMM


@dataclass
class DeepGemmMoeQuantInfo(MoeQuantInfo):  # DeepGEMM MoE量化信息数据类
    w13_weight: torch.Tensor  # W1+W3门控上投影权重
    w2_weight: torch.Tensor  # W2下投影权重
    use_fp8: bool  # 是否使用FP8量化
    w13_scale: Optional[torch.Tensor] = None  # W1+W3权重缩放因子
    w2_scale: Optional[torch.Tensor] = None  # W2权重缩放因子
    block_shape: Optional[List[int]] = None  # 分块量化形状
    # DSV4 mxfp4 layout flag; selects recipe_a=(1,128)/recipe_b=(1,32) downstream.  # DSV4 mxfp4布局标志；下游选择recipe_a=(1,128)/recipe_b=(1,32)。
    is_fp4_experts: bool = False  # 是否为FP4专家


class DeepGemmRunnerCore(MoeRunnerCore):  # DeepGEMM MoE Runner核心实现类
    def __init__(self, config: MoeRunnerConfig):  # 初始化DeepGEMM Runner核心
        super().__init__(config)  # 调用父类初始化
        assert self.config.activation == "silu"  # 仅支持SiLU激活
        assert self.config.is_gated  # 必须使用门控结构
        self.swiglu_limit = self.config.swiglu_limit  # SwiGLU限幅值
        self.use_swizzle = False  # 是否使用swizzle布局
        if envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get():  # 如果启用了mega-moe内存修复
            assert envs.SGLANG_OPT_SWIGLU_CLAMP_FUSION.get()  # 必须同时启用SwiGLU限幅融合
            assert envs.SGLANG_OPT_USE_JIT_EP_ACTIVATION.get()  # 必须同时启用JIT EP激活
            self.use_swizzle = True  # 启用swizzle布局

    def run(  # 执行DeepGEMM MoE计算
        self,
        runner_input: DeepGemmRunnerInput,  # DeepGEMM Runner输入
        quant_info: DeepGemmMoeQuantInfo,  # DeepGEMM量化信息
        running_state: dict,  # 运行时状态
        hooks: Optional[Any] = None,  # 钩子函数
    ) -> DeepGemmRunnerOutput:  # 返回DeepGEMM Runner输出
        weight_dtype = quant_info.w13_weight.dtype  # 获取权重数据类型
        if not runner_input.use_masked_gemm:  # 连续GEMM模式
            if weight_dtype == torch.bfloat16:  # BF16权重
                hidden_states = self._run_bf16_contiguous_gemm(  # 执行BF16连续GEMM
                    runner_input, quant_info, running_state
                )
            else:  # FP8权重
                hidden_states = self._run_contiguous_gemm(  # 执行FP8连续GEMM
                    runner_input, quant_info, running_state
                )
        else:  # 掩码GEMM模式
            if weight_dtype == torch.bfloat16:  # BF16权重
                hidden_states = self._run_masked_bf16_gemm(  # 执行BF16掩码GEMM
                    runner_input, quant_info, running_state
                )
            else:  # FP8权重
                hidden_states = self._run_masked_gemm(  # 执行FP8掩码GEMM
                    runner_input, quant_info, running_state
                )
        return DeepGemmRunnerOutput(hidden_states=hidden_states)  # 返回输出

    def _run_contiguous_gemm(  # 执行FP8连续GEMM（DeepEP normal路径）
        self,
        runner_input: DeepGemmRunnerInput,  # DeepGEMM Runner输入
        quant_info: DeepGemmMoeQuantInfo,  # DeepGEMM量化信息
        running_state: dict,  # 运行时状态
    ) -> torch.Tensor:  # 返回输出隐藏状态
        from sglang.jit_kernel.dsv4 import silu_and_mul_contig_post_quant  # 导入连续SiLU+Mul+量化JIT内核
        from sglang.srt.layers.moe.ep_moe.kernels import tma_align_input_scale  # 导入TMA对齐输入缩放
        from sglang.srt.layers.quantization.fp8_kernel import (  # 导入FP8量化内核
            create_per_token_group_quant_fp8_output_scale,  # 创建逐token组FP8量化输出缩放
        )

        hidden_states = runner_input.hidden_states  # 获取隐藏状态
        hidden_states_scale = runner_input.hidden_states_scale  # 获取隐藏状态缩放因子
        all_tokens = running_state["all_tokens"]  # 获取总token数
        hidden_states_device = running_state["hidden_states_device"]  # 获取设备
        hidden_states_dtype = running_state["hidden_states_dtype"]  # 获取数据类型
        hidden_states_shape = running_state["hidden_states_shape"]  # 获取形状
        m_indices = runner_input.m_indices  # 获取行索引

        N = quant_info.w13_weight.size(1)  # W13输出维度
        K = hidden_states_shape[1]  # 输入隐藏维度
        scale_block_size = 128  # 量化分块大小

        recipe_a, recipe_b = (  # FP4专家使用(1,128)/(1,32)配方
            ((1, 128), (1, 32)) if quant_info.is_fp4_experts else (None, None)
        )

        w13_weight_fp8 = (  # W13 FP8权重和缩放
            quant_info.w13_weight,
            quant_info.w13_scale,
        )
        w2_weight_fp8 = (quant_info.w2_weight, quant_info.w2_scale)  # W2 FP8权重和缩放

        gateup_output = torch.empty(  # 分配gate+up投影输出缓冲区
            (all_tokens, N),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        if deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:  # 如果需要TMA对齐缩放
            hidden_states_scale = tma_align_input_scale(hidden_states_scale)  # 对齐输入缩放

        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(  # 执行分组GEMM（FP8输入+FP8权重->BF16输出，连续模式）
            (hidden_states, hidden_states_scale),  # 输入和缩放
            w13_weight_fp8,  # W13权重
            gateup_output,  # 输出
            m_indices,  # 行索引
            recipe_a=recipe_a,  # FP4配方a
            recipe_b=recipe_b,  # FP4配方b
        )

        dispose_tensor(hidden_states)  # 释放输入隐藏状态
        dispose_tensor(hidden_states_scale)  # 释放输入缩放因子

        if envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get():  # 优化路径：融合SiLU+Mul+量化
            swiglu_limit_arg: Optional[float] = self.swiglu_limit  # 获取SwiGLU限幅参数

            down_input_fp8 = torch.empty(  # 分配下投影FP8输入缓冲区
                (all_tokens, N // 2),
                device=gateup_output.device,
                dtype=torch.float8_e4m3fn,
            )
            down_input_scale = create_per_token_group_quant_fp8_output_scale(  # 创建FP8量化输出缩放缓冲区
                x_shape=(all_tokens, N // 2),
                device=gateup_output.device,
                group_size=scale_block_size,
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )
            silu_and_mul_contig_post_quant(  # 融合执行SiLU+Mul+FP8量化
                input=gateup_output,  # gate+up投影输出
                output=down_input_fp8,  # 量化后的下投影输入
                output_scale=down_input_scale,  # 量化缩放因子
                quant_group_size=scale_block_size,  # 量化分组大小
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,  # 是否使用UE8M0缩放
                transposed=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,  # 是否转置
                swiglu_limit=swiglu_limit_arg,  # SwiGLU限幅值
                swizzle=self.use_swizzle,  # 是否使用swizzle布局
            )
            del gateup_output  # 释放gate+up输出
        else:  # 回退路径：先BF16 SiLU+Mul，再单独FP8量化
            # Hacky byte-equal fallback that reproduces the optimize-branch  # 字节等价的回退路径，精确复现优化分支
            # code path exactly: bf16 silu_and_mul then a separate per-token  # 的代码路径：BF16 silu_and_mul，然后单独的逐token
            # group fp8 quant. Kept behind the mega-moe-memory flag.  # 组FP8量化。隐藏在mega-moe-memory标志之后。
            from sglang.srt.layers.quantization.fp8_kernel import (  # 导入FP8量化函数
                sglang_per_token_group_quant_fp8,
            )

            if self.swiglu_limit is not None:  # 如果设置了SwiGLU限幅
                gateup_output = _apply_swiglu_limit(  # 应用SwiGLU限幅
                    gateup_output, swiglu_limit=self.swiglu_limit
                )

            if not _is_musa:  # 非MUSA环境
                down_input = torch.empty(  # 分配BF16下投影输入缓冲区
                    (all_tokens, N // 2),
                    device=gateup_output.device,
                    dtype=torch.bfloat16,
                )
                _legacy_silu_and_mul(gateup_output.view(-1, N), down_input)  # 执行旧版SiLU+Mul
            else:  # MUSA环境
                down_input = _silu_and_mul_musa(gateup_output.view(-1, N))  # 使用MUSA版SiLU+Mul
            del gateup_output  # 释放gate+up输出

            down_input_fp8, down_input_scale = sglang_per_token_group_quant_fp8(  # 执行逐token组FP8量化
                down_input,  # BF16输入
                scale_block_size,  # 量化分块大小
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )
            del down_input  # 释放BF16下投影输入

        down_output = torch.empty(  # 分配最终输出缓冲区
            (all_tokens, K),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        if deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:  # 如果需要TMA对齐缩放
            down_input_scale = tma_align_input_scale(down_input_scale)  # 对齐下投影缩放

        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(  # 执行W2分组GEMM（FP8->BF16，连续模式）
            (down_input_fp8, down_input_scale),  # 输入和缩放
            w2_weight_fp8,  # W2权重
            down_output,  # 输出
            m_indices,  # 行索引
            recipe_a=recipe_a,  # FP4配方a
            recipe_b=recipe_b,  # FP4配方b
        )

        return down_output  # 返回最终输出

    def _run_bf16_contiguous_gemm(  # 执行BF16连续GEMM（DeepEP normal路径，BF16权重）
        self,
        runner_input: DeepGemmRunnerInput,  # DeepGEMM Runner输入
        quant_info: DeepGemmMoeQuantInfo,  # DeepGEMM量化信息
        running_state: dict,  # 运行时状态
    ) -> torch.Tensor:  # 返回输出隐藏状态

        hidden_states = runner_input.hidden_states  # 获取隐藏状态
        all_tokens = running_state["all_tokens"]  # 获取总token数
        hidden_states_device = running_state["hidden_states_device"]  # 获取设备
        hidden_states_shape = running_state["hidden_states_shape"]  # 获取形状
        m_indices = runner_input.m_indices  # 获取行索引

        N = quant_info.w13_weight.size(1)  # W13输出维度
        K = hidden_states_shape[1]  # 输入隐藏维度

        w13_weight = quant_info.w13_weight  # W13权重
        w2_weight = quant_info.w2_weight  # W2权重

        # GroupGemm-1: (M, K) (E, N, K) -> (M, N)  # 分组GEMM-1: (M, K) (E, N, K) -> (M, N)
        gateup_output = torch.empty(  # 分配gate+up投影输出缓冲区
            (all_tokens, N),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )

        deep_gemm_wrapper.grouped_gemm_nt_bf16_contig(  # 执行BF16连续分组GEMM
            hidden_states,  # 输入
            w13_weight,  # W13权重
            gateup_output,  # 输出
            m_indices,  # 行索引
        )

        dispose_tensor(hidden_states)  # 释放输入隐藏状态

        # Act: (M, N) -> (M, N/2)  # 激活: (M, N) -> (M, N/2)
        if not _is_musa:  # 非MUSA环境
            down_input = torch.empty(  # 分配下投影输入缓冲区
                (
                    all_tokens,
                    N // 2,
                ),
                device=gateup_output.device,
                dtype=torch.bfloat16,
            )
            _legacy_silu_and_mul(gateup_output.view(-1, N), down_input)  # 执行SiLU+Mul
        else:  # MUSA环境
            down_input = _silu_and_mul_musa(gateup_output.view(-1, N))  # 使用MUSA版SiLU+Mul
        del gateup_output  # 释放gate+up输出

        # GroupGemm-2: (M, N/2) (E, K, N/2) -> (M, K)  # 分组GEMM-2: (M, N/2) (E, K, N/2) -> (M, K)
        down_output = torch.empty(  # 分配最终输出缓冲区
            (all_tokens, K),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        deep_gemm_wrapper.grouped_gemm_nt_bf16_contig(  # 执行W2 BF16连续分组GEMM
            down_input,  # 输入
            w2_weight,  # W2权重
            down_output,  # 输出
            m_indices,  # 行索引
        )

        return down_output  # 返回最终输出

    def _run_masked_gemm(  # 执行FP8掩码GEMM（标准/低延迟路径）
        self,
        runner_input: DeepGemmRunnerInput,  # DeepGEMM Runner输入
        quant_info: DeepGemmMoeQuantInfo,  # DeepGEMM量化信息
        running_state: dict,  # 运行时状态
    ) -> torch.Tensor:  # 返回输出隐藏状态
        from sglang.srt.layers import deep_gemm_wrapper  # 导入DeepGEMM封装

        hidden_states = runner_input.hidden_states  # 获取隐藏状态
        hidden_states_scale = runner_input.hidden_states_scale  # 获取缩放因子
        masked_m = runner_input.masked_m  # 获取每组实际行数
        expected_m = runner_input.expected_m  # 获取预期最大行数

        w13_weight = quant_info.w13_weight  # W13权重
        w2_weight = quant_info.w2_weight  # W2权重
        w13_scale = quant_info.w13_scale  # W13缩放因子
        w2_scale = quant_info.w2_scale  # W2缩放因子

        recipe_a, recipe_b = (  # FP4专家配方
            ((1, 128), (1, 32)) if quant_info.is_fp4_experts else (None, None)
        )

        hidden_states_device = running_state["hidden_states_device"]  # 获取设备

        # GroupGemm-0  # 分组GEMM-0（W13门控上投影）
        if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:  # 如果使用UE8M0缩放格式
            if hidden_states_scale.dtype != torch.int:  # 如果缩放还不是e8m0格式
                b, s_mn, s_k = hidden_states_scale.shape  # 获取缩放形状
                assert (
                    s_mn % 4 == 0 and s_k % 4 == 0
                ), f"scales must be aligned to 4, but got ({b}, {s_mn}, {s_k})"  # 缩放必须4对齐
                hidden_states_scale = _cast_to_e8m0_with_rounding_up(  # 转换为e8m0格式
                    hidden_states_scale
                )
        elif deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:  # 如果需要TMA对齐缩放
            hidden_states_scale = deep_gemm_wrapper.get_mn_major_tma_aligned_tensor(  # TMA对齐缩放
                hidden_states_scale
            )

        num_groups, m, k = hidden_states.shape  # 获取分组数、行数、列数
        n = w13_weight.size(1)  # W13输出维度
        gateup_output = torch.empty(  # 分配gate+up输出缓冲区
            (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
        )
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(  # 执行FP8掩码分组GEMM（W13）
            (hidden_states, hidden_states_scale),  # 输入和缩放
            (w13_weight, w13_scale),  # W13权重和缩放
            gateup_output,  # 输出
            masked_m,  # 每组实际行数
            expected_m,  # 预期最大行数
            recipe_a=recipe_a,  # FP4配方a
            recipe_b=recipe_b,  # FP4配方b
        )
        dispose_tensor(hidden_states)  # 释放输入隐藏状态
        dispose_tensor(hidden_states_scale)  # 释放输入缩放因子

        swiglu_limit_arg: Optional[float] = None  # 初始化SwiGLU限幅参数
        if self.swiglu_limit is not None:  # 如果设置了SwiGLU限幅
            # DeepSeek V4: clamped swiglu requires JIT EP activation; the  # DeepSeek V4：限幅SwiGLU需要JIT EP激活；
            # FAST_ACT fused-quant path doesn't carry a swiglu_limit arg.  # FAST_ACT融合量化路径不携带swiglu_limit参数。
            assert (
                not _MASKED_GEMM_FAST_ACT
            ), "DeepSeek V4 does not support SGLANG_MASKED_GEMM_FAST_ACT"  # V4不支持FAST_ACT
            assert (
                envs.SGLANG_OPT_USE_JIT_EP_ACTIVATION.get()
            ), "DeepSeek V4 requires SGLANG_OPT_USE_JIT_EP_ACTIVATION=True"  # V4需要JIT EP激活

            if envs.SGLANG_OPT_SWIGLU_CLAMP_FUSION.get():  # 如果启用了SwiGLU限幅融合
                swiglu_limit_arg = self.swiglu_limit  # 传递限幅参数
            else:  # 未启用融合，手动应用限幅
                gateup_output = einops.rearrange(  # 重排为2D
                    gateup_output, "grp tok hidden -> (grp tok) hidden"
                )
                gateup_output = _apply_swiglu_limit(  # 应用SwiGLU限幅
                    gateup_output, swiglu_limit=self.swiglu_limit
                )
                gateup_output = einops.rearrange(  # 重排回3D
                    gateup_output, "(grp tok) hidden -> grp tok hidden", grp=num_groups
                )

        # Act  # 激活（SiLU+Mul+FP8量化）
        down_input, down_input_scale = _varlen_deep_gemm_silu_mul_quant(  # 执行SiLU+Mul+量化
            gateup_output,
            masked_m,
            group_size=128,
            topk=self.config.top_k,
            swiglu_limit=swiglu_limit_arg,
            swizzle=self.use_swizzle,
        )
        del gateup_output  # 释放gate+up输出

        # GroupGemm-1  # 分组GEMM-1（W2下投影）
        n = w2_weight.shape[1]  # W2输出维度

        if deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:  # 如果需要TMA对齐缩放
            down_input_scale = deep_gemm_wrapper.get_mn_major_tma_aligned_tensor(  # TMA对齐下投影缩放
                down_input_scale
            )

        down_output = torch.empty(  # 分配最终输出缓冲区
            (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
        )

        down_gemm_overlap_args = running_state.get("down_gemm_overlap_args", None)  # 获取下投影GEMM重叠参数
        if down_gemm_overlap_args is None:  # 无重叠参数
            gemm_overlap_args_dict = {}  # 空参数
        else:  # 有重叠参数
            down_gemm_overlap_args.start_event.record()  # 记录开始事件
            max_block_n = (  # 根据GPU型号和预期行数选择最大块N
                160 if (_DEEPGEMM_ON_H20 and runner_input.expected_m <= 64) else 256
            )
            gemm_overlap_args_dict = {  # 构造重叠参数字典
                "overlap_args": down_gemm_overlap_args,  # 重叠参数
                "max_block_n": max_block_n,  # 最大块N
            }

        deep_gemm_return_value = deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(  # 执行FP8掩码分组GEMM（W2）
            (down_input, down_input_scale),  # 输入和缩放
            (w2_weight, w2_scale),  # W2权重和缩放
            down_output,  # 输出
            masked_m,  # 每组实际行数
            expected_m,  # 预期最大行数
            recipe_a=recipe_a,  # FP4配方a
            recipe_b=recipe_b,  # FP4配方b
            **gemm_overlap_args_dict,  # 重叠参数
        )
        meta_overlap_args = running_state.get("meta_overlap_args", None)  # 获取元数据重叠参数
        if meta_overlap_args is not None:  # 如果有元数据重叠参数
            block_m, threshold = deep_gemm_return_value  # 解包返回值
            meta_overlap_args["block_m"] = block_m  # 保存块M
            meta_overlap_args["threshold"] = threshold  # 保存阈值

        return down_output  # 返回最终输出

    def _run_masked_bf16_gemm(  # 执行BF16掩码GEMM
        self,
        runner_input: DeepGemmRunnerInput,  # DeepGEMM Runner输入
        quant_info: DeepGemmMoeQuantInfo,  # DeepGEMM量化信息
        running_state: dict,  # 运行时状态
    ) -> torch.Tensor:  # 返回输出隐藏状态
        from sglang.srt.layers import deep_gemm_wrapper  # 导入DeepGEMM封装
        from sglang.srt.layers.moe.ep_moe.kernels import silu_and_mul_masked_fwd  # 导入掩码SiLU+Mul内核

        hidden_states = runner_input.hidden_states  # 获取隐藏状态
        masked_m = runner_input.masked_m  # 获取每组实际行数
        expected_m = runner_input.expected_m  # 获取预期最大行数

        w13_weight = quant_info.w13_weight  # W13权重
        w2_weight = quant_info.w2_weight  # W2权重

        hidden_states_device = running_state["hidden_states_device"]  # 获取设备

        # GroupGemm-0  # 分组GEMM-0（W13门控上投影）
        num_groups, m, k = hidden_states.shape  # 获取分组数、行数、列数
        n = w13_weight.size(1)  # W13输出维度
        gateup_output = torch.empty(  # 分配gate+up输出缓冲区
            (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
        )
        deep_gemm_wrapper.grouped_gemm_nt_bf16_masked(  # 执行BF16掩码分组GEMM（W13）
            hidden_states,  # 输入
            w13_weight,  # W13权重
            gateup_output,  # 输出
            masked_m,  # 每组实际行数
            expected_m,  # 预期最大行数
        )
        dispose_tensor(hidden_states)  # 释放输入隐藏状态

        down_input = torch.empty(  # 分配下投影输入缓冲区
            (
                gateup_output.shape[0],  # 分组数
                gateup_output.shape[1],  # 行数
                gateup_output.shape[2] // 2,  # 列数减半（gate和up拼接后分离）
            ),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )

        # Act  # 激活（SiLU+Mul）
        silu_and_mul_masked_fwd(gateup_output, down_input, masked_m)  # 执行掩码SiLU+Mul
        del gateup_output  # 释放gate+up输出

        # GroupGemm-1  # 分组GEMM-1（W2下投影）
        n = w2_weight.shape[1]  # W2输出维度

        down_output = torch.empty(  # 分配最终输出缓冲区
            (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
        )
        deep_gemm_wrapper.grouped_gemm_nt_bf16_masked(  # 执行BF16掩码分组GEMM（W2）
            down_input,  # 输入
            w2_weight,  # W2权重
            down_output,  # 输出
            masked_m,  # 每组实际行数
            expected_m,  # 预期最大行数
        )
        # Note: BF16 masked gemm doesn't support overlap_args, so no return value unpack  # 注意：BF16掩码GEMM不支持overlap_args，因此无返回值解包

        return down_output  # 返回最终输出

    @property
    def runner_backend(self) -> MoeRunnerBackend:  # 返回DeepGEMM后端标识
        return MoeRunnerBackend.DEEP_GEMM


@register_pre_permute("standard", "deep_gemm")  # 注册standard到deep_gemm的前置换
def pre_permute_standard_to_deep_gemm(  # 将标准调度输出转换为DeepGEMM Runner输入
    dispatch_output: StandardDispatchOutput,  # 标准调度输出
    quant_info: DeepGemmMoeQuantInfo,  # DeepGEMM量化信息
    runner_config: MoeRunnerConfig,  # Runner配置
    running_state: dict,  # 运行时状态
) -> DeepGemmRunnerInput:  # 返回DeepGEMM Runner输入
    from sglang.srt.layers.moe.ep_moe.kernels import moe_ep_deepgemm_preprocess  # 导入DeepGEMM预处理内核

    hidden_states, topk_output = (  # 解包调度输出
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
    )
    topk_weights, topk_ids, _ = topk_output  # 解包Top-K输出

    hidden_states_shape = hidden_states.shape  # 获取隐藏状态形状
    hidden_states_dtype = hidden_states.dtype  # 获取隐藏状态数据类型
    hidden_states_device = hidden_states.device  # 获取隐藏状态设备
    hidden_states_ref = hidden_states  # 保存引用以便后续释放

    topk_weights, topk_ids = topk_weights, topk_ids  # 保留Top-K权重和索引

    # PreReorder  # 预重排
    output_dtype = (  # 确定输出数据类型
        torch.bfloat16  # BF16权重使用BF16
        if quant_info.w13_weight.dtype == torch.bfloat16
        else torch.float8_e4m3fn  # FP8权重使用FP8
    )
    masked_m, expected_m, src2dst, hidden_states, hidden_states_scale = (  # 执行预处理
        moe_ep_deepgemm_preprocess(
            topk_ids,  # Top-K专家索引
            runner_config.num_local_experts,  # 本地专家数
            hidden_states,  # 隐藏状态
            runner_config.top_k,  # Top-K值
            quant_info.block_shape,  # 量化分块形状
            output_dtype=output_dtype,  # 输出数据类型
        )
    )

    dispose_tensor(hidden_states_ref)  # 释放原始隐藏状态引用

    running_state["topk_ids"] = topk_ids  # 保存Top-K索引到运行状态
    running_state["topk_weights"] = topk_weights  # 保存Top-K权重到运行状态
    running_state["hidden_states_shape"] = hidden_states_shape  # 保存形状
    running_state["hidden_states_dtype"] = hidden_states_dtype  # 保存数据类型
    running_state["hidden_states_device"] = hidden_states_device  # 保存设备
    running_state["src2dst"] = src2dst  # 保存源到目标映射

    return DeepGemmRunnerInput(  # 构造并返回DeepGEMM Runner输入
        hidden_states=hidden_states,  # 预处理后的隐藏状态
        hidden_states_scale=hidden_states_scale,  # 预处理后的缩放因子
        use_masked_gemm=True,  # 使用掩码GEMM模式
        masked_m=masked_m,  # 每组实际行数
        expected_m=expected_m,  # 预期最大行数
    )


@register_post_permute("deep_gemm", "standard")  # 注册deep_gemm到standard的后置换
def post_permute_deep_gemm_to_standard(  # 将DeepGEMM Runner输出转换为标准Combine输入
    runner_output: DeepGemmRunnerOutput,  # DeepGEMM Runner输出
    quant_info: DeepGemmMoeQuantInfo,  # DeepGEMM量化信息
    runner_config: MoeRunnerConfig,  # Runner配置
    running_state: dict,  # 运行时状态
) -> StandardCombineInput:  # 返回标准Combine输入
    from sglang.srt.layers.moe.ep_moe.kernels import post_reorder_triton_kernel  # 导入后重排Triton内核
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput  # 导入标准Combine输入类

    hidden_states_shape = running_state["hidden_states_shape"]  # 获取原始形状
    hidden_states_dtype = running_state["hidden_states_dtype"]  # 获取原始数据类型
    hidden_states_device = running_state["hidden_states_device"]  # 获取原始设备
    src2dst = running_state["src2dst"]  # 获取源到目标映射
    topk_ids = running_state["topk_ids"]  # 获取Top-K索引
    topk_weights = running_state["topk_weights"]  # 获取Top-K权重

    output = torch.empty(  # 分配输出缓冲区
        hidden_states_shape, dtype=hidden_states_dtype, device=hidden_states_device
    )
    post_reorder_triton_kernel[(hidden_states_shape[0],)](  # 执行后重排（Triton内核）
        runner_output.hidden_states,  # Runner输出
        output,  # 重排后的输出
        src2dst,  # 源到目标映射
        topk_ids,  # Top-K索引
        topk_weights,  # Top-K权重
        runner_config.top_k,  # Top-K值
        hidden_states_shape[1],  # 隐藏维度
        BLOCK_SIZE=512,  # Triton块大小
    )

    dispose_tensor(runner_output.hidden_states)  # 释放Runner输出

    if runner_config.routed_scaling_factor is not None:  # 如果有路由缩放因子
        output *= runner_config.routed_scaling_factor  # 应用路由缩放

    return StandardCombineInput(  # 构造并返回标准Combine输入
        hidden_states=output,
    )


@register_pre_permute("deepep_ll", "deep_gemm")  # 注册deepep_ll到deep_gemm的前置换
def pre_permute_deepep_ll_to_deep_gemm(  # 将DeepEP低延迟调度输出转换为DeepGEMM Runner输入
    dispatch_output: DeepEPLLDispatchOutput,  # DeepEP低延迟调度输出
    quant_info: DeepGemmMoeQuantInfo,  # DeepGEMM量化信息
    runner_config: MoeRunnerConfig,  # Runner配置
    running_state: dict,  # 运行时状态
) -> DeepGemmRunnerInput:  # 返回DeepGEMM Runner输入
    hidden_states, hidden_states_scale, topk_ids, topk_weights, masked_m, expected_m = (  # 解包调度输出
        dispatch_output
    )

    running_state["topk_ids"] = topk_ids  # 保存Top-K索引
    running_state["topk_weights"] = topk_weights  # 保存Top-K权重
    running_state["hidden_states_shape"] = hidden_states.shape  # 保存形状
    running_state["hidden_states_dtype"] = hidden_states.dtype  # 保存数据类型
    running_state["hidden_states_device"] = hidden_states.device  # 保存设备

    return DeepGemmRunnerInput(  # 构造并返回DeepGEMM Runner输入
        hidden_states=hidden_states,  # 隐藏状态
        hidden_states_scale=hidden_states_scale,  # 缩放因子
        use_masked_gemm=True,  # 使用掩码GEMM模式
        masked_m=masked_m,  # 每组实际行数
        expected_m=expected_m,  # 预期最大行数
    )


@register_post_permute("deep_gemm", "deepep_ll")  # 注册deep_gemm到deepep_ll的后置换
def post_permute_deep_gemm_to_deepep_ll(  # 将DeepGEMM Runner输出转换为DeepEP低延迟Combine输入
    runner_output: DeepGemmRunnerOutput,  # DeepGEMM Runner输出
    quant_info: DeepGemmMoeQuantInfo,  # DeepGEMM量化信息
    runner_config: MoeRunnerConfig,  # Runner配置
    running_state: dict,  # 运行时状态
) -> DeepEPLLCombineInput:  # 返回DeepEP低延迟Combine输入
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput  # 导入DeepEP低延迟Combine输入类

    return DeepEPLLCombineInput(  # 构造并返回DeepEP低延迟Combine输入
        hidden_states=runner_output.hidden_states,  # 隐藏状态
        topk_ids=running_state["topk_ids"],  # Top-K索引
        topk_weights=running_state["topk_weights"],  # Top-K权重
    )


@register_pre_permute("deepep_normal", "deep_gemm")  # 注册deepep_normal到deep_gemm的前置换
def pre_permute_deepep_normal_to_deep_gemm(  # 将DeepEP普通调度输出转换为DeepGEMM Runner输入
    dispatch_output: DeepEPNormalDispatchOutput,  # DeepEP普通调度输出
    quant_info: DeepGemmMoeQuantInfo,  # DeepGEMM量化信息
    runner_config: MoeRunnerConfig,  # Runner配置
    running_state: dict,  # 运行时状态
) -> DeepGemmRunnerInput:  # 返回DeepGEMM Runner输入
    from sglang.srt.layers.moe.ep_moe.kernels import ep_scatter  # 导入EP scatter内核

    (
        hidden_states,  # 隐藏状态
        hidden_states_scale,  # 缩放因子
        topk_ids,  # Top-K索引
        topk_weights,  # Top-K权重
        num_recv_tokens_per_expert,  # 每个专家接收的token数
    ) = dispatch_output  # 解包调度输出
    assert runner_config.activation == "silu"  # 仅支持SiLU激活

    all_tokens = sum(num_recv_tokens_per_expert)  # 计算总token数
    running_state["all_tokens"] = all_tokens  # 保存到运行状态

    K = hidden_states.shape[1]  # 获取隐藏维度

    hidden_states_shape = hidden_states.shape  # 获取形状
    hidden_states_device = hidden_states.device  # 获取设备
    hidden_states_dtype = hidden_states.dtype  # 获取数据类型

    running_state["hidden_states_shape"] = hidden_states_shape  # 保存形状
    running_state["hidden_states_device"] = hidden_states_device  # 保存设备
    running_state["hidden_states_dtype"] = hidden_states_dtype  # 保存数据类型
    running_state["topk_ids"] = topk_ids  # 保存Top-K索引
    running_state["topk_weights"] = topk_weights  # 保存Top-K权重

    input_tensor = torch.empty(  # 分配scatter后的输入缓冲区
        (all_tokens, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:  # 如果使用UE8M0缩放格式
        # TODO check whether need `zeros`  # TODO 检查是否需要`zeros`
        input_tensor_scale = torch.zeros(  # 分配零初始化的缩放缓冲区
            (ceil_div(K // 128, 4), all_tokens),  # UE8M0格式形状
            device=hidden_states.device,
            dtype=torch.int,  # e8m0以int存储
        ).transpose(0, 1)  # 转置为(all_tokens, K//128//4)
    else:  # 非UE8M0格式
        input_tensor_scale = torch.empty(  # 分配缩放缓冲区
            (all_tokens, K // 128),
            device=hidden_states.device,
            dtype=torch.float32,
        )
    m_indices = torch.empty(all_tokens, device=hidden_states.device, dtype=torch.int32)  # 行索引缓冲区
    output_index = torch.empty_like(topk_ids)  # 输出索引缓冲区

    if get_offloader().forbid_copy_engine_usage:  # 如果禁止使用拷贝引擎
        num_recv_tokens_per_expert_gpu = copy_list_to_gpu_no_ce(  # 不使用拷贝引擎传输
            num_recv_tokens_per_expert
        )
    else:  # 允许使用拷贝引擎
        num_recv_tokens_per_expert_gpu = torch.tensor(  # 创建CPU张量
            num_recv_tokens_per_expert,
            dtype=torch.int32,
            pin_memory=True,  # 固定内存
            device="cpu",
        ).cuda(non_blocking=True)  # 异步传输到GPU
    expert_start_loc = torch.empty_like(num_recv_tokens_per_expert_gpu)  # 专家起始位置缓冲区

    ep_scatter(  # 执行EP scatter操作
        hidden_states,  # 输入隐藏状态
        hidden_states_scale,  # 输入缩放因子
        topk_ids,  # Top-K索引
        num_recv_tokens_per_expert_gpu,  # 每个专家接收的token数
        expert_start_loc,  # 专家起始位置
        input_tensor,  # scatter输出
        input_tensor_scale,  # scatter缩放输出
        m_indices,  # 行索引输出
        output_index,  # 输出索引
        scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,  # UE8M0缩放标志
    )
    dispose_tensor(hidden_states)  # 释放原始隐藏状态
    if hidden_states_scale is not None:  # 如果缩放因子存在
        dispose_tensor(hidden_states_scale)  # 释放原始缩放因子

    running_state["output_index"] = output_index  # 保存输出索引

    return DeepGemmRunnerInput(  # 构造并返回DeepGEMM Runner输入
        hidden_states=input_tensor,  # scatter后的输入
        hidden_states_scale=input_tensor_scale,  # scatter后的缩放
        use_masked_gemm=False,  # 使用连续GEMM模式（非掩码）
        m_indices=m_indices,  # 行索引
    )


@register_post_permute("deep_gemm", "deepep_normal")  # 注册deep_gemm到deepep_normal的后置换
def post_permute_deep_gemm_to_deepep_normal(  # 将DeepGEMM Runner输出转换为DeepEP普通Combine输入
    runner_output: DeepGemmRunnerOutput,  # DeepGEMM Runner输出
    quant_info: DeepGemmMoeQuantInfo,  # DeepGEMM量化信息
    runner_config: MoeRunnerConfig,  # Runner配置
    running_state: dict,  # 运行时状态
) -> DeepEPNormalCombineInput:  # 返回DeepEP普通Combine输入
    from sglang.srt.layers.moe.ep_moe.kernels import ep_gather  # 导入EP gather内核
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalCombineInput  # 导入DeepEP普通Combine输入类

    hidden_states = runner_output.hidden_states  # 获取Runner输出
    topk_ids = running_state["topk_ids"]  # 获取Top-K索引
    topk_weights = running_state["topk_weights"]  # 获取Top-K权重
    output_index = running_state["output_index"]  # 获取输出索引

    gather_out = torch.empty(  # 分配gather输出缓冲区
        running_state["hidden_states_shape"],
        device=running_state["hidden_states_device"],
        dtype=torch.bfloat16,
    )
    ep_gather(hidden_states, topk_ids, topk_weights, output_index, gather_out)  # 执行EP gather操作

    return DeepEPNormalCombineInput(  # 构造并返回DeepEP普通Combine输入
        hidden_states=gather_out,  # gather后的输出
        topk_ids=running_state["topk_ids"],  # Top-K索引
        topk_weights=running_state["topk_weights"],  # Top-K权重
    )


def _varlen_deep_gemm_silu_mul_quant(  # 变长SiLU+Mul+FP8量化函数
    gateup_output: torch.Tensor,  # gate+up投影输出
    masked_m: Optional[torch.Tensor],  # 每组实际行数
    group_size: int,  # 量化分组大小
    topk: int,  # Top-K值
    swiglu_limit: Optional[float] = None,  # SwiGLU限幅值
    swizzle: bool = False,  # 是否使用swizzle布局
) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回(量化后的输入, 缩放因子)
    from sglang.srt.layers.moe.ep_moe.kernels import silu_and_mul_masked_post_quant_fwd  # 导入掩码SiLU+Mul+量化内核
    from sglang.srt.layers.quantization.fp8_kernel import (  # 导入FP8量化函数
        sglang_per_token_group_quant_8bit,
    )

    if _MASKED_GEMM_FAST_ACT:  # 如果使用快速激活路径
        assert not swizzle, (  # swizzle与FAST_ACT不兼容
            "SGLANG_OPT_FIX_MEGA_MOE_MEMORY is incompatible with "
            "SGLANG_MASKED_GEMM_FAST_ACT (swizzled layout only supported by JIT act)"
        )
        assert (
            swiglu_limit is None
        ), "swiglu_limit (DeepSeek V4) is not supported together with SGLANG_MASKED_GEMM_FAST_ACT"  # swiglu_limit与FAST_ACT不兼容
        return sglang_per_token_group_quant_8bit(  # 使用8bit量化（融合SiLU+Mul）
            x=gateup_output,  # 输入
            dst_dtype=torch.float8_e4m3fn,  # 目标数据类型
            group_size=group_size,  # 量化分组大小
            masked_m=masked_m,  # 每组实际行数
            column_major_scales=True,  # 列主序缩放
            scale_tma_aligned=True,  # TMA对齐缩放
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,  # UE8M0缩放标志
            fuse_silu_and_mul=True,  # 融合SiLU+Mul
            enable_v2=True,  # 启用v2版本
        )

    assert masked_m is not None  # 掩码GEMM路径必须提供masked_m
    hidden_states_device = gateup_output.device  # 获取设备
    E, N, D_2 = gateup_output.shape  # 获取形状：专家数、行数、gate+up维度
    D = D_2 // 2  # 下投影维度（gate和up各占一半）
    del D_2  # 释放变量
    G = D // group_size  # 量化分组数
    down_input = torch.empty(  # 分配FP8下投影输入缓冲区
        (E, N, D),
        device=hidden_states_device,
        dtype=torch.float8_e4m3fn,
    )

    use_jit_ep_activation = envs.SGLANG_OPT_USE_JIT_EP_ACTIVATION.get()  # 是否使用JIT EP激活
    if N % 4 != 0 or G % 4 != 0:  # 维度非4对齐时禁用JIT EP激活
        use_jit_ep_activation = False

    if use_jit_ep_activation:  # JIT EP激活路径
        packed_ue8m0 = deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0  # UE8M0标志
        down_input_scale = torch.empty(  # 分配缩放缓冲区
            (E, G // 4, N) if packed_ue8m0 else (E, N, G),  # UE8M0与非UE8M0形状不同
            device=hidden_states_device,
            dtype=torch.int32 if packed_ue8m0 else torch.float32,  # UE8M0用int32存储
        )
        silu_and_mul_masked_post_quant(  # 执行JIT融合SiLU+Mul+量化
            gateup_output,  # 输入
            down_input,  # 量化输出
            down_input_scale,  # 缩放输出
            group_size,  # 量化分组大小
            masked_m,  # 每组实际行数
            scale_ue8m0=packed_ue8m0,  # UE8M0标志
            topk=topk,  # Top-K值
            transposed=packed_ue8m0,  # 是否转置
            swiglu_limit=swiglu_limit,  # SwiGLU限幅值
            swizzle=swizzle,  # swizzle布局
        )
        if packed_ue8m0:  # UE8M0格式需要转置
            down_input_scale = down_input_scale.transpose(-1, -2)  # 转置缩放因子
    else:  # 非JIT路径
        assert (
            swiglu_limit is None
        ), "swiglu_limit (DeepSeek V4) requires SGLANG_OPT_USE_JIT_EP_ACTIVATION=True"  # swiglu_limit需要JIT EP激活
        assert (
            not swizzle
        ), "SGLANG_OPT_FIX_MEGA_MOE_MEMORY requires SGLANG_OPT_USE_JIT_EP_ACTIVATION=True"  # mega-moe-memory修复需要JIT EP激活
        down_input_scale = torch.empty(  # 分配缩放缓冲区
            (E, N, G),
            device=hidden_states_device,
            dtype=torch.float32,
        )
        silu_and_mul_masked_post_quant_fwd(  # 执行标准SiLU+Mul+量化
            gateup_output,  # 输入
            down_input,  # 量化输出
            down_input_scale,  # 缩放输出
            group_size,  # 量化分组大小
            masked_m,  # 每组实际行数
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,  # UE8M0标志
        )
    return down_input, down_input_scale  # 返回量化输入和缩放因子


def _apply_swiglu_limit(  # 对gate+up输出应用SwiGLU限幅
    gateup_output: torch.Tensor, swiglu_limit: float  # gate+up输出和限幅值
) -> torch.Tensor:  # 返回限幅后的输出
    assert swiglu_limit == 10  # 当前仅支持限幅值10

    num_tokens, hidden_size_x2 = gateup_output.shape  # 获取形状
    assert gateup_output.dtype == torch.bfloat16  # 仅支持BF16

    gate, up = torch.chunk(gateup_output, chunks=2, dim=-1)  # 分离gate和up
    assert gate.shape == (num_tokens, hidden_size_x2 // 2)  # 验证gate形状
    assert up.shape == (num_tokens, hidden_size_x2 // 2)  # 验证up形状

    up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)  # up双向限幅
    gate = torch.clamp(gate, max=swiglu_limit)  # gate仅上限限幅

    out = torch.cat([gate, up], dim=-1)  # 重新拼接gate和up
    assert out.shape == (num_tokens, hidden_size_x2)  # 验证输出形状
    return out  # 返回限幅后的输出