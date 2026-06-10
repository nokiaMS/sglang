# MoE 运行器主模块
# 本模块实现了 MoeRunner 类，负责 MoE（混合专家）层的运行器核心逻辑。
# 根据不同的运行器后端（Triton、DeepGEMM、Marlin、FlashInfer等）选择相应的执行路径，
# 支持融合函数路径和运行器核心路径，并处理 LoRA 和 GEMM 重叠等高级功能。

from __future__ import annotations  # 启用延迟类型注解求值

import logging  # 导入日志模块
import os  # 导入操作系统模块
from typing import TYPE_CHECKING, Any, Optional  # 导入类型提示工具

from sglang.srt.layers.moe.moe_runner.base import (  # 从MoE运行器基类导入
    FusedOpPool,  # 融合操作池
    MoeRunnerConfig,  # MoE运行器配置类
    PermuteMethodPool,  # 排列方法池
)
from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmRunnerCore  # 导入DeepGEMM运行器核心
from sglang.srt.layers.moe.moe_runner.triton import TritonRunnerCore  # 导入Triton运行器核心
from sglang.srt.layers.moe.moe_runner.triton_kernels import TritonKernelsRunnerCore  # 导入TritonKernels运行器核心
from sglang.srt.layers.moe.utils import get_moe_a2a_backend  # 导入MoE A2A后端获取函数

if TYPE_CHECKING:  # 仅在类型检查时导入，避免循环依赖
    from sglang.srt.batch_overlap.single_batch_overlap import DownGemmOverlapArgs  # 下行GEMM重叠参数
    from sglang.srt.layers.moe.moe_runner.base import MoeQuantInfo  # MoE量化信息基类
    from sglang.srt.layers.moe.token_dispatcher.base import CombineInput, DispatchOutput  # 合并输入和分发输出
    from sglang.srt.layers.moe.utils import MoeRunnerBackend  # MoE运行器后端枚举
    from sglang.srt.lora.lora_moe_runners import LoRAHooks  # LoRA MoE钩子

logger = logging.getLogger(__name__)  # 创建模块级日志记录器


class MoeRunner:
    """MoE运行器类，负责管理和执行混合专家模型的前向计算"""

    def __init__(
        self,
        runner_backend: MoeRunnerBackend,
        config: MoeRunnerConfig,
        lora_enabled: bool = False,
    ):
        """初始化MoE运行器

        Args:
            runner_backend: 运行器后端类型
            config: MoE运行器配置
            lora_enabled: 是否启用LoRA
        """
        self.runner_backend = runner_backend  # 存储运行器后端类型
        self.config = config  # 存储运行器配置
        self.lora_enabled = lora_enabled  # 存储LoRA启用状态

        self.fused_func = None  # 初始化融合函数为None

        if runner_backend.is_triton():  # 如果是Triton后端
            self.runner_core = TritonRunnerCore(config)  # 创建Triton运行器核心
        elif runner_backend.is_triton_kernels():  # 如果是TritonKernels后端
            self.runner_core = TritonKernelsRunnerCore(config)  # 创建TritonKernels运行器核心
        elif runner_backend.is_deep_gemm():  # 如果是DeepGEMM后端
            self.runner_core = DeepGemmRunnerCore(config)  # 创建DeepGEMM运行器核心
        elif runner_backend.is_aiter():  # 如果是Aiter后端
            from sglang.srt.layers.moe.moe_runner.aiter import AiterRunnerCore  # 惰性导入Aiter运行器核心

            self.runner_core = AiterRunnerCore(config)  # 创建Aiter运行器核心
        elif runner_backend.is_marlin():  # 如果是Marlin后端
            if lora_enabled:  # 如果启用了LoRA
                from sglang.srt.lora.lora_moe_runner_marlin import MarlinLoraRunnerCore  # 导入Marlin LoRA运行器核心

                self.runner_core = MarlinLoraRunnerCore(config)  # 创建Marlin LoRA运行器核心
            else:
                self.runner_core = None  # Marlin only supports fused path / Marlin仅支持融合路径
        elif (
            runner_backend.is_flashinfer_trtllm()
            or runner_backend.is_flashinfer_trtllm_routed()
        ):  # 如果是FlashInfer TRT-LLM后端
            self.runner_core = None  # FlashInfer TRT-LLM only supports fused path / FlashInfer TRT-LLM仅支持融合路径
        elif runner_backend.is_flashinfer_cutedsl():  # 如果是FlashInfer CuteDSL后端
            self.runner_core = None  # FlashInfer CuteDSL only supports fused path / FlashInfer CuteDSL仅支持融合路径
        elif runner_backend.is_flashinfer_mxfp4():  # 如果是FlashInfer MXFP4后端
            self.runner_core = None  # FlashInfer MXFP4 only supports fused path / FlashInfer MXFP4仅支持融合路径
        else:
            raise NotImplementedError(f"Unsupported runner backend: {runner_backend}")  # 不支持的运行器后端

        # Skip fused func if LoRA is enabled (LoRA requires non-fused path)
        # 如果启用LoRA则跳过融合函数（LoRA需要非融合路径）
        if not lora_enabled:  # 如果未启用LoRA
            a2a_backend_name = get_moe_a2a_backend().value  # 获取A2A后端名称
            runner_backend_name = runner_backend.value  # 获取运行器后端名称

            # TODO(cwan): add a server argument to disable fused func
            # TODO(cwan): 添加服务器参数以禁用融合函数
            self.fused_func = FusedOpPool.get_fused_func(
                a2a_backend_name, runner_backend_name
            )  # 从融合操作池获取融合函数

            if self.runner_core is None and self.fused_func is None:  # 如果运行器核心和融合函数都不存在
                raise NotImplementedError(
                    f"Runner backend {runner_backend} requires a fused func for a2a backend "
                    f"{a2a_backend_name}, but none is registered."
                    f"运行器后端 {runner_backend} 需要 a2a 后端 "
                    f"{a2a_backend_name} 的融合函数，但未注册任何融合函数。"
                )

        self.down_gemm_overlap_args: Optional[DownGemmOverlapArgs] = None  # 下行GEMM重叠参数
        self.meta_overlap_args: Optional[dict] = None  # 元数据重叠参数

        SGLANG_CI_DISABLE_MOE_FUSED_FUNC = os.environ.get(
            "SGLANG_CI_DISABLE_MOE_FUSED_FUNC", "0"
        )  # 获取CI禁用MoE融合函数环境变量
        if SGLANG_CI_DISABLE_MOE_FUSED_FUNC == "1":  # 如果CI禁用MoE融合函数
            logger.info(
                "SGLANG_CI_DISABLE_MOE_FUSED_FUNC is set to 1, disabling fused func"
            )  # 记录禁用信息
            self.fused_func = None  # 禁用融合函数

    def run(
        self, dispatch_output: DispatchOutput, quant_info: MoeQuantInfo, lora_info=None
    ) -> CombineInput:
        """执行MoE前向计算

        Args:
            dispatch_output: 分发输出
            quant_info: 量化信息
            lora_info: LoRA信息

        Returns:
            合并输入
        """
        if self.fused_func is not None and not self.lora_enabled:  # 如果融合函数存在且未启用LoRA
            return self.fused_func(dispatch_output, quant_info, self.config)  # 直接调用融合函数

        assert self.runner_core is not None  # 断言运行器核心必须存在

        def _maybe_build_lora_hooks(_runner_input: Any) -> LoRAHooks:
            """可能构建LoRA钩子

            Args:
                _runner_input: 运行器输入

            Returns:
                LoRA钩子或None
            """
            from sglang.srt.layers.moe.token_dispatcher.base import DispatchOutput  # 导入分发输出类
            from sglang.srt.lora.lora_moe_runners import build_lora_hooks  # 导入LoRA钩子构建函数

            if isinstance(_runner_input, DispatchOutput):  # 如果输入是分发输出类型
                hidden_states, topk_ids = (
                    _runner_input.hidden_states,
                    _runner_input.topk_output.topk_ids,
                )  # 从分发输出提取隐藏状态和TopK ID
            else:
                hidden_states = _runner_input.hidden_states  # 从运行器输入提取隐藏状态
                topk_ids = getattr(_runner_input, "topk_ids", None)  # 尝试获取TopK ID
            if self.lora_enabled and lora_info is not None:  # 如果LoRA已启用且LoRA信息存在
                return build_lora_hooks(
                    hidden_states,
                    lora_info,
                    topk_ids,
                )  # 构建并返回LoRA钩子
            return None  # 不构建LoRA钩子

        # Runners that handle dispatch_output directly (e.g., MarlinRunnerCore)
        # bypass the pre-permute step and do their own alignment internally.
        # 直接处理 dispatch_output 的运行器（如 MarlinRunnerCore）
        # 绕过预排列步骤并在内部进行自己的对齐。
        if hasattr(self.runner_core, "run_from_dispatch"):  # 如果运行器核心支持从分发输出运行
            hooks = _maybe_build_lora_hooks(dispatch_output)  # 构建LoRA钩子
            return self.runner_core.run_from_dispatch(
                dispatch_output, quant_info, self.config, hooks=hooks
            )  # 直接从分发输出运行

        dispatch_format = dispatch_output.format.value  # 获取分发格式值
        runner_format = self.runner_core.runner_backend.value  # 获取运行器后端格式值
        self.pre_permute_func = PermuteMethodPool.get_pre_permute(
            dispatch_format, runner_format
        )  # 从排列方法池获取预排列函数

        running_state = {}  # 初始化运行状态字典
        if self.down_gemm_overlap_args is not None:  # 如果下行GEMM重叠参数存在
            running_state["down_gemm_overlap_args"] = self.down_gemm_overlap_args  # 存入运行状态
        if self.meta_overlap_args is not None:  # 如果元数据重叠参数存在
            running_state["meta_overlap_args"] = self.meta_overlap_args  # 存入运行状态

        runner_input = self.pre_permute_func(
            dispatch_output, quant_info, self.config, running_state
        )  # 执行预排列，将分发输出转换为运行器输入

        hooks = _maybe_build_lora_hooks(runner_input)  # 为运行器输入构建LoRA钩子

        runner_output = self.runner_core.run(
            runner_input, quant_info, running_state, hooks=hooks
        )  # 运行核心计算
        runner_format = self.runner_core.runner_backend.value  # 获取运行器后端格式值
        combine_format = dispatch_output.format.value  # 获取合并格式值
        self.post_permute_func = PermuteMethodPool.get_post_permute(
            runner_format, combine_format
        )  # 从排列方法池获取后排列函数
        combine_input = self.post_permute_func(
            runner_output, quant_info, self.config, running_state
        )  # 执行后排列，将运行器输出转换为合并输入

        return combine_input  # 返回合并输入

    def set_overlap_args(
        self, down_gemm_overlap_args: DownGemmOverlapArgs, meta_overlap_args: dict
    ):
        """设置重叠参数

        Args:
            down_gemm_overlap_args: 下行GEMM重叠参数
            meta_overlap_args: 元数据重叠参数
        """
        self.down_gemm_overlap_args = down_gemm_overlap_args  # 存储下行GEMM重叠参数
        self.meta_overlap_args = meta_overlap_args  # 存储元数据重叠参数

    def clear_overlap_args(self) -> None:
        """清除重叠参数"""
        self.down_gemm_overlap_args = None  # 清除下行GEMM重叠参数
        self.meta_overlap_args = None  # 清除元数据重叠参数
