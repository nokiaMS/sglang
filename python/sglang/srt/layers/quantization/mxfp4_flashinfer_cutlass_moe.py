# 文件说明：DeepSeek-V4 MXFP4专家后端，基于FlashInfer的SM90 cutlass混合输入MoE GEMM实现
# 本模块实现了MXFP4量化下MoE（混合专家）层的FlashInfer SM90 cutlass后端，
# 支持GEMM1 + SwiGLU + GEMM2的融合计算，适用于PD分离式预填充工作节点。
"""DeepSeek-V4 MXFP4 expert backend backed by FlashInfer's SM90 cutlass # DeepSeek-V4 MXFP4专家后端，基于FlashInfer的SM90 cutlass
mixed-input MoE GEMM (FlashInfer PR #3084). # 混合输入MoE GEMM（FlashInfer PR #3084）。

Sibling of :class:`Mxfp4MarlinMoEMethod` and :class:`Mxfp4FlashinferTrtllmMoEMethod`. # 与Mxfp4MarlinMoEMethod和Mxfp4FlashinferTrtllmMoEMethod为兄弟类。
Wired into :func:`Fp8MoEConfig.get_quant_method` when # 在Fp8MoEConfig.get_quant_method中接入，当
``is_fp4_experts=True`` and ``--moe-runner-backend flashinfer_mxfp4`` is # is_fp4_experts=True且--moe-runner-backend flashinfer_mxfp4
selected on a Hopper (SM90) device. SM100 still routes to # 在Hopper(SM90)设备上被选中时。SM100仍路由到
:class:`Mxfp4FlashinferTrtllmMoEMethod` (trtllm-gen). # Mxfp4FlashinferTrtllmMoEMethod（trtllm-gen）。

Performance trade-off vs Marlin (kernel-level on H100, GPT-OSS-like body): # 相比Marlin的性能权衡（H100上内核级，GPT-OSS风格主体）：
  - decode  (M <=   64) :  Marlin     +12-15 % # 解码(M<=64)：Marlin快12-15%
  - tie     (M ~=  256) # 持平(M约256)
  - prefill (M >= 1024) :  FlashInfer +24-36 % # 预填充(M>=1024)：FlashInfer快24-36%

PD-disaggregated prefill workers are the natural fit; decode workers should # PD分离式预填充工作节点是自然选择；解码工作节点应
keep the Marlin default. # 保持Marlin默认后端。
"""

from __future__ import annotations # 启用延迟类型注解评估

import logging # 导入日志模块
import os # 导入操作系统模块
from typing import TYPE_CHECKING # 导入类型检查常量

import torch # 导入PyTorch
from torch.nn import Module # 导入神经网络模块基类
from torch.nn.parameter import Parameter # 导入参数类

from sglang.srt.layers.moe.topk import TopKOutputChecker # 导入TopK输出检查器
from sglang.srt.utils import is_flashinfer_available, log_info_on_rank0 # 导入FlashInfer可用性检查和rank0日志工具

# Silence the TRT-LLM cutlass autotune trace embedded inside FlashInfer's # 静默FlashInfer内嵌的TRT-LLM cutlass自动调优跟踪，
# cutlass_fused_moe. Its C++ logger reads TLLM_LOG_LEVEL on first kernel launch; # 其C++日志器在首次内核启动时读取TLLM_LOG_LEVEL；
# setdefault preserves any explicit user override. # setdefault保留用户显式覆盖的值。
os.environ.setdefault("TLLM_LOG_LEVEL", "INFO") # 设置默认日志级别为INFO，避免TRT-LLM输出过多调试信息

if is_flashinfer_available(): # 如果FlashInfer可用
    try: # 尝试导入SM90 cutlass混合输入MoE相关函数
        from flashinfer.fused_moe import ( # 从FlashInfer融合MoE模块导入
            interleave_moe_scales_for_sm90_mixed_gemm, # 导入SM90混合GEMM的尺度交错函数
            interleave_moe_weights_for_sm90_mixed_gemm, # 导入SM90混合GEMM的权重交错函数
        )

        _FI_HAS_SM90_CUTLASS_MXFP4 = True # 标记FlashInfer支持SM90 cutlass MXFP4
    except ImportError: # 如果导入失败（版本过旧）
        interleave_moe_scales_for_sm90_mixed_gemm = None # 尺度交错函数置空
        interleave_moe_weights_for_sm90_mixed_gemm = None # 权重交错函数置空
        _FI_HAS_SM90_CUTLASS_MXFP4 = False # 标记不支持
else: # FlashInfer不可用
    _FI_HAS_SM90_CUTLASS_MXFP4 = False # 标记不支持SM90 cutlass MXFP4

logger = logging.getLogger(__name__) # 获取当前模块的日志记录器

if TYPE_CHECKING: # 仅在类型检查时导入，避免运行时循环依赖
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput # 导入组合输入和分发输出类型

# MXFP4 group/block size (E8M0 scale per 32 fp4 weights). # MXFP4分组/块大小（每32个fp4权重一个E8M0缩放因子）。
_GROUP_SIZE = 32 # 分组大小，每32个FP4权重共享一个E8M0缩放因子


class Mxfp4FlashinferCutlassMoEMethod: # MXFP4 FlashInfer Cutlass MoE方法类
    """DeepSeek-V4 W4A16 MXFP4 MoE via FlashInfer's SM90 mixed-input cutlass # 通过FlashInfer的SM90混合输入cutlass实现的DeepSeek-V4 W4A16 MXFP4 MoE
    grouped GEMM. The fused kernel does GEMM1 + clamped SwiGLU + GEMM2 in one # 分组GEMM。融合内核在一次调用中完成GEMM1+限幅SwiGLU+GEMM2，
    call after a one-shot weight/scale interleave at load time.""" # 在加载时进行一次性权重/尺度交错后执行。

    def __init__(self, fp8_method, prefix: str): # 初始化方法，接收FP8方法实例和前缀名
        if not _FI_HAS_SM90_CUTLASS_MXFP4: # 检查FlashInfer是否支持SM90 cutlass MXFP4
            raise RuntimeError( # 抛出运行时错误
                "Mxfp4FlashinferCutlassMoEMethod requires FlashInfer >= 0.6.11 " # 需要FlashInfer 0.6.11或更高版本
                "(PR #3084 SM90 mixed-input helpers). Older builds lack " # （PR #3084 SM90混合输入辅助函数）。旧版本缺少
                "interleave_moe_{weights,scales}_for_sm90_mixed_gemm; " # interleave_moe_{weights,scales}_for_sm90_mixed_gemm函数；
                "either upgrade flashinfer-python or fall back to " # 请升级flashinfer-python或回退到
                "--moe-runner-backend marlin." # --moe-runner-backend marlin后端。
            )
        self._fp8 = fp8_method # 保存FP8基础方法引用
        self.prefix = prefix # 保存层前缀名
        self._swiglu_alpha_tensor: torch.Tensor | None = None # SwiGLU的alpha缩放张量（标准SiLU为1.0）
        self._swiglu_beta_tensor: torch.Tensor | None = None # SwiGLU的beta偏移张量（标准SiLU为0.0）
        self._swiglu_limit_tensor: torch.Tensor | None = None # SwiGLU的激活限幅张量

    # --- Lifecycle --------------------------------------------------------- # --- 生命周期方法 ---

    def create_weights( # 创建权重参数方法
        self,
        layer: Module, # MoE层模块
        num_experts: int, # 专家数量
        hidden_size: int, # 隐藏层维度
        intermediate_size_per_partition: int, # 每个分区的中间层维度
        params_dtype, # 参数数据类型
        **extra_weight_attrs, # 额外权重属性
    ):
        # SM90 mixed-input GEMM: contraction dim K must be a multiple of 128 # SM90混合输入GEMM：收缩维度K必须是128的倍数
        # (interleave factor = 128 / group_size = 4). For DSv4 (hidden=7168, # （交错因子=128/group_size=4）。对于DSv4（hidden=7168，
        # inter=2048) both are already multiples of 128; we assert rather than # inter=2048）两者都已是128的倍数；我们使用断言而不是
        # silently pad here, since padding the FP8-base buffers in-place would # 静默填充，因为原地填充FP8基础缓冲区
        # require deeper changes. # 需要更深层的修改。
        if hidden_size % 128 != 0 or intermediate_size_per_partition % 128 != 0: # 检查隐藏维度和中间维度是否为128的倍数
            raise ValueError( # 抛出数值错误
                "Mxfp4FlashinferCutlassMoEMethod requires hidden_size and " # Mxfp4FlashinferCutlassMoEMethod要求hidden_size和
                "intermediate_size_per_partition to be multiples of 128 " # intermediate_size_per_partition必须是128的倍数
                f"(got hidden={hidden_size}, " # （得到hidden=
                f"intermediate={intermediate_size_per_partition})." # intermediate=
            )
        # Raw weight shapes match what the fp8 base method allocates for fp4 # 原始权重形状与fp8基础方法为fp4
        # experts (uint8 4-bit packed weights, fp32 E8M0 scales). Delegate. # 专家分配的形状一致（uint8 4位打包权重，fp32 E8M0缩放）。委托处理。
        self._fp8.create_weights( # 委托FP8基础方法创建权重
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    def create_moe_runner(self, layer: Module, moe_runner_config) -> None: # 创建MoE运行器方法
        from sglang.srt.layers.moe.moe_runner.runner import MoeRunner # 导入MoE运行器
        from sglang.srt.layers.moe.utils import MoeRunnerBackend # 导入MoE运行器后端枚举

        self.moe_runner_config = moe_runner_config # 保存MoE运行器配置

        # DSv4 uses standard SwiGLU plus a config-driven activation clamp. # DSv4使用标准SwiGLU加配置驱动的激活限幅。
        # We pass all three (alpha, beta, limit) as explicit per-expert tensors # 我们将三个参数（alpha、beta、limit）都作为显式的逐专家张量传入
        # rather than mixing tensors with None: the cutlass SwiGLU kernel # 而不是混合张量和None：cutlass SwiGLU内核
        # branches on whether each is None, and partial-None inputs land in # 会根据每个是否为None分支，部分为None的输入会进入
        # less-tested code paths. ``alpha=1.0``, ``beta=0.0`` reproduce plain # 较少测试的代码路径。``alpha=1.0``、``beta=0.0``复现普通
        # ``silu(gate) * up``; ``limit`` enforces the activation clamp the # ``silu(gate) * up``；``limit``执行训练时使用的激活限幅，
        # checkpoint was trained with. # 即检查点训练时的限幅。
        swiglu_limit = getattr(moe_runner_config, "swiglu_limit", None) # 获取SwiGLU限幅配置
        if swiglu_limit is not None: # 如果配置了限幅值
            E = layer.num_local_experts # 获取本地专家数量
            device = layer.w13_weight.device # 获取权重所在设备
            self._swiglu_alpha_tensor = torch.ones( # 创建alpha张量，全1（标准SiLU门控）
                E, dtype=torch.float32, device=device
            )
            self._swiglu_beta_tensor = torch.zeros( # 创建beta张量，全0（标准up通路）
                E, dtype=torch.float32, device=device
            )
            self._swiglu_limit_tensor = torch.full( # 创建限幅张量，填充为配置的限幅值
                (E,), float(swiglu_limit), dtype=torch.float32, device=device
            )
        else: # 如果没有配置限幅值
            self._swiglu_alpha_tensor = None # alpha张量置空
            self._swiglu_beta_tensor = None # beta张量置空
            self._swiglu_limit_tensor = None # 限幅张量置空

        # Register the fused func at runner construction so the FusedOpPool # 在运行器构造时注册融合函数，以便FusedOpPool
        # lookup at `MoeRunner.__init__` finds it. # 在MoeRunner.__init__查找时能找到。
        import sglang.srt.layers.moe.moe_runner.flashinfer_mxfp4  # noqa: F401 # 导入FlashInfer MXFP4运行器模块（注册融合函数）

        self.runner = MoeRunner(MoeRunnerBackend.FLASHINFER_MXFP4, moe_runner_config) # 创建FlashInfer MXFP4 MoE运行器

    def process_weights_after_loading(self, layer: Module) -> None: # 加载后处理权重方法，进行权重重排和交错
        from sglang.srt.layers.quantization.utils import reorder_w1w3_to_w3w1 # 导入权重重排工具函数

        # Run the fp8 base hook first (ROCm normalization, mxfp8 requant, ...). # 先运行FP8基础钩子（ROCm归一化、mxfp8重量化等）。
        self._fp8.process_weights_after_loading(layer) # 调用FP8基础方法的后加载处理

        if getattr(layer, "_mega_moe_weights_built", False): # 如果已经构建了mega MoE权重
            return # 直接返回，跳过处理

        # cutlass_fused_moe expects fc1 in [w3; w1] = [up; gate] order, just # cutlass_fused_moe期望fc1以[w3; w1]=[up; gate]顺序排列，就像
        # like the trtllm-gen path. The HF / FP8 loader emits [w1; w3]. # trtllm-gen路径一样。HF/FP8加载器输出的是[w1; w3]顺序。
        w13, w13_s = reorder_w1w3_to_w3w1( # 将w13权重和缩放因子从[w1;w3]重排为[w3;w1]
            layer.w13_weight.data, layer.w13_weight_scale_inv.data
        )
        layer.w13_weight = Parameter(w13, requires_grad=False) # 更新w13权重为重排后的参数
        layer.w13_weight_scale_inv = Parameter(w13_s, requires_grad=False) # 更新w13缩放因子为重排后的参数

        log_info_on_rank0( # 在rank0上记录信息
            logger,
            f"Preparing DSv4 MXFP4 experts for FlashInfer SM90 cutlass " # 正在为FlashInfer SM90 cutlass准备DSv4 MXFP4专家
            f"(layer: {self.prefix})...", # （层：{前缀}）...
        )

        # FP8 base stores scales as fp32 numerical values (= 2**e). The # FP8基础方法以fp32数值（=2**e）存储缩放因子。
        # FlashInfer SM90 helper reads raw E8M0 bytes (uint8 with the # FlashInfer SM90辅助函数读取原始E8M0字节（uint8带
        # exponent + 127 bias). Cast through float8_e8m0fnu to extract the # 指数+127偏移）。通过float8_e8m0fnu转换以提取
        # raw byte without losing the exponent. # 原始字节而不丢失指数。
        w13_scale_u8 = ( # 将w13缩放因子转换为E8M0原始字节
            layer.w13_weight_scale_inv.data.to(torch.float8_e8m0fnu) # 先转为float8_e8m0fnu类型
            .view(torch.uint8) # 再按uint8查看以获取原始字节
            .contiguous() # 确保内存连续
        )
        w2_scale_u8 = ( # 将w2缩放因子转换为E8M0原始字节
            layer.w2_weight_scale_inv.data.to(torch.float8_e8m0fnu) # 先转为float8_e8m0fnu类型
            .view(torch.uint8) # 再按uint8查看以获取原始字节
            .contiguous() # 确保内存连续
        )

        # C++ byte interleave on packed 4-bit weights. # 对打包的4位权重进行C++字节交错。
        w13_il = interleave_moe_weights_for_sm90_mixed_gemm( # 对w13权重进行SM90混合GEMM交错
            layer.w13_weight.data.view(torch.uint8).contiguous(), "fp4" # 将权重按uint8查看并进行fp4交错
        )
        w2_il = interleave_moe_weights_for_sm90_mixed_gemm( # 对w2权重进行SM90混合GEMM交错
            layer.w2_weight.data.view(torch.uint8).contiguous(), "fp4" # 将权重按uint8查看并进行fp4交错
        )
        # Pure-PyTorch reshape+permute on E8M0 block scales. # 对E8M0块缩放因子进行纯PyTorch reshape+permute操作。
        w13_s_il = interleave_moe_scales_for_sm90_mixed_gemm( # 对w13缩放因子进行SM90混合GEMM交错
            w13_scale_u8, group_size=_GROUP_SIZE # 使用指定的分组大小
        )
        w2_s_il = interleave_moe_scales_for_sm90_mixed_gemm( # 对w2缩放因子进行SM90混合GEMM交错
            w2_scale_u8, group_size=_GROUP_SIZE # 使用指定的分组大小
        )

        layer.w13_weight = Parameter(w13_il, requires_grad=False) # 更新w13权重为交错后的参数
        layer.w2_weight = Parameter(w2_il, requires_grad=False) # 更新w2权重为交错后的参数
        layer.w13_weight_scale_inv = Parameter(w13_s_il, requires_grad=False) # 更新w13缩放因子为交错后的参数
        layer.w2_weight_scale_inv = Parameter(w2_s_il, requires_grad=False) # 更新w2缩放因子为交错后的参数

        layer._dsv4_mxfp4_backend = "flashinfer_cutlass_sm90" # 标记使用flashinfer_cutlass_sm90后端
        torch.cuda.empty_cache() # 清空CUDA缓存，释放不再需要的GPU内存

    # --- Forward ----------------------------------------------------------- # --- 前向推理方法 ---

    def apply( # 应用方法，执行MoE前向推理
        self,
        layer: Module, # MoE层模块
        dispatch_output: "DispatchOutput", # 分发输出，包含隐藏状态和TopK结果
    ) -> "CombineInput": # 返回组合输入
        from sglang.srt.layers.moe.moe_runner.flashinfer_mxfp4 import ( # 导入FlashInfer MXFP4 Cutlass MoE量化信息类
            FlashInferMxfp4CutlassMoeQuantInfo,
        )

        # DSv4 always feeds StandardDispatchOutput; the fused func tolerates # DSv4始终提供StandardDispatchOutput；融合函数也能容忍
        # bypassed too but we keep the strict check here as a contract guard. # bypassed格式，但这里保持严格检查作为契约保障。
        topk_output = dispatch_output.topk_output # 获取TopK路由输出
        if not TopKOutputChecker.format_is_standard(topk_output): # 检查TopK输出格式是否为标准格式
            raise ValueError(f"Unsupported topk output format: {topk_output.format}") # 不支持则抛出错误

        quant_info = FlashInferMxfp4CutlassMoeQuantInfo( # 构建FlashInfer MXFP4 Cutlass MoE量化信息
            w13_weight=layer.w13_weight, # FC1权重（gate+up，交错后）
            w2_weight=layer.w2_weight, # FC2权重（down投影）
            w13_weight_scale=layer.w13_weight_scale_inv, # FC1权重的缩放因子
            w2_weight_scale=layer.w2_weight_scale_inv, # FC2权重的缩放因子
            w13_bias=None,  # DSv4 has no MoE expert bias. # DSv4没有MoE专家偏置。
            w2_bias=None, # FC2无偏置
            swiglu_alpha=self._swiglu_alpha_tensor,  # ones: standard SiLU gate # 全1：标准SiLU门控
            swiglu_beta=self._swiglu_beta_tensor,  # zeros: standard up # 全0：标准up通路
            swiglu_limit=self._swiglu_limit_tensor, # SwiGLU激活限幅张量
            moe_tp_size=layer.moe_tp_size, # MoE张量并行大小
            moe_tp_rank=layer.moe_tp_rank, # MoE张量并行秩
            moe_ep_size=layer.moe_ep_size, # MoE专家并行大小
            moe_ep_rank=layer.moe_ep_rank, # MoE专家并行秩
            padded_hidden=None,  # DSv4 hidden_size is already a multiple of 128. # DSv4的hidden_size已是128的倍数，无需填充。
        )
        return self.runner.run(dispatch_output, quant_info) # 运行MoE运行器并返回结果
