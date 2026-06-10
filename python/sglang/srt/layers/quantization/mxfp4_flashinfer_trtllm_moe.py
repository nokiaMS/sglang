# 文件说明：DeepSeek-V4 MXFP4专家后端，基于FlashInfer的TRT-LLM FP4 MoE内核实现
# 本模块实现了MXFP4量化下MoE层的FlashInfer TRT-LLM后端，
# 支持SM100架构上的FP4块缩放路由MoE计算，包含TopK ID打包和权重混洗等功能。
from __future__ import annotations # 启用延迟类型注解评估

import logging # 导入日志模块
from typing import TYPE_CHECKING # 导入类型检查常量

import torch # 导入PyTorch
import triton # 导入Triton编译框架
import triton.language as tl # 导入Triton语言
from torch.nn import Module # 导入神经网络模块基类
from torch.nn.parameter import Parameter # 导入参数类

from sglang.srt.distributed import get_tp_group # 导入张量并行组获取函数
from sglang.srt.distributed.device_communicators.pynccl_allocator import ( # 导入对称内存使用函数
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric # 导入分配对称性检查函数
from sglang.srt.layers.moe.utils import RoutingMethodType # 导入路由方法类型枚举
from sglang.srt.server_args import get_global_server_args # 导入全局服务器参数获取函数
from sglang.srt.utils import ( # 导入工具函数
    is_flashinfer_available, # FlashInfer可用性检查
    log_info_on_rank0, # rank0日志工具
    set_weight_attrs, # 权重属性设置工具
)
from sglang.srt.utils.common import is_sm100_supported, next_power_of_2 # 导入SM100支持检查和2的幂向上取整函数

_MXFP8_QUANTIZE_BACKEND = "cute-dsl" if is_sm100_supported() else "cuda" # 根据是否支持SM100选择MXFP8量化后端

if is_flashinfer_available(): # 如果FlashInfer可用
    from flashinfer import mxfp8_quantize, shuffle_matrix_a, shuffle_matrix_sf_a # 导入MXFP8量化和矩阵混洗函数
    from flashinfer.fp4_quantization import block_scale_interleave # 导入块缩放交错函数
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe # 导入TRT-LLM FP4块缩放路由MoE内核
    from flashinfer.fused_moe.core import ( # 导入融合MoE核心函数
        _maybe_get_cached_w3_w1_permute_indices, # 获取缓存的w3w1排列索引
        get_w2_permute_indices_with_cache, # 获取带缓存的w2排列索引
    )

logger = logging.getLogger(__name__) # 获取当前模块的日志记录器

if TYPE_CHECKING: # 仅在类型检查时导入，避免运行时循环依赖
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput # 导入组合输入和分发输出类型

from sglang.srt.utils.common import get_bool_env_var # 导入布尔环境变量获取函数

_USE_OFFICIAL_SHUFFLE = get_bool_env_var( # 是否使用官方混洗方法的标志
    "SGLANG_MXFP4_USE_OFFICIAL_SHUFFLE", default="true" # 默认为True，使用官方混洗
)


class PackTopkIds: # TopK ID打包工具类，将topk_ids和topk_weights打包为单个int32

    @classmethod # 类方法：执行TopK ID打包
    def execute(
        cls, topk_ids: torch.Tensor, topk_weights: torch.Tensor # 输入topk ID和权重张量
    ) -> torch.Tensor: # 返回打包后的int32张量
        return cls.triton(topk_ids, topk_weights) # 调用Triton实现进行打包

    @classmethod # 类方法：朴素Python实现打包
    def vanilla(
        cls, topk_ids: torch.Tensor, topk_weights: torch.Tensor # 输入topk ID和权重张量
    ) -> torch.Tensor: # 返回打包后的int32张量
        weight_bits = ( # 将权重转为bfloat16再提取低16位
            topk_weights.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF # 转为bfloat16，按int16查看，取低16位
        )
        return (topk_ids.to(torch.int32) << 16) | weight_bits # ID左移16位，与权重低16位或运算打包

    @classmethod # 类方法：Triton加速实现打包
    def triton(cls, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor: # 输入topk ID和权重，返回打包张量
        assert ( # 断言形状匹配
            topk_ids.shape == topk_weights.shape
        ), f"shape mismatch: {topk_ids.shape=} vs {topk_weights.shape=}" # 形状不匹配则报错
        assert topk_ids.ndim >= 1, f"expected >=1D, got {topk_ids.shape=}" # 断言至少1维

        assert ( # 断言topk_ids为int32类型
            topk_ids.dtype == torch.int32
        ), f"topk_ids must be int32, got {topk_ids.dtype}" # 类型不对则报错
        assert ( # 断言topk_weights为float32类型
            topk_weights.dtype == torch.float32
        ), f"topk_weights must be float32, got {topk_weights.dtype}" # 类型不对则报错

        assert topk_ids.is_contiguous(), "topk_ids must be contiguous" # 断言topk_ids内存连续
        assert topk_weights.is_contiguous(), "topk_weights must be contiguous" # 断言topk_weights内存连续

        out = torch.empty_like(topk_ids, dtype=torch.int32) # 创建与topk_ids同形状的int32输出张量
        numel = out.numel() # 获取元素总数
        if numel == 0: # 如果没有元素
            return out # 直接返回空张量

        BLOCK_SIZE = 1024 # 每个线程块处理1024个元素
        grid = (triton.cdiv(numel, BLOCK_SIZE),) # 计算网格大小
        _pack_topk_ids_triton_kernel[grid]( # 调用Triton内核进行打包
            topk_ids,
            topk_weights,
            out,
            numel,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out # 返回打包结果


@triton.jit # Triton JIT编译的TopK ID打包内核
def _pack_topk_ids_triton_kernel(
    topk_ids_ptr, # topk ID指针
    topk_weights_ptr, # topk权重指针
    out_ptr, # 输出指针
    numel, # 元素总数
    BLOCK_SIZE: tl.constexpr, # 线程块大小（编译时常量）
):
    pid = tl.program_id(0) # 获取当前程序ID
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) # 计算当前线程块处理的偏移量
    mask = offsets < numel # 生成有效元素掩码

    ids = tl.load(topk_ids_ptr + offsets, mask=mask, other=0) # 加载topk ID，越界用0填充
    w = tl.load(topk_weights_ptr + offsets, mask=mask, other=0.0) # 加载topk权重，越界用0.0填充

    w_bf16 = w.to(tl.bfloat16) # 将权重转为bfloat16
    w_i16 = w_bf16.to(tl.int16, bitcast=True) # 按位转换为int16
    w_i32 = w_i16.to(tl.int32) & 0xFFFF # 转为int32并取低16位

    ids_i32 = ids.to(tl.int32) # 将ID转为int32

    packed = (ids_i32 << 16) | w_i32 # 将ID左移16位后与权重或运算打包

    tl.store(out_ptr + offsets, packed, mask=mask) # 存储打包结果


class Mxfp4FlashinferTrtllmMoEMethod: # MXFP4 FlashInfer TRT-LLM MoE方法类

    def __init__(self, fp8_method, prefix: str): # 初始化方法，接收FP8方法实例和前缀名
        self._fp8 = fp8_method # 保存FP8基础方法引用
        self.prefix = prefix # 保存层前缀名
        self.flashinfer_mxfp4_moe_precision = ( # 获取MXFP4 MoE精度配置
            get_global_server_args().flashinfer_mxfp4_moe_precision # 从全局服务器参数读取
        )

    def create_moe_runner(self, layer, moe_runner_config): # 创建MoE运行器方法
        self.moe_runner_config = moe_runner_config # 保存MoE运行器配置

        swiglu_limit = moe_runner_config.swiglu_limit # 获取SwiGLU限幅配置
        assert ( # 断言限幅值非空
            swiglu_limit is not None
        ), f"swiglu_limit must be non-None for DeepSeek V4 (got {swiglu_limit!r})" # DeepSeek V4必须设置限幅值
        self._gemm1_clamp_limit_tensor = ( # 创建GEMM1限幅张量
            torch.full( # 创建填充指定值的张量
                (layer.num_local_experts,), # 形状为本地专家数量
                swiglu_limit, # 填充值为SwiGLU限幅
                dtype=torch.float32, # 数据类型为float32
                device=layer.w13_weight.device, # 设备与权重一致
            )
            if swiglu_limit is not None # 如果有限幅值
            else None # 否则为None
        )

    def create_weights( # 创建权重参数方法
        self,
        layer, # MoE层模块
        num_experts: int, # 专家数量
        hidden_size: int, # 隐藏层维度
        intermediate_size_per_partition: int, # 每个分区的中间层维度
        params_dtype, # 参数数据类型
        **extra_weight_attrs, # 额外权重属性
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported # 导入权重缩放支持类型枚举

        fp4_block_k = 32 # FP4块大小（每32个权重一个缩放因子）

        w13_weight = Parameter( # 创建FC1权重参数（gate+up，4位打包）
            torch.empty( # 创建空张量
                num_experts, # 专家数量
                2 * intermediate_size_per_partition, # gate+up的中间维度
                hidden_size // 2, # FP4打包后K维度减半
                dtype=torch.int8, # int8存储（每个元素包含2个4位权重）
            ),
            requires_grad=False, # 不需要梯度
        )
        w2_weight = Parameter( # 创建FC2权重参数（down投影，4位打包）
            torch.empty( # 创建空张量
                num_experts, # 专家数量
                hidden_size, # 隐藏维度
                intermediate_size_per_partition // 2, # FP4打包后中间维度减半
                dtype=torch.int8, # int8存储
            ),
            requires_grad=False, # 不需要梯度
        )
        layer.register_parameter("w13_weight", w13_weight) # 注册w13权重参数
        set_weight_attrs(w13_weight, extra_weight_attrs) # 设置权重属性
        layer.register_parameter("w2_weight", w2_weight) # 注册w2权重参数
        set_weight_attrs(w2_weight, extra_weight_attrs) # 设置权重属性

        w13_weight_scale = Parameter( # 创建FC1权重缩放因子参数
            torch.ones( # 创建全1张量
                num_experts, # 专家数量
                2 * intermediate_size_per_partition, # gate+up的中间维度
                hidden_size // fp4_block_k, # 每个块一个缩放因子
                dtype=torch.float32, # float32存储
            ),
            requires_grad=False, # 不需要梯度
        )
        w2_weight_scale = Parameter( # 创建FC2权重缩放因子参数
            torch.ones( # 创建全1张量
                num_experts, # 专家数量
                hidden_size, # 隐藏维度
                intermediate_size_per_partition // fp4_block_k, # 每个块一个缩放因子
                dtype=torch.float32, # float32存储
            ),
            requires_grad=False, # 不需要梯度
        )
        w13_weight_scale.format_ue8m0 = False # 标记缩放因子不是E8M0格式
        w2_weight_scale.format_ue8m0 = False # 标记缩放因子不是E8M0格式
        scale_attrs = dict(extra_weight_attrs) # 复制权重属性
        scale_attrs["quant_method"] = FusedMoeWeightScaleSupported.BLOCK.value # 标记使用块级缩放
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale) # 注册w13缩放因子参数
        set_weight_attrs(w13_weight_scale, scale_attrs) # 设置缩放因子属性
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale) # 注册w2缩放因子参数
        set_weight_attrs(w2_weight_scale, scale_attrs) # 设置缩放因子属性

    def process_weights_after_loading(self, layer: Module) -> None: # 加载后处理权重方法，进行重排和混洗
        from sglang.srt.layers.quantization.utils import reorder_w1w3_to_w3w1 # 导入权重重排工具函数

        self._fp8.process_weights_after_loading(layer) # 调用FP8基础方法的后加载处理

        if getattr(layer, "_mega_moe_weights_built", False): # 如果已经构建了mega MoE权重
            return # 直接返回，跳过处理

        w13_w, w13_s = reorder_w1w3_to_w3w1( # 将w13权重和缩放因子从[w1;w3]重排为[w3;w1]
            layer.w13_weight.data, layer.w13_weight_scale_inv.data
        )
        layer.w13_weight = Parameter(w13_w, requires_grad=False) # 更新w13权重为重排后的参数
        layer.w13_weight_scale_inv = Parameter(w13_s, requires_grad=False) # 更新w13缩放因子为重排后的参数

        log_info_on_rank0( # 在rank0上记录信息
            logger,
            f"Shuffling FP4 expert weights for TRT-LLM MxFP4 kernel " # 正在为TRT-LLM MxFP4内核混洗FP4专家权重
            f"(layer: {self.prefix})...", # （层：{前缀}）...
        )

        w13 = layer.w13_weight.data # 获取w13权重数据
        w2 = layer.w2_weight.data # 获取w2权重数据
        w13_scale = layer.w13_weight_scale_inv.data # 获取w13缩放因子数据
        w2_scale = layer.w2_weight_scale_inv.data # 获取w2缩放因子数据
        num_experts = w13.shape[0] # 获取专家数量

        if w13_scale.dtype == torch.float32: # 如果缩放因子是float32类型
            w13_scale = w13_scale.to(torch.float8_e8m0fnu) # 转换为E8M0格式
            w2_scale = w2_scale.to(torch.float8_e8m0fnu) # 转换为E8M0格式

        epilogue_tile_m = 128 # 尾声平铺的M维度大小
        g1_w, g1_s, g2_w, g2_s = [], [], [], [] # 初始化存储混洗后权重和缩放因子的列表
        if _USE_OFFICIAL_SHUFFLE: # 如果使用官方混洗方法
            cache: dict = {} # 创建排列索引缓存
            for i in range(num_experts): # 遍历每个专家
                w13_u8 = w13[i].view(torch.uint8) # 将w13权重按uint8查看
                w13_s_u8 = w13_scale[i].view(torch.uint8) # 将w13缩放因子按uint8查看
                w2_u8 = w2[i].view(torch.uint8) # 将w2权重按uint8查看
                w2_s_u8 = w2_scale[i].view(torch.uint8) # 将w2缩放因子按uint8查看

                perm = _maybe_get_cached_w3_w1_permute_indices( # 获取w3w1排列索引（带缓存）
                    cache,
                    w13_u8,
                    epilogue_tile_m,
                )
                g1_w.append(w13_u8[perm.to(w13_u8.device)].contiguous()) # 应用排列并添加到列表
                perm_sf = _maybe_get_cached_w3_w1_permute_indices( # 获取w3w1缩放因子排列索引（带缓存）
                    cache,
                    w13_s_u8,
                    epilogue_tile_m,
                    num_elts_per_sf=16, # 每个缩放因子对应16个元素
                )
                g1_s.append( # 应用排列和块缩放交错后添加到列表
                    block_scale_interleave(
                        w13_s_u8[perm_sf.to(w13_s_u8.device)].contiguous()
                    )
                )

                perm = get_w2_permute_indices_with_cache( # 获取w2排列索引（带缓存）
                    cache,
                    w2_u8,
                    epilogue_tile_m,
                )
                g2_w.append(w2_u8[perm.to(w2_u8.device)].contiguous()) # 应用排列并添加到列表
                perm_sf = get_w2_permute_indices_with_cache( # 获取w2缩放因子排列索引（带缓存）
                    cache,
                    w2_s_u8,
                    epilogue_tile_m,
                    num_elts_per_sf=16, # 每个缩放因子对应16个元素
                )
                g2_s.append( # 应用排列和块缩放交错后添加到列表
                    block_scale_interleave(
                        w2_s_u8[perm_sf.to(w2_s_u8.device)].contiguous()
                    )
                )
        else: # 不使用官方混洗方法
            for i in range(num_experts): # 遍历每个专家
                g1_w.append(shuffle_matrix_a(w13[i].view(torch.uint8), epilogue_tile_m)) # 使用FlashInfer混洗w13权重
                g1_s.append( # 使用FlashInfer混洗w13缩放因子
                    shuffle_matrix_sf_a(w13_scale[i].view(torch.uint8), epilogue_tile_m)
                )
                g2_w.append(shuffle_matrix_a(w2[i].view(torch.uint8), epilogue_tile_m)) # 使用FlashInfer混洗w2权重
                g2_s.append( # 使用FlashInfer混洗w2缩放因子
                    shuffle_matrix_sf_a(w2_scale[i].view(torch.uint8), epilogue_tile_m)
                )

        layer.w13_weight = Parameter(torch.stack(g1_w), requires_grad=False) # 将混洗后的w13权重堆叠并更新
        layer.w13_weight_scale_inv = Parameter( # 将混洗后的w13缩放因子堆叠、格式转换并更新
            torch.stack(g1_s)
            .view(torch.float8_e4m3fn) # 按e4m3格式查看
            .reshape(num_experts, w13.shape[1], -1), # 重塑形状
            requires_grad=False,
        )
        layer.w2_weight = Parameter(torch.stack(g2_w), requires_grad=False) # 将混洗后的w2权重堆叠并更新
        layer.w2_weight_scale_inv = Parameter( # 将混洗后的w2缩放因子堆叠、格式转换并更新
            torch.stack(g2_s)
            .view(torch.float8_e4m3fn) # 按e4m3格式查看
            .reshape(num_experts, w2.shape[1], -1), # 重塑形状
            requires_grad=False,
        )

        self._register_static_scale_ones(layer) # 注册静态缩放1.0张量
        torch.cuda.empty_cache() # 清空CUDA缓存，释放不再需要的GPU内存

    def _register_static_scale_ones(self, layer: Module) -> None: # 注册静态缩放因子为1.0的张量
        device = layer.w13_weight.device # 获取权重所在设备
        for name in ( # 遍历需要注册的缩放因子名称
            "output1_scale_scalar", # 输出1缩放标量
            "output1_scale_gate_scalar", # 输出1门控缩放标量
            "output2_scale_scalar", # 输出2缩放标量
        ):
            layer.register_buffer( # 注册缓冲区
                name,
                torch.ones(layer.num_local_experts, device=device, dtype=torch.float32), # 全1张量
                persistent=False, # 非持久化，不保存到状态字典
            )

    def apply( # 应用方法，执行MoE前向推理
        self,
        layer: Module, # MoE层模块
        dispatch_output: DispatchOutput, # 分发输出
    ) -> CombineInput: # 返回组合输入
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput # 导入标准组合输入类
        from sglang.srt.layers.moe.topk import TopKOutputChecker # 导入TopK输出检查器

        hidden_states = dispatch_output.hidden_states # 获取隐藏状态
        topk_output = dispatch_output.topk_output # 获取TopK路由输出

        w13 = layer.w13_weight # 获取FC1权重
        w2 = layer.w2_weight # 获取FC2权重
        w13_scale = layer.w13_weight_scale_inv # 获取FC1缩放因子
        w2_scale = layer.w2_weight_scale_inv # 获取FC2缩放因子

        intermediate_size = w2.shape[2] * 2 if w2.dtype == torch.uint8 else w2.shape[2] # 计算中间维度（uint8时需要乘2）
        hidden_size = w13.shape[2] * 2 if w13.dtype == torch.uint8 else w13.shape[2] # 计算隐藏维度（uint8时需要乘2）

        num_local_experts = layer.num_local_experts # 获取本地专家数量
        if w13_scale.dim() == 2: # 如果缩放因子是2维的
            w13_scale = w13_scale.reshape(num_local_experts, 2 * intermediate_size, -1) # 重塑为3维
        if w2_scale.dim() == 2: # 如果缩放因子是2维的
            w2_scale = w2_scale.reshape(num_local_experts, hidden_size, -1) # 重塑为3维

        if TopKOutputChecker.format_is_standard(topk_output): # 如果TopK输出为标准格式
            topk_ids = topk_output.topk_ids # 获取topk ID
            topk_weights = topk_output.topk_weights # 获取topk权重
        elif TopKOutputChecker.format_is_bypassed(topk_output): # 如果TopK输出为旁路格式
            raise NotImplementedError( # 抛出未实现错误
                "the old code in this branch is WRONG. e.g. it does not consider HashTopK, and may miss args" # 旧代码有误，未考虑HashTopK等
            )
        else: # 其他不支持的格式
            raise ValueError(f"Unsupported topk output format: {topk_output.format}") # 抛出不支持错误

        packed_topk = PackTopkIds.execute(topk_ids, topk_weights) # 打包topk ID和权重

        precision = self.flashinfer_mxfp4_moe_precision # 获取精度配置
        if precision == "bf16": # 如果使用bf16精度（不量化激活）
            assert hidden_states.dtype == torch.bfloat16 # 断言输入为bfloat16
            x_quant = hidden_states # 直接使用原始隐藏状态
            x_scale = None # 无缩放因子
            origin_dim = x_quant.shape[-1] # 记录原始维度
            if hidden_size != origin_dim: # 如果需要填充
                x_quant = torch.nn.functional.pad( # 对隐藏状态进行零填充
                    x_quant,
                    (0, hidden_size - origin_dim), # 填充到最后维度
                    mode="constant", # 常量填充
                    value=0.0, # 填充值0
                )
        elif precision == "default": # 默认精度（使用MXFP8量化激活）
            x_quant, x_scale = mxfp8_quantize( # 对输入进行MXFP8量化
                hidden_states, # 输入隐藏状态
                False, # 不转置
                alignment=hidden_size, # 对齐到隐藏维度
                backend=_MXFP8_QUANTIZE_BACKEND, # 使用选定的量化后端
            )
            x_scale = x_scale.view(torch.float8_e4m3fn).reshape( # 将缩放因子重塑
                *hidden_states.shape[:-1], -1 # 保留前面的维度，最后维度自动计算
            )
        else: # 不支持的精度配置
            raise NotImplementedError(f"Unsupported mxfp4 moe precision: {precision}") # 抛出未实现错误

        with use_symmetric_memory( # 使用对称内存分配输出张量
            get_tp_group(), disabled=not is_allocation_symmetric() # 仅在对称分配时启用
        ):
            num_tokens = x_quant.shape[0] # 获取token数量
            out_hidden_size = ( # 计算输出隐藏维度
                x_quant.shape[-1] * 2 # 如果是uint8（4位打包），乘2
                if x_quant.dtype == torch.uint8 # 判断是否为uint8
                else x_quant.shape[-1] # 否则直接使用
            )
            symm_output = torch.empty( # 创建对称内存输出张量
                num_tokens, out_hidden_size, dtype=torch.bfloat16, device=x_quant.device # bfloat16类型
            )

        output = trtllm_fp4_block_scale_routed_moe( # 调用TRT-LLM FP4块缩放路由MoE内核
            topk_ids=packed_topk, # 打包的topk ID（含权重）
            routing_bias=None, # 无路由偏置
            hidden_states=x_quant, # 量化后的隐藏状态
            hidden_states_scale=x_scale, # 隐藏状态的缩放因子
            gemm1_weights=w13, # GEMM1权重（FC1）
            gemm1_weights_scale=w13_scale, # GEMM1缩放因子
            gemm1_bias=None, # 无GEMM1偏置
            gemm1_alpha=None, # 无GEMM1 alpha
            gemm1_beta=None, # 无GEMM1 beta
            gemm1_clamp_limit=self._gemm1_clamp_limit_tensor, # GEMM1的SwiGLU限幅张量
            gemm2_weights=w2, # GEMM2权重（FC2）
            gemm2_weights_scale=w2_scale, # GEMM2缩放因子
            gemm2_bias=None, # 无GEMM2偏置
            output1_scale_scalar=layer.output1_scale_scalar, # 输出1缩放标量
            output1_scale_gate_scalar=layer.output1_scale_gate_scalar, # 输出1门控缩放标量
            output2_scale_scalar=layer.output2_scale_scalar, # 输出2缩放标量
            num_experts=layer.num_experts, # 专家总数
            top_k=packed_topk.shape[1], # top-k值
            n_group=1, # 分组数
            topk_group=1, # topk分组数
            intermediate_size=intermediate_size, # 中间维度
            local_expert_offset=layer.moe_ep_rank * layer.num_local_experts, # 本地专家偏移
            local_num_experts=num_local_experts, # 本地专家数量
            routed_scaling_factor=1.0, # 路由缩放因子
            routing_method_type=int(RoutingMethodType.TopK), # 路由方法类型
            do_finalize=True, # 是否执行最终化
            tune_max_num_tokens=next_power_of_2(x_quant.shape[0]), # 调优最大token数为2的幂
            output=symm_output, # 输出张量
        )[0] # 取第一个返回值

        return StandardCombineInput(hidden_states=output) # 返回标准组合输入


def maybe_fuse_routed_scale_and_shared_add( # 可能融合路由缩放与共享加法
    experts, # 专家模块
    routed: torch.Tensor, # 路由专家输出
    shared: torch.Tensor | None, # 共享专家输出（可能为空）
    routed_scaling_factor: float, # 路由缩放因子
) -> torch.Tensor: # 返回最终输出张量
    # When MxFP4 fusion is on, the upstream `routed *= scale` is skipped and # 当MxFP4融合开启时，上游的`routed *= scale`被跳过，
    # the scaling is folded into the shared-add via `shared.add_(routed, # 缩放被折叠到共享加法中，通过`shared.add_(routed,
    # alpha=scale)`. With no shared output, the missing scale is applied # alpha=scale)`。如果没有共享输出，缺失的缩放
    # in-place. Otherwise `routed` is already scale-final and we just add # 原地应用。否则`routed`已经是最终缩放值，只需加上
    # `shared` (or pass through if there is none). # `shared`（如果没有则直接通过）。
    from sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe import ( # 导入FlashInfer Cutlass MoE方法
        Mxfp4FlashinferCutlassMoEMethod,
    )
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import ( # 导入Marlin MoE方法
        Mxfp4MarlinMoEMethod,
    )

    fused = isinstance( # 检查是否为融合量化方法
        experts.quant_method,
        (
            Mxfp4FlashinferTrtllmMoEMethod, # TRT-LLM MoE方法
            Mxfp4FlashinferCutlassMoEMethod, # FlashInfer Cutlass MoE方法
            Mxfp4MarlinMoEMethod, # Marlin MoE方法
        ),
    )
    if fused: # 如果是融合方法
        if shared is not None: # 如果有共享专家输出
            return shared.add_(routed, alpha=routed_scaling_factor) # 融合缩放和加法：shared += alpha * routed
        return routed.mul_(routed_scaling_factor) # 无共享输出时，原地应用缩放
    if shared is not None: # 非融合方法，有共享专家输出
        routed += shared # 路由输出加上共享输出（路由已在别处缩放）
    return routed # 返回最终输出
