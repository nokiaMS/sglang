# CPU图运行器模块，使用CPU上的torch.compile加速模型解码推理
# Copyright 2023-2024 SGLang Team  # 版权所有2023-2024 SGLang团队
# Licensed under the Apache License, Version 2.0 (the "License");  # 根据Apache许可证2.0版授权
# you may not use this file except in compliance with the License.  # 除非遵守许可证，否则不得使用此文件
# You may obtain a copy of the License at  # 可在以下地址获取许可证
#
#     http://www.apache.org/licenses/LICENSE-2.0  # Apache许可证2.0的URL
#
# Unless required by applicable law or agreed to in writing, software  # 除非适用法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 分发的软件按"原样"提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何明示或暗示的保证
# See the License for the specific language governing permissions and  # 请参阅许可证以了解管理权限和
# limitations under the License.  # 限制的特定语言
# ==============================================================================
"""Run the model with cpu torch compile."""  # 使用CPU torch.compile运行模型

# The implementation of CPUGraphRunner follows the CudaGraphRunner  # CPUGraphRunner的实现参照了CudaGraphRunner

from __future__ import annotations  # 启用延迟注解评估

import bisect  # 导入二分查找模块
import logging  # 导入日志模块
from contextlib import contextmanager  # 导入上下文管理器装饰器
from typing import TYPE_CHECKING, Callable, Optional, Union  # 导入类型提示

import psutil  # 导入系统进程和内存监控模块
import torch  # 导入PyTorch框架
import tqdm  # 导入进度条模块

from sglang.srt.distributed import get_tensor_model_parallel_rank  # 导入获取张量模型并行排名函数
from sglang.srt.distributed.parallel_state import GroupCoordinator  # 导入并行组协调器类
from sglang.srt.layers.logits_processor import LogitsProcessorOutput  # 导入逻辑处理器输出类
from sglang.srt.model_executor.forward_batch_info import (  # 导入前向批处理信息模块
    CaptureHiddenMode,  # 捕获隐藏状态模式
    ForwardBatch,  # 前向批处理类
    ForwardMode,  # 前向模式枚举
    PPProxyTensors,  # 流水线并行代理张量
    enable_num_token_non_padded,  # 启用非填充token数量标志
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context  # 导入前向上下文类和上下文管理器
from sglang.srt.utils import (  # 导入工具函数
    log_info_on_rank0,  # 在rank 0上记录信息日志
    require_attn_tp_gather,  # 是否需要注意力TP聚合
    require_gathered_buffer,  # 是否需要聚合缓冲区
    require_mlp_sync,  # 是否需要MLP同步
    require_mlp_tp_gather,  # 是否需要MLP TP聚合
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_compile  # 导入torch.compile猴子补丁函数

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器

if TYPE_CHECKING:  # 仅在类型检查时执行导入
    from sglang.srt.model_executor.model_runner import ModelRunner  # 导入模型运行器类


@contextmanager
def patch_model(  # 上下文管理器，对模型进行补丁以兼容torch.compile
    model: torch.nn.Module,  # PyTorch模型
    enable_compile: bool,  # 是否启用编译
    num_tokens: int,  # token数量
    tp_group: GroupCoordinator,  # 张量并行组协调器
):
    """Patch the model to make it compatible with torch.compile"""  # 对模型进行补丁以兼容torch.compile
    backup_ca_comm = None  # 备份自定义全归约通信对象

    try:  # 尝试执行
        if enable_compile:  # 如果启用编译
            backup_ca_comm = tp_group.ca_comm  # 备份原始通信对象
            # Use custom-allreduce here.  # 在此处使用自定义全归约
            # We found the custom allreduce is much faster than the built-in allreduce in torch,  # 我们发现自定义全归约比PyTorch内置的全归约快得多
            # even with ENABLE_INTRA_NODE_COMM=1.  # 即使启用了ENABLE_INTRA_NODE_COMM=1
            # tp_group.ca_comm = None  # 注释掉的代码：设置为None
            yield torch.compile(  # 编译模型的前向方法
                torch.no_grad()(model.forward),  # 禁用梯度计算
                dynamic=False,  # 禁用动态形状
            )
        else:  # 如果不启用编译
            yield model.forward  # 直接使用原始前向方法
    finally:  # 最终恢复
        if enable_compile:  # 如果启用了编译
            tp_group.ca_comm = backup_ca_comm  # 恢复原始通信对象


def set_torch_compile_config():  # 设置torch.compile的编译配置
    import torch._dynamo.config  # 导入Dynamo配置模块
    import torch._inductor.config  # 导入Inductor配置模块

    torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future  # 实验性特性，减少编译时间，未来将默认开启
    torch._inductor.config.freezing = True  # 启用冻结优化
    torch._dynamo.config.accumulated_cache_size_limit = 1024  # 设置累积缓存大小限制
    if hasattr(torch._dynamo.config, "cache_size_limit"):  # 如果存在缓存大小限制属性
        torch._dynamo.config.cache_size_limit = 1024  # 设置缓存大小限制
    monkey_patch_torch_compile()  # 应用torch.compile猴子补丁


def get_batch_sizes_to_capture(model_runner: ModelRunner):  # 获取需要捕获的批处理大小列表
    # torch compile speeds up decoding by reducing python overhead on CPU  # torch.compile通过减少CPU上的Python开销来加速解码
    server_args = model_runner.server_args  # 获取服务器参数
    # Note that we reuse server_args.cuda_graph_bs here.  # 注意这里复用了server_args.cuda_graph_bs
    # Users can customize the batch sizes supported by cpu_graph, such as:  # 用户可以自定义cpu_graph支持的批处理大小，例如：
    # --cuda-graph-bs 1 2 4 8 16  # 命令行参数示例
    capture_bs = server_args.cuda_graph_bs  # 获取要捕获的批处理大小列表
    assert (
        max(capture_bs) <= server_args.torch_compile_max_bs
    ), f"{capture_bs=}, {server_args.torch_compile_max_bs=}"  # 断言最大批处理大小不超过编译最大限制
    capture_bs = [bs for bs in capture_bs if bs <= model_runner.req_to_token_pool.size]  # 过滤掉超过请求到token池大小的批处理大小
    capture_bs = list(sorted(set(capture_bs)))  # 去重并排序
    assert len(capture_bs) > 0 and capture_bs[0] > 0, f"{capture_bs=}"  # 断言列表非空且第一个元素大于0
    return capture_bs  # 返回捕获批处理大小列表


def register_fake_ops():  # 注册所有自定义sgl_kernel CPU算子的伪/元实现，以支持torch.compile
    """
    Registers fake/meta implementations for all custom sgl_kernel CPU operators
    using torch.library.register_fake to support torch.compile
    """  # 使用torch.library.register_fake注册所有自定义sgl_kernel CPU算子的伪/元实现以支持torch.compile

    none_return_ops = [  # 无返回值的算子列表
        "shm_allreduce",  # 共享内存全归约
        "bmm_cpu",  # CPU批量矩阵乘法
        "fused_add_rmsnorm_cpu",  # CPU融合加法RMS归一化
        "decode_attention_cpu",  # CPU解码注意力
        "extend_attention_cpu",  # CPU扩展注意力
        "gemma_fused_add_rmsnorm_cpu",  # CPU Gemma融合加法RMS归一化
        "layernorm_cpu",  # CPU层归一化
        "fused_add_layernorm_cpu",  # CPU融合加法层归一化
    ]
    for op in none_return_ops:  # 遍历无返回值算子

        @torch.library.register_fake(f"sgl_kernel::{op}")  # 注册伪实现
        def _(*args, **kwargs):  # 伪实现函数，不返回任何值
            return

    for op in [  # 返回与输入形状相同张量的算子列表
        "rmsnorm_cpu",  # CPU RMS归一化
        "l2norm_cpu",  # CPU L2归一化
        "fused_experts_cpu",  # CPU融合专家
        "fused_rmsnorm_gated_cpu",  # CPU融合RMS归一化门控
        "shared_expert_cpu",  # CPU共享专家
        "causal_conv1d_update_cpu",  # CPU因果卷积1D更新
        "causal_conv1d_fwd_cpu",  # CPU因果卷积1D前向
        "gemma_rmsnorm_cpu",  # CPU Gemma RMS归一化
        "gemma3_rmsnorm_cpu",  # CPU Gemma3 RMS归一化
        "gemma4_rmsnorm_cpu",  # CPU Gemma4 RMS归一化
    ]:

        @torch.library.register_fake(f"sgl_kernel::{op}")  # 注册伪实现
        def _(input, *args, **kwargs):  # 伪实现函数，返回与输入形状相同的空张量
            return torch.empty_like(input)

    @torch.library.register_fake("sgl_kernel::qkv_proj_with_rope")  # 注册qkv_proj_with_rope算子的伪实现
    def _(  # 伪实现函数，模拟QKV投影与RoPE的输出形状
        hidden_states,  # 隐藏状态
        q_a_proj_weight,  # Q_a投影权重
        q_b_proj_weight,  # Q_b投影权重
        kv_a_proj_weight,  # KV_a投影权重
        w_kc,  # KC权重
        q_a_layernorm_weight,  # Q_a层归一化权重
        kv_a_layernorm_weight,  # KV_a层归一化权重
        positions,  # 位置编码
        cos_sin_cache,  # 余弦正弦缓存
        eps,  # epsilon值
        use_int8_w8a8,  # 是否使用int8量化
        use_fp8_w8a16,  # 是否使用fp8量化
        q_a_proj_scale,  # Q_a投影缩放因子
        q_b_proj_scale,  # Q_b投影缩放因子
        kv_a_proj_scale,  # KV_a投影缩放因子
        is_vnni,  # 是否为VNNI格式
        block_size,  # 块大小
    ):
        num_seqs = hidden_states.shape[0]  # 序列数量
        num_heads = w_kc.shape[0]  # 注意力头数量
        kv_lora_rank = w_kc.shape[1]  # KV LoRA秩
        qk_rope_head_dim = kv_a_proj_weight.shape[0] - kv_lora_rank  # QK RoPE头维度
        q_input = torch.empty(  # 创建Q输出张量
            num_seqs,  # 序列数
            num_heads,  # 头数
            kv_lora_rank + qk_rope_head_dim,  # 每头维度
            dtype=hidden_states.dtype,  # 数据类型
            device=hidden_states.device,  # 设备
        )
        k_input = torch.empty(  # 创建K输出张量
            num_seqs,  # 序列数
            1,  # KV头数为1
            kv_lora_rank + qk_rope_head_dim,  # 每头维度
            dtype=hidden_states.dtype,  # 数据类型
            device=hidden_states.device,  # 设备
        )
        v_input = k_input.narrow(-1, 0, kv_lora_rank)  # V输入是K输入的前kv_lora_rank列
        return q_input, k_input, v_input  # 返回QKV输出

    @torch.library.register_fake("sgl_kernel::rotary_embedding_cpu")  # 注册rotary_embedding_cpu算子的伪实现
    def _(positions, query, key, head_size, cos_sin_cache, is_neox):  # 伪实现函数，模拟旋转嵌入的输出形状
        if query.ndim == 2:  # 如果查询是2维的
            return query, key  # 直接返回输入（无需修改）
        else:  # 如果是3维或更高
            return torch.empty_like(query), torch.empty_like(key)  # 返回与输入形状相同的空张量

    @torch.library.register_fake("sgl_kernel::multimodal_rotary_embedding_cpu")  # 注册multimodal_rotary_embedding_cpu算子的伪实现
    def _(  # 伪实现函数，模拟多模态旋转嵌入的输出形状
        positions,  # 位置编码
        query,  # 查询张量
        key,  # 键张量
        head_size,  # 头大小
        cos_sin_cache,  # 余弦正弦缓存
        mrope_section,  # 多模态RoPE分段
        mrope_interleaved,  # 多模态RoPE是否交错
        is_neox,  # 是否为NeoX格式
    ):
        return query, key  # 返回输入查询和键

    @torch.library.register_fake("sgl_kernel::qkv_proj_with_rope_fused_weight")  # 注册qkv_proj_with_rope_fused_weight算子的伪实现
    def _(  # 伪实现函数，模拟融合权重的QKV投影与RoPE的输出形状
        hidden_states,  # 隐藏状态
        q_a_proj_weight,  # Q_a投影权重
        q_b_proj_weight,  # Q_b投影权重
        w_kc,  # KC权重
        q_a_layernorm_weight,  # Q_a层归一化权重
        kv_a_layernorm_weight,  # KV_a层归一化权重
        positions,  # 位置编码
        cos_sin_cache,  # 余弦正弦缓存
        eps,  # epsilon值
        use_int8_w8a8,  # 是否使用int8量化
        use_fp8_w8a16,  # 是否使用fp8量化
        qkv_a_proj_scale,  # QKV_a投影缩放因子
        q_b_proj_scale,  # Q_b投影缩放因子
        w_scale,  # 权重缩放因子
        is_vnni,  # 是否为VNNI格式
        block_size,  # 块大小
        q_lora_rank,  # Q LoRA秩
        kv_lora_rank,  # KV LoRA秩
        qk_rope_head_dim,  # QK RoPE头维度
    ):
        num_seqs = hidden_states.shape[0]  # 序列数量
        num_heads = w_kc.shape[0]  # 注意力头数量
        kv_lora_rank = w_kc.shape[1]  # KV LoRA秩
        weight_chunks = torch.split(  # 按Q和KV维度分割权重
            q_a_proj_weight, [q_lora_rank, kv_lora_rank + qk_rope_head_dim], dim=0
        )
        qk_rope_head_dim = weight_chunks[1].shape[0] - kv_lora_rank  # 重新计算QK RoPE头维度
        q_input = torch.empty(  # 创建Q输出张量
            num_seqs,  # 序列数
            num_heads,  # 头数
            kv_lora_rank + qk_rope_head_dim,  # 每头维度
            dtype=hidden_states.dtype,  # 数据类型
            device=hidden_states.device,  # 设备
        )
        k_input = torch.empty(  # 创建K输出张量
            num_seqs,  # 序列数
            1,  # KV头数为1
            kv_lora_rank + qk_rope_head_dim,  # 每头维度
            dtype=hidden_states.dtype,  # 数据类型
            device=hidden_states.device,  # 设备
        )
        v_input = k_input.narrow(-1, 0, kv_lora_rank)  # V输入是K输入的前kv_lora_rank列
        return q_input, k_input, v_input  # 返回QKV输出

    def get_n_size(mat2, is_vnni):  # 获取矩阵乘法输出的N维度大小
        tile_n = 16  # VNNI分块大小为16
        if mat2.dtype == torch.float32:  # 如果是float32类型
            return mat2.shape[1]  # 直接返回第二维度
        if not is_vnni and mat2.dim() == 2 and mat2.shape[0] < tile_n:  # 如果不是VNNI格式且维度小于分块大小
            return mat2.shape[1]  # 返回第二维度
        return mat2.shape[0]  # 否则返回第一维度

    @torch.library.register_fake("sgl_kernel::weight_packed_linear")  # 注册weight_packed_linear算子的伪实现
    def _(mat1, mat2, bias, is_vnni):  # 伪实现函数，模拟打包权重线性层的输出形状
        M = mat1.shape[0]  # 矩阵1的行数
        N = get_n_size(mat2, is_vnni)  # 获取输出列数
        return mat1.new_empty(M, N)  # 返回形状为(M, N)的空张量

    @torch.library.register_fake("sgl_kernel::per_token_quant_int8_cpu")  # 注册per_token_quant_int8_cpu算子的伪实现
    def _(input):  # 伪实现函数，模拟逐token int8量化的输出形状
        M = input.shape[0]  # token数量
        K = input.shape[1]  # 特征维度
        Aq = input.new_empty(M, K, dtype=torch.int8)  # 量化后的int8张量
        As = input.new_empty(M, dtype=torch.float32)  # 量化缩放因子
        return Aq, As  # 返回量化结果和缩放因子

    @torch.library.register_fake("sgl_kernel::int8_scaled_mm_cpu")  # 注册int8_scaled_mm_cpu算子的伪实现
    def _(mat1, mat2, scales1, scales2, bias, out_dtype, is_vnni):  # 伪实现函数，模拟int8缩放矩阵乘法的输出形状
        M = mat1.shape[0]  # 矩阵1的行数
        N = mat2.shape[0]  # 矩阵2的行数（即输出列数）
        out = mat1.new_empty(M, N, dtype=out_dtype)  # 创建输出张量
        return out  # 返回输出张量

    @torch.library.register_fake("sgl_kernel::grouped_topk_cpu")  # 注册grouped_topk_cpu算子的伪实现
    def _(  # 伪实现函数，模拟分组TopK的输出形状
        hidden_states,  # 隐藏状态
        gating_output,  # 门控输出
        topk,  # TopK值
        renormalize,  # 是否重新归一化
        num_expert_group,  # 专家组数量
        topk_group,  # 每组TopK值
        num_fused_shared_experts,  # 融合共享专家数量
        routed_scaling_factor,  # 路由缩放因子
        num_token_non_padded,  # 非填充token数量
    ):
        num_tokens = hidden_states.shape[0]  # token数量
        shape = (num_tokens, topk)  # 输出形状
        device = hidden_states.device  # 设备
        topk_weights = torch.empty(shape, device=device, dtype=torch.float32)  # TopK权重
        topk_ids = torch.empty(shape, device=device, dtype=torch.int)  # TopK索引
        return topk_weights, topk_ids  # 返回TopK权重和索引

    @torch.library.register_fake("sgl_kernel::biased_grouped_topk_cpu")  # 注册biased_grouped_topk_cpu算子的伪实现
    def _(  # 伪实现函数，模拟带偏置的分组TopK的输出形状
        hidden_states,  # 隐藏状态
        gating_output,  # 门控输出
        correction_bias,  # 校正偏置
        topk,  # TopK值
        renormalize,  # 是否重新归一化
        num_expert_group,  # 专家组数量
        topk_group,  # 每组TopK值
        num_fused_shared_experts,  # 融合共享专家数量
        routed_scaling_factor,  # 路由缩放因子
        num_token_non_padded,  # 非填充token数量
    ):
        num_tokens = hidden_states.shape[0]  # token数量
        shape = (num_tokens, topk)  # 输出形状
        device = hidden_states.device  # 设备
        topk_weights = torch.empty(shape, device=device, dtype=torch.float32)  # TopK权重
        topk_ids = torch.empty(shape, device=device, dtype=torch.int)  # TopK索引
        return topk_weights, topk_ids  # 返回TopK权重和索引

    @torch.library.register_fake("sgl_kernel::topk_sigmoid_cpu")  # 注册topk_sigmoid_cpu算子的伪实现
    def _(hidden_states, gating_output, topk, renormalize):  # 伪实现函数，模拟Sigmoid TopK的输出形状
        num_tokens = hidden_states.shape[0]  # token数量
        shape = (num_tokens, topk)  # 输出形状
        return (
            torch.empty(shape, device=hidden_states.device, dtype=torch.float),  # TopK权重
            torch.empty(shape, device=hidden_states.device, dtype=torch.int),  # TopK索引
        )

    @torch.library.register_fake("sgl_kernel::topk_softmax_cpu")  # 注册topk_softmax_cpu算子的伪实现
    def _(  # 伪实现函数，模拟Softmax TopK的输出形状
        hidden_states,  # 隐藏状态
        gating_output,  # 门控输出
        topk,  # TopK值
        renormalize,  # 是否重新归一化
    ):
        num_tokens = hidden_states.shape[0]  # token数量
        shape = (num_tokens, topk)  # 输出形状
        return (
            torch.empty(shape, device=hidden_states.device, dtype=torch.float),  # TopK权重
            torch.empty(shape, device=hidden_states.device, dtype=torch.int),  # TopK索引
        )

    for act_op in [  # 激活函数算子列表
        "silu_and_mul_cpu",  # CPU SiLU与乘法融合
        "gelu_tanh_and_mul_cpu",  # CPU GELU(tanh)与乘法融合
        "gelu_and_mul_cpu",  # CPU GELU与乘法融合
    ]:

        @torch.library.register_fake(f"sgl_kernel::{act_op}")  # 注册激活函数算子的伪实现
        def _(input):  # 伪实现函数，模拟激活函数的输出形状
            sizes = list(input.shape)  # 获取输入形状
            last_dim = input.dim() - 1  # 最后一个维度的索引
            d = sizes[last_dim] // 2  # 最后维度减半（因为是门控融合操作）
            sizes[last_dim] = d  # 更新最后维度大小
            return input.new_empty(sizes)  # 返回半宽的空张量

    @torch.library.register_fake("sgl_kernel::int8_scaled_mm_with_quant")  # 注册int8_scaled_mm_with_quant算子的伪实现
    def _(  # 伪实现函数，模拟带量化的int8缩放矩阵乘法的输出形状
        mat1,  # 输入矩阵1
        mat2,  # 输入矩阵2
        scales2,  # 矩阵2的缩放因子
        bias,  # 偏置
        out_dtype,  # 输出数据类型
        is_vnni,  # 是否为VNNI格式
    ):
        M = mat1.shape[0]  # 矩阵1的行数
        N = mat2.shape[0]  # 矩阵2的行数（输出列数）
        return mat1.new_empty(M, N, dtype=out_dtype)  # 返回输出张量

    @torch.library.register_fake("sgl_kernel::fp8_scaled_mm_cpu")  # 注册fp8_scaled_mm_cpu算子的伪实现
    def _(  # 伪实现函数，模拟FP8缩放矩阵乘法的输出形状
        mat1,  # 输入矩阵1
        mat2,  # 输入矩阵2
        scales2,  # 矩阵2的缩放因子
        block_size,  # 块大小
        bias,  # 偏置
        out_dtype,  # 输出数据类型
        is_vnni,  # 是否为VNNI格式
    ):
        M = mat1.shape[0]  # 矩阵1的行数
        N = mat2.shape[0]  # 矩阵2的行数（输出列数）
        return mat1.new_empty(M, N, dtype=out_dtype)  # 返回输出张量

    @torch.library.register_fake("sgl_kernel::fused_linear_sigmoid_mul")  # 注册fused_linear_sigmoid_mul算子的伪实现
    def _(  # 伪实现函数，模拟融合线性sigmoid乘法的输出形状
        mat1,  # 输入矩阵1
        mat2,  # 输入矩阵2（权重）
        bias,  # 偏置
        is_vnni,  # 是否为VNNI格式
        post_mul_mat,  # 后乘矩阵
    ):
        M = mat1.shape[0]  # 输入矩阵1的行数
        N = post_mul_mat.shape[1]  # 后乘矩阵的列数
        return mat1.new_empty(M, N)  # 返回输出张量

    @torch.library.register_fake("sgl_kernel::fused_qkvzba_split_reshape_cat_cpu")  # 注册fused_qkvzba_split_reshape_cat_cpu算子的伪实现
    def _(mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v):  # 伪实现函数，模拟融合QKVZBA分割重排拼接的输出形状
        batch = mixed_qkvz.shape[0]  # 批大小
        qkv_dim = num_heads_qk * head_qk * 2 + num_heads_v * head_v  # QKV总维度
        mixed_qkv = mixed_qkvz.new_empty(batch, qkv_dim)  # 混合QKV张量
        z = mixed_qkvz.new_empty(batch, num_heads_v, head_v)  # Z张量
        b = mixed_ba.new_empty(batch, num_heads_v)  # B张量
        a = mixed_ba.new_empty(batch, num_heads_v)  # A张量
        return mixed_qkv, z, b, a  # 返回所有输出

    @torch.library.register_fake(  # 注册fused_qkvzba_split_reshape_cat_contiguous_cpu算子的伪实现
        "sgl_kernel::fused_qkvzba_split_reshape_cat_contiguous_cpu"
    )
    def _(mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v):  # 伪实现函数，模拟连续内存版本的融合QKVZBA分割重排拼接的输出形状
        batch = mixed_qkvz.shape[0]  # 批大小
        qkv_dim = num_heads_qk * head_qk * 2 + num_heads_v * head_v  # QKV总维度
        mixed_qkv = mixed_qkvz.new_empty(batch, qkv_dim)  # 混合QKV张量
        z = mixed_qkvz.new_empty(batch, num_heads_v, head_v)  # Z张量
        b = mixed_ba.new_empty(batch, num_heads_v)  # B张量
        a = mixed_ba.new_empty(batch, num_heads_v)  # A张量
        return mixed_qkv, z, b, a  # 返回所有输出

    @torch.library.register_fake(  # 注册fused_sigmoid_gating_delta_rule_update_cpu算子的伪实现
        "sgl_kernel::fused_sigmoid_gating_delta_rule_update_cpu"
    )
    def _(  # 伪实现函数，模拟融合sigmoid门控delta规则更新的输出形状
        A_log,  # A的对数值
        dt_bias,  # dt偏置
        q,  # 查询张量
        k,  # 键张量
        v,  # 值张量
        a,  # A张量
        b,  # B张量
        initial_state_source,  # 初始状态来源
        initial_state_indices,  # 初始状态索引
        cu_seqlens,  # 累积序列长度
        use_qk_l2norm_in_kernel,  # 是否在内核中使用QK L2归一化
        softplus_beta=1.0,  # softplus的beta参数
        softplus_threshold=20.0,  # softplus的阈值参数
    ):
        assert q.dim() == 4  # 断言查询张量为4维
        assert v.dim() == 4  # 断言值张量为4维
        batch_size = q.shape[1]  # 批大小
        seq_len = q.shape[0]  # 序列长度
        v_num_heads = v.shape[2]  # V头数
        v_head_dim = v.shape[3]  # V头维度
        return q.new_empty(batch_size, seq_len, v_num_heads, v_head_dim)  # 返回输出张量

    @torch.library.register_fake("sgl_kernel::fused_gdn_gating_cpu")  # 注册fused_gdn_gating_cpu算子的伪实现
    def _(A_log, a, b, dt_bias):  # 伪实现函数，模拟融合GDN门控的输出形状
        batch = a.shape[0]  # 批大小
        num_heads = a.shape[1]  # 头数
        out = a.new_empty(1, batch, num_heads, dtype=torch.float)  # 输出张量
        beta = b.new_empty(1, batch, num_heads)  # beta张量
        return out, beta  # 返回输出和beta

    @torch.library.register_fake("sgl_kernel::chunk_gated_delta_rule_cpu")  # 注册chunk_gated_delta_rule_cpu算子的伪实现
    def _(  # 伪实现函数，模拟分块门控delta规则的输出形状
        query,  # 查询张量
        key,  # 键张量
        value,  # 值张量
        g,  # 门控张量
        beta,  # beta张量
        initial_state,  # 初始状态
        output_final_state,  # 是否输出最终状态
        cu_seqlens,  # 累积序列长度
        head_first,  # 是否头在前
        use_qk_l2norm_in_kernel,  # 是否在内核中使用QK L2归一化
        eps,  # epsilon值
    ):
        output = torch.empty_like(value)  # 创建与值形状相同的输出张量
        assert initial_state is not None  # 断言初始状态不为None
        final_state = initial_state.to(torch.float32)  # 将初始状态转为float32

        return output, final_state  # 返回输出和最终状态


# TODO Remove unnecessary settings for CPUGraphRunner.  # TODO 移除CPUGraphRunner中不必要的设置
# Re-abstract the graph runner and restructure CPUGraphRunner to reuse the same logic.  # 重新抽象图运行器并重构CPUGraphRunner以复用相同逻辑
class CPUGraphRunner:  # CPU图运行器类，使用CPU torch.compile运行模型前向传播
    """A CPUGraphRunner runs the forward pass of a model with cpu torch.compile."""  # CPUGraphRunner使用CPU torch.compile运行模型前向传播

    def __init__(self, model_runner: ModelRunner):  # 初始化CPU图运行器
        # Parse args  # 解析参数
        self.model_runner = model_runner  # 保存模型运行器引用
        self.device = model_runner.device  # 保存设备信息
        self.graphs = {}  # 存储编译后的图（按批处理大小索引）
        self.output_buffers = {}  # 存储输出缓冲区
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile  # 是否启用torch.compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding  # 是否禁用填充
        self.is_encoder_decoder = model_runner.model_config.is_encoder_decoder  # 是否为编码器-解码器模型
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)  # 是否需要聚合缓冲区
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)  # 是否需要MLP TP聚合
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)  # 是否需要MLP同步
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)  # 是否需要注意力TP聚合
        self.enable_two_batch_overlap = (  # 是否启用双批次重叠
            model_runner.server_args.enable_two_batch_overlap
        )
        self.speculative_algorithm = model_runner.server_args.speculative_algorithm  # 推测算法
        self.enable_profile_cuda_graph = (  # 是否启用CUDA图性能分析
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.tp_size = model_runner.server_args.tp_size  # 张量并行大小
        self.dp_size = model_runner.server_args.dp_size  # 数据并行大小
        self.pp_size = model_runner.server_args.pp_size  # 流水线并行大小

        self.capture_forward_mode = ForwardMode.DECODE  # 捕获时的前向模式为解码
        self.capture_hidden_mode = CaptureHiddenMode.NULL  # 捕获隐藏状态模式为NULL
        self.num_tokens_per_bs = 1  # 每个批处理大小的token数

        # If returning hidden states is enabled, set initial capture hidden mode to full to avoid double-capture on startup  # 如果启用了返回隐藏状态，将初始捕获隐藏模式设置为FULL以避免启动时双重捕获
        if model_runner.server_args.enable_return_hidden_states:  # 如果启用了返回隐藏状态
            self.capture_hidden_mode = CaptureHiddenMode.FULL  # 设置捕获隐藏模式为FULL

        assert (
            not self.model_runner.server_args.enable_lora
        ), "CPUGraphRunner does not support LoRA yet."  # 断言不支持LoRA
        assert (
            not self.enable_two_batch_overlap
        ), "CPUGraphRunner does not support two batch overlap yet."  # 断言不支持双批次重叠
        assert (
            not self.require_mlp_tp_gather
        ), "CPUGraphRunner does not support MLP TP gather yet."  # 断言不支持MLP TP聚合
        assert (
            not self.require_mlp_sync
        ), "CPUGraphRunner does not support MLP sync yet."  # 断言不支持MLP同步
        assert (
            not self.require_gathered_buffer
        ), "CPUGraphRunner does not support gathered buffer yet."  # 断言不支持聚合缓冲区
        assert (
            model_runner.spec_algorithm.is_none()
        ), "CPUGraphRunner does not support speculative inference yet."  # 断言不支持推测推理
        # TODO add compile support for encoder-decoder models  # TODO 为编码器-解码器模型添加编译支持
        assert (
            not self.is_encoder_decoder
        ), "CPUGraphRunner does not support encoder-decoder models yet."  # 断言不支持编码器-解码器模型
        assert self.dp_size == 1, "CPUGraphRunner does not support DP yet."  # 断言不支持数据并行
        assert self.pp_size == 1, "CPUGraphRunner does not support PP yet."  # 断言不支持流水线并行

        # Batch sizes to capture  # 要捕获的批处理大小
        self.capture_bs = get_batch_sizes_to_capture(model_runner)  # 获取捕获批处理大小列表
        log_info_on_rank0(logger, f"Capture cpu graph bs {self.capture_bs}")  # 记录捕获信息
        self.captured_forward_batches = {}  # 存储已捕获的前向批处理
        # Attention backend  # 注意力后端
        self.max_bs = max(self.capture_bs)  # 最大批处理大小
        self.max_num_token = self.max_bs * self.num_tokens_per_bs  # 最大token数量
        self.model_runner.attn_backend.init_cpu_graph_state(  # 初始化注意力后端的CPU图状态
            self.max_bs, self.max_num_token
        )

        self.seq_len_fill_value = (  # 序列长度填充值
            self.model_runner.attn_backend.get_cpu_graph_seq_len_fill_value()
        )

        if self.enable_torch_compile:  # 如果启用了torch.compile
            register_fake_ops()  # 注册伪算子
            set_torch_compile_config()  # 设置编译配置

        # Graph inputs  # 图输入
        with torch.device(self.device):  # 在指定设备上创建输入张量
            self.input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)  # 输入token ID
            self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int64)  # 请求池索引
            self.seq_lens = torch.full(  # 序列长度张量
                (self.max_bs,), self.seq_len_fill_value, dtype=torch.int64
            )
            self.out_cache_loc = torch.zeros((self.max_num_token,), dtype=torch.int64)  # 输出缓存位置
            self.positions = torch.zeros((self.max_num_token,), dtype=torch.int64)  # 位置编码
            self.mrope_positions = torch.zeros((3, self.max_bs), dtype=torch.int64)  # 多模态RoPE位置
            self.num_token_non_padded = torch.zeros((1,), dtype=torch.int64)  # 非填充token数量
            self.custom_mask = torch.ones(  # 自定义注意力掩码
                (
                    (self.seq_lens.sum().item() + self.max_num_token)
                    * self.num_tokens_per_bs
                ),
                dtype=torch.bool,
                device=self.device,
            )

        # Capture  # 捕获图
        try:  # 尝试捕获
            self.capture()  # 执行捕获
        except RuntimeError as e:  # 捕获运行时错误
            raise Exception(
                f"Capture CPU graph failed: {e}\n{CPU_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def can_run(self, forward_batch: ForwardBatch):  # 判断当前CPU图运行器是否能处理给定的前向批处理
        is_bs_supported = (  # 判断批处理大小是否支持
            forward_batch.batch_size in self.graphs
            if self.disable_padding
            else forward_batch.batch_size <= self.max_bs
        )

        requested_capture_hidden_mode = max(  # 计算请求的捕获隐藏模式
            forward_batch.capture_hidden_mode,
            (
                forward_batch.spec_info.capture_hidden_mode
                if getattr(forward_batch.spec_info, "capture_hidden_mode", None)
                is not None
                else CaptureHiddenMode.NULL
            ),
        )
        capture_hidden_mode_matches = (  # 判断捕获隐藏模式是否匹配
            requested_capture_hidden_mode == CaptureHiddenMode.NULL
            or requested_capture_hidden_mode == self.capture_hidden_mode
        )

        return is_bs_supported and capture_hidden_mode_matches  # 返回批处理大小和隐藏模式是否都匹配

    def capture(self) -> None:  # 捕获所有批处理大小的CPU计算图
        capture_range = (  # 创建捕获范围，rank 0显示进度条
            tqdm.tqdm(list(reversed(self.capture_bs)))
            if get_tensor_model_parallel_rank() == 0
            else reversed(self.capture_bs)
        )
        for bs in capture_range:  # 遍历所有批处理大小
            if get_tensor_model_parallel_rank() == 0:  # 仅在rank 0上更新进度条
                avail_mem = psutil.virtual_memory().available / (1 << 30)  # 获取可用内存（GB）
                capture_range.set_description(
                    f"Capturing batches ({bs=} {avail_mem=:.2f} GB)"
                )

            with patch_model(  # 使用补丁模型上下文
                self.model_runner.model,
                bs in self.capture_bs,
                num_tokens=bs * self.num_tokens_per_bs,
                tp_group=self.model_runner.tp_group,
            ) as forward:
                (
                    graph,
                    output_buffers,
                ) = self.capture_one_batch_size(bs, forward)  # 捕获单个批处理大小的图
                self.graphs[bs] = graph  # 保存编译后的图
                self.output_buffers[bs] = output_buffers  # 保存输出缓冲区

        # Re-init states for qwen3-next as  # 重新初始化qwen3-next的状态，因为
        # torch.compile may change the states  # torch.compile可能会改变状态
        self._reset_mamba_cache_if_needed()

    def _reset_mamba_cache_if_needed(self) -> None:  # 如果需要，重置Mamba缓存状态

        mamba_pool = getattr(self.model_runner.req_to_token_pool, "mamba_pool", None)  # 获取Mamba池
        if mamba_pool is None:  # 如果不存在则返回
            return
        mamba_cache = getattr(mamba_pool, "mamba_cache", None)  # 获取Mamba缓存
        if mamba_cache is None:  # 如果不存在则返回
            return

        def _zero_nested(obj):  # 递归将嵌套结构中的张量清零
            if isinstance(obj, torch.Tensor):  # 如果是张量
                obj.zero_()  # 原地清零
            elif isinstance(obj, (list, tuple)):  # 如果是列表或元组
                for it in obj:  # 遍历每个元素
                    _zero_nested(it)  # 递归清零

        for v in vars(mamba_cache).values():  # 遍历Mamba缓存的所有属性
            _zero_nested(v)  # 递归清零

    def capture_one_batch_size(self, bs: int, forward: Callable):  # 捕获单个批处理大小的CPU计算图
        num_tokens = bs * self.num_tokens_per_bs  # 计算总token数

        # Graph inputs  # 图输入
        input_ids = self.input_ids[:num_tokens]  # 截取输入token ID
        req_pool_indices = self.req_pool_indices[:bs]  # 截取请求池索引
        seq_lens = self.seq_lens[:bs]  # 截取序列长度
        out_cache_loc = self.out_cache_loc[:num_tokens]  # 截取输出缓存位置
        positions = self.positions[:num_tokens]  # 截取位置编码
        mrope_positions = self.mrope_positions[:, :num_tokens]  # 截取多模态RoPE位置
        self.num_token_non_padded[...] = num_tokens  # 设置非填充token数量

        spec_info = self.get_spec_info(num_tokens)  # 获取推测信息
        if self.capture_hidden_mode != CaptureHiddenMode.FULL:  # 如果不是FULL模式
            self.capture_hidden_mode = (  # 更新捕获隐藏模式
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            )

        forward_batch = ForwardBatch(  # 创建前向批处理对象
            forward_mode=self.capture_forward_mode,  # 前向模式
            batch_size=bs,  # 批处理大小
            input_ids=input_ids,  # 输入token ID
            req_pool_indices=req_pool_indices,  # 请求池索引
            seq_lens=seq_lens,  # 序列长度
            out_cache_loc=out_cache_loc,  # 输出缓存位置
            seq_lens_sum=seq_lens.sum().item(),  # 序列长度总和
            return_logprob=False,  # 不返回对数概率
            positions=positions,  # 位置编码
            mrope_positions=mrope_positions,  # 多模态RoPE位置
            spec_algorithm=self.model_runner.spec_algorithm,  # 推测算法
            spec_info=spec_info,  # 推测信息
            capture_hidden_mode=self.capture_hidden_mode,  # 捕获隐藏模式
            num_token_non_padded=self.num_token_non_padded,  # 非填充token数量
            global_forward_mode=self.capture_forward_mode,  # 全局前向模式
        )
        with forward_context(  # 使用前向上下文
            ForwardContext(attn_backend=self.model_runner.attn_backend)
        ):
            self.model_runner.attn_backend.init_forward_metadata_capture_cpu_graph(  # 初始化注意力前向元数据
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                None,
                forward_batch.forward_mode,
                forward_batch.spec_info,
            )
            with torch.no_grad():  # 禁用梯度计算
                self.model_runner.tp_group.barrier()  # 同步TP组
                self.model_runner.model.forward(  # 执行模型前向传播
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                )

            # Run and capture  # 运行并捕获
            def run_once():  # 执行一次前向传播的内部函数
                # Clean intermediate result cache for DP attention  # 清除DP注意力的中间结果缓存
                forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = (
                    None
                )
                logits_output_or_pp_proxy_tensors = forward(  # 执行编译后的前向传播
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                )
                return logits_output_or_pp_proxy_tensors  # 返回输出

            with torch.no_grad():  # 禁用梯度计算
                for _ in range(2):  # 运行两次以预热编译
                    self.model_runner.tp_group.barrier()  # 同步TP组
                    out = run_once()  # 执行一次前向传播
                # Save the captured forward_batch  # 保存已捕获的前向批处理
                self.captured_forward_batches[bs] = forward_batch
                return forward, out  # 返回编译后的前向函数和输出

    def recapture_if_needed(self, forward_batch: ForwardBatch):  # 如果需要，重新捕获CPU计算图

        # If the required capture_hidden_mode changes, we need to recapture the graph  # 如果所需的捕获隐藏模式发生变化，需要重新捕获图

        # These are the different factors that can influence the capture_hidden_mode  # 以下是可以影响捕获隐藏模式的不同因素
        capture_hidden_mode_required_by_forward_batch = (  # 前向批处理要求的捕获隐藏模式
            forward_batch.capture_hidden_mode
        )
        capture_hidden_mode_required_by_spec_info = getattr(  # 推测信息要求的捕获隐藏模式
            forward_batch.spec_info, "capture_hidden_mode", CaptureHiddenMode.NULL
        )
        capture_hidden_mode_required_for_returning_hidden_states = (  # 返回隐藏状态要求的捕获隐藏模式
            CaptureHiddenMode.FULL
            if self.model_runner.server_args.enable_return_hidden_states
            else CaptureHiddenMode.NULL
        )

        # Determine the highest capture_hidden_mode required  # 确定所需的最高捕获隐藏模式
        # (If we have FULL, we can emulate LAST or NULL)  # （如果有FULL，可以模拟LAST或NULL）
        # (If we have LAST, we can emulate NULL)  # （如果有LAST，可以模拟NULL）
        required_capture_hidden_mode = max(  # 取三者中的最大值作为所需模式
            capture_hidden_mode_required_by_forward_batch,
            capture_hidden_mode_required_by_spec_info,
            capture_hidden_mode_required_for_returning_hidden_states,
        )

        # If the current hidden mode is no longer aligned with the required hidden mode, we need to set it to what is required and re-capture  # 如果当前隐藏模式与所需模式不一致，需要设置为所需模式并重新捕获
        if self.capture_hidden_mode != required_capture_hidden_mode:  # 如果模式不匹配
            self.capture_hidden_mode = required_capture_hidden_mode  # 更新捕获隐藏模式
            self.capture()  # 重新捕获图

    def prepare_replay(  # 准备重放操作，填充捕获的批处理输入
        self,
        forward_batch: ForwardBatch,  # 前向批处理
    ):
        self.recapture_if_needed(forward_batch)  # 检查是否需要重新捕获

        raw_bs = forward_batch.batch_size  # 原始批处理大小
        if raw_bs in self.graphs:  # 如果批处理大小已有对应的图
            self.model_runner.attn_backend.init_forward_metadata(forward_batch)  # 初始化注意力前向元数据
            return forward_batch  # 直接返回原始批处理

        raw_num_token = raw_bs * self.num_tokens_per_bs  # 原始token数量
        index = bisect.bisect_left(self.capture_bs, raw_bs)  # 二分查找合适的捕获批处理大小
        bs = self.capture_bs[index]  # 获取大于等于原始大小的捕获批处理大小
        assert bs > raw_bs  # 断言捕获大小大于原始大小
        self.raw_bs = raw_bs  # 保存原始批处理大小
        self.raw_num_token = raw_num_token  # 保存原始token数量
        self.bs = bs  # 保存捕获批处理大小

        captured_forward_batch = self.captured_forward_batches[bs]  # 获取已捕获的前向批处理
        assert captured_forward_batch is not None  # 断言不为None
        captured_forward_batch.seq_lens.fill_(self.seq_len_fill_value)  # 用填充值填充序列长度
        captured_forward_batch.out_cache_loc.zero_()  # 清零输出缓存位置
        # Pair with seq_lens fill: padded rows must point at reserved  # 与序列长度填充配对：填充行必须指向保留的
        # req_pool slot 0 (req_to_token[0, :] is all zeros from init).  # 请求池槽0（req_to_token[0, :]初始化时全为零）
        captured_forward_batch.req_pool_indices.zero_()  # 清零请求池索引
        captured_forward_batch.input_ids[:raw_num_token].copy_(forward_batch.input_ids)  # 复制输入token ID
        captured_forward_batch.req_pool_indices[:raw_bs].copy_(  # 复制请求池索引
            forward_batch.req_pool_indices
        )
        captured_forward_batch.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)  # 复制序列长度
        captured_forward_batch.out_cache_loc[:raw_num_token].copy_(  # 复制输出缓存位置
            forward_batch.out_cache_loc
        )
        captured_forward_batch.positions[:raw_num_token].copy_(forward_batch.positions)  # 复制位置编码
        if forward_batch.mrope_positions is not None:  # 如果存在多模态RoPE位置
            self.mrope_positions[:, :raw_num_token].copy_(forward_batch.mrope_positions)  # 复制多模态RoPE位置

        if self.is_encoder_decoder:  # 如果是编码器-解码器模型
            captured_forward_batch.encoder_lens[:raw_bs].copy_(  # 复制编码器长度
                forward_batch.encoder_lens
            )
        if enable_num_token_non_padded():  # 如果启用了非填充token数量
            captured_forward_batch.num_token_non_padded.copy_(  # 复制非填充token数量
                forward_batch.num_token_non_padded
            )

        self.model_runner.attn_backend.init_forward_metadata(captured_forward_batch)  # 初始化注意力前向元数据
        return captured_forward_batch  # 返回准备好的前向批处理

    def replay(  # 重放已捕获的CPU计算图
        self,
        forward_batch: ForwardBatch,  # 前向批处理
        skip_attn_backend_init: bool = False,  # 是否跳过注意力后端初始化
        pp_proxy_tensors: Optional[PPProxyTensors] = None,  # 流水线并行代理张量
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:  # 返回逻辑处理器输出或代理张量
        assert (
            pp_proxy_tensors is None
        ), "PPProxyTensors is not supported in CPUGraphRunner yet."  # 断言不支持PP代理张量

        prepared_forward_batch = self.prepare_replay(forward_batch)  # 准备重放
        output = self.graphs[prepared_forward_batch.batch_size](  # 执行已捕获的图
            prepared_forward_batch.input_ids,
            prepared_forward_batch.positions,
            prepared_forward_batch,
        )
        if forward_batch.batch_size in self.graphs:  # 如果批处理大小有对应图
            return output  # 直接返回输出

        assert isinstance(output, LogitsProcessorOutput)  # 断言输出为逻辑处理器输出
        return LogitsProcessorOutput(  # 返回截取后的输出
            next_token_logits=output.next_token_logits[: self.raw_num_token],  # 截取下一个token逻辑值
            hidden_states=(
                output.hidden_states[: self.raw_num_token]
                if output.hidden_states is not None
                else None
            ),  # 截取隐藏状态
        )

    def get_spec_info(self, num_tokens: int):  # 获取推测算法信息
        spec_info = None  # 初始化为None
        if (  # 如果使用Eagle或Standalone推测算法
            self.model_runner.spec_algorithm.is_eagle()
            or self.model_runner.spec_algorithm.is_standalone()
        ):
            from sglang.srt.speculative.eagle_info import EagleVerifyInput  # 导入Eagle验证输入类

            if self.model_runner.is_draft_worker:  # 如果是草稿工作者
                raise RuntimeError("This should not happen.")  # 抛出运行时错误
            else:  # 如果是验证工作者
                spec_info = EagleVerifyInput(  # 创建Eagle验证输入
                    draft_token=None,  # 草稿token
                    custom_mask=self.custom_mask,  # 自定义掩码
                    positions=None,  # 位置编码
                    retrieve_index=None,  # 检索索引
                    retrieve_next_token=None,  # 检索下一个token
                    retrieve_next_sibling=None,  # 检索下一个兄弟
                    retrieve_cum_len=None,  # 检索累积长度
                    spec_steps=self.model_runner.server_args.speculative_num_steps,  # 推测步数
                    topk=self.model_runner.server_args.speculative_eagle_topk,  # TopK值
                    draft_token_num=self.model_runner.server_args.speculative_num_draft_tokens,  # 草稿token数
                    capture_hidden_mode=CaptureHiddenMode.FULL,  # 捕获隐藏模式为FULL
                    seq_lens_sum=None,  # 序列长度总和
                    seq_lens_cpu=None,  # CPU序列长度
                )

        return spec_info  # 返回推测信息


CPU_GRAPH_CAPTURE_FAILED_MSG = (  # CPU图捕获失败的错误消息
    "Possible solutions:\n"  # 可能的解决方案：
    "1. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)\n"  # 1. 将--mem-fraction-static设为更小的值
    "2. set --torch-compile-max-bs to a smaller value (e.g., 8)\n"  # 2. 将--torch-compile-max-bs设为更小的值
    "3. disable torch compile by not using --enable-torch-compile\n"  # 3. 不使用--enable-torch-compile来禁用torch编译
    "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"  # 在GitHub上提交问题
)
