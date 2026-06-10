# 路由专家捕获模块
# 实现RoutedExpertsCapturer，用于捕获MoE模型中每层的路由专家索引
# 支持DP注意力、DeepEP全聚合以及融合共享专家的截断处理

from typing import Optional  # 可选类型注解

import numpy as np  # NumPy数值计算库
import pybase64  # Base64编解码库
import torch  # PyTorch深度学习框架

from sglang.srt.configs.model_config import ModelConfig  # 模型配置类
from sglang.srt.layers.dp_attention import (  # DP注意力相关工具
    attn_tp_all_gather_into_tensor,  # 注意力TP全聚合到张量
    get_attention_tp_size,  # 获取注意力TP大小
    get_dp_local_slice_cpu,  # 获取DP本地切片（CPU端）
    is_dp_attention_enabled,  # 判断是否启用DP注意力
)
from sglang.srt.layers.moe import get_moe_a2a_backend  # 获取MoE全连接后端
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # 前向批处理信息
from sglang.srt.server_args import get_global_server_args  # 获取全局服务器参数
from sglang.srt.state_capturer.base import BaseTopkCapturer  # Topk捕获器基类


class RoutedExpertsCapturer(BaseTopkCapturer):  # 路由专家捕获器，继承自Topk捕获器基类
    """Capturer for routed experts with host buffer.

    Routed experts share a global device buffer across DP ranks (indexed by
    dp_rank), so `_get_local_slice` overrides the default to apply DP-rank-aware
    slicing. The device cache also holds extra columns for any fused shared
    experts; the host cache and user-facing return drop them via the
    [:topk_size] truncation.
    """
    # 带主机缓冲区的路由专家捕获器。
    # 路由专家跨DP rank共享全局设备缓冲区（按dp_rank索引），
    # 因此_get_local_slice重写默认实现以应用DP rank感知的切片。
    # 设备缓存还包含融合共享专家的额外列；主机缓存和面向用户的返回
    # 通过[:topk_size]截断丢弃这些额外列。

    @staticmethod
    def create(  # 静态工厂方法，创建路由专家捕获器
        enable: bool,  # 是否启用
        model_config: ModelConfig,  # 模型配置
        num_fused_shared_experts: int,  # 融合共享专家数量
        num_tokens: int,  # token总数
        max_running_requests: int,  # 最大运行请求数
        device: str,  # 设备类型
    ) -> Optional["RoutedExpertsCapturer"]:
        if not enable:  # 如果未启用
            return None  # 返回None
        return RoutedExpertsCapturer(  # 创建并返回路由专家捕获器
            model_config,  # 模型配置
            num_tokens=num_tokens,  # token总数
            max_running_requests=max_running_requests,  # 最大运行请求数
            num_fused_shared_experts=num_fused_shared_experts,  # 融合共享专家数量
            device=device,  # 设备类型
        )

    def __init__(  # 初始化路由专家捕获器
        self,
        model_config: ModelConfig,  # 模型配置
        num_tokens: int,  # token总数
        max_running_requests: int,  # 最大运行请求数
        num_fused_shared_experts: int,  # 融合共享专家数量
        device: str,  # 设备类型
    ):
        self.num_fused_shared_experts = num_fused_shared_experts  # 保存融合共享专家数量
        topk_size = model_config.hf_text_config.num_experts_per_tok  # 从模型配置获取每token的专家数
        num_layers = model_config.hf_text_config.num_hidden_layers  # 从模型配置获取隐藏层数

        server_args = get_global_server_args()  # 获取全局服务器参数
        # Scale by dp_size so the buffer covers the full DP-concatenated batch.
        # _get_local_slice indexes into [attention_dp_rank * cuda_graph_batch, ...)
        # and otherwise overflows on dp_rank > 0 when max_running_requests >
        # chunked_prefill_size.
        # 按dp_size缩放以使缓冲区覆盖完整的DP拼接批次。
        # _get_local_slice按[attention_dp_rank * cuda_graph_batch, ...)索引，
        # 否则当max_running_requests > chunked_prefill_size时在dp_rank > 0处溢出。
        # FIXME: spec decoding's num_verify_tokens is still not accounted for.
        # FIXME: 推测解码的num_verify_tokens仍未计入。
        max_batch_size = max(  # 计算最大批大小
            server_args.chunked_prefill_size * server_args.dp_size,  # 分块预填充大小乘以DP大小
            max_running_requests * server_args.dp_size,  # 最大运行请求数乘以DP大小
        )

        super().__init__(  # 调用父类初始化
            num_tokens=num_tokens,  # token总数
            max_batch_size=max_batch_size,  # 最大批大小
            num_layers=num_layers,  # 层数
            topk_size=topk_size,  # topk大小
            device=device,  # 设备类型
            name="routed_experts",  # 捕获器名称
            device_topk_size=topk_size + num_fused_shared_experts,  # 设备端topk大小包含融合共享专家
        )

        # DeepEP a2a path: each attn-TP rank only sees its scattered slice of
        # topk_ids. All-gather across attn-TP at capture time so device_cache
        # holds the full batch and the existing _get_local_slice / D2H sync
        # paths work unchanged. Pre-allocate the gather target.
        # DeepEP全连接路径：每个注意力TP rank只能看到其分散的topk_ids切片。
        # 在捕获时跨注意力TP执行全聚合，使device_cache持有完整批次，
        # 现有的_get_local_slice / D2H同步路径无需更改。预分配聚合目标。
        if get_moe_a2a_backend().is_deepep():  # 如果使用DeepEP后端
            attn_tp_size = get_attention_tp_size() if is_dp_attention_enabled() else 1  # 获取注意力TP大小
            self.gather_buffer = torch.empty(  # 预分配聚合缓冲区
                (
                    self.device_cache.buffer.shape[0] * attn_tp_size,  # 按TP大小扩展第一维
                    self.device_cache.buffer.shape[2],  # 保持topk维度
                ),
                dtype=torch.int32,  # 32位整数类型
                device=device,  # 存储设备
            )

    def capture(self, layer_id: int, topk_indices: torch.Tensor):  # 捕获指定层的路由专家topk索引
        if get_moe_a2a_backend().is_deepep():  # 如果使用DeepEP后端
            local_topk = topk_indices  # 保存本地topk索引
            topk_indices = self.gather_buffer[  # 从聚合缓冲区获取全量topk空间
                : local_topk.size(0) * get_attention_tp_size()  # 按TP大小切片
            ]
            attn_tp_all_gather_into_tensor(topk_indices, local_topk)  # 执行跨TP全聚合
        super().capture(layer_id, topk_indices)  # 调用父类捕获方法

    def _get_local_slice(  # 获取当前DP rank的本地设备缓存切片
        self,
        forward_batch: ForwardBatch,  # 前向批处理信息
        can_run_graph: bool,  # 是否可以运行CUDA图
        cuda_graph_batch: Optional[int],  # CUDA图批大小
    ) -> torch.Tensor:
        # Under DeepEP, capture() already attn_tp_all_gathered into the head of
        # the per-rank buffer, so the local DP rank's data lives at [0:N_local]
        # rather than at the global [start_pos:end_pos] offset.
        # 在DeepEP下，capture()已将注意力TP全聚合到每个rank缓冲区的头部，
        # 因此本地DP rank的数据位于[0:N_local]而非全局[start_pos:end_pos]偏移处。
        if is_dp_attention_enabled() and not get_moe_a2a_backend().is_deepep():  # DP注意力启用且非DeepEP
            # GPU->CPU sync would break overlap; operate on CPU directly.
            # GPU到CPU同步会破坏重叠；直接在CPU上操作。
            local_start_pos, local_num_tokens = get_dp_local_slice_cpu(  # 获取DP本地切片的起始位置和token数
                forward_batch, can_run_graph, cuda_graph_batch
            )
            local_end_pos = local_start_pos + local_num_tokens  # 计算结束位置
        else:  # DeepEP路径或非DP注意力
            local_start_pos, local_end_pos = 0, forward_batch.out_cache_loc.shape[0]  # 从0到本地token数
        return self.device_cache.buffer[  # 返回本地切片
            local_start_pos:local_end_pos, :, : self.topk_size  # 截断到topk_size
        ]


_global_expert_capturer: Optional[RoutedExpertsCapturer] = None  # 全局路由专家捕获器单例


def get_global_experts_capturer() -> Optional[RoutedExpertsCapturer]:  # 获取全局路由专家捕获器
    return _global_expert_capturer  # 返回全局单例


def set_global_experts_capturer(capturer: Optional[RoutedExpertsCapturer]):  # 设置全局路由专家捕获器
    global _global_expert_capturer  # 声明全局变量
    _global_expert_capturer = capturer  # 更新全局单例


def extract_routed_experts_from_meta_info(data):  # 从元信息中提取路由专家索引
    # To solve the performance issue, we return the experts_ids in base64
    # We left this function for user to change it back to normal int32
    # See detokenizer_manager::_extract_routed_experts
    # 为解决性能问题，我们以base64格式返回expert_ids
    # 保留此函数供用户将其转换回普通int32
    # 参见detokenizer_manager::_extract_routed_experts
    routed_experts_base64 = data["meta_info"].get("routed_experts", None)  # 获取base64编码的路由专家索引
    routed_experts = np.frombuffer(  # 从缓冲区创建NumPy数组
        pybase64.b64decode(routed_experts_base64.encode("utf-8")), dtype=np.int32  # base64解码为int32
    )
    return routed_experts  # 返回路由专家索引数组
