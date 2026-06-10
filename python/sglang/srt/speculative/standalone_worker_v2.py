# 独立推测解码工作器V2模块，支持重叠调度。
# 实现了StandaloneDraftWorker（使用独立嵌入层/语言模型头的草稿工作器）
# 和StandaloneWorkerV2（支持重叠调度的主工作器）。
# 适用于草稿模型和目标模型结构不同、不共享嵌入层的场景。

import contextlib  # 导入上下文管理工具
import logging  # 导入日志模块
from typing import Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch

from sglang.srt.environ import envs  # 导入环境变量
from sglang.srt.layers.moe.utils import speculative_moe_backend_context  # 导入推测MoE后端上下文
from sglang.srt.managers.tp_worker import TpModelWorker  # 导入张量并行工作器
from sglang.srt.server_args import ServerArgs  # 导入服务器参数
from sglang.srt.speculative.adaptive_runtime_state import (  # 导入自适应运行时状态
    AdaptiveController,
)
from sglang.srt.speculative.eagle_utils import TreeMaskMode  # 导入树掩码模式
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2  # 导入EAGLE V2工作器
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # 导入推测算法
from sglang.srt.speculative.spec_utils import draft_tp_context  # 导入草稿TP上下文
from sglang.srt.utils import empty_context, get_bool_env_var, is_cuda  # 导入工具函数

if is_cuda():  # 如果是CUDA设备
    from sgl_kernel import segment_packbits  # noqa: F401  # 导入segment_packbits

logger = logging.getLogger(__name__)  # 获取日志记录器
SGLANG_RETURN_ORIGINAL_LOGPROB = get_bool_env_var("SGLANG_RETURN_ORIGINAL_LOGPROB")  # 获取原始logprob环境变量


def _get_plan_stream(
    device: str,  # 设备类型
) -> Tuple[any, contextlib.AbstractContextManager]:
    """获取计划流及其上下文管理器，用于重叠计划与计算。"""
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():  # 如果启用重叠计划流
        plan_stream = torch.get_device_module(device).Stream()  # 创建计划流
        plan_stream_ctx = torch.get_device_module(device).stream(plan_stream)  # 创建流上下文
        return plan_stream, plan_stream_ctx  # 返回流和上下文
    else:
        return None, contextlib.nullcontext()  # 返回None和空上下文


class StandaloneDraftWorker(EagleDraftWorker):
    """独立草稿工作器，不与目标模型共享嵌入层和语言模型头。"""
    """Custom EagleDraftWorker that doesn't share embeddings/lm_head with target model."""

    def __init__(
        self,
        server_args: ServerArgs,  # 服务器参数
        gpu_id: int,  # GPU ID
        tp_rank: int,  # 张量并行排名
        dp_rank: int,  # 数据并行排名
        moe_ep_rank: int,  # MoE专家并行排名
        attn_cp_rank: int,  # 注意力上下文并行排名
        moe_dp_rank: int,  # MoE数据并行排名
        nccl_port: int,  # NCCL端口
        target_worker: TpModelWorker,  # 目标工作器
    ):
        """初始化独立草稿工作器，使用独立的嵌入层和语言模型头。"""
        # copy args
        self.server_args = server_args  # 服务器参数
        self.gpu_id = gpu_id  # GPU ID
        self.tp_rank = tp_rank  # TP排名
        self.dp_rank = dp_rank  # DP排名
        self.moe_ep_rank = moe_ep_rank  # MoE EP排名
        self.nccl_port = nccl_port  # NCCL端口
        self.target_worker = target_worker  # 目标工作器
        self.attn_cp_rank = attn_cp_rank  # 注意力CP排名
        self.moe_dp_rank = moe_dp_rank  # MoE DP排名

        # Args for easy access
        self.device = server_args.device  # 设备
        self.topk = server_args.speculative_eagle_topk  # topk参数
        self.speculative_num_steps = server_args.speculative_num_steps  # 推测步数
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens  # 推测草稿token数
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(  # 解析推测算法
            server_args.speculative_algorithm
        )

        # Pre-allocated constants for the topk=1 chain fast path in draft_forward.
        self._topk1_parents_prealloc = None  # topk=1链式快速路径的预分配父节点
        self._topk1_score_indices_prealloc = None  # topk=1链式快速路径的预分配分数索引
        self._rebuild_topk1_chain_buffers()  # 重建topk=1链式缓冲区

        # Set constant
        from sglang.srt.speculative.eagle_info import EagleDraftInput  # 导入EAGLE草稿输入

        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(  # 设置每次解码的分配长度
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )

        # Do not capture cuda graph in `TpModelWorker` init,
        # will capture later with init_cuda_graphs()
        backup_disable_cuda_graph = server_args.disable_cuda_graph  # 备份CUDA图设置
        server_args.disable_cuda_graph = True  # 临时禁用CUDA图

        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (  # 共享内存池分配器
            target_worker.get_memory_pool()
        )
        with empty_context():  # 空上下文
            # Init draft worker
            self.draft_worker = TpModelWorker(  # 创建草稿工作器
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # spec workers don't support pipeline parallelism  # 不支持流水线并行
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,  # 标记为草稿工作器
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                memory_pool_config=target_worker.model_runner.memory_pool_config,
            )

        # Alias for better readability
        self.draft_runner = self.draft_worker.model_runner  # 草稿模型运行器别名

        self.init_token_map()  # 初始化token映射
        self.init_lm_head()  # 初始化语言模型头（独立，不共享）

        # Init attention backend and cuda graphs
        self.draft_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph  # 恢复CUDA图设置
        self.draft_tp_context = (  # 设置草稿TP上下文
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with (
            self.draft_tp_context(self.draft_runner.tp_group),  # 草稿TP上下文
            speculative_moe_backend_context(),  # 推测MoE后端上下文
        ):
            self.init_attention_backend()  # 初始化注意力后端
            self.init_cuda_graphs()  # 初始化CUDA图
        self.tree_mask_mode = TreeMaskMode.FULL_MASK  # 树掩码模式

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)  # 获取计划流

    def init_lm_head(self):
        """重写初始化语言模型头，不与目标模型共享嵌入层和语言模型头。"""
        """Override to prevent sharing embeddings and lm_head with target model."""
        # For standalone worker, we don't share embeddings and lm_head
        # The draft model uses its own embeddings and lm_head
        pass  # 不做任何操作，使用草稿模型自身的嵌入和头


class StandaloneWorkerV2(EAGLEWorkerV2):
    """独立推测解码工作器V2，使用独立的草稿模型，支持重叠调度。"""

    def __init__(
        self,
        server_args: ServerArgs,  # 服务器参数
        gpu_id: int,  # GPU ID
        tp_rank: int,  # 张量并行排名
        dp_rank: Optional[int],  # 数据并行排名
        moe_ep_rank: int,  # MoE专家并行排名
        attn_cp_rank: int,  # 注意力上下文并行排名
        moe_dp_rank: int,  # MoE数据并行排名
        nccl_port: int,  # NCCL端口
        target_worker: TpModelWorker,  # 目标工作器
    ):
        """初始化独立推测解码工作器V2，创建独立草稿工作器。"""
        # Parse arguments
        self.server_args = server_args  # 服务器参数
        self.topk = server_args.speculative_eagle_topk  # topk参数
        self.speculative_num_steps = server_args.speculative_num_steps  # 推测步数
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens  # 推测草稿token数
        self.gpu_id = gpu_id  # GPU ID
        self.device = server_args.device  # 设备
        self._target_worker = target_worker  # 目标工作器
        self.page_size = server_args.page_size  # 页面大小
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(  # 解析推测算法
            server_args.speculative_algorithm
        )

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (  # 共享内存池
            target_worker.get_memory_pool()
        )

        # Override the context length of the draft model to be the same as the target model.
        server_args.context_length = target_worker.model_runner.model_config.context_len  # 覆盖上下文长度

        # Create our custom draft worker that doesn't share embeddings/lm_head
        self._draft_worker = StandaloneDraftWorker(  # 创建独立草稿工作器
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
        )

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(  # 每topk新增页数占位张量
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)  # 扩展长度占位张量

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)  # 获取计划流

        # TODO: Adaptive speculative
        self.adaptive_controller: Optional[AdaptiveController] = None  # 自适应控制器（待实现）
