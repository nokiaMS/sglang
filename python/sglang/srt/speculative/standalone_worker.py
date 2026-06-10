# 独立推测解码工作器模块。
# 实现了StandaloneWorker，继承自EAGLEWorker，
# 使用独立的嵌入层和语言模型头，不与目标模型共享。
# 适用于草稿模型和目标模型结构不同的场景。

import logging  # 导入日志模块
from typing import Optional  # 导入可选类型

import torch  # 导入PyTorch

from sglang.srt.layers.moe.utils import (  # 导入MoE工具
    speculative_moe_a2a_backend_context,  # 推测MoE a2a后端上下文
    speculative_moe_backend_context,  # 推测MoE后端上下文
)
from sglang.srt.managers.tp_worker import TpModelWorker  # 导入张量并行工作器
from sglang.srt.server_args import ServerArgs  # 导入服务器参数
from sglang.srt.speculative.adaptive_runtime_state import (  # 导入自适应运行时状态
    AdaptiveController,
)
from sglang.srt.speculative.eagle_worker import EAGLEWorker  # 导入EAGLE工作器
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # 导入推测算法
from sglang.srt.speculative.spec_utils import draft_tp_context, load_token_map  # 导入推测工具
from sglang.srt.utils import empty_context, get_bool_env_var, is_cuda  # 导入工具函数

if is_cuda():  # 如果是CUDA设备
    from sgl_kernel import segment_packbits  # noqa: F401  # 导入segment_packbits

logger = logging.getLogger(__name__)  # 获取日志记录器
SGLANG_RETURN_ORIGINAL_LOGPROB = get_bool_env_var("SGLANG_RETURN_ORIGINAL_LOGPROB")  # 获取原始logprob环境变量


class StandaloneWorker(EAGLEWorker):
    """独立推测解码工作器，继承自EAGLEWorker，使用独立的嵌入层和语言模型头。"""

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
        """初始化独立推测解码工作器，设置参数、内存池和注意力后端。"""
        # Parse arguments
        self.server_args = server_args  # 保存服务器参数
        self.topk = server_args.speculative_eagle_topk  # topk参数
        self.speculative_num_steps = server_args.speculative_num_steps  # 推测步数
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens  # 推测草稿token数
        self.gpu_id = gpu_id  # GPU ID
        self.device = server_args.device  # 设备
        self.target_worker = target_worker  # 目标工作器
        self.page_size = server_args.page_size  # 页面大小
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(  # 解析推测算法
            server_args.speculative_algorithm
        )

        # TODO: Adaptive speculative
        self.adaptive_controller: Optional[AdaptiveController] = None  # 自适应控制器（待实现）

        # Override the context length of the draft model to be the same as the target model.
        server_args.context_length = target_worker.model_runner.model_config.context_len  # 覆盖上下文长度

        # Do not capture cuda graph in `super().__init__()`
        # It will be captured later.
        backup_disable_cuda_graph = server_args.disable_cuda_graph  # 备份CUDA图设置
        server_args.disable_cuda_graph = True  # 临时禁用CUDA图
        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (  # 共享内存池分配器
            target_worker.get_memory_pool()
        )

        # Load hot token ids
        if server_args.speculative_token_map is not None:  # 如果指定了token映射
            self.hot_token_id = load_token_map(server_args.speculative_token_map)  # 加载token映射
            server_args.json_model_override_args = (  # 设置JSON模型覆盖参数
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None  # 无token映射

        # Init draft worker
        with (
            empty_context(),  # 空上下文
            speculative_moe_backend_context(),  # 推测MoE后端上下文
            speculative_moe_a2a_backend_context(),  # 推测MoE a2a后端上下文
        ):
            TpModelWorker.__init__(  # 直接调用TpModelWorker初始化（跳过EAGLEWorker）
                self,
                server_args=server_args,  # 服务器参数
                gpu_id=gpu_id,  # GPU ID
                tp_rank=tp_rank,  # TP排名
                pp_rank=0,  # spec workers don't support pipeline parallelism  # 不支持流水线并行
                dp_rank=dp_rank,  # DP排名
                moe_ep_rank=moe_ep_rank,  # MoE EP排名
                attn_cp_rank=attn_cp_rank,  # 注意力CP排名
                moe_dp_rank=moe_dp_rank,  # MoE DP排名
                nccl_port=nccl_port,  # NCCL端口
                is_draft_worker=True,  # 标记为草稿工作器
                req_to_token_pool=self.req_to_token_pool,  # 请求到token映射池
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,  # token到KV池分配器
                memory_pool_config=target_worker.model_runner.memory_pool_config,  # 内存池配置
            )

        # Init attention backend and cuda graphs
        self.draft_model_runner.server_args.disable_cuda_graph = (  # 恢复CUDA图设置
            backup_disable_cuda_graph
        )
        self.draft_tp_context = (  # 设置草稿TP上下文
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with (
            self.draft_tp_context(self.draft_model_runner.tp_group),  # 草稿TP上下文
            speculative_moe_backend_context(),  # 推测MoE后端上下文
            speculative_moe_a2a_backend_context(),  # 推测MoE a2a后端上下文
        ):
            self.init_attention_backend()  # 初始化注意力后端
            self.init_cuda_graphs()  # 初始化CUDA图

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(  # 每topk新增页数占位张量
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)  # 扩展长度占位张量
