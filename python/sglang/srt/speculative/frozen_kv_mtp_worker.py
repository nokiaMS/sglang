# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# 冻结KV MTP（多token预测）草稿工作器模块。
# 该工作器只读取目标模型的KV缓存，不进行助手侧的KV扩展。
# 复用EAGLE的验证输入/输出契约，但拥有独立的种子和循环草稿逻辑。
# 适用于推测解码中无需扩展KV缓存的MTP场景。
"""Frozen-KV MTP draft worker.

The assistant reads target KV only. It reuses EAGLE's verify input/output
contract, but owns the seed and recurrent draft loop because there is no
assistant-side KV extension.
"""

from __future__ import annotations  # 启用延迟类型注解评估

import logging  # 导入日志模块
from typing import List, Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput  # 导入logits处理器输出类
from sglang.srt.layers.moe.utils import (  # 导入混合专家工具函数
    speculative_moe_a2a_backend_context,  # 推测解码MoE all-to-all后端上下文
    speculative_moe_backend_context,  # 推测解码MoE后端上下文
)
from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1  # 导入为推测解码v1添加输出logprob的函数
from sglang.srt.managers.schedule_batch import ScheduleBatch  # 导入调度批次类
from sglang.srt.managers.scheduler import GenerationBatchResult  # 导入生成批次结果类
from sglang.srt.managers.tp_worker import TpModelWorker  # 导入张量并行模型工作器
from sglang.srt.model_executor.forward_batch_info import (  # 导入前向批次信息
    CaptureHiddenMode,  # 隐藏状态捕获模式
    ForwardBatch,  # 前向批次
    ForwardMode,  # 前向模式
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context  # 导入前向上下文
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig  # 导入内存池配置类
from sglang.srt.observability.req_time_stats import set_time_batch  # 导入请求时间统计设置函数
from sglang.srt.observability.trace import get_global_tracing_enabled  # 导入全局追踪启用检查函数
from sglang.srt.server_args import ServerArgs  # 导入服务器参数类
from sglang.srt.speculative.eagle_utils import (  # 导入EAGLE工具函数
    build_tree_kernel_efficient,  # 高效构建树核
    organize_draft_results,  # 组织草稿结果
)
from sglang.srt.speculative.frozen_kv_mtp_info import (  # 导入冻结KV MTP信息类
    FrozenKVMTPContext,  # 冻结KV MTP上下文
    FrozenKVMTPDraftExtendInput,  # 冻结KV MTP草稿扩展输入
    FrozenKVMTPDraftInput,  # 冻结KV MTP草稿输入
    FrozenKVMTPVerifyInput,  # 冻结KV MTP验证输入
    FrozenKVMTPVerifyOutput,  # 冻结KV MTP验证输出
)
from sglang.srt.speculative.frozen_kv_mtp_utils import (  # 导入冻结KV MTP工具函数
    capture_for_decode,  # 为解码捕获
    expand_for_topk_draft,  # 为topk草稿扩展
    frozen_kv_target_view,  # 冻结KV目标视图
    position_for_batch,  # 批次位置计算
    select_last_extend_hidden,  # 选择最后扩展隐藏状态
    select_last_verified_seed,  # 选择最后验证种子
    set_frozen_kv_positions,  # 设置冻结KV位置
    target_kv_pool_view,  # 目标KV池视图
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # 导入推测算法枚举
from sglang.srt.speculative.spec_utils import (  # 导入推测解码工具函数
    draft_tp_context,  # 草稿张量并行上下文
    fast_topk,  # 快速topk选择
    generate_token_bitmask,  # 生成token位掩码
    select_top_k_tokens,  # 选择top-k token
)
from sglang.srt.utils import empty_context  # 导入空上下文工具
from sglang.srt.utils.async_probe import (  # 导入异步探测函数
    maybe_detect_inf,  # 检测无穷大
    maybe_detect_nan,  # 检测NaN
    maybe_detect_oob,  # 检测越界
)

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class FrozenKVMTPWorker(TpModelWorker):
    """冻结KV MTP工作器；与EAGLEWorker具有相同的构造函数签名。
    入口方法为:meth:`forward_batch_generation`。
    该工作器只读取目标KV缓存，不进行助手侧的KV扩展。
    """
    """Frozen-KV MTP worker; same constructor shape as EAGLEWorker. Entry:
    :meth:`forward_batch_generation` (stubs for now).
    """

    def __init__(
        self,
        server_args: ServerArgs,  # 服务器参数
        gpu_id: int,  # GPU设备ID
        tp_rank: int,  # 张量并行排名
        dp_rank: Optional[int],  # 数据并行排名，可选
        moe_ep_rank: int,  # 混合专家专家并行排名
        attn_cp_rank: int,  # 注意力上下文并行排名
        moe_dp_rank: int,  # 混合专家数据并行排名
        nccl_port: int,  # NCCL通信端口
        target_worker: TpModelWorker,  # 目标模型工作器
    ):
        """初始化冻结KV MTP工作器，设置草稿模型和注意力后端。"""
        self.server_args = server_args  # 保存服务器参数
        self.topk = server_args.speculative_eagle_topk  # 获取topk参数
        self.speculative_num_steps = server_args.speculative_num_steps  # 获取推测步数
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens  # 获取推测草稿token数
        self.gpu_id = gpu_id  # 保存GPU ID
        self.device = server_args.device  # 保存设备类型
        self.target_worker = target_worker  # 保存目标工作器引用
        self.page_size = server_args.page_size  # 保存页面大小
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(  # 从字符串解析推测算法
            server_args.speculative_algorithm  # 传入算法名称
        )
        assert self.speculative_algorithm.is_frozen_kv_mtp(), (  # 断言算法类型为冻结KV MTP
            "FrozenKVMTPWorker should only be instantiated for "
            "SpeculativeAlgorithm.FROZEN_KV_MTP, got "
            f"{self.speculative_algorithm.name}. The dispatch happens in "
            "arg_groups.speculative_hook.handle_speculative_decoding -> "
            "_resolve_speculative_algorithm_alias."
        )

        # Assistant reads target KV directly, so its context length must match the target.
        server_args.context_length = target_worker.model_runner.model_config.context_len  # 设置草稿模型上下文长度与目标模型一致

        # Defer cuda graph capture; we do it ourselves below.
        backup_disable_cuda_graph = server_args.disable_cuda_graph  # 备份原始的cuda图禁用设置
        server_args.disable_cuda_graph = True  # 临时禁用cuda图捕获

        # Draft attention uses target req_to_token + KV allocator (read-only).
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (  # 获取目标工作器的内存池
            target_worker.get_memory_pool()  # 从目标工作器获取内存池
        )

        target_cfg = target_worker.model_runner.memory_pool_config  # 获取目标工作器的内存池配置
        draft_pool_config = MemoryPoolConfig(  # 创建草稿模型的内存池配置
            max_total_num_tokens=64,  # Dummy value  # 虚拟值
            max_running_requests=target_cfg.max_running_requests,  # 最大运行请求数与目标一致
        )

        self.hot_token_id = None  # 热门token ID初始化为None

        with (
            empty_context()  # 使用空上下文
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():  # 使用推测解码MoE后端上下文
            super().__init__(  # 调用父类初始化
                server_args=server_args,  # 服务器参数
                gpu_id=gpu_id,  # GPU ID
                tp_rank=tp_rank,  # 张量并行排名
                pp_rank=0,  # 流水线并行排名为0（不支持）
                dp_rank=dp_rank,  # 数据并行排名
                moe_ep_rank=moe_ep_rank,  # 混合专家专家并行排名
                attn_cp_rank=attn_cp_rank,  # 注意力上下文并行排名
                moe_dp_rank=moe_dp_rank,  # 混合专家数据并行排名
                nccl_port=nccl_port,  # NCCL端口
                is_draft_worker=True,  # 标记为草稿工作器
                req_to_token_pool=self.req_to_token_pool,  # 请求到token映射池
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,  # token到KV池分配器
                memory_pool_config=draft_pool_config,  # 内存池配置
            )

        embed, head = self.target_worker.model_runner.model.get_embed_and_head()  # 获取目标模型的嵌入层和语言模型头
        if hasattr(self.draft_model_runner.model, "set_embed_and_head"):  # 检查草稿模型是否支持设置嵌入和头
            self.draft_model_runner.model.set_embed_and_head(embed, head)  # 设置草稿模型的嵌入和头
        else:
            logger.debug(  # 记录调试信息
                "Draft model %s does not implement set_embed_and_head; "
                "skipping target-embedding bind in Frozen-KV MTP skeleton.",
                type(self.draft_model_runner.model).__name__,  # 草稿模型类名
            )

        self.kv_context: Optional["FrozenKVMTPContext"] = None  # 初始化KV上下文为None
        if hasattr(self.draft_model_runner.model, "bind_frozen_kv_context"):  # 检查草稿模型是否支持绑定冻结KV上下文
            self._bind_kv_context()  # 绑定冻结KV上下文

        self.draft_model_runner.server_args.disable_cuda_graph = (  # 恢复草稿模型的cuda图设置
            backup_disable_cuda_graph  # 使用备份的设置
        )

        self.draft_tp_context = (  # 设置草稿张量并行上下文
            draft_tp_context if server_args.enable_dp_attention else empty_context  # 根据是否启用DP注意力选择上下文
        )

        self.draft_attn_backend = self._init_draft_attn_backend()  # 初始化草稿注意力后端
        self.draft_model_runner.draft_attn_backend = self.draft_attn_backend  # 设置草稿模型运行器的注意力后端
        self.cuda_graph_runner = None  # 初始化CUDA图运行器为None

        with (
            self.draft_tp_context(self.draft_model_runner.tp_group),  # 使用草稿TP上下文
            speculative_moe_backend_context(),  # 使用推测MoE后端上下文
            speculative_moe_a2a_backend_context(),  # 使用推测MoE a2a后端上下文
        ):
            self.init_cuda_graphs()  # 初始化CUDA图

    @property
    def draft_model_runner(self):
        """获取草稿模型运行器。"""
        return self.model_runner  # 返回模型运行器

    def get_attn_backend(self):  # pragma: no cover - exposed for adaptive
        """获取注意力后端，供自适应机制使用。"""
        return self.draft_attn_backend  # 返回草稿注意力后端

    def clear_cache_pool(self):
        """清空缓存池，由于共享目标工作器的分配器，此处无需操作。"""
        pass  # 无操作

    def _resolve_draft_backend_type(self) -> str:
        """解析草稿后端类型，优先使用推测草稿注意力后端设置。"""
        return (  # 返回解析后的后端类型
            self.server_args.speculative_draft_attention_backend  # 优先使用推测草稿注意力后端
            or self.server_args.decode_attention_backend  # 其次使用解码注意力后端
            or self.server_args.attention_backend  # 最后使用通用注意力后端
        )

    def _init_draft_attn_backend(self):
        """初始化草稿注意力后端，topk>1时仅支持triton后端。"""
        if self.topk == 1:  # 如果topk为1
            return self.draft_model_runner.attn_backend  # 直接使用草稿模型的注意力后端

        backend_type = self._resolve_draft_backend_type()  # 解析后端类型
        if backend_type != "triton":  # 检查是否为triton后端
            raise ValueError(  # 非triton后端抛出异常
                "Frozen-KV MTP topk > 1 currently supports only the triton "
                f"attention backend, got {backend_type}."
            )
        return self._init_triton_draft_attn_backend()  # 初始化triton草稿注意力后端

    def _init_triton_draft_attn_backend(self):
        """初始化Triton草稿注意力后端，配置KV索引指针缓冲区。"""
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend  # 导入Triton注意力后端

        max_bs = self.req_to_token_pool.size * self.topk  # 计算最大批次大小
        kv_indptr_buf = torch.zeros(  # 创建KV索引指针缓冲区
            (max_bs + 1,), dtype=torch.int32, device=self.draft_model_runner.device  # 形状为max_bs+1，int32类型
        )
        return TritonAttnBackend(  # 返回Triton注意力后端实例
            self.draft_model_runner,  # 草稿模型运行器
            skip_prefill=True,  # 跳过预填充
            kv_indptr_buf=kv_indptr_buf,  # KV索引指针缓冲区
        )

    def _bind_kv_context(self) -> None:
        """绑定冻结KV上下文到草稿模型，构建与目标模型的KV关联。"""
        draft_model = self.draft_model_runner.model  # 获取草稿模型
        if not hasattr(draft_model, "build_frozen_kv_mtp_context") or not hasattr(  # 检查草稿模型是否支持构建和绑定冻结KV上下文
            draft_model, "bind_frozen_kv_context"
        ):
            logger.debug(  # 记录调试信息
                "Draft model %s does not implement Frozen-KV MTP context hooks; "
                "skipping frozen-kv bind.",
                type(draft_model).__name__,  # 草稿模型类名
            )
            return  # 不支持则直接返回

        ctx = draft_model.build_frozen_kv_mtp_context(  # 构建冻结KV MTP上下文
            target_model=self.target_worker.model_runner.model,  # 目标模型
            target_token_to_kv_pool=self.target_worker.model_runner.token_to_kv_pool,  # 目标token到KV池
        )
        draft_model.bind_frozen_kv_context(ctx)  # 将上下文绑定到草稿模型
        self.kv_context = ctx  # 保存KV上下文引用

    def _frozen_kv_target_view(self, forward_batch: ForwardBatch):
        """获取前向批次的冻结KV目标视图。"""
        return frozen_kv_target_view(  # 调用工具函数获取目标视图
            forward_batch, self.kv_context, self.draft_attn_backend  # 传入前向批次、KV上下文和注意力后端
        )

    def _target_kv_pool_view(self, forward_batch: ForwardBatch):
        """获取前向批次的目标KV池视图。"""
        return target_kv_pool_view(  # 调用工具函数获取KV池视图
            forward_batch, self.kv_context, self.draft_attn_backend  # 传入前向批次、KV上下文和注意力后端
        )

    def _set_positions(self, forward_batch: ForwardBatch) -> None:
        """设置前向批次的冻结KV位置信息。"""
        set_frozen_kv_positions(forward_batch, self.topk)  # 调用工具函数设置位置

    def _expand_for_topk_draft(self, forward_batch: ForwardBatch) -> None:
        """为topk草稿扩展前向批次。"""
        expand_for_topk_draft(forward_batch, self.topk)  # 调用工具函数进行扩展

    def _position_for_batch(self, batch: ScheduleBatch) -> torch.Tensor:
        """计算批次的position tensor。"""
        return position_for_batch(batch)  # 调用工具函数计算位置

    @property
    def _recurrent_hidden_size(self) -> int:
        """获取草稿模型的循环隐藏状态维度大小。"""
        return int(self.draft_model_runner.model.backbone_hidden_size)  # 返回骨干隐藏维度

    def _init_frozen_kv_metadata(self, forward_batch: ForwardBatch) -> None:
        """初始化冻结KV元数据，设置序列长度总和和注意力前向元数据。"""
        if forward_batch.forward_mode.is_idle():  # 如果是空闲模式
            return  # 直接返回
        if forward_batch.seq_lens_cpu is not None:  # 如果CPU序列长度存在
            forward_batch.seq_lens_sum = forward_batch.seq_lens_cpu.sum().item()  # 使用CPU版本计算总和
        else:
            forward_batch.seq_lens_sum = torch.sum(forward_batch.seq_lens).item()  # 使用GPU版本计算总和
        with self._frozen_kv_target_view(forward_batch):  # 使用冻结KV目标视图
            self.draft_attn_backend.init_forward_metadata(forward_batch)  # 初始化注意力前向元数据

    def _init_frozen_kv_metadata_capture_cuda_graph(
        self, forward_batch: ForwardBatch  # 前向批次
    ) -> None:
        """在CUDA图捕获阶段初始化冻结KV元数据。"""
        with self._frozen_kv_target_view(forward_batch):  # 使用冻结KV目标视图
            self.draft_attn_backend.init_forward_metadata_capture_cuda_graph(  # 初始化CUDA图捕获的前向元数据
                forward_batch.batch_size,  # 批次大小
                forward_batch.positions.numel(),  # 位置数量
                forward_batch.req_pool_indices,  # 请求池索引
                forward_batch.seq_lens,  # 序列长度
                encoder_lens=None,  # 编码器长度为None
                forward_mode=ForwardMode.DECODE,  # 前向模式为解码
                spec_info=None,  # 推测信息为None
            )

    def _init_frozen_kv_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int, seq_lens_sum: int  # 前向批次, # 批次大小, # 序列长度总和
    ) -> None:
        """在CUDA图重放阶段初始化冻结KV元数据。"""
        with self._frozen_kv_target_view(forward_batch):  # 使用冻结KV目标视图
            self.draft_attn_backend.init_forward_metadata_replay_cuda_graph(  # 初始化CUDA图重放的前向元数据
                bs,  # 批次大小
                forward_batch.req_pool_indices[:bs],  # 请求池索引（截取）
                forward_batch.seq_lens[:bs],  # 序列长度（截取）
                seq_lens_sum,  # 序列长度总和
                encoder_lens=None,  # 编码器长度为None
                forward_mode=ForwardMode.DECODE,  # 前向模式为解码
                spec_info=None,  # 推测信息为None
                seq_lens_cpu=(  # CPU序列长度
                    forward_batch.seq_lens_cpu[:bs]  # 截取前bs个
                    if forward_batch.seq_lens_cpu is not None  # 如果CPU序列长度存在
                    else None  # 否则为None
                ),
            )

    def init_cuda_graphs(self) -> None:
        """初始化CUDA图，捕获冻结KV MTP草稿的CUDA图。"""
        if self.server_args.disable_cuda_graph or self.speculative_num_steps <= 1:  # 如果禁用CUDA图或步数<=1
            return  # 直接返回
        if self.target_worker.device != "cuda":  # 如果目标设备不是CUDA
            logger.info(  # 记录信息
                "Frozen-KV MTP draft CUDA graph is only supported on CUDA; "
                "running the draft loop eagerly on %s.",
                self.target_worker.device,  # 设备类型
            )
            return  # 直接返回

        from sglang.srt.speculative.frozen_kv_mtp_cuda_graph_runner import (  # 导入CUDA图运行器
            FrozenKVMTPCudaGraphRunner,
        )

        logger.info("Capture Frozen-KV MTP draft cuda graph begin.")  # 记录开始捕获
        self.cuda_graph_runner = FrozenKVMTPCudaGraphRunner(self)  # 创建CUDA图运行器
        logger.info("Capture Frozen-KV MTP draft cuda graph end.")  # 记录捕获完成

    def _select_last_extend_hidden(
        self, batch: ScheduleBatch, hidden_states: torch.Tensor  # 调度批次, # 隐藏状态张量
    ) -> torch.Tensor:
        """选择每个请求最后扩展步骤的隐藏状态。"""
        return select_last_extend_hidden(batch, hidden_states)  # 调用工具函数选择

    def _select_last_verified_seed(
        self, draft_input: FrozenKVMTPDraftExtendInput  # 冻结KV MTP草稿扩展输入
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """从草稿扩展输入中选择最后验证的种子token和隐藏状态。"""
        return select_last_verified_seed(draft_input)  # 调用工具函数选择

    def _capture_for_decode(
        self, logits_output: LogitsProcessorOutput, draft_input: FrozenKVMTPDraftInput  # logits输出, # 冻结KV MTP草稿输入
    ) -> None:
        """为解码捕获logits输出中的topk信息到草稿输入。"""
        capture_for_decode(logits_output, draft_input, self.topk)  # 调用工具函数捕获

    def _run_assistant_seed_step(
        self,
        batch: ScheduleBatch,  # 调度批次
        last_token_ids: torch.Tensor,  # 最后token ID
        last_hidden_states: torch.Tensor,  # 最后隐藏状态
        seq_lens_cpu: Optional[torch.Tensor] = None,  # CPU序列长度，可选
        mm_input_embeds: Optional[torch.Tensor] = None,  # 多模态输入嵌入，可选
        draft_input: Optional[FrozenKVMTPDraftInput] = None,  # 草稿输入，可选
    ) -> None:
        """运行助手种子步骤，将种子输入存储到batch.spec_info上；
        前向运行在捕获的草稿图内执行（见draft_forward的种子迭代）。
        """
        del seq_lens_cpu, mm_input_embeds, draft_input  # 删除未使用的参数

        if batch.forward_mode.is_idle() or last_token_ids.numel() == 0:  # 如果是空闲模式或没有token
            batch.spec_info = FrozenKVMTPDraftInput.create_idle_input(  # 创建空闲输入
                device=batch.device,  # 设备
                hidden_size=self._recurrent_hidden_size,  # 隐藏维度
                dtype=self.model_config.dtype,  # 数据类型
                topk=self.topk,  # topk参数
                capture_hidden_mode=CaptureHiddenMode.LAST,  # 捕获模式为LAST
            )
            return  # 返回

        stashed = FrozenKVMTPDraftInput()  # 创建新的草稿输入
        stashed.bonus_tokens = last_token_ids.to(torch.int64)  # 设置bonus token
        stashed.hidden_states = last_hidden_states  # 设置隐藏状态
        # Real-shaped zeros so inherited `filter_batch`/`merge_batch` can slice
        # them between iters; overwritten by the captured seed iter.
        bs = last_token_ids.shape[0]  # 获取批次大小
        device = last_token_ids.device  # 获取设备
        stashed.topk_p = torch.zeros(  # 初始化topk概率为零
            (bs, self.topk), device=device, dtype=torch.float32  # 形状为(bs, topk)
        )
        stashed.topk_index = torch.zeros(  # 初始化topk索引为零
            (bs, self.topk), device=device, dtype=torch.int64  # 形状为(bs, topk)
        )
        stashed.capture_hidden_mode = CaptureHiddenMode.LAST  # 设置捕获模式为LAST
        stashed.num_tokens_per_req = 1  # 每个请求1个token
        stashed.num_tokens_for_logprob_per_req = 1  # 每个请求1个logprob token
        batch.spec_info = stashed  # 将草稿输入安装到批次上

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """执行推测解码的前向批次生成，包括扩展、草稿、验证和草稿扩展流程。"""
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:  # 如果是扩展模式或批次中包含扩展
            (
                logits_output,  # logits输出
                next_token_ids,  # 下一个token ID
                seq_lens_cpu,  # CPU序列长度
                can_run_cuda_graph,  # 是否可运行CUDA图
            ) = self.forward_target_extend(batch)  # 运行目标扩展
            with (
                self.draft_tp_context(self.draft_model_runner.tp_group),  # 草稿TP上下文
                speculative_moe_backend_context(),  # 推测MoE后端上下文
                speculative_moe_a2a_backend_context(),  # 推测MoE a2a后端上下文
            ):
                self.forward_draft_extend(  # 运行草稿扩展
                    batch,  # 调度批次
                    logits_output.hidden_states,  # 隐藏状态
                    next_token_ids,  # 下一个token ID
                    seq_lens_cpu,  # CPU序列长度
                    logits_output.mm_input_embeds,  # 多模态输入嵌入
                )
            return GenerationBatchResult(  # 返回生成批次结果
                logits_output=logits_output,  # logits输出
                next_token_ids=next_token_ids,  # 下一个token ID
                num_correct_drafts=0,  # 正确草稿数为0
                can_run_cuda_graph=can_run_cuda_graph,  # 是否可运行CUDA图
            )

        set_time_batch(batch.reqs, "set_spec_draft_start_time", trace_only=True)  # 设置草稿开始时间
        with (
            self.draft_tp_context(self.draft_model_runner.tp_group),  # 草稿TP上下文
            speculative_moe_backend_context(),  # 推测MoE后端上下文
            speculative_moe_a2a_backend_context(),  # 推测MoE a2a后端上下文
        ):
            verify_input = self.draft(batch)  # 运行草稿生成
        set_time_batch(batch.reqs, "set_spec_draft_end_time", trace_only=True)  # 设置草稿结束时间
        set_time_batch(batch.reqs, "set_spec_verify_start_time", trace_only=True)  # 设置验证开始时间

        # Install verify_input as `batch.spec_info` for the verify forward.
        batch.spec_info = verify_input  # 安装验证输入为批次的spec_info
        verify_output = self.verify(batch)  # 运行验证

        if get_global_tracing_enabled():  # 如果启用了全局追踪
            for idx, req in enumerate(batch.reqs):  # 遍历请求
                num_correct_drafts = verify_output.num_correct_drafts_per_req_cpu[idx]  # 获取每个请求的正确草稿数
                req.time_stats.set_spec_verify_end_time(  # 设置验证结束时间
                    num_correct_drafts=num_correct_drafts  # 正确草稿数
                )

        set_time_batch(batch.reqs, "set_spec_draft_extend_start_time", trace_only=True)  # 设置草稿扩展开始时间
        with (
            self.draft_tp_context(self.draft_model_runner.tp_group),  # 草稿TP上下文
            speculative_moe_backend_context(),  # 推测MoE后端上下文
            speculative_moe_a2a_backend_context(),  # 推测MoE a2a后端上下文
        ):
            draft_extend_input = verify_output.draft_extend_input  # 获取草稿扩展输入
            if (
                self.server_args.enable_dp_attention  # 是否启用DP注意力
                or draft_extend_input.input_ids.shape[0] > 0  # 或草稿扩展输入有token
            ):
                # Install draft_extend_input as `batch.spec_info` for the seed
                # step; `_run_assistant_seed_step` replaces it with a fresh
                # `FrozenKVMTPDraftInput` for next iter.
                batch.spec_info = draft_extend_input  # 安装草稿扩展输入
                self.forward_draft_extend_after_decode(batch)  # 运行解码后草稿扩展
        set_time_batch(batch.reqs, "set_spec_draft_extend_end_time", trace_only=True)  # 设置草稿扩展结束时间

        return GenerationBatchResult(  # 返回生成批次结果
            logits_output=verify_output.logits_output,  # logits输出
            next_token_ids=verify_output.accept_tokens,  # 接受的token
            num_correct_drafts=sum(verify_output.num_correct_drafts_per_req_cpu),  # 总正确草稿数
            num_correct_drafts_per_req_cpu=verify_output.num_correct_drafts_per_req_cpu,  # 每请求正确草稿数
            can_run_cuda_graph=verify_output.can_run_cuda_graph,  # 是否可运行CUDA图
        )

    def forward_target_extend(
        self, batch: ScheduleBatch  # 调度批次
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, Optional[torch.Tensor], bool]:
        """运行目标模型扩展，获取完整隐藏状态以预填充草稿模型的KV缓存。"""
        batch.capture_hidden_mode = CaptureHiddenMode.FULL  # 设置捕获模式为FULL
        batch_result = self.target_worker.forward_batch_generation(batch)  # 运行目标工作器前向生成
        return (  # 返回结果元组
            batch_result.logits_output,  # logits输出
            batch_result.next_token_ids,  # 下一个token ID
            batch.seq_lens_cpu,  # CPU序列长度
            batch_result.can_run_cuda_graph,  # 是否可运行CUDA图
        )

    def forward_draft_extend(
        self,
        batch: ScheduleBatch,  # 调度批次
        hidden_states: torch.Tensor,  # 隐藏状态
        next_token_ids: torch.Tensor,  # 下一个token ID
        seq_lens_cpu: Optional[torch.Tensor],  # CPU序列长度
        mm_input_embeds: Optional[torch.Tensor] = None,  # 多模态输入嵌入，可选
    ) -> None:
        """运行草稿模型扩展，选择最后扩展隐藏状态并运行助手种子步骤。"""
        last_hidden = self._select_last_extend_hidden(batch, hidden_states)  # 选择最后扩展隐藏状态
        self._run_assistant_seed_step(  # 运行助手种子步骤
            batch,  # 调度批次
            next_token_ids,  # 下一个token ID
            last_hidden,  # 最后隐藏状态
            seq_lens_cpu=seq_lens_cpu,  # CPU序列长度
            mm_input_embeds=mm_input_embeds,  # 多模态输入嵌入
        )

    def forward_draft_extend_after_decode(self, batch: ScheduleBatch) -> None:
        """解码后运行草稿扩展，处理验证结果并为下一轮迭代准备种子。"""
        draft_extend_input: FrozenKVMTPDraftExtendInput = batch.spec_info  # 获取草稿扩展输入
        input_is_idle = batch.forward_mode.is_idle()  # 检查是否为空闲模式

        if not input_is_idle and draft_extend_input.input_ids.shape[0] == 0:  # 如果非空闲但无输入token
            # All reqs finished. Install an idle FrozenKVMTPDraftInput so the
            # next-iter draft sees a valid spec_info.
            batch = batch.copy()  # 复制批次
            batch.prepare_for_idle()  # 准备空闲状态
            batch.spec_info = FrozenKVMTPDraftInput.create_idle_input(  # 创建空闲输入
                device=self.device,  # 设备
                hidden_size=self._recurrent_hidden_size,  # 隐藏维度
                dtype=self.model_config.dtype,  # 数据类型
                topk=self.topk,  # topk参数
                capture_hidden_mode=CaptureHiddenMode.LAST,  # 捕获模式为LAST
            )
            return  # 返回

        if batch.forward_mode.is_idle():  # 如果是空闲模式
            return  # 直接返回

        seq_lens_backup = batch.seq_lens.clone()  # 备份序列长度
        seq_lens_cpu_backup = batch.seq_lens_cpu.clone()  # 备份CPU序列长度
        req_pool_indices_backup = batch.req_pool_indices  # 备份请求池索引

        try:
            # Verify may leave finished requests in ScheduleBatch; seed only
            # the unfinished reqs carried by `draft_extend_input`.
            batch.seq_lens = draft_extend_input.seq_lens  # 设置为草稿扩展输入的序列长度
            batch.seq_lens_cpu = draft_extend_input.seq_lens_cpu  # 设置为草稿扩展输入的CPU序列长度
            batch.req_pool_indices = draft_extend_input.req_pool_indices  # 设置为草稿扩展输入的请求池索引

            last_token_ids, last_hidden = self._select_last_verified_seed(  # 选择最后验证的种子
                draft_extend_input  # 草稿扩展输入
            )
            # `_run_assistant_seed_step` constructs a fresh `FrozenKVMTPDraftInput`
            # and installs it on `batch.spec_info` for next iter.
            self._run_assistant_seed_step(  # 运行助手种子步骤
                batch,  # 调度批次
                last_token_ids,  # 最后token ID
                last_hidden,  # 最后隐藏状态
                seq_lens_cpu=draft_extend_input.seq_lens_cpu,  # CPU序列长度
            )
        finally:
            batch.seq_lens = seq_lens_backup  # 恢复序列长度
            batch.seq_lens_cpu = seq_lens_cpu_backup  # 恢复CPU序列长度
            batch.req_pool_indices = req_pool_indices_backup  # 恢复请求池索引

    def draft(self, batch: ScheduleBatch):
        """运行草稿生成，通过多步推测生成草稿token并构建树掩码。"""
        if batch.forward_mode.is_idle():  # 如果是空闲模式
            return FrozenKVMTPVerifyInput.create_idle_input(  # 创建空闲验证输入
                self.topk,  # topk参数
                self.speculative_num_steps,  # 推测步数
                self.speculative_num_draft_tokens,  # 推测草稿token数
            )

        batch.maybe_evict_swa()  # 可能驱逐滑动窗口注意力缓存
        for req in batch.reqs:  # 遍历请求
            req.decode_batch_idx += 1  # 递增解码批次索引

        spec_info = batch.spec_info  # 获取推测信息
        assert isinstance(spec_info, FrozenKVMTPDraftInput)  # 断言类型

        if batch.sampling_info.penalizer_orchestrator.is_required:  # 如果需要惩罚器
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(  # 累积输出token
                spec_info.bonus_tokens.to(torch.int64)  # bonus token转为int64
            )

        spec_info.capture_hidden_mode = CaptureHiddenMode.LAST  # 设置捕获模式为LAST
        spec_info.num_tokens_per_req = self.topk  # 每个请求的token数为topk
        spec_info.num_tokens_for_logprob_per_req = self.topk  # 每个请求的logprob token数为topk
        spec_info.positions = self._position_for_batch(batch)  # 计算位置
        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()  # 计算序列长度总和
        batch.return_hidden_states = False  # 不返回隐藏状态

        forward_batch = ForwardBatch.init_new(batch, self.draft_model_runner)  # 初始化前向批次
        assert forward_batch.capture_hidden_mode == CaptureHiddenMode.LAST  # 断言捕获模式
        self._set_positions(forward_batch)  # 设置位置
        self._expand_for_topk_draft(forward_batch)  # 扩展topk草稿

        can_run_cuda_graph = self.cuda_graph_runner and self.cuda_graph_runner.can_run(  # 检查是否可运行CUDA图
            forward_batch  # 前向批次
        )
        if can_run_cuda_graph:  # 如果可以运行CUDA图
            parent_list, top_scores_index, draft_tokens = self.cuda_graph_runner.replay(  # 重放CUDA图
                forward_batch  # 前向批次
            )
        else:
            forward_batch.can_run_dp_cuda_graph = False  # 标记不可运行DP CUDA图
            parent_list, top_scores_index, draft_tokens = self.draft_forward(  # 运行草稿前向
                forward_batch  # 前向批次
            )

        (
            tree_mask,  # 树掩码
            position,  # 位置
            retrieve_index,  # 检索索引
            retrieve_next_token,  # 检索下一个token
            retrieve_next_sibling,  # 检索下一个兄弟节点
            draft_tokens,  # 草稿token
        ) = build_tree_kernel_efficient(  # 高效构建树核
            spec_info.bonus_tokens,  # bonus token
            parent_list,  # 父节点列表
            top_scores_index,  # top分数索引
            draft_tokens,  # 草稿token
            batch.seq_lens,  # 序列长度
            batch.seq_lens_sum,  # 序列长度总和
            self.topk,  # topk参数
            self.speculative_num_steps,  # 推测步数
            self.speculative_num_draft_tokens,  # 推测草稿token数
        )

        return FrozenKVMTPVerifyInput(  # 返回验证输入
            draft_token=draft_tokens,  # 草稿token
            custom_mask=tree_mask,  # 自定义掩码
            positions=position,  # 位置
            retrieve_index=retrieve_index,  # 检索索引
            retrieve_next_token=retrieve_next_token,  # 检索下一个token
            retrieve_next_sibling=retrieve_next_sibling,  # 检索下一个兄弟节点
            retrieve_cum_len=None,  # 检索累积长度为None
            spec_steps=self.speculative_num_steps,  # 推测步数
            topk=self.topk,  # topk参数
            draft_token_num=self.speculative_num_draft_tokens,  # 草稿token数
            capture_hidden_mode=CaptureHiddenMode.FULL,  # 捕获模式为FULL
            seq_lens_sum=batch.seq_lens_sum,  # 序列长度总和
            seq_lens_cpu=batch.seq_lens_cpu,  # CPU序列长度
        )

    def draft_forward(
        self, forward_batch: ForwardBatch, skip_attn_backend_init: bool = False  # 前向批次, # 是否跳过注意力后端初始化
    ):
        """运行草稿前向，通过种子迭代和循环迭代生成草稿token。"""
        spec_info = forward_batch.spec_info  # 获取推测信息
        assert isinstance(spec_info, FrozenKVMTPDraftInput)  # 断言类型

        score_list: List[torch.Tensor] = []  # 分数列表
        token_list: List[torch.Tensor] = []  # token列表
        parents_list: List[torch.Tensor] = []  # 父节点列表

        # Seed + recurrent iters share the same `seq_lens - 1` rope position,
        # so one init covers the loop. Must run even at num_steps == 1.
        if not skip_attn_backend_init:  # 如果不跳过注意力后端初始化
            self._init_frozen_kv_metadata(forward_batch)  # 初始化冻结KV元数据

        # Seed iter: assistant forward on (bonus_token, target_h) to produce
        # iter-0 `(topk_p, topk_index, hidden_states)`. For topk>1, replicate
        # to `bs*topk` to match kernel shapes, then slice back per-req.
        bonus_tokens = spec_info.bonus_tokens  # 获取bonus token
        target_hidden = spec_info.hidden_states  # 获取目标隐藏状态
        if self.topk > 1:  # 如果topk>1
            seed_input_ids = bonus_tokens.repeat_interleave(self.topk, dim=0)  # 复制bonus token
            seed_prev_hidden = target_hidden.repeat_interleave(self.topk, dim=0)  # 复制隐藏状态
        else:
            seed_input_ids = bonus_tokens  # 直接使用bonus token
            seed_prev_hidden = target_hidden  # 直接使用隐藏状态

        forward_batch.input_ids = seed_input_ids  # 设置输入ID
        forward_batch.spec_info.hidden_states = seed_prev_hidden  # 设置隐藏状态
        self._set_positions(forward_batch)  # 设置位置

        with (
            self._target_kv_pool_view(forward_batch),  # 使用目标KV池视图
            forward_context(ForwardContext(attn_backend=self.draft_attn_backend)),  # 使用前向上下文
        ):
            seed_output = self.draft_model_runner.forward(  # 运行草稿模型前向
                forward_batch, skip_attn_backend_init=True  # 跳过注意力后端初始化
            ).logits_output  # 获取logits输出

        maybe_detect_nan(  # 检测NaN
            seed_output.next_token_logits, "frozen_kv_mtp_draft: seed iter"  # 种子迭代
        )

        if self.topk > 1:  # 如果topk>1
            seed_next_logits = seed_output.next_token_logits[:: self.topk]  # 每topk取一个
            seed_hidden_per_req = seed_output.hidden_states[:: self.topk]  # 每topk取一个隐藏状态
        else:
            seed_next_logits = seed_output.next_token_logits  # 直接使用logits
            seed_hidden_per_req = seed_output.hidden_states  # 直接使用隐藏状态

        probs = torch.softmax(seed_next_logits, dim=-1)  # 计算softmax概率
        topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)  # 快速topk选择
        maybe_detect_oob(  # 检测越界
            topk_index,  # topk索引
            0,  # 下界
            seed_next_logits.shape[-1],  # 上界
            "frozen_kv_mtp_draft: seed topk_index OOB",  # 错误信息
        )
        hidden_states = seed_hidden_per_req  # 保存隐藏状态

        scores = None  # 初始化分数为None
        for i in range(self.speculative_num_steps):  # 循环推测步数
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(  # 选择top-k token
                i, topk_p, topk_index, hidden_states, scores, self.topk  # 步骤索引、概率、索引、隐藏状态、分数、topk
            )
            score_list.append(tree_info[0])  # 添加分数
            token_list.append(tree_info[1])  # 添加token
            parents_list.append(tree_info[2])  # 添加父节点

            if i == self.speculative_num_steps - 1:  # 如果是最后一步
                break  # 跳出循环

            forward_batch.input_ids = input_ids  # 设置输入ID
            forward_batch.spec_info.hidden_states = hidden_states  # 设置隐藏状态
            self._set_positions(forward_batch)  # 设置位置

            with (
                self._target_kv_pool_view(forward_batch),  # 使用目标KV池视图
                forward_context(ForwardContext(attn_backend=self.draft_attn_backend)),  # 使用前向上下文
            ):
                logits_output = self.draft_model_runner.forward(  # 运行草稿模型前向
                    forward_batch, skip_attn_backend_init=True  # 跳过注意力后端初始化
                ).logits_output  # 获取logits输出

            maybe_detect_nan(  # 检测NaN
                logits_output.next_token_logits, f"frozen_kv_mtp_draft step {i}"  # 步骤i
            )
            maybe_detect_inf(  # 检测无穷大
                logits_output.next_token_logits, f"frozen_kv_mtp_draft step {i}"  # 步骤i
            )
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)  # 计算softmax概率
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)  # 快速topk选择
            maybe_detect_oob(  # 检测越界
                topk_index,  # topk索引
                0,  # 下界
                logits_output.next_token_logits.shape[-1],  # 上界
                "frozen_kv_mtp_draft: topk_index OOB",  # 错误信息
            )
            hidden_states = logits_output.hidden_states  # 更新隐藏状态

        return organize_draft_results(  # 返回组织后的草稿结果
            score_list, token_list, parents_list, self.speculative_num_draft_tokens  # 分数列表、token列表、父节点列表、草稿token数
        )

    def verify(self, batch: ScheduleBatch):
        """运行验证步骤，用目标模型验证草稿token并确定接受的token。"""
        spec_info: FrozenKVMTPVerifyInput = batch.spec_info  # 获取验证输入
        seq_lens_pre_verify = batch.seq_lens.clone()  # 备份验证前的序列长度
        spec_info.prepare_for_verify(batch, self.page_size)  # 准备验证
        spec_info.num_tokens_per_req = self.speculative_num_steps + 1  # 每请求token数为步数+1
        batch.return_hidden_states = False  # 不返回隐藏状态
        batch.forward_mode = (  # 设置前向模式
            ForwardMode.TARGET_VERIFY  # 目标验证模式
            if not batch.forward_mode.is_idle()  # 如果非空闲
            else ForwardMode.IDLE  # 否则空闲模式
        )

        if batch.has_grammar:  # 如果有语法约束
            retrieve_next_token_cpu = spec_info.retrieve_next_token.cpu()  # 获取CPU上的下一个token检索
            retrieve_next_sibling_cpu = spec_info.retrieve_next_sibling.cpu()  # 获取CPU上的兄弟节点检索
            draft_tokens_cpu = spec_info.draft_token.view(  # 获取CPU上的草稿token
                spec_info.retrieve_next_token.shape  # 调整视图形状
            ).cpu()

        batch.seq_lens_cpu_cache = spec_info.seq_lens_cpu  # 缓存CPU序列长度
        batch_result = self.target_worker.forward_batch_generation(  # 运行目标工作器前向生成
            batch, is_verify=True  # 标记为验证
        )
        logits_output, can_run_cuda_graph = (  # 获取logits输出和CUDA图标志
            batch_result.logits_output,  # logits输出
            batch_result.can_run_cuda_graph,  # 是否可运行CUDA图
        )

        vocab_mask = None  # 初始化词汇掩码为None
        if batch.has_grammar:  # 如果有语法约束
            vocab_mask = generate_token_bitmask(  # 生成token位掩码
                batch.reqs,  # 请求列表
                spec_info,  # 验证输入
                retrieve_next_token_cpu,  # CPU上的下一个token检索
                retrieve_next_sibling_cpu,  # CPU上的兄弟节点检索
                draft_tokens_cpu,  # CPU上的草稿token
                batch.sampling_info.vocab_size,  # 词汇大小
            )
            if vocab_mask is not None:  # 如果掩码不为None
                assert spec_info.grammar is not None  # 断言语法约束存在
                vocab_mask = vocab_mask.to(spec_info.retrieve_next_token.device)  # 将掩码移到设备上
                batch.sampling_info.vocab_mask = None  # 清除采样信息的词汇掩码

        maybe_detect_nan(logits_output.next_token_logits, "frozen_kv_mtp_verify")  # 检测NaN
        maybe_detect_inf(logits_output.next_token_logits, "frozen_kv_mtp_verify")  # 检测无穷大

        spec_info.hidden_states = logits_output.hidden_states  # 保存隐藏状态
        res: FrozenKVMTPVerifyOutput = spec_info.verify(  # 运行验证
            batch,  # 调度批次
            logits_output,  # logits输出
            self.token_to_kv_pool_allocator,  # token到KV池分配器
            self.page_size,  # 页面大小
            vocab_mask,  # 词汇掩码
        )

        logits_output.next_token_logits = logits_output.next_token_logits[  # 截取接受的logits
            res.accept_indices  # 接受索引
        ]
        logits_output.hidden_states = logits_output.hidden_states[res.accept_indices]  # 截取接受的隐藏状态

        if (  # 检查是否有混合架构配置
            self.target_worker.model_runner.hybrid_gdn_config is not None
            or self.target_worker.model_runner.mamba2_config is not None
            or self.target_worker.model_runner.hybrid_lightning_config is not None
        ):
            logger.warning(  # 记录警告
                "Frozen-KV MTP does not implement mamba state updates; "
                "targets with recurrent state should not use this path."
            )

        if batch.return_logprob:  # 如果需要返回logprob
            add_output_logprobs_for_spec_v1(batch, res, logits_output)  # 添加输出logprob

        batch.forward_mode = (  # 恢复前向模式
            ForwardMode.DECODE if not batch.forward_mode.is_idle() else ForwardMode.IDLE  # 解码或空闲
        )

        del seq_lens_pre_verify  # 删除备份
        res.can_run_cuda_graph = can_run_cuda_graph  # 设置CUDA图标志
        return res  # 返回验证结果
