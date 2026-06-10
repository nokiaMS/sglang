# Copyright 2023-2024 SGLang Team
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
# 多层EAGLE工作器模块。
# 实现多层EAGLE推测解码工作器，支持EAGLE和EAGLE3算法，
# 包含草稿生成、验证和草稿扩展等核心流程。
# 多层EAGLE在每一步使用独立的MTP模型运行器。

import logging  # 导入日志模块
import time  # 导入时间模块
from typing import TYPE_CHECKING, List, Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch

from sglang.srt.layers.dp_attention import get_attention_tp_group  # 导入DP注意力TP组
from sglang.srt.layers.logits_processor import LogitsProcessorOutput  # 导入logits处理器输出
from sglang.srt.layers.moe.utils import speculative_moe_backend_context  # 导入推测MoE后端上下文
from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1  # 导入推测解码v1 logprob
from sglang.srt.managers.schedule_batch import ScheduleBatch  # 导入调度批次
from sglang.srt.managers.scheduler import GenerationBatchResult  # 导入生成批次结果
from sglang.srt.managers.tp_worker import TpModelWorker  # 导入张量并行工作器
from sglang.srt.model_executor.forward_batch_info import (  # 导入前向批次信息
    CaptureHiddenMode,  # 隐藏状态捕获模式
    ForwardBatch,  # 前向批次
    ForwardMode,  # 前向模式
)
from sglang.srt.observability.req_time_stats import set_time_batch  # 导入请求时间统计
from sglang.srt.observability.trace import get_global_tracing_enabled  # 导入全局追踪检查
from sglang.srt.server_args import ServerArgs  # 导入服务器参数
from sglang.srt.speculative.draft_utils import DraftBackendFactory  # 导入草稿后端工厂
from sglang.srt.speculative.eagle_info import (  # 导入EAGLE信息类
    EagleDraftExtendInput,  # EAGLE草稿扩展输入
    EagleDraftInput,  # EAGLE草稿输入
    EagleVerifyInput,  # EAGLE验证输入
    EagleVerifyOutput,  # EAGLE验证输出
)
from sglang.srt.speculative.eagle_utils import (  # 导入EAGLE工具函数
    apply_eagle_prefill_input_rotation,  # 应用EAGLE预填充输入旋转
    build_tree_kernel_efficient,  # 高效构建树核
    organize_draft_results,  # 组织草稿结果
)
from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (  # 导入多层EAGLE草稿扩展CUDA图运行器
    MultiLayerEagleDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # 导入推测算法枚举
from sglang.srt.speculative.spec_utils import (  # 导入推测解码工具
    draft_tp_context,  # 草稿TP上下文
    fast_topk,  # 快速topk
    generate_token_bitmask,  # 生成token位掩码
    load_token_map,  # 加载token映射
    select_top_k_tokens,  # 选择top-k token
)
from sglang.srt.utils import empty_context, get_available_gpu_memory, is_cuda, is_npu  # 导入工具函数
from sglang.srt.utils.async_probe import maybe_detect_nan  # 导入NaN检测

if TYPE_CHECKING:  # 类型检查时导入
    from sglang.srt.model_executor.model_runner import ModelRunner  # 导入模型运行器

_is_npu = is_npu()  # 检查是否为NPU设备

if is_cuda():  # 如果是CUDA设备
    from sgl_kernel import segment_packbits  # noqa: F401  # 导入segment_packbits

logger = logging.getLogger(__name__)  # 获取日志记录器


class MultiLayerEagleWorker(TpModelWorker):
    """多层EAGLE工作器，继承自TpModelWorker，实现多层推测解码。"""

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
        """初始化多层EAGLE工作器，设置草稿模型和注意力后端。"""
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
        self.draft_extend_attn_backend_list = []  # 草稿扩展注意力后端列表

        # Override the context length of the draft model to be the same as the target model.
        server_args.context_length = target_worker.model_runner.model_config.context_len  # 覆盖草稿模型上下文长度

        # Do not capture cuda graph in `super().__init__()`
        # It will be captured later.
        backup_disable_cuda_graph = server_args.disable_cuda_graph  # 备份CUDA图禁用设置
        server_args.disable_cuda_graph = True  # 临时禁用CUDA图捕获
        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (  # 共享内存池分配器
            target_worker.get_memory_pool()
        )

        # Load hot token ids
        if self.speculative_algorithm.is_eagle3():  # 如果是EAGLE3算法
            if server_args.speculative_token_map is not None:  # 如果指定了token映射
                logger.warning(  # 记录警告
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None  # EAGLE3不需要外部token映射
        elif server_args.speculative_token_map is not None:  # 如果指定了token映射（非EAGLE3）
            self.hot_token_id = load_token_map(server_args.speculative_token_map)  # 加载token映射
            server_args.json_model_override_args = (  # 设置JSON模型覆盖参数
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None  # 无token映射

        # Init draft worker
        if server_args.enable_dp_attention and self.speculative_algorithm.is_eagle3():  # 如果启用DP注意力和EAGLE3
            ctx = draft_tp_context(get_attention_tp_group())  # 使用DP注意力TP上下文
        else:
            ctx = empty_context()  # 使用空上下文
        with ctx, speculative_moe_backend_context():  # 在上下文中初始化
            super().__init__(  # 调用父类初始化
                server_args=server_args,  # 服务器参数
                gpu_id=gpu_id,  # GPU ID
                tp_rank=tp_rank,  # TP排名
                pp_rank=0,  # spec workers don't support pipeline parallelism  # 流水线并行排名为0
                dp_rank=dp_rank,  # DP排名
                moe_ep_rank=moe_ep_rank,  # MoE EP排名
                attn_cp_rank=attn_cp_rank,  # 注意力CP排名
                moe_dp_rank=moe_dp_rank,  # MoE DP排名
                nccl_port=nccl_port,  # NCCL端口
                is_draft_worker=True,  # 标记为草稿工作器
                req_to_token_pool=self.req_to_token_pool,  # 请求到token映射池
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,  # token到KV池分配器
                memory_pool_config=target_worker.model_runner.memory_pool_config,  # 内存池配置
                is_multi_layer_eagle=True,  # 标记为多层EAGLE
            )

        self.eagle_use_aux_hidden_state = False  # 是否使用辅助隐藏状态
        if self.speculative_algorithm.is_eagle3():  # 如果是EAGLE3
            eagle_config = getattr(  # 获取EAGLE配置
                self.model_runner.model_config.hf_config, "eagle_config", {}
            )
            self.eagle_use_aux_hidden_state = eagle_config.get(  # 获取辅助隐藏状态设置
                "use_aux_hidden_state", True
            )

        embed, head = self.target_worker.model_runner.model.get_embed_and_head()  # 获取嵌入层和语言模型头

        if self.speculative_algorithm.is_eagle3():  # 如果是EAGLE3
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            if (  # 如果草稿模型需要从目标模型加载lm_head
                hasattr(self.draft_model_runner.model, "load_lm_head_from_target")
                and self.draft_model_runner.model.load_lm_head_from_target
            ):
                self.draft_model_runner.model.set_embed_and_head(embed, head)  # 设置嵌入和头
            else:
                self.draft_model_runner.model.set_embed(embed)  # 仅设置嵌入

            # grab hot token ids
            if self.draft_model_runner.model.hot_token_id is not None:  # 如果草稿模型有热门token ID
                self.hot_token_id = self.draft_model_runner.model.hot_token_id.to(  # 转移到嵌入设备
                    embed.device
                )

        else:
            if self.hot_token_id is not None:  # 如果有热门token ID（非EAGLE3）
                head = head.clone()  # 克隆语言模型头
                self.hot_token_id = self.hot_token_id.to(head.device)  # 转移到头设备
                head.data = head.data[self.hot_token_id]  # 截取热门token对应的头

            # Share the embedding and lm_head
            for i in range(self.speculative_num_steps):  # 遍历每步
                self.mtp_model_runner(i).model.set_embed_and_head(embed, head)  # 设置嵌入和头

        # Init attention backend and cuda graphs
        for i in range(self.speculative_num_steps):  # 遍历每步
            self.mtp_model_runner(i).server_args.disable_cuda_graph = (  # 恢复CUDA图设置
                backup_disable_cuda_graph
            )
        self.draft_tp_context = (  # 设置草稿TP上下文
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with (
            self.draft_tp_context(self.mtp_model_runner(0).tp_group),  # 草稿TP上下文
            speculative_moe_backend_context(),  # 推测MoE后端上下文
        ):
            self.init_attention_backend()  # 初始化注意力后端
            self.init_cuda_graphs()  # 初始化CUDA图

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(  # 每topk新增页数的占位张量
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)  # 扩展长度的占位张量

    def init_attention_backend(self):
        """初始化注意力后端，为每步创建草稿扩展注意力后端。"""
        # Create multi-step attn backends and cuda graph runners
        for step in range(self.speculative_num_steps):  # 遍历每步
            draft_backend_factory = DraftBackendFactory(  # 创建草稿后端工厂
                self.server_args,  # 服务器参数
                self.mtp_model_runner(step),  # 当前步的模型运行器
                self.topk,  # topk参数
                self.speculative_num_steps,  # 推测步数
            )

            # Initialize draft extend attention backend (respects speculative_attention_mode setting)
            self.draft_extend_attn_backend_list.append(  # 添加到注意力后端列表
                draft_backend_factory.create_draft_extend_backend()
            )

    def init_cuda_graphs(self):
        """捕获CUDA图，为每步创建草稿扩展CUDA图运行器。"""
        """Capture cuda graphs."""
        self.cuda_graph_runner_for_draft_extend_list = []  # 草稿扩展CUDA图运行器列表

        if self.server_args.disable_cuda_graph:  # 如果禁用CUDA图
            return

        # Capture extend
        for step in range(self.speculative_num_steps):  # 遍历每步
            if self.draft_extend_attn_backend_list[step] and not _is_npu:  # 如果注意力后端存在且非NPU
                tic = time.perf_counter()  # 记录开始时间
                before_mem = get_available_gpu_memory(self.device, self.gpu_id)  # 获取可用内存
                logger.info(  # 记录信息
                    f"Capture draft extend cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
                )
                self.cuda_graph_runner_for_draft_extend_list.append(  # 添加CUDA图运行器
                    MultiLayerEagleDraftExtendCudaGraphRunner(self, step)
                )
                after_mem = get_available_gpu_memory(self.device, self.gpu_id)  # 获取捕获后内存
                logger.info(  # 记录信息
                    f"Capture draft extend cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
                )

    def mtp_model_runner(self, layer_id: int) -> ModelRunner:
        """获取指定层的MTP模型运行器。"""
        return self.model_runner_list[layer_id]  # 返回模型运行器

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """运行推测解码前向生成，处理扩展和解码两种模式。
        NOTE: 批次的许多状态在执行过程中会被修改，不保证最终输出批次与输入状态相同。
        """
        """Run speculative decoding forward.

        NOTE: Many states of batch is modified as you go through. It is not guaranteed that
        the final output batch have the same state as the input.

        Args:
            batch: The batch to run forward. The state of the batch is modified as it runs.
        Returns:
            A tuple of the final logit output of the target model, next tokens accepted,
            the batch id (used for overlap schedule), and number of accepted tokens.
        """
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:  # 如果是扩展模式
            (
                logits_output,  # logits输出
                next_token_ids,  # 下一个token ID
                seq_lens_cpu,  # CPU序列长度
                can_run_cuda_graph,  # 是否可运行CUDA图
            ) = self.forward_target_extend(batch)  # 运行目标扩展
            with (
                self.draft_tp_context(self.mtp_model_runner(0).tp_group),  # 草稿TP上下文
                speculative_moe_backend_context(),  # 推测MoE后端上下文
            ):
                self.forward_draft_extend(  # 运行草稿扩展
                    batch, logits_output.hidden_states, next_token_ids, seq_lens_cpu
                )
            return GenerationBatchResult(  # 返回生成结果
                logits_output=logits_output,  # logits输出
                next_token_ids=next_token_ids,  # 下一个token ID
                num_correct_drafts=0,  # 正确草稿数为0
                can_run_cuda_graph=can_run_cuda_graph,  # 是否可运行CUDA图
            )
        else:
            set_time_batch(batch.reqs, "set_spec_draft_start_time", trace_only=True)  # 设置草稿开始时间

            with (
                self.draft_tp_context(self.mtp_model_runner(0).tp_group),  # 草稿TP上下文
                speculative_moe_backend_context(),  # 推测MoE后端上下文
            ):
                verify_input = self.draft(batch)  # 运行草稿生成

            set_time_batch(batch.reqs, "set_spec_draft_end_time", trace_only=True)  # 设置草稿结束时间
            set_time_batch(batch.reqs, "set_spec_verify_start_time", trace_only=True)  # 设置验证开始时间

            # Install verify_input as `batch.spec_info` for the verify forward.
            batch.spec_info = verify_input  # 安装验证输入
            verify_output = self.verify(batch)  # 运行验证

            if get_global_tracing_enabled():  # 如果启用全局追踪
                for idx, req in enumerate(batch.reqs):  # 遍历请求
                    num_correct_drafts = verify_output.num_correct_drafts_per_req_cpu[  # 获取正确草稿数
                        idx
                    ]
                    req.time_stats.set_spec_verify_end_time(  # 设置验证结束时间
                        num_correct_drafts=num_correct_drafts
                    )

            set_time_batch(  # 设置草稿扩展开始时间
                batch.reqs, "set_spec_draft_extend_start_time", trace_only=True
            )

            with (
                self.draft_tp_context(self.mtp_model_runner(0).tp_group),  # 草稿TP上下文
                speculative_moe_backend_context(),  # 推测MoE后端上下文
            ):
                # NOTE: We should use `check_forward_draft_extend_after_decode`
                # when DP attention is enabled, but it is slow. Skip it for now.
                draft_extend_input = verify_output.draft_extend_input  # 获取草稿扩展输入
                if (
                    self.server_args.enable_dp_attention  # 启用DP注意力
                    or draft_extend_input.input_ids.shape[0] > 0  # 或有草稿扩展token
                ):
                    # decode is not finished; install draft_extend_input for
                    # the extend forward, then install the next-iter
                    # EagleDraftInput it returns.
                    batch.spec_info = draft_extend_input  # 安装草稿扩展输入
                    next_draft_input = self.forward_draft_extend_after_decode(batch)  # 运行解码后草稿扩展
                    batch.spec_info = next_draft_input  # 安装下一轮草稿输入
                else:
                    # All reqs finished and dp_attention isn't forcing extend.
                    # Install an idle EagleDraftInput so next iter's scheduler
                    # ops (merge_batch / filter_batch) see well-typed empty
                    # tensors instead of None.
                    self._draft_preprocess_idle(batch)  # 处理空闲状态

            set_time_batch(  # 设置草稿扩展结束时间
                batch.reqs, "set_spec_draft_extend_end_time", trace_only=True
            )

            return GenerationBatchResult(  # 返回生成结果
                logits_output=verify_output.logits_output,  # logits输出
                next_token_ids=verify_output.accept_tokens,  # 接受的token
                num_correct_drafts=sum(verify_output.num_correct_drafts_per_req_cpu),  # 总正确草稿数
                num_correct_drafts_per_req_cpu=verify_output.num_correct_drafts_per_req_cpu,  # 每请求正确草稿数
                can_run_cuda_graph=verify_output.can_run_cuda_graph,  # 是否可运行CUDA图
            )

    def forward_target_extend(
        self, batch: ScheduleBatch  # 调度批次
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, Optional[torch.Tensor], bool]:
        """运行目标模型扩展，获取完整隐藏状态以预填充草稿KV缓存。"""
        """Run the target extend.

        Args:
            batch: The batch to run. States could be modified.

        Returns:
            logits_output: The output of logits. It will contain the full hidden states.
            next_token_ids: Next token ids generated.
            seq_lens_cpu: CPU copy of sequence lengths for the draft prefill path.
            can_run_cuda_graph: Whether the target prefill ran with cuda graph.
        """
        # Forward with the target model and get hidden states.
        # We need the full hidden states to prefill the KV cache of the draft model.
        capture_mode = (  # 确定捕获模式
            CaptureHiddenMode.NULL  # 独立模式不捕获
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.FULL  # 其他模式捕获全部
        )
        batch.capture_hidden_mode = capture_mode  # 设置捕获模式
        batch.return_hidden_states_before_norm = True  # 返回归一化前的隐藏状态
        batch_result = self.target_worker.forward_batch_generation(batch)  # 运行目标前向
        logits_output, next_token_ids = (  # 获取logits和下一个token
            batch_result.logits_output,
            batch_result.next_token_ids,
        )
        return (  # 返回结果
            logits_output,
            next_token_ids,
            batch.seq_lens_cpu,
            batch_result.can_run_cuda_graph,
        )

    def _draft_preprocess_decode(self, batch: ScheduleBatch):
        """草稿解码预处理，委托给EAGLEWorker实现。"""
        from sglang.srt.speculative.eagle_worker import EAGLEWorker  # 导入EAGLE工作器

        # FIXME: migrate multi-layer eagle worker to eagle worker
        return EAGLEWorker._draft_preprocess_decode(self, batch)  # 委托给EAGLE工作器

    def _draft_preprocess_idle(self, batch: ScheduleBatch):
        """草稿空闲预处理，委托给EAGLEWorker实现。"""
        from sglang.srt.speculative.eagle_worker import EAGLEWorker  # 导入EAGLE工作器

        # FIXME: migrate multi-layer eagle worker to eagle worker
        return EAGLEWorker._draft_preprocess_idle(self, batch)  # 委托给EAGLE工作器

    def draft(self, batch: ScheduleBatch):
        """运行草稿生成，通过多步MTP模型生成草稿token并构建树掩码。"""
        # Parse args
        if batch.forward_mode.is_idle():  # 如果是空闲模式
            self._draft_preprocess_idle(batch)  # 处理空闲
        else:
            self._draft_preprocess_decode(batch)  # 处理解码

        spec_info = batch.spec_info  # 获取推测信息
        assert isinstance(spec_info, EagleDraftInput)  # 断言类型

        draft_capture_mode = (  # 确定草稿捕获模式
            CaptureHiddenMode.NULL  # 独立模式
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.LAST  # 其他模式捕获最后
        )
        spec_info.capture_hidden_mode = draft_capture_mode  # 设置捕获模式
        spec_info.num_tokens_per_req = self.topk  # 每请求token数
        spec_info.num_tokens_for_logprob_per_req = self.topk  # 每请求logprob token数
        batch.return_hidden_states = False  # 不返回隐藏状态

        # Get forward batch
        forward_batch = ForwardBatch.init_new(batch, self.mtp_model_runner(0))  # 初始化前向批次
        assert forward_batch.capture_hidden_mode == draft_capture_mode  # 断言捕获模式
        forward_batch.can_run_dp_cuda_graph = False  # 不运行DP CUDA图
        forward_batch.return_hidden_states_before_norm = True  # 返回归一化前隐藏状态

        # Parse args
        assert isinstance(spec_info, EagleDraftInput)  # 断言类型
        topk_p, topk_index, hidden_states = (  # 获取topk概率、索引和隐藏状态
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        maybe_detect_nan(topk_p, "draft: NaN in initial topk_p from spec_info")  # 检测NaN

        # Return values
        score_list: List[torch.Tensor] = []  # 分数列表
        token_list: List[torch.Tensor] = []  # token列表
        parents_list: List[torch.Tensor] = []  # 父节点列表

        # Forward multiple steps
        scores = None  # 初始化分数
        input_ids, hidden_states, scores, tree_info = select_top_k_tokens(  # 选择top-k token
            0, topk_p, topk_index, hidden_states, scores, self.topk
        )
        if self.speculative_num_steps == 1:  # 如果只有一步
            score_list.append(tree_info[0])  # 添加分数
            token_list.append(tree_info[1])  # 添加token
            parents_list.append(tree_info[2])  # 添加父节点
        else:
            for i in range(self.speculative_num_steps):  # 遍历多步
                score_list.append(tree_info[0][:, :, i].unsqueeze(-1))  # 添加每步分数
                token_index = tree_info[1][:, i].unsqueeze(-1)  # 获取每步token索引
                token_list.append(token_index)  # 添加token
                if i == 0:  # 第一步
                    parents_list.append(tree_info[2])  # 添加父节点
                else:
                    parents_list.append(  # 后续步使用步数作为父节点
                        torch.full(
                            (tree_info[2].size(0), 1),  # 批次大小
                            i,  # 步数
                            dtype=torch.long,
                            device=self.device,
                        )
                    )

        parent_list, top_scores_index, draft_tokens = organize_draft_results(  # 组织草稿结果
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )

        if batch.forward_mode.is_idle():  # 如果是空闲模式
            return EagleVerifyInput.create_idle_input(  # 创建空闲验证输入
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        (
            tree_mask,  # 树掩码
            position,  # 位置
            retrieve_index,  # 检索索引
            retrieve_next_token,  # 检索下一个token
            retrieve_next_sibling,  # 检索下一个兄弟节点
            draft_tokens,  # 草稿token
        ) = build_tree_kernel_efficient(  # 构建树核
            spec_info.bonus_tokens,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
        )

        target_capture_mode = (  # 确定目标捕获模式
            CaptureHiddenMode.NULL  # 独立模式
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.FULL  # 其他模式
        )
        return EagleVerifyInput(  # 返回验证输入
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.server_args.speculative_num_draft_tokens,
            capture_hidden_mode=target_capture_mode,
            seq_lens_sum=forward_batch.seq_lens_sum,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
        )

    def clear_cache_pool(self):
        """清空缓存池，由于共享目标工作器的分配器，无需操作。"""
        # allocator and kv cache pool are shared with target worker
        pass

    def verify(self, batch: ScheduleBatch):
        """运行验证步骤，用目标模型验证草稿token并确定接受的token。"""
        spec_info: EagleVerifyInput = batch.spec_info  # 获取验证输入
        spec_info.prepare_for_verify(batch, self.page_size)  # 准备验证
        batch.return_hidden_states = False  # 不返回隐藏状态
        batch.forward_mode = (  # 设置前向模式
            ForwardMode.TARGET_VERIFY  # 目标验证
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE  # 空闲
        )

        if batch.has_grammar:  # 如果有语法约束
            retrieve_next_token_cpu = spec_info.retrieve_next_token.cpu()  # 获取CPU端下一个token
            retrieve_next_sibling_cpu = spec_info.retrieve_next_sibling.cpu()  # 获取CPU端兄弟节点
            draft_tokens_cpu = spec_info.draft_token.view(  # 获取CPU端草稿token
                spec_info.retrieve_next_token.shape
            ).cpu()

        # Forward
        batch.seq_lens_cpu_cache = spec_info.seq_lens_cpu  # 缓存CPU序列长度
        batch.return_hidden_states_before_norm = True  # 返回归一化前隐藏状态
        batch_result = self.target_worker.forward_batch_generation(  # 运行目标前向
            batch, is_verify=True
        )
        logits_output, can_run_cuda_graph = (  # 获取结果
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        vocab_mask = None  # 初始化词汇掩码
        if batch.has_grammar:  # 如果有语法约束
            # Generate the logit mask for structured output.
            # Overlap the CPU operations for bitmask generation with the forward pass.
            vocab_mask = generate_token_bitmask(  # 生成token位掩码
                batch.reqs,
                spec_info,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )

            if vocab_mask is not None:  # 如果掩码存在
                assert spec_info.grammar is not None  # 断言语法约束存在
                vocab_mask = vocab_mask.to(spec_info.retrieve_next_token.device)  # 转移到设备
                # NOTE (sk): otherwise, this vocab mask will be the one from the previous extend stage
                # and will be applied to produce wrong results
                batch.sampling_info.vocab_mask = None  # 清除旧掩码

        maybe_detect_nan(logits_output.next_token_logits, "verify: target model logits")  # 检测NaN

        spec_info.hidden_states = logits_output.hidden_states  # 保存隐藏状态
        res: EagleVerifyOutput = spec_info.verify(  # 运行验证
            batch,
            logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
            vocab_mask,
        )

        # Post process based on verified outputs.
        # Pick indices that we care (accepted)
        logits_output.next_token_logits = logits_output.next_token_logits[  # 截取接受位置的logits
            res.accept_indices
        ]
        logits_output.hidden_states = logits_output.hidden_states[res.accept_indices]  # 截取接受位置的隐藏状态

        if self.target_worker.model_runner.hybrid_gdn_config is not None:  # 如果有混合GDN配置
            num_correct_drafts = torch.tensor(  # 创建正确草稿数张量
                res.num_correct_drafts_per_req_cpu,
                device=logits_output.hidden_states.device,
                dtype=torch.int64,
            )

            # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
            # res.accept_indices.shape[0] > 0 skips DP attn idle batch
            if spec_info.topk > 1 and res.accept_indices.shape[0] > 0:  # topk>1且有接受token
                # accept_indices=[0,2,3,4,5,7,9,10,11], num_accept_tokens=[4, 3, 2], cumulative_num_accept_tokens=[4, 7, 9]
                # first_token_indices_per_req=prepend(0, accept_indices[cumulative_num_accept_tokens[:-1]]) = [0, 5, 10]
                # last_token_indices_per_req=accept_indices[cumulative_num_accept_tokens - 1] = [4, 9, 11] (last token ID of each req)
                # last_correct_step_indices = [4,4,1]; those are the per-req spec-decoding step offsets that contain the correct mamba caches
                # equivalent: last_correct_step_indices = last_token_indices_per_req - first_token_indices_per_req;
                # `accepted_indices_offset` equals `first_token_indices_per_req` because the first accepted slot of each req is its "current token" at logical position i * draft_token_num.
                cumulative_num_accept_tokens = torch.cumsum(  # 计算累积接受token数
                    num_correct_drafts + 1, dim=0
                )
                accepted_indices_offset = torch.arange(  # 计算每个请求的起始索引偏移
                    0,
                    len(batch.seq_lens) * self.speculative_num_draft_tokens,
                    step=self.speculative_num_draft_tokens,
                    dtype=num_correct_drafts.dtype,
                    device=num_correct_drafts.device,
                )
                last_correct_step_indices = (  # 计算最后正确步骤索引
                    res.accept_indices[cumulative_num_accept_tokens - 1]
                    - accepted_indices_offset
                )
            else:
                last_correct_step_indices = num_correct_drafts  # 直接使用正确草稿数
            self.target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify(  # 更新mamba状态
                last_correct_step_indices=last_correct_step_indices,
                mamba_track_indices=None,
                mamba_steps_to_track=None,
                model=self.target_worker.model_runner.model,
            )

        if batch.return_logprob:  # 如果需要返回logprob
            add_output_logprobs_for_spec_v1(batch, res, logits_output)  # 添加输出logprob

        # Prepare the batch for the next draft forwards.
        batch.forward_mode = (  # 恢复前向模式
            ForwardMode.DECODE if not batch.forward_mode.is_idle() else ForwardMode.IDLE
        )

        res.can_run_cuda_graph = can_run_cuda_graph  # 设置CUDA图标志
        return res  # 返回验证结果

    def forward_draft_extend(
        self,
        batch: ScheduleBatch,  # 调度批次
        hidden_states: torch.Tensor,  # 隐藏状态
        next_token_ids: torch.Tensor,  # 下一个token ID
        seq_lens_cpu: Optional[torch.Tensor],  # CPU序列长度
    ):
        """运行草稿模型扩展，预填充草稿模型的KV缓存。"""
        """Run draft model extend. This API modifies the states of the batch.

        Args:
            batch: The batch to run.
            hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        batch.spec_info = EagleDraftInput(  # 创建草稿输入
            hidden_states=hidden_states,
            bonus_tokens=next_token_ids,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )
        batch.return_hidden_states = False  # 不返回隐藏状态
        apply_eagle_prefill_input_rotation(batch, next_token_ids)  # 应用EAGLE预填充输入旋转
        capture_mode = (  # 确定捕获模式
            CaptureHiddenMode.NULL
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.LAST
        )
        batch.spec_info.capture_hidden_mode = capture_mode  # 设置捕获模式
        batch.seq_lens_cpu_cache = seq_lens_cpu  # 缓存CPU序列长度
        forward_batch = ForwardBatch.init_new(batch, self.mtp_model_runner(0))  # 初始化前向批次
        forward_batch.return_logprob = False  # 不返回logprob
        forward_batch.return_hidden_states_before_norm = True  # 返回归一化前隐藏状态
        topk_p_list = []  # topk概率列表
        topk_index_list = []  # topk索引列表
        for step in range(self.speculative_num_steps):  # 遍历每步
            logits_output = (  # 运行模型前向
                self.mtp_model_runner(step).forward(forward_batch).logits_output
            )
            maybe_detect_nan(  # 检测NaN
                logits_output.next_token_logits,
                f"draft_extend_for_prefill step {step}",
            )
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)  # 计算softmax
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)  # 快速topk
            topk_p_list.append(topk_p)  # 添加概率
            topk_index_list.append(topk_index)  # 添加索引
            pt = 0  # 指针归零
            if forward_batch.extend_seq_lens is not None:  # 如果有扩展序列长度
                for i, extend_len in enumerate(forward_batch.extend_seq_lens):  # 遍历扩展长度
                    input_ids = forward_batch.input_ids[pt : pt + extend_len]  # 获取输入ID
                    forward_batch.input_ids[pt : pt + extend_len] = torch.cat(  # 旋转输入ID
                        (input_ids[1:], topk_index[i].reshape(1))
                    )
                    pt += extend_len  # 移动指针

        assert isinstance(forward_batch.spec_info, EagleDraftInput)  # 断言类型
        assert forward_batch.spec_info is batch.spec_info  # 断言一致性
        forward_batch.spec_info.topk_p = torch.cat(topk_p_list, dim=1)  # 合并topk概率
        forward_batch.spec_info.topk_index = torch.cat(topk_index_list, dim=1)  # 合并topk索引

    def forward_draft_extend_after_decode(
        self, batch: ScheduleBatch  # 调度批次
    ) -> EagleDraftInput:
        """解码后运行草稿扩展，为下一轮迭代准备草稿输入。"""
        draft_extend_input: EagleDraftExtendInput = batch.spec_info  # 获取草稿扩展输入

        # Backup fields that will be modified in-place
        seq_lens_backup = batch.seq_lens.clone()  # 备份序列长度
        seq_lens_cpu_backup = batch.seq_lens_cpu.clone()  # 备份CPU序列长度
        req_pool_indices_backup = batch.req_pool_indices  # 备份请求池索引
        return_logprob_backup = batch.return_logprob  # 备份logprob标志

        input_is_idle = batch.forward_mode.is_idle()  # 检查是否空闲

        draft_extend_capture_mode = (  # 确定草稿扩展捕获模式
            CaptureHiddenMode.NULL
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.LAST
        )
        if draft_extend_input.input_ids.shape[0] == 0:  # 如果没有输入token
            # Single source for hidden_size via hidden_size_for(self) (incl.
            # EAGLE-3 aux widening). Two stub origins from verify(): fully-idle
            # batch and active batch with all reqs finished.
            batch = batch.copy()  # 复制批次
            batch.prepare_for_idle()  # 准备空闲
            draft_extend_input = EagleDraftExtendInput.create_idle_input(  # 创建空闲输入
                device=self.device,
                hidden_size=EagleDraftExtendInput.hidden_size_for(self),
                dtype=EagleDraftExtendInput.dtype_for(self),
                capture_hidden_mode=draft_extend_capture_mode,
            )
            batch.spec_info = draft_extend_input  # 安装空闲输入

        # Phase 1: prepare extend (kernel writes draft_extend_input.{positions, bonus_tokens})
        draft_extend_input.num_tokens_per_req = self.speculative_num_steps + 1  # 设置每请求token数
        draft_extend_input.num_tokens_for_logprob_per_req = 1  # 设置每请求logprob token数
        draft_extend_input.prepare_extend_after_decode(  # 准备解码后扩展
            batch,
            speculative_num_steps=self.speculative_num_steps,
        )
        batch.forward_mode = (  # 设置前向模式
            ForwardMode.DRAFT_EXTEND
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )

        batch.return_hidden_states = False  # 不返回隐藏状态
        draft_extend_input.capture_hidden_mode = draft_extend_capture_mode  # 设置捕获模式
        forward_batch = ForwardBatch.init_new(batch, self.mtp_model_runner(0))  # 初始化前向批次
        assert forward_batch.capture_hidden_mode == draft_extend_capture_mode  # 断言捕获模式
        forward_batch.return_hidden_states_before_norm = True  # 返回归一化前隐藏状态
        if forward_batch.seq_lens_cpu is not None:  # 如果CPU序列长度存在
            forward_batch.seq_lens_sum = forward_batch.seq_lens_cpu.sum().item()  # 计算总和
        else:
            forward_batch.seq_lens_sum = batch.seq_lens.sum().item()  # 使用GPU版本
        topk_p_list = []  # topk概率列表
        topk_index_list = []  # topk索引列表
        # Run
        for step in range(self.speculative_num_steps):  # 遍历每步
            can_cuda_graph = len(  # 检查是否可用CUDA图
                self.cuda_graph_runner_for_draft_extend_list
            ) and self.cuda_graph_runner_for_draft_extend_list[step].can_run(
                forward_batch
            )
            if can_cuda_graph:  # 如果可用CUDA图
                logits_output = self.cuda_graph_runner_for_draft_extend_list[  # 使用CUDA图运行
                    step
                ].replay(forward_batch)
            else:
                forward_batch.can_run_dp_cuda_graph = False  # 标记不可用DP CUDA图
                if not forward_batch.forward_mode.is_idle():  # 如果非空闲
                    self.mtp_model_runner(step).attn_backend.init_forward_metadata(  # 初始化注意力元数据
                        forward_batch
                    )
                logits_output = (  # 运行模型前向
                    self.mtp_model_runner(step)
                    .forward(forward_batch, skip_attn_backend_init=True)
                    .logits_output
                )

            maybe_detect_nan(  # 检测NaN
                logits_output.next_token_logits,
                f"draft_extend_after_decode step {step} (cuda_graph={can_cuda_graph})",
            )
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)  # 计算softmax
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)  # 快速topk
            topk_p_list.append(topk_p)  # 添加概率
            topk_index_list.append(topk_index)  # 添加索引
            pt = 0  # 指针归零
            if forward_batch.extend_seq_lens is not None:  # 如果有扩展序列长度
                for i, extend_len in enumerate(forward_batch.extend_seq_lens):  # 遍历
                    input_ids = forward_batch.input_ids[pt : pt + extend_len]  # 获取输入ID
                    forward_batch.input_ids[pt : pt + extend_len] = torch.cat(  # 旋转输入ID
                        (input_ids[1:], topk_index[i].reshape(1))
                    )
                    pt += extend_len  # 移动指针

        # Phase 3: assemble next-iter EagleDraftInput from extend output
        next_decode_capture_mode = (  # 确定下一轮捕获模式
            CaptureHiddenMode.NULL
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.LAST
        )
        next_draft_input = EagleDraftInput(  # 创建下一轮草稿输入
            bonus_tokens=draft_extend_input.bonus_tokens,
            hidden_states=logits_output.hidden_states,
            topk_p=torch.cat(topk_p_list, dim=1),
            topk_index=torch.cat(topk_index_list, dim=1),
            capture_hidden_mode=next_decode_capture_mode,
        )

        # Restore batch fields. `seq_lens` etc. were modified by
        # `prepare_extend_after_decode`. Caller installs `next_draft_input` as
        # `batch.spec_info`.
        batch.forward_mode = (  # 恢复前向模式
            ForwardMode.DECODE if not input_is_idle else ForwardMode.IDLE
        )
        batch.seq_lens = seq_lens_backup  # 恢复序列长度
        batch.seq_lens_cpu = seq_lens_cpu_backup  # 恢复CPU序列长度
        batch.req_pool_indices = req_pool_indices_backup  # 恢复请求池索引
        batch.return_logprob = return_logprob_backup  # 恢复logprob标志
        return next_draft_input  # 返回下一轮草稿输入
