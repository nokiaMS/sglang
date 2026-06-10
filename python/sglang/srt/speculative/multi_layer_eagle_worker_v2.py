# 多层EAGLE工作器V2模块，支持重叠调度。
# 实现了MultiLayerEagleDraftWorker（草稿工作器）和MultiLayerEagleWorkerV2（主工作器），
# 支持链式MTP隐藏状态传播、CUDA图捕获和重放、以及验证-草稿扩展重叠执行。
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

import contextlib  # 导入上下文管理工具
import logging  # 导入日志模块
from typing import TYPE_CHECKING, List, Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch

from sglang.srt.environ import envs  # 导入环境变量
from sglang.srt.layers.moe.utils import speculative_moe_backend_context  # 导入推测MoE后端上下文
from sglang.srt.layers.utils.logprob import compute_spec_v2_logprobs  # 导入推测v2 logprob计算
from sglang.srt.managers.io_struct import (  # 导入IO结构
    UpdateWeightFromDiskReqInput,  # 从磁盘更新权重请求
    UpdateWeightsFromIPCReqInput,  # 从IPC更新权重请求
)
from sglang.srt.managers.schedule_batch import ScheduleBatch  # 导入调度批次
from sglang.srt.managers.scheduler import GenerationBatchResult  # 导入生成批次结果
from sglang.srt.managers.tp_worker import TpModelWorker  # 导入张量并行工作器
from sglang.srt.model_executor.forward_batch_info import (  # 导入前向批次信息
    CaptureHiddenMode,  # 隐藏状态捕获模式
    ForwardBatch,  # 前向批次
)
from sglang.srt.server_args import ServerArgs  # 导入服务器参数
from sglang.srt.speculative.base_spec_worker import BaseDraftWorker, BaseSpecWorker  # 导入基础推测工作器
from sglang.srt.speculative.draft_utils import DraftBackendFactory  # 导入草稿后端工厂
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput  # 导入EAGLE信息类
from sglang.srt.speculative.eagle_info_v2 import fill_bonus_tokens  # 导入bonus token填充函数
from sglang.srt.speculative.eagle_utils import TreeMaskMode, build_tree_kernel_efficient  # 导入EAGLE工具
from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (  # 导入多层CUDA图运行器
    MultiLayerEagleMultiStepDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.multi_layer_eagle_utils import (  # 导入多层EAGLE工具
    assign_hidden_states_pool_triton,  # 隐藏状态池分配
    rotate_input_ids_triton,  # 输入ID旋转
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # 导入推测算法
from sglang.srt.speculative.spec_utils import (  # 导入推测工具
    draft_tp_context,  # 草稿TP上下文
    record_stream_each,  # 记录流
    record_stream_for_v2_verify,  # 为v2验证记录流
    select_top_k_tokens,  # 选择top-k token
)
from sglang.srt.utils.async_probe import (  # 导入异步探测
    maybe_detect_inf,  # 检测无穷大
    maybe_detect_nan,  # 检测NaN
    maybe_detect_oob,  # 检测越界
)
from sglang.srt.utils.common import empty_context, fast_topk  # 导入通用工具

if TYPE_CHECKING:  # 类型检查时导入
    from sglang.srt.model_executor.model_runner import ModelRunner, ModelRunnerOutput


logger = logging.getLogger(__name__)  # 获取日志记录器


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


class MultiLayerEagleDraftWorker(BaseDraftWorker):
    """多层EAGLE草稿工作器，管理草稿模型的多步运行。"""

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
        """初始化多层EAGLE草稿工作器，创建草稿模型运行器和注意力后端。"""
        # copy args
        self.server_args = server_args  # 保存服务器参数
        self.gpu_id = gpu_id  # GPU ID
        self.tp_rank = tp_rank  # TP排名
        self.dp_rank = dp_rank  # DP排名
        self.moe_ep_rank = moe_ep_rank  # MoE EP排名
        self.nccl_port = nccl_port  # NCCL端口
        self.target_worker = target_worker  # 目标工作器
        self.draft_extend_attn_backend_list = []  # 草稿扩展注意力后端列表
        self.model_config = target_worker.model_config  # 模型配置

        # Args for easy access
        self.device = server_args.device  # 设备
        self.topk = server_args.speculative_eagle_topk  # topk参数
        self.speculative_num_steps = server_args.speculative_num_steps  # 推测步数
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens  # 推测草稿token数
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(  # 解析推测算法
            server_args.speculative_algorithm
        )

        # Set constant
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
        with empty_context(), speculative_moe_backend_context():  # 在上下文中初始化
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
                is_multi_layer_eagle=True,  # 标记为多层EAGLE
            )

        # Alias for better readability
        self.draft_runner_list: List[ModelRunner] = self.draft_worker.model_runner_list  # 草稿运行器列表
        # Match `EagleDraftWorker.draft_runner` so `_draft_runner_of(self)` works
        # for the EagleDraftInput shape classmethods.
        self.draft_runner: ModelRunner = self.draft_runner_list[0]  # 第一个运行器的别名

        # Chain-style MTP: each step propagates its own output hidden states to the
        # next step.  Non-chain: each step uses the target model's hidden states.
        draft_arch = self.draft_worker.model_config.hf_config.architectures[0]  # 获取草稿架构名
        self.chain_mtp_hidden_states = draft_arch in ["Step3p5MTP"]  # 是否为链式MTP

        self.init_lm_head()  # 初始化语言模型头

        # KV cache reversion buffer; sized to mirror req_to_token (indexed by
        # req_pool_idx).
        self.req_to_hidden_states_pool = torch.empty(  # 创建请求到隐藏状态池
            (
                self.req_to_token_pool.req_to_token.shape[0],  # 请求数
                self.speculative_num_steps - 1,  # 步数-1
                self.model_config.hidden_size,  # 隐藏维度
            ),
            dtype=self.model_config.dtype,  # 数据类型
            device=self.device,  # 设备
        )

        # Init attention backend and cuda graphs
        for i in range(self.speculative_num_steps):  # 遍历每步
            self.draft_runner_list[i].server_args.disable_cuda_graph = (  # 恢复CUDA图设置
                backup_disable_cuda_graph
            )
        self.draft_tp_context = (  # 设置草稿TP上下文
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with (
            self.draft_tp_context(self.draft_runner_list[0].tp_group),  # 草稿TP上下文
            speculative_moe_backend_context(),  # 推测MoE后端上下文
        ):
            self.init_attention_backend()  # 初始化注意力后端
            self.init_cuda_graphs()  # 初始化CUDA图

        self.tree_mask_mode = TreeMaskMode.FULL_MASK  # 树掩码模式

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)  # 获取计划流

    def mtp_model_runner(self, step: int):
        """获取指定步骤的MTP模型运行器。"""
        return self.draft_runner_list[step]  # 返回对应步骤的运行器

    def init_lm_head(self):
        """初始化语言模型头，共享目标模型的嵌入层和语言模型头。"""
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()  # 获取嵌入和头
        # Share the embedding and lm_head
        for i in range(self.speculative_num_steps):  # 遍历每步
            self.draft_runner_list[i].model.set_embed_and_head(embed, head)  # 设置嵌入和头

    def init_attention_backend(self):
        """初始化注意力后端，为每步创建草稿扩展后端。"""
        # Create attn backends
        self.draft_extend_attn_backend_list = []  # 初始化列表
        for step in range(self.speculative_num_steps):  # 遍历每步
            draft_backend_factory = DraftBackendFactory(  # 创建后端工厂
                self.server_args,
                self.draft_runner_list[step],
                self.topk,
                self.speculative_num_steps,
            )
            self.draft_extend_attn_backend_list.append(  # 添加草稿扩展后端
                draft_backend_factory.create_draft_extend_backend()
            )
            self.draft_runner_list[step].attn_backend = (  # 设置模型运行器的注意力后端
                self.draft_extend_attn_backend_list[-1]
            )

    def init_cuda_graphs(self):
        """捕获CUDA图，创建多步草稿扩展CUDA图运行器。"""
        """Capture cuda graphs."""
        self.cuda_graph_runner = None  # 草稿CUDA图运行器
        self.cuda_graph_runner_for_draft_extend = None  # 草稿扩展CUDA图运行器

        if self.server_args.disable_cuda_graph:  # 如果禁用CUDA图
            return

        self.cuda_graph_runner_for_draft_extend = (  # 创建多步草稿扩展CUDA图运行器
            MultiLayerEagleMultiStepDraftExtendCudaGraphRunner(self)
        )

    def reset_cuda_graph_buffers(self, forward_batch, batch_result):
        """重置CUDA图缓冲区，根据验证结果更新正确草稿数。"""
        if self.cuda_graph_runner_for_draft_extend:  # 如果CUDA图运行器存在
            self.cuda_graph_runner_for_draft_extend.reset_buffers(  # 重置缓冲区
                forward_batch, batch_result
            )

    def draft(self, batch: ScheduleBatch):
        """运行草稿生成，通过多步推测生成草稿token并构建验证输入。"""
        draft_input: EagleDraftInput = batch.spec_info  # 获取草稿输入
        forward_batch, can_cuda_graph = draft_input.prepare_for_v2_draft(  # 准备v2草稿
            self.req_to_token_pool,
            batch,
            self.cuda_graph_runner,
            self.draft_runner_list[0],
            self.topk,
            self.speculative_num_steps,
        )

        # Run draft
        parent_list, top_scores_index, draft_tokens = self.draft_forward(forward_batch)  # 运行草稿前向

        if batch.forward_mode.is_idle():  # 如果是空闲模式
            return EagleVerifyInput.create_idle_input(  # 创建空闲验证输入
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        # Build tree mask
        # Directly write to cuda graph buffers for verify attn
        tree_mask_buf, position_buf = (  # 获取验证缓冲区
            self.target_worker.model_runner.attn_backend.get_verify_buffers_to_fill_after_draft()
        )
        (
            tree_mask,  # 树掩码
            position,  # 位置
            retrieve_index,  # 检索索引
            retrieve_next_token,  # 检索下一个token
            retrieve_next_sibling,  # 检索下一个兄弟节点
            draft_tokens,  # 草稿token
        ) = build_tree_kernel_efficient(  # 构建树核
            draft_input.bonus_tokens,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            self.tree_mask_mode,
            tree_mask_buf,
            position_buf,
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
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=None,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        """运行草稿前向，通过topk选择和树构建生成草稿token。"""
        # Parse args
        spec_info: EagleDraftInput = forward_batch.spec_info  # 获取草稿输入
        topk_p, topk_index, hidden_states = (  # 获取topk概率、索引和隐藏状态
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")  # 检测NaN

        # Return values
        score_list: List[torch.Tensor] = []  # 分数列表
        token_list: List[torch.Tensor] = []  # token列表
        parents_list: List[torch.Tensor] = []  # 父节点列表

        # Forward multiple steps
        scores = None  # 初始化分数
        _, hidden_states, scores, tree_info = select_top_k_tokens(  # 选择top-k token
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
                            (tree_info[2].size(0), 1),
                            i,
                            dtype=torch.long,
                            device="cuda",
                        )
                    )

        # Organize the results
        score_list = torch.cat(score_list, dim=1).flatten(  # 合并并展平分数列表
            1
        )  # b, n, topk; n= 1 + (num_steps-1) * self.topk
        ss_token_list = torch.cat(  # 合并token列表
            token_list, dim=1
        )  # b, (self.topk + (num_steps-1) * self.topk)
        top_scores = torch.topk(  # 取top-k分数
            score_list, self.speculative_num_draft_tokens - 1, dim=-1
        )
        top_scores_index = top_scores.indices  # top-k索引
        top_scores_index = torch.sort(top_scores_index).values  # 排序索引
        maybe_detect_oob(  # 检测越界
            top_scores_index,
            0,
            ss_token_list.shape[1],
            "draft_forward: top_scores_index OOB for gather on ss_token_list",
        )
        draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)  # 收集草稿token

        if len(parents_list) > 1:  # 如果有多步
            parent_list = torch.cat(parents_list[:-1], dim=1)  # 合并除最后一步外的父节点
        else:
            batch_size = parents_list[0].shape[0]  # 批次大小
            parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)  # 空父节点列表

        return parent_list, top_scores_index, draft_tokens  # 返回父节点、top分数索引和草稿token

    def draft_extend(self):
        """草稿扩展的占位方法（V2中在别处实现）。"""
        pass  # 占位

    def _draft_extend_for_prefill(
        self,
        batch: ScheduleBatch,  # 调度批次
        target_hidden_states: torch.Tensor,  # 目标隐藏状态
        next_token_ids: torch.Tensor,  # 下一个token ID
    ):
        """预填充阶段运行草稿扩展，正确填充KV缓存。"""
        """
        Run draft model extend to correctly fill the KV cache.

        Args:
            batch: The batch to run.
            target_hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        # Construct spec_info
        next_draft_input = EagleDraftInput(  # 创建草稿输入
            hidden_states=target_hidden_states,
            bonus_tokens=next_token_ids,
            # draft mode is same with decode mode, only 1 token per req
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )

        batch.spec_info = next_draft_input  # 安装草稿输入

        # Chain-style MTP needs FULL to get all-token hidden states;
        # non-chain only needs LAST (the target model's hidden states).
        # STANDALONE skips hidden states end-to-end.
        if self.speculative_algorithm.is_standalone():  # 独立模式
            draft_capture_hidden_mode = CaptureHiddenMode.NULL
        elif self.chain_mtp_hidden_states:  # 链式MTP
            draft_capture_hidden_mode = CaptureHiddenMode.FULL
        else:  # 其他
            draft_capture_hidden_mode = CaptureHiddenMode.LAST

        # Run forward
        batch.capture_hidden_mode = draft_capture_hidden_mode  # 设置捕获模式
        batch.return_hidden_states_before_norm = True  # 返回归一化前隐藏状态
        forward_batch = ForwardBatch.init_new(batch, self.draft_runner_list[0])  # 初始化前向批次

        # Construct input_ids
        # TODO: same chunked-prefill chain divergence as PR #26329.
        if not batch.forward_mode.is_idle():  # 如果非空闲
            rotate_input_ids_triton(  # 旋转输入ID
                forward_batch.input_ids,
                forward_batch.extend_start_loc,
                forward_batch.extend_seq_lens,
                next_token_ids,
            )

        topk_p_list = []  # topk概率列表
        topk_index_list = []  # topk索引列表
        for step in range(self.speculative_num_steps):  # 遍历每步
            output: ModelRunnerOutput = self.draft_runner_list[step].forward(  # 运行草稿前向
                forward_batch
            )
            maybe_detect_nan(  # 检测NaN
                output.logits_output.next_token_logits,
                f"draft_extend_for_prefill step {step}",
            )
            maybe_detect_inf(  # 检测无穷大
                output.logits_output.next_token_logits,
                f"draft_extend_for_prefill step {step}",
            )
            probs = torch.softmax(output.logits_output.next_token_logits, dim=-1)  # 计算softmax
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)  # 快速topk
            topk_p_list.append(topk_p)  # 添加概率
            topk_index_list.append(topk_index)  # 添加索引
            # Chain-style: use this step's output hidden_states as next step's input
            if (  # 链式MTP：使用当前步输出作为下一步输入
                self.chain_mtp_hidden_states
                and step < self.speculative_num_steps - 1
                and output.logits_output.hidden_states is not None
            ):
                forward_batch.spec_info.hidden_states = (  # 更新隐藏状态
                    output.logits_output.hidden_states
                )
            if forward_batch.extend_seq_lens is not None:  # 如果有扩展序列长度
                rotate_input_ids_triton(  # 旋转输入ID
                    forward_batch.input_ids,
                    forward_batch.extend_start_loc,
                    forward_batch.extend_seq_lens,
                    topk_index,
                )
        next_draft_input.topk_p = torch.cat(topk_p_list, dim=1)  # 合并topk概率
        next_draft_input.topk_index = torch.cat(topk_index_list, dim=1)  # 合并topk索引

        # Update req_to_hidden_states_pool for KV Cache reversion
        if forward_batch.extend_seq_lens is not None:  # 如果有扩展序列长度
            assign_hidden_states_pool_triton(  # 更新隐藏状态池
                target_hidden_states,
                forward_batch.req_pool_indices,
                self.req_to_hidden_states_pool,
                self.speculative_num_steps - 1,
                forward_batch.batch_size,
                forward_batch.extend_seq_lens,
                forward_batch.extend_start_loc,
            )
        return next_draft_input  # 返回草稿输入

    def _draft_extend_for_decode(
        self, batch: ScheduleBatch, batch_result: GenerationBatchResult  # 调度批次, # 生成批次结果
    ):
        """解码阶段运行草稿扩展，填充KV缓存并为下一轮准备草稿输入。"""
        # Batch 2: Draft extend
        draft_input = EagleDraftInput(  # 创建草稿输入
            hidden_states=batch_result.logits_output.hidden_states,
            num_tokens_per_req=self.speculative_num_steps + 1,
            num_tokens_for_logprob_per_req=1,
        )

        # Prepare for draft extend in a separate stream
        # Notice that here we use batch_result.next_token_ids as the input ids
        with self.plan_stream_ctx:  # 在计划流中准备
            forward_batch = draft_input.prepare_for_extend_to_fill_draft_kvcache(  # 准备扩展
                batch,
                batch_result.next_token_ids,
                self.speculative_num_draft_tokens,
                self.draft_runner_list[0],
                self.cuda_graph_runner_for_draft_extend,
            )
            forward_batch.return_hidden_states_before_norm = True  # 返回归一化前隐藏状态

        if self.plan_stream:  # 如果有计划流
            torch.get_device_module(self.device).current_stream().wait_stream(  # 等待计划流完成
                self.plan_stream
            )
        # Run draft extend batch in the main compute stream
        can_cuda_graph = (  # 检查是否可用CUDA图
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run(forward_batch)
        )
        ret_topk_p_list = []  # 返回的topk概率列表
        ret_topk_index_list = []  # 返回的topk索引列表
        next_token_ids_backup = batch_result.next_token_ids.clone()  # 备份下一个token ID

        if can_cuda_graph:  # 如果可用CUDA图
            self.reset_cuda_graph_buffers(forward_batch, batch_result)  # 重置缓冲区
        else:
            logger.warning_once(  # 记录警告
                f"can't use cuda graph for draft extend! may have correctness issue!"
            )
            select_index = (  # 计算选择索引
                torch.arange(len(batch.seq_lens), device=self.device)
                * self.speculative_num_draft_tokens
                + batch_result.accept_lens
                - 1
            )

        for step in range(self.speculative_num_steps):  # 遍历每步
            if can_cuda_graph:  # 如果可用CUDA图
                draft_logits_output = (  # 使用CUDA图运行
                    self.cuda_graph_runner_for_draft_extend.get_runner(step).replay(
                        forward_batch, init_state=(step == 0)
                    )
                )
                ret_topk_p, ret_topk_index = (  # 获取topk结果
                    draft_logits_output.topk_p,
                    draft_logits_output.topk_index,
                )
            else:
                draft_logits_output = self.draft_runner_list[step].forward(  # 运行草稿前向
                    forward_batch, skip_attn_backend_init=True
                )
                probs = torch.softmax(  # 计算softmax
                    draft_logits_output.logits_output.next_token_logits[select_index],
                    dim=-1,
                )
                ret_topk_p, ret_topk_index = fast_topk(probs, self.topk, dim=-1)  # 快速topk
                # Chain-style: use this step's output hidden_states as next step's input
                if (  # 链式MTP：使用当前步输出作为下一步输入
                    self.chain_mtp_hidden_states
                    and step < self.speculative_num_steps - 1
                    and draft_logits_output.logits_output.hidden_states is not None
                ):
                    forward_batch.spec_info.hidden_states = (  # 更新隐藏状态
                        draft_logits_output.logits_output.hidden_states
                    )
                if forward_batch.extend_seq_lens is not None:  # 如果有扩展序列长度
                    rotate_input_ids_triton(  # 旋转输入ID
                        forward_batch.input_ids,
                        forward_batch.extend_start_loc,
                        forward_batch.extend_seq_lens,
                        ret_topk_index,
                        select_index,
                    )
            ret_topk_p_list.append(ret_topk_p)  # 添加概率
            ret_topk_index_list.append(ret_topk_index)  # 添加索引

        # Update req_to_hidden_states_pool for KV Cache reversion
        if (  # 更新隐藏状态池
            forward_batch.extend_seq_lens is not None
            and self.cuda_graph_runner_for_draft_extend is not None
        ):
            if can_cuda_graph:  # 如果使用CUDA图
                last_runner = self.cuda_graph_runner_for_draft_extend.get_last_runner()  # 获取最后运行器
                hidden_states = last_runner.buffers.hidden_states  # 隐藏状态
                req_pool_indices = last_runner.buffers.req_pool_indices  # 请求池索引
                extend_seq_lens = last_runner.buffers.extend_seq_lens  # 扩展序列长度
                extend_start_loc = last_runner.buffers.extend_start_loc  # 扩展起始位置
            else:
                hidden_states = draft_logits_output.logits_output.hidden_states  # 隐藏状态
                req_pool_indices = forward_batch.req_pool_indices  # 请求池索引
                extend_seq_lens = forward_batch.extend_seq_lens  # 扩展序列长度
                extend_start_loc = forward_batch.extend_start_loc  # 扩展起始位置
            assign_hidden_states_pool_triton(  # 分配隐藏状态池
                hidden_states,
                req_pool_indices,
                self.req_to_hidden_states_pool,
                self.speculative_num_steps - 1,
                forward_batch.batch_size,
                extend_seq_lens,
                extend_start_loc,
            )

        # Reorganize the spec info for the next batch
        batch_result.next_token_ids = next_token_ids_backup  # 恢复下一个token ID
        # Construct the return values
        next_draft_input = batch_result.next_draft_input  # 获取下一轮草稿输入
        (
            next_draft_input.topk_p,  # topk概率
            next_draft_input.topk_index,  # topk索引
            next_draft_input.hidden_states,  # 隐藏状态
        ) = (
            torch.cat(ret_topk_p_list, dim=1).clone(),  # 合并并克隆概率
            torch.cat(ret_topk_index_list, dim=1).clone(),  # 合并并克隆索引
            None,  # 隐藏状态设为None
        )


class MultiLayerEagleWorkerV2(BaseSpecWorker):
    """多层EAGLE工作器V2，支持重叠调度，协调草稿和验证的执行。"""

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
        """初始化多层EAGLE工作器V2，创建草稿工作器并设置参数。"""
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

        self._draft_worker = MultiLayerEagleDraftWorker(  # 创建草稿工作器
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
        self.num_new_pages_per_topk = torch.empty(  # 占位张量
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)  # 占位张量

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)  # 获取计划流

    @property
    def target_worker(self):
        """获取目标工作器。"""
        return self._target_worker

    @property
    def draft_worker(self):
        """获取草稿工作器。"""
        return self._draft_worker

    @property
    def spec_v2_attn_backends(self) -> tuple:
        """获取V2推测解码的所有注意力后端（目标+草稿扩展）。"""
        return (
            self._target_worker.model_runner.attn_backend,  # 目标注意力后端
            *self._draft_worker.draft_extend_attn_backend_list,  # 草稿扩展注意力后端列表
        )

    def clear_cache_pool(self):
        """清空缓存池，共享目标工作器的分配器，无需操作。"""
        # allocator and kv cache pool are shared with target worker, which are cleared in scheduler
        pass

    def forward_batch_generation(self, batch: ScheduleBatch, on_publish=None):
        """运行推测解码前向生成，处理扩展和解码两种模式。"""
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:  # 如果是扩展模式
            # Target prefill
            target_capture_mode = (  # 确定目标捕获模式
                CaptureHiddenMode.NULL
                if self.speculative_algorithm.is_standalone()
                else CaptureHiddenMode.FULL
            )
            batch.capture_hidden_mode = target_capture_mode  # 设置捕获模式
            batch_output = self.target_worker.forward_batch_generation(batch)  # 运行目标前向

            # Spec_v2 convention: batch.seq_lens = length BEFORE this iter's tokens.
            # Extend processed L prompt tokens; next verify iter expects same L.
            batch_output.new_seq_lens = batch.seq_lens  # 设置新序列长度
            # Publish before draft_extend so the fence is at target-end.
            if on_publish is not None:  # 如果有发布回调
                on_publish(batch_output.new_seq_lens)  # 发布新序列长度

            # Chain-style MTP needs FULL to get all-token hidden states;
            # non-chain only needs LAST (the target model's hidden states).
            batch_output.next_draft_input = self.draft_worker._draft_extend_for_prefill(  # 运行草稿扩展预填充
                batch,
                batch_output.logits_output.hidden_states,
                batch_output.next_token_ids,
            )
            return batch_output  # 返回输出
        else:
            if batch.spec_info is None:  # 如果没有推测信息
                capture_mode = (  # 确定捕获模式
                    CaptureHiddenMode.NULL
                    if self.speculative_algorithm.is_standalone()
                    else CaptureHiddenMode.LAST
                )
                batch.spec_info = EagleDraftInput.create_idle_input(  # 创建空闲草稿输入
                    device=self.device,
                    hidden_size=EagleDraftInput.hidden_size_for(self.draft_worker),
                    dtype=EagleDraftInput.dtype_for(self.draft_worker),
                    topk=self.topk * self.speculative_num_steps,
                    capture_hidden_mode=capture_mode,
                )
            verify_input: EagleVerifyInput = self.draft_worker.draft(batch)  # 运行草稿
            assert verify_input.is_verify_input()  # 断言为验证输入
            batch.spec_info = verify_input  # 安装验证输入
            batch_output = self.verify(batch)  # 运行验证
            # Publish before draft_extend so the fence is at verify-end.
            if on_publish is not None:  # 如果有发布回调
                on_publish(batch_output.new_seq_lens)  # 发布新序列长度
            self.draft_worker._draft_extend_for_decode(batch, batch_output)  # 运行解码后草稿扩展
            return batch_output  # 返回输出

    def verify(
        self,
        batch: ScheduleBatch,  # 调度批次
    ):
        """运行验证步骤，用目标模型验证草稿token并确定接受的token。"""
        fwd_stream = torch.get_device_module(self.device).current_stream()  # 获取前向流
        verify_input: EagleVerifyInput = batch.spec_info  # 获取验证输入
        record_stream_for_v2_verify(batch, verify_input, fwd_stream)  # 记录流

        bs = len(batch.seq_lens)  # 批次大小

        # Batch 1: Target verify
        # Prepare for target verify in a separate stream
        with self.plan_stream_ctx:  # 在计划流中准备
            verify_forward_batch, can_run_cuda_graph = (  # 准备验证前向批次
                verify_input.prepare_for_v2_verify(
                    self.req_to_token_pool,
                    batch,
                    self.target_worker,
                )
            )

        # Cover post-prepare rebinds: draft_token, plan_stream-allocated out_cache_loc.
        record_stream_each((batch.input_ids, batch.out_cache_loc), fwd_stream)  # 记录流

        # Correct some buffers due to the overlap plan
        if self.plan_stream:  # 如果有计划流
            torch.get_device_module(self.device).current_stream().wait_stream(  # 等待计划流
                self.plan_stream
            )

            # Some values such as custom_mask and position depend on the output of draft,
            # so the previous plan step used the wrong values. Here, we need to run the related
            # computation again to update them to the correct values.
            self.target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(  # 更新验证缓冲区
                verify_input,
                (
                    self.target_worker.model_runner.graph_runner.bs  # CUDA图批次大小
                    if can_run_cuda_graph
                    else None
                ),
            )
        # Run target verify batch in the main compute stream
        forward_batch_output = self.target_worker.forward_batch_generation(  # 运行目标验证前向
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = forward_batch_output.logits_output  # 获取logits

        # Sample
        maybe_detect_nan(logits_output.next_token_logits, "verify: target model logits")  # 检测NaN
        maybe_detect_inf(logits_output.next_token_logits, "verify: target model logits")  # 检测无穷大
        (
            predict,  # 预测token
            accept_lens,  # 接受长度
            accept_index,  # 接受索引
        ) = verify_input.sample(batch, logits_output)  # 采样
        new_seq_lens = batch.seq_lens + accept_lens  # 计算新序列长度

        if not batch.forward_mode.is_idle():  # 如果非空闲
            accept_tokens = predict[accept_index]  # 获取接受token
            bonus_tokens = torch.empty_like(accept_lens, dtype=torch.int32)  # 创建bonus token张量
            fill_bonus_tokens[(bs,)](  # 填充bonus token
                accept_tokens,
                accept_lens,
                bonus_tokens,
                self.speculative_num_draft_tokens,
            )
        else:
            bonus_tokens = torch.empty((0,), device=self.device, dtype=torch.int32)  # 空bonus token

        if batch.return_logprob and not batch.forward_mode.is_idle():  # 如果需要logprob
            compute_spec_v2_logprobs(  # 计算v2 logprob
                batch, logits_output, predict, accept_index, self.speculative_num_steps
            )

        next_draft_input = EagleDraftInput(bonus_tokens=bonus_tokens)  # 创建下一轮草稿输入
        # verify_forward_batch transitively holds verify-time GPU tensors that
        # must outlive the imminent batch.input_ids rebind; scheduler pins it
        # in batch_record_buf via extra_keep_alive_refs. See EAGLEWorkerV2.verify.
        return GenerationBatchResult(  # 返回生成批次结果
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            speculative_num_draft_tokens=self.speculative_num_draft_tokens,
            next_draft_input=next_draft_input,
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            routed_experts_output=forward_batch_output.routed_experts_output,
            indexer_topk_output=forward_batch_output.indexer_topk_output,
            extra_keep_alive_refs=[verify_forward_batch],
        )

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """从磁盘更新所有步骤的草稿模型权重。"""
        for i in range(self.speculative_num_steps):  # 遍历每步
            success, message = self._draft_worker.draft_runner_list[  # 更新权重
                i
            ].update_weights_from_disk(
                recv_req.model_path,
                recv_req.load_format,
                recapture_cuda_graph=recv_req.recapture_cuda_graph,
            )
            if not success:  # 如果失败
                return success, message  # 返回失败信息
        return True, "Succeeded to update model weights."  # 返回成功

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """从IPC更新所有步骤的草稿模型权重。"""
        for i in range(self.speculative_num_steps):  # 遍历每步
            success, message = self._draft_worker.draft_runner_list[  # 更新权重
                i
            ].update_weights_from_ipc(recv_req)
            if not success:  # 如果失败
                return success, message  # 返回失败信息
        return True, "Succeeded to update model weights."  # 返回成功
