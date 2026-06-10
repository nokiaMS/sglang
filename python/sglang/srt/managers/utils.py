# 管理器工具模块 - 定义生成批结果、嵌入批结果、输入验证等通用工具函数和数据结构

from __future__ import annotations  # 启用延迟类型注解评估

import dataclasses  # 导入数据类模块
import logging  # 导入日志模块
from dataclasses import dataclass  # 导入数据类装饰器
from typing import TYPE_CHECKING, Any, List, Optional, Union  # 导入类型提示

import torch  # 导入PyTorch

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX  # 导入健康检查请求ID前缀
from sglang.srt.eplb.expert_distribution import ExpertDistributionMetrics  # 导入专家分布指标
from sglang.srt.layers.logits_processor import LogitsProcessorOutput  # 导入logits处理器输出
from sglang.srt.managers.schedule_batch import Req  # 导入请求类
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors  # 导入流水线并行代理张量
from sglang.srt.server_args import ServerArgs  # 导入服务器参数
from sglang.srt.state_capturer.base import TopkCaptureOutput  # 导入TopK捕获输出

if TYPE_CHECKING:  # 仅用于类型检查时导入
    from sglang.srt.managers.scheduler import GenerationBatchResult  # 导入生成批结果
    from sglang.srt.speculative.eagle_info import EagleDraftInput  # 导入Eagle草稿输入


logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


@dataclasses.dataclass
class GenerationBatchResult:  # 生成批结果数据类
    logits_output: Optional[LogitsProcessorOutput] = None  # logits处理器输出
    pp_hidden_states_proxy_tensors: Optional[PPProxyTensors] = None  # 流水线并行隐藏状态代理张量
    next_token_ids: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None  # 下一个token ID
    num_correct_drafts: int = 0  # no bonus included  正确草稿数（不含bonus）
    num_correct_drafts_per_req_cpu: Optional[List[int]] = None  # 每个请求的正确草稿数（CPU）
    can_run_cuda_graph: bool = False  # 是否可以运行CUDA图

    # PP skip output comm: True when output send/recv was skipped and  流水线并行跳过输出通信：当输出发送/接收被跳过且
    # next_token_ids are placeholder zeros. Used by process_batch_result_prefill  next_token_ids为占位零时为True。用于process_batch_result_prefill
    # to validate that skipped output is never consumed.  验证跳过的输出从未被消费。
    skipped_output_comm: bool = False  # 是否跳过了输出通信

    # For output processing  用于输出处理
    extend_input_len_per_req: Optional[List[int]] = None  # 每个请求的扩展输入长度
    extend_logprob_start_len_per_req: Optional[List[int]] = None  # 每个请求的扩展logprob起始长度

    # For overlap scheduling  用于重叠调度
    copy_done: Optional[torch.cuda.Event] = None  # CUDA事件：异步拷贝完成
    delay_sample_func: Optional[callable] = None  # 延迟采样函数
    future_indices: Optional[torch.Tensor] = None  # 未来索引
    speculative_num_draft_tokens: Optional[int] = None  # 推测草稿token数量

    # FIXME(lsyin): maybe move to a better place?  FIXME(lsyin): 可能移到更好的位置？
    # sync path: forward stream -> output processor  同步路径：前向流 -> 输出处理器
    accept_lens: Optional[torch.Tensor] = None  # 接受长度

    # Next-iter seq_lens; published via on_publish.  下一次迭代的序列长度；通过on_publish发布。
    new_seq_lens: Optional[torch.Tensor] = None  # 新序列长度

    # relay path: forward stream -> next step forward  中继路径：前向流 -> 下一步前向
    next_draft_input: Optional[EagleDraftInput] = None  # 下一个草稿输入

    # Refs the worker wants scheduler to keep alive for the same 2-iter window  worker希望调度器在同一2迭代窗口内保持存活的引用
    # as batch_record_buf. Used for cross-stream tensor lifetime (e.g. a spec  用于跨流张量生命周期（例如推测
    # V2 verify ForwardBatch whose tensors must outlive mid-iter SB rebinds).  V2验证的ForwardBatch，其张量必须存活过中间迭代的SB重新绑定）。
    extra_keep_alive_refs: Optional[List[Any]] = None  # 额外保持存活的引用

    # Routed experts: pending async D2H for overlap scheduling  路由专家：重叠调度的待处理异步D2H
    routed_experts_output: Optional[TopkCaptureOutput] = None  # 路由专家输出
    indexer_topk_output: Optional[TopkCaptureOutput] = None  # 索引器TopK输出

    # metrics  指标
    expert_distribution_metrics: Optional[ExpertDistributionMetrics] = None  # 专家分布指标

    # Forward pass metrics (FPM) — GPU-accurate timing via CUDA events  前向传播指标（FPM）——通过CUDA事件的GPU精确计时
    fpm_start_event: Optional[torch.cuda.Event] = None  # FPM开始事件
    fpm_end_event: Optional[torch.cuda.Event] = None  # FPM结束事件

    def copy_to_cpu(self, return_logprob: bool, return_hidden_states: bool = True):  # 将张量拷贝到CPU（重叠调度用）
        """Copy tensors to CPU in overlap scheduling.
        Only the tensors which are needed for processing results are copied,
        e.g., next_token_ids, logits outputs
        在重叠调度中将张量拷贝到CPU。仅拷贝处理结果所需的张量，例如next_token_ids、logits输出。
        """
        if return_logprob:  # 如果需要返回logprob
            if self.logits_output.next_token_logprobs is not None:  # 下一个token的logprobs
                self.logits_output.next_token_logprobs = (
                    self.logits_output.next_token_logprobs.to("cpu", non_blocking=True)  # 异步拷贝到CPU
                )
            if self.logits_output.input_token_logprobs is not None:  # 输入token的logprobs
                self.logits_output.input_token_logprobs = (
                    self.logits_output.input_token_logprobs.to("cpu", non_blocking=True)  # 异步拷贝到CPU
                )
            if self.logits_output.next_token_top_logprobs_val is not None:  # 下一个token的top logprobs值
                self.logits_output.next_token_top_logprobs_val = [
                    v.to("cpu", non_blocking=True) if torch.is_tensor(v) else v  # 张量则异步拷贝
                    for v in self.logits_output.next_token_top_logprobs_val
                ]
            if self.logits_output.next_token_top_logprobs_idx is not None:  # 下一个token的top logprobs索引
                self.logits_output.next_token_top_logprobs_idx = [
                    x.to("cpu", non_blocking=True) if torch.is_tensor(x) else x  # 张量则异步拷贝
                    for x in self.logits_output.next_token_top_logprobs_idx
                ]
            if self.logits_output.next_token_token_ids_logprobs_val is not None:  # 下一个token ID的logprobs值
                self.logits_output.next_token_token_ids_logprobs_val = [
                    v.to("cpu", non_blocking=True) if torch.is_tensor(v) else v  # 张量则异步拷贝
                    for v in self.logits_output.next_token_token_ids_logprobs_val
                ]
        if return_hidden_states and self.logits_output.hidden_states is not None:  # 如果需要隐藏状态且存在
            self.logits_output.hidden_states = self.logits_output.hidden_states.to(
                "cpu", non_blocking=True  # 异步拷贝到CPU
            )
        self.next_token_ids = self.next_token_ids.to("cpu", non_blocking=True)  # 拷贝next_token_ids到CPU

        if self.accept_lens is not None:  # 如果接受长度存在
            self.accept_lens = self.accept_lens.to("cpu", non_blocking=True)  # 拷贝到CPU

        if self.routed_experts_output is not None:  # 如果路由专家输出存在
            self.routed_experts_output.copy_to_cpu()  # 拷贝到CPU

        if self.indexer_topk_output is not None:  # 如果索引器TopK输出存在
            self.indexer_topk_output.copy_to_cpu()  # 拷贝到CPU

        if (x := self.expert_distribution_metrics) is not None:  # 如果专家分布指标存在
            x.copy_to_cpu()  # 拷贝到CPU

        self.copy_done.record()  # 记录CUDA事件完成

    @classmethod
    def from_pp_proxy(  # 从流水线并行代理创建GenerationBatchResult
        cls, logits_output, next_pp_outputs: PPProxyTensors, can_run_cuda_graph  # logits输出、PP代理张量、是否可运行CUDA图
    ):
        # TODO(lsyin): refactor PP and avoid using dict  TODO(lsyin): 重构PP，避免使用字典
        proxy_dict = next_pp_outputs.tensors  # 获取代理张量字典
        return cls(
            logits_output=logits_output,  # logits输出
            pp_hidden_states_proxy_tensors=None,  # 不需要PP隐藏状态代理
            next_token_ids=next_pp_outputs["next_token_ids"],  # 下一个token ID
            extend_input_len_per_req=proxy_dict.get("extend_input_len_per_req", None),  # 扩展输入长度
            extend_logprob_start_len_per_req=proxy_dict.get(
                "extend_logprob_start_len_per_req", None  # 扩展logprob起始长度
            ),
            can_run_cuda_graph=can_run_cuda_graph,  # 是否可运行CUDA图
        )


def validate_input_length(  # 验证输入长度
    req: Req, max_req_input_len: int, allow_auto_truncate: bool  # 请求、最大输入长度、是否允许自动截断
) -> Optional[str]:  # 返回错误消息或None（验证通过）
    """Validate and potentially truncate input length.
    验证并可能截断输入长度。

    Args:
        req: The request containing input_ids to validate  包含要验证的input_ids的请求
        max_req_input_len: Maximum allowed input length  最大允许输入长度
        allow_auto_truncate: Whether to truncate long inputs  是否截断过长输入

    Returns:
        Error message if validation fails, None if successful  验证失败返回错误消息，成功返回None
    """
    if len(req.origin_input_ids) >= max_req_input_len:  # 如果输入长度超过最大值
        if allow_auto_truncate:  # 如果允许自动截断
            logger.warning(
                "Request length is longer than the KV cache pool size or "  # 请求长度超过KV缓存池大小或
                "the max context length. Truncated. "  "最大上下文长度。已截断。"
                f"{len(req.origin_input_ids)=}, {max_req_input_len=}."  # 原始长度和最大长度
            )
            req.origin_input_ids = req.origin_input_ids[:max_req_input_len]  # 截断输入
            return None  # 返回None表示成功
        else:  # 不允许截断
            error_msg = (
                f"Input length ({len(req.origin_input_ids)} tokens) exceeds "  # 输入长度超出
                f"the maximum allowed length ({max_req_input_len} tokens). "  "最大允许长度"
                f"Use a shorter input or enable --allow-auto-truncate."  "使用更短的输入或启用 --allow-auto-truncate"
            )
            return error_msg  # 返回错误消息

    return None  # 验证通过


def get_logprob_dict_from_result(result: GenerationBatchResult) -> dict:  # 从生成批结果中提取logprob字典

    logits_output = result.logits_output  # 获取logits输出
    assert logits_output is not None  # 断言logits输出不为None

    return {
        "extend_input_len_per_req": result.extend_input_len_per_req,  # 扩展输入长度
        "extend_logprob_start_len_per_req": result.extend_logprob_start_len_per_req,  # 扩展logprob起始长度
        "next_token_logprobs": result.logits_output.next_token_logprobs,  # 下一个token logprobs
        "next_token_top_logprobs_val": result.logits_output.next_token_top_logprobs_val,  # 下一个token top logprobs值
        "next_token_top_logprobs_idx": result.logits_output.next_token_top_logprobs_idx,  # 下一个token top logprobs索引
        "next_token_token_ids_logprobs_val": result.logits_output.next_token_token_ids_logprobs_val,  # 下一个token ID logprobs值
        "next_token_token_ids_logprobs_idx": result.logits_output.next_token_token_ids_logprobs_idx,  # 下一个token ID logprobs索引
        "input_token_logprobs": result.logits_output.input_token_logprobs,  # 输入token logprobs
        "input_top_logprobs_val": result.logits_output.input_top_logprobs_val,  # 输入top logprobs值
        "input_top_logprobs_idx": result.logits_output.input_top_logprobs_idx,  # 输入top logprobs索引
        "input_token_ids_logprobs_val": result.logits_output.input_token_ids_logprobs_val,  # 输入token ID logprobs值
        "input_token_ids_logprobs_idx": result.logits_output.input_token_ids_logprobs_idx,  # 输入token ID logprobs索引
    }


def get_logprob_from_pp_outputs(  # 从流水线并行输出中提取logprob
    next_pp_outputs: PPProxyTensors,  # 流水线并行代理张量
) -> tuple[LogitsProcessorOutput, list[int], list[int]]:  # 返回（logits输出，扩展输入长度，扩展logprob起始长度）
    logits_output = LogitsProcessorOutput(
        # Do not send logits and hidden states because they are large  不发送logits和隐藏状态因为它们太大
        next_token_logits=None,  # 不发送logits
        hidden_states=None,  # 不发送隐藏状态
        next_token_logprobs=next_pp_outputs["next_token_logprobs"],  # 下一个token logprobs
        next_token_top_logprobs_val=next_pp_outputs["next_token_top_logprobs_val"],  # top logprobs值
        next_token_top_logprobs_idx=next_pp_outputs["next_token_top_logprobs_idx"],  # top logprobs索引
        next_token_token_ids_logprobs_val=next_pp_outputs[
            "next_token_token_ids_logprobs_val"  # token ID logprobs值
        ],
        next_token_token_ids_logprobs_idx=next_pp_outputs[
            "next_token_token_ids_logprobs_idx"  # token ID logprobs索引
        ],
        input_token_logprobs=next_pp_outputs["input_token_logprobs"],  # 输入token logprobs
        input_top_logprobs_val=next_pp_outputs["input_top_logprobs_val"],  # 输入top logprobs值
        input_top_logprobs_idx=next_pp_outputs["input_top_logprobs_idx"],  # 输入top logprobs索引
        input_token_ids_logprobs_val=next_pp_outputs["input_token_ids_logprobs_val"],  # 输入token ID logprobs值
        input_token_ids_logprobs_idx=next_pp_outputs["input_token_ids_logprobs_idx"],  # 输入token ID logprobs索引
    )
    extend_input_len_per_req = next_pp_outputs["extend_input_len_per_req"]  # 扩展输入长度
    extend_logprob_start_len_per_req = next_pp_outputs[
        "extend_logprob_start_len_per_req"  # 扩展logprob起始长度
    ]

    return logits_output, extend_input_len_per_req, extend_logprob_start_len_per_req  # 返回结果


def get_alloc_len_per_decode(server_args: Optional[ServerArgs] = None) -> int:  # 获取解码时每个请求的分配长度
    if server_args is None:  # 如果没有提供服务器参数
        from sglang.srt.server_args import get_global_server_args  # 延迟导入

        server_args = get_global_server_args()  # 获取全局服务器参数

    if server_args.speculative_algorithm is None:  # 如果没有推测算法
        return 1  # 返回1

    # Spec v1:  推测V1：
    # 1) alloc topk * num_steps when draft decoding and then restore the allocation  1) 草稿解码时分配topk * num_steps，然后恢复分配
    # 2) alloc num_draft_tokens when verifying the drafts  2) 验证草稿时分配num_draft_tokens
    # Sepc v2: allocate max(topk * num_steps, num_draft_tokens)  推测V2：分配max(topk * num_steps, num_draft_tokens)

    spec_steps = server_args.speculative_num_steps or 1  # 推测步数，默认1
    spec_topk = server_args.speculative_eagle_topk or 1  # Eagle topk，默认1
    spec_tokens = server_args.max_speculative_num_draft_tokens  # 最大推测草稿token数
    page_size = server_args.page_size  # 页面大小

    if page_size == 1 or spec_topk == 1:  # 页面大小为1或topk为1
        return max(spec_steps * spec_topk, spec_tokens)  # 返回两者中的较大值
    else:  # 其他情况暂不支持
        raise NotImplementedError(
            "get_alloc_len_per_decode not implemented for page_size > 1 and spec_topk > 1"  # 未实现page_size>1和spec_topk>1的情况
        )


@dataclass
class EmbeddingBatchResult:  # 嵌入批结果数据类
    """Result from an embedding/classification forward pass.
    嵌入/分类前向传播的结果。

    Attributes:
        embeddings: Model output — pooled embeddings or classification logits.
            模型输出——池化嵌入或分类logits。
        pooled_hidden_states: Raw hidden states before the task head.  Present
            only when the batch contained ``return_pooled_hidden_states=True``
            requests.  Tensor (uniform shapes) or list of tensors (MIS).
            任务头之前的原始隐藏状态。仅在批中包含``return_pooled_hidden_states=True``
            请求时存在。张量（统一形状）或张量列表（MIS）。
        copy_done: CUDA event recorded after the async CPU copy completes.
            异步CPU拷贝完成后记录的CUDA事件。
    """

    embeddings: torch.Tensor  # 嵌入/分类logits
    pooled_hidden_states: Optional[torch.Tensor] = None  # 池化隐藏状态
    copy_done: Optional[torch.cuda.Event] = None  # 拷贝完成事件

    @property
    def can_run_cuda_graph(self) -> bool:  # 嵌入批不支持CUDA图
        return False

    def copy_to_cpu(self):  # 将嵌入和池化隐藏状态拷贝到CPU（重叠调度用）
        """Copy embeddings and pooled hidden states to CPU for overlap scheduling."""  # 将嵌入和池化隐藏状态拷贝到CPU用于重叠调度
        if isinstance(self.embeddings, torch.Tensor):  # 如果嵌入是张量
            self.copy_done = torch.get_device_module(self.embeddings.device).Event()  # 创建CUDA事件
            self.embeddings = self.embeddings.to("cpu", non_blocking=True)  # 异步拷贝到CPU
        else:  # 如果嵌入是列表
            assert isinstance(self.embeddings, list)  # 断言是列表
            if len(self.embeddings) == 0:  # 空列表
                return

            self.copy_done = torch.get_device_module(self.embeddings[0].device).Event()  # 创建CUDA事件
            self.embeddings = [
                emb.to("cpu", non_blocking=True) for emb in self.embeddings  # 异步拷贝每个嵌入
            ]

        if self.pooled_hidden_states is not None:  # 如果池化隐藏状态存在
            if isinstance(self.pooled_hidden_states, list):  # 如果是列表
                self.pooled_hidden_states = [
                    t.to("cpu", non_blocking=True) for t in self.pooled_hidden_states  # 异步拷贝每个张量
                ]
            else:  # 如果是单个张量
                self.pooled_hidden_states = self.pooled_hidden_states.to(
                    "cpu", non_blocking=True  # 异步拷贝到CPU
                )

        self.copy_done.record()  # 记录CUDA事件完成


def is_health_check_generate_req(recv_req):  # 判断请求是否为健康检查请求
    rid = getattr(recv_req, "rid", None)  # 获取请求ID
    return rid is not None and rid.startswith(HEALTH_CHECK_RID_PREFIX)  # 以健康检查前缀开头
