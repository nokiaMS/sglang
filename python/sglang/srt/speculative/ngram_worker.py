# N-Gram推测解码工作器
# 本文件实现了基于N-Gram的推测解码（Speculative Decoding）工作器。
# N-Gram推测解码利用历史token序列的N-Gram模式来预测未来可能的token，
# 生成候选draft token树，然后由目标模型（target model）进行验证，
# 从而加速推理过程。与使用小型draft模型的Eagle等方式不同，
# N-Gram方法通过CPU端的trie结构查找来生成draft token，无需额外的draft模型。

import logging
from typing import List, Optional

import numpy as np
import torch
from sgl_kernel.speculative import reconstruct_indices_from_tree_mask

from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.observability.trace import get_global_tracing_enabled
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import generate_token_bitmask

logger = logging.getLogger(__name__)


# 是否使用全注意力掩码（FULL_MASK）模式
# FULL_MASK比QLEN_MASK更保守，但不需要flashinfer的额外支持
USE_FULL_MASK = True


class NGRAMWorker:
    """N-Gram推测解码工作器

    基于N-Gram匹配的推测解码工作器，通过CPU端的trie结构
    从历史token序列中查找可能的后续token，构建draft token树，
    然后交由目标模型进行批量验证，以加速推理。
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # 保存目标工作器引用，draft验证依赖目标模型执行前向推理
        self.target_worker = target_worker
        # 复用目标工作器的model_runner，无需独立加载模型
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        # 每次推测生成的draft token数量
        self.draft_token_num: int = server_args.speculative_num_draft_tokens
        # trie的最大搜索深度，即N-Gram中的N值上限
        self.max_trie_depth: int = server_args.speculative_ngram_max_trie_depth

        # 最大批处理大小，用于预分配GPU张量
        self.max_batch_size = target_worker.max_running_requests
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"

        # 预分配GPU张量，避免推理时的动态内存分配开销
        self._init_preallocated_tensors()

        # 初始化N-Gram语料库（CPU端的trie结构），用于存储和检索token序列模式
        self.ngram_corpus = NgramCorpus(
            min_bfs_breadth=server_args.speculative_ngram_min_bfs_breadth,
            max_bfs_breadth=server_args.speculative_ngram_max_bfs_breadth,
            match_type=server_args.speculative_ngram_match_type,
            capacity=server_args.speculative_ngram_capacity,
            max_trie_depth=server_args.speculative_ngram_max_trie_depth,
            draft_token_num=server_args.speculative_num_draft_tokens,
            external_sam_budget=server_args.speculative_ngram_external_sam_budget,
            external_corpus_max_tokens=server_args.speculative_ngram_external_corpus_max_tokens,
        )
        # 如果指定了外部语料库路径，则加载外部语料库
        if server_args.speculative_ngram_external_corpus_path is not None:
            from sglang.srt.speculative.cpp_ngram.external_corpus import (
                iter_external_corpus_chunks,
            )

            corpus_path = server_args.speculative_ngram_external_corpus_path
            # 将外部语料库分块读取，避免内存溢出
            chunks = list(
                iter_external_corpus_chunks(
                    corpus_path,
                    target_worker.tokenizer,
                    server_args.speculative_ngram_external_corpus_max_tokens,
                )
            )
            loaded = self.add_external_corpus(corpus_path, chunks)
            self.commit_corpus_load(corpus_path, loaded)
            logger.info(
                "Loaded external ngram corpus '%s' (%d tokens).",
                corpus_path,
                loaded,
            )

    def clear_cache_pool(self):
        """清空N-Gram语料库缓存池，重置所有trie结构"""
        self.ngram_corpus.reset()

    def update_weights_from_tensor(self, recv_req):
        # NGRAM has no draft weights of its own — the n-gram corpus is a CPU
        # lookup structure built from request token streams — and its
        # `model_runner` is shared with the target worker. The scheduler
        # mixin dispatches via `self.draft_worker or self.tp_worker`, so
        # without this method any caller of `update_weights_from_tensor`
        # under `--speculative-algorithm NGRAM` raises AttributeError.
        # N-Gram工作器没有自己的draft模型权重，直接委托给目标工作器更新权重
        return self.target_worker.update_weights_from_tensor(recv_req)

    def add_external_corpus(self, corpus_id: str, token_chunks: list[list[int]]) -> int:
        """添加外部语料库到N-Gram索引中，返回加载的token数量"""
        return self.ngram_corpus.load_external_corpus_named(corpus_id, token_chunks)

    def commit_corpus_load(self, corpus_id: str, loaded_token_count: int) -> None:
        """提交外部语料库加载，使其对查询可见"""
        self.ngram_corpus.commit_external_corpus_load(corpus_id, loaded_token_count)

    def remove_external_corpus(self, corpus_id: str) -> None:
        """移除指定的外部语料库"""
        self.ngram_corpus.remove_external_corpus(corpus_id)

    def list_external_corpora(self) -> dict[str, int]:
        """列出所有已加载的外部语料库及其token数量"""
        return self.ngram_corpus.list_external_corpora()

    def _efficient_concat_last_n(self, seq1: List[int], seq2: List[int], n: int):
        """高效拼接两个序列的最后n个元素

        优先从seq2取末尾元素，不足时从seq1补充，
        避免不必要的完整序列拼接操作。
        """
        seq2_len = len(seq2)
        # 如果seq2长度已足够，直接取seq2的最后n个元素
        if seq2_len >= n:
            return seq2[-n:]

        # 否则从seq1补充所需数量
        need_from_seq1 = n - seq2_len
        return seq1[-need_from_seq1:] + seq2

    def _init_preallocated_tensors(self):
        """预分配GPU张量以避免推理时的动态内存分配

        为draft token、树掩码、检索索引等预分配固定大小的GPU内存，
        并按批大小预先切片，减少运行时的GPU内存操作开销。
        """
        max_total_drafts = self.max_batch_size * self.draft_token_num
        max_total_mask_size = (
            self.max_batch_size * self.draft_token_num * self.draft_token_num
        )

        # draft token存储
        self.draft_tokens = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        # 树结构检索索引，用于从验证结果中恢复正确token
        self.retrieve_indexes = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        # 树结构中每个节点的下一个token指针
        self.retrieve_next_token = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        # 树结构中每个节点的兄弟节点指针（用于广度优先遍历）
        self.retrieve_next_sibling = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        # draft token的位置编码
        self.positions = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        # 树形注意力掩码，控制draft token之间的注意力可见性
        self.tree_mask = torch.empty(
            (max_total_mask_size,), dtype=torch.bool, device=self.device
        )

        # 按批大小预先切片，避免运行时重复创建视图
        self.draft_tokens_batch = []
        self.tree_mask_batch = []
        self.retrieve_indexes_batch = []
        self.retrieve_next_token_batch = []
        self.retrieve_next_sibling_batch = []
        self.positions_batch = []

        for bs in range(0, self.max_batch_size + 1):
            self.retrieve_indexes_batch.append(self.retrieve_indexes[:bs, :])
            self.retrieve_next_token_batch.append(self.retrieve_next_token[:bs, :])
            self.retrieve_next_sibling_batch.append(self.retrieve_next_sibling[:bs, :])
            self.positions_batch.append(self.positions[: bs * self.draft_token_num])
            self.draft_tokens_batch.append(
                self.draft_tokens[: bs * self.draft_token_num]
            )
            self.tree_mask_batch.append(
                self.tree_mask[: bs * self.draft_token_num * self.draft_token_num]
            )

    def _prepare_draft_tokens(
        self, batch: ScheduleBatch
    ) -> tuple[np.ndarray, np.ndarray]:
        """从N-Gram语料库中准备draft token

        对批次中每个请求，提取其最近的token序列，
        在trie结构中查找匹配的N-Gram模式，生成候选draft token及对应的树掩码。
        """
        bs = batch.batch_size()

        # 确保语料库的并发写入操作已完成，保证读取一致性
        self.ngram_corpus.synchronize()
        req_ids = []
        batch_tokens = []
        total_lens = []
        for req in batch.reqs:
            # 拼接输入和输出token的最后max_trie_depth个元素作为trie查找的key
            check_token = self._efficient_concat_last_n(
                req.origin_input_ids, req.output_ids, self.max_trie_depth
            )
            req_ids.append(req.rid)
            batch_tokens.append(check_token)
            total_lens.append(len(req.origin_input_ids) + len(req.output_ids))
        # 批量在trie中查找匹配的draft token和对应的树掩码
        req_drafts, mask = self.ngram_corpus.batch_get(
            req_ids, batch_tokens, total_lens
        )
        total_draft_token_num = len(req_drafts)

        # Check if speculative decoding is needed; here we always enforce it
        # 验证draft token数量与批次大小和draft_token_num的乘积一致
        assert (
            total_draft_token_num == bs * self.draft_token_num
        ), f"{total_draft_token_num=}, {bs=}, {self.draft_token_num=}"
        return req_drafts, mask

    def _prepare_for_speculative_decoding(self, batch: ScheduleBatch):
        """为推测解码准备批次数据

        如果当前是extend（预填充）模式，则跳过推测解码准备。
        否则，从N-Gram语料库获取draft token，构建树掩码和位置信息，
        并将批次设置为TARGET_VERIFY模式以便后续验证。
        """
        # extend模式下不进行推测解码
        if batch.forward_mode.is_extend():
            return

        bs = batch.batch_size()

        # 获取预分配的张量视图，避免动态内存分配
        retrieve_index = self.retrieve_indexes_batch[bs]
        retrieve_next_token = self.retrieve_next_token_batch[bs]
        retrieve_next_sibling = self.retrieve_next_sibling_batch[bs]
        positions = self.positions_batch[bs]
        tree_mask = self.tree_mask_batch[bs]
        draft_tokens = self.draft_tokens_batch[bs]

        # 从N-Gram语料库获取draft token和树掩码
        req_drafts, mask = self._prepare_draft_tokens(batch)
        # 将CPU数据异步拷贝到预分配的GPU张量中
        tree_mask.copy_(torch.from_numpy(mask), non_blocking=True)
        draft_tokens.copy_(torch.from_numpy(req_drafts), non_blocking=True)

        # 从树掩码重建检索索引、位置编码和树结构指针
        reconstruct_indices_from_tree_mask(
            tree_mask,
            batch.seq_lens,
            positions,  # mutable 会被原地修改，填充位置编码
            retrieve_index,  # mutable 会被原地修改，填充检索索引
            retrieve_next_token,  # mutable 会被原地修改，填充next token指针
            retrieve_next_sibling,  # mutable 会被原地修改，填充兄弟节点指针
            bs,
            self.draft_token_num,
        )

        # NOTE: QLEN_MASK is faster than FULL_MASK, but requires corresponding changes in flashinfer.
        # Testing shows about 8% performance improvement (the effect is roughly proportional to batch size).
        # FULL_MASK模式：为每个请求构建包含完整序列长度的注意力掩码
        if USE_FULL_MASK:
            tree_mask = []
            mask = mask.reshape(
                batch.batch_size(), self.draft_token_num, self.draft_token_num
            )
            for i, req in enumerate(batch.reqs):
                seq_len = len(req.origin_input_ids) + len(req.output_ids)
                # 构建draft token对原始序列的注意力掩码（全1，即全部可见）
                req_mask = torch.ones((self.draft_token_num, seq_len - 1)).cuda()
                # 拼接原始序列掩码和draft token之间的树掩码
                req_mask = torch.cat(
                    (req_mask, torch.from_numpy(mask[i]).cuda()), dim=1
                ).to(torch.bool)
                tree_mask.append(req_mask.flatten())
            tree_mask = torch.cat(tree_mask, dim=0)

        # 设置批次的推测算法类型和前向模式
        batch.spec_algorithm = SpeculativeAlgorithm.NGRAM
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        # 创建N-Gram验证输入信息，包含draft token树的所有元数据
        batch.spec_info = NgramVerifyInput(
            draft_tokens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            self.draft_token_num,
        )
        # 为验证阶段准备KV缓存等资源
        batch.spec_info.prepare_for_verify(batch, self.page_size)

    def _update_ngram_corpus(self, batch: ScheduleBatch):
        """用当前批次的token序列更新N-Gram语料库

        将每个请求的输入+输出token序列插入trie结构，
        以便后续请求可以利用这些模式进行推测。
        """
        batch_tokens = []
        for req in batch.reqs:
            # FIXME: Whether to insert 'extend' into the cache or not, after testing,
            # there is not much difference, so we will not insert it for now.
            # if batch.forward_mode.is_extend():
            #     put_ids = req.origin_input_ids + req.output_ids
            # else:
            # 只插入最近max_trie_depth个token，控制trie的搜索空间
            put_ids = self._efficient_concat_last_n(
                req.origin_input_ids, req.output_ids, self.max_trie_depth
            )
            batch_tokens.append(put_ids)
        # 批量将token序列插入语料库的trie结构
        self.ngram_corpus.batch_put(batch_tokens)

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """执行推测解码的生成前向传播

        流程：
        1. 准备draft token（N-Gram查找）
        2. 如果是TARGET_VERIFY模式：
           a. 由目标模型对draft token进行前向推理
           b. 验证draft token的正确性，接受匹配的token
           c. 更新N-Gram语料库
        3. 如果是DECODE/EXTEND模式：直接由目标模型执行常规推理
        """
        set_time_batch(batch.reqs, "set_spec_draft_start_time", trace_only=True)

        # 准备推测解码所需的draft token和元数据
        self._prepare_for_speculative_decoding(batch)

        set_time_batch(batch.reqs, "set_spec_draft_end_time", trace_only=True)

        spec_info = batch.spec_info
        num_correct_drafts = 0
        accept_lens = None
        num_correct_drafts_per_req_cpu = None

        if batch.forward_mode.is_target_verify():
            # TARGET_VERIFY模式：目标模型验证draft token
            if batch.has_grammar:
                # 将树结构指针提前拷贝到CPU，用于生成grammar的bitmask
                retrieve_next_token_cpu = spec_info.retrieve_next_token.cpu()
                retrieve_next_sibling_cpu = spec_info.retrieve_next_sibling.cpu()
                draft_tokens_cpu = spec_info.draft_token.view(
                    spec_info.retrieve_next_token.shape
                ).cpu()

            set_time_batch(batch.reqs, "set_spec_verify_start_time", trace_only=True)

            # 由目标模型对draft token执行前向推理
            batch_result = self.target_worker.forward_batch_generation(
                batch, is_verify=True
            )
            logits_output, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.can_run_cuda_graph,
            )

            verify_input: NgramVerifyInput = batch.spec_info
            vocab_mask = None
            if batch.has_grammar:
                # Generate the logit mask for structured output.
                # Overlap the CPU operations for bitmask generation with the forward pass.
                # 为结构化输出（如JSON、正则等约束）生成词表掩码
                vocab_mask = generate_token_bitmask(
                    batch.reqs,
                    verify_input,
                    retrieve_next_token_cpu,
                    retrieve_next_sibling_cpu,
                    draft_tokens_cpu,
                    batch.sampling_info.vocab_size,
                )

                if vocab_mask is not None:
                    assert verify_input.grammar is not None
                    vocab_mask = vocab_mask.to(verify_input.retrieve_next_token.device)
                    # NOTE (sk): otherwise, this vocab mask will be the one from the previous extend stage
                    # and will be applied to produce wrong results
                    # 清除上一阶段的vocab_mask，避免对当前验证产生错误影响
                    batch.sampling_info.vocab_mask = None

            # 验证draft token：对比目标模型输出与draft token，确定接受哪些token
            logits_output, next_token_ids, num_correct_drafts = verify_input.verify(
                batch, logits_output, self.page_size, vocab_mask
            )
            # 获取每个请求的正确draft token数量，用于指标统计
            num_correct_drafts_per_req_cpu = (
                verify_input.num_correct_drafts.cpu().tolist()
            )

            if get_global_tracing_enabled():
                for idx, req in enumerate(batch.reqs):
                    num_correct_drafts = (
                        verify_input.num_correct_drafts[idx].item()
                        if verify_input.num_correct_drafts is not None
                        else 0
                    )
                    req.time_stats.set_spec_verify_end_time(
                        num_correct_drafts=num_correct_drafts
                    )

            # Store accept_lens (with bonus) for per-request metrics; downstream
            # subtracts 1 to recover drafts-only counts.
            # 存储接受长度（含bonus token），用于每请求级别的指标统计
            accept_lens = verify_input.num_accept_tokens
            if batch.return_logprob:
                # 为推测解码补全输出logprob信息
                add_output_logprobs_for_spec_v1(batch, verify_input, logits_output)
            # 用验证通过的token更新N-Gram语料库
            self._update_ngram_corpus(batch)
            # Clean up per-request match state for finished/retracted requests.
            # State entries are created in _prepare_draft_tokens and cleaned here.
            # If a request is removed without passing through verify, the entry
            # persists until reset(); this is acceptable because MatchState is small.
            # 清理已完成或被撤回请求的匹配状态，释放内存
            finished_req_ids = []
            for req in batch.reqs:
                if req.finished() or req.is_retracted:
                    finished_req_ids.append(req.rid)
            if finished_req_ids:
                self.ngram_corpus.erase_match_state(finished_req_ids)
            # 验证完成后将前向模式恢复为DECODE
            batch.forward_mode = ForwardMode.DECODE

        else:
            # 非TARGET_VERIFY模式（如EXTEND），直接由目标模型执行常规推理
            batch_result = self.target_worker.forward_batch_generation(batch)
            logits_output, next_token_ids, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.next_token_ids,
                batch_result.can_run_cuda_graph,
            )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            num_correct_drafts=num_correct_drafts,
            num_correct_drafts_per_req_cpu=num_correct_drafts_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
            accept_lens=accept_lens,
        )
