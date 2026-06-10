# DFlash投机解码的信息类模块
# 定义DFlash草稿输入(DFlashDraftInput)和验证输入(DFlashVerifyInput)数据类，
# 包含验证准备、注意力参数生成和验证结果处理等核心逻辑。
from __future__ import annotations  # 启用延迟注解求值

from dataclasses import dataclass  # 导入数据类装饰器
from typing import List, Tuple  # 导入列表和元组类型

import torch  # 导入PyTorch

from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton  # 导入flashinfer KV索引创建工具
from sglang.srt.layers.logits_processor import LogitsProcessorOutput  # 导入logits处理器输出类
from sglang.srt.layers.sampler import apply_custom_logit_processor  # 导入自定义logit处理器
from sglang.srt.managers.schedule_batch import ScheduleBatch  # 导入调度批次类
from sglang.srt.mem_cache.common import (  # 导入内存缓存通用工具
    alloc_paged_token_slots_extend,  # 分配分页token槽位（extend模式）
    alloc_token_slots,  # 分配token槽位
    get_last_loc,  # 获取最后位置
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode  # 导入隐藏状态捕获模式
from sglang.srt.speculative.dflash_utils import (  # 导入DFlash工具函数
    compute_dflash_correct_drafts_and_bonus,  # 计算贪心验证的正确草稿数和奖励token
    compute_dflash_sampling_correct_drafts_and_bonus,  # 计算采样验证的正确草稿数和奖励token
    is_dflash_sampling_verify_available,  # 检查采样验证是否可用
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType  # 导入投机输入基类和枚举
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func  # 导入请求到token池映射函数


def _compute_paged_keep_slots(
    *,
    prefix_lens: torch.Tensor,  # 前缀长度
    commit_lens: torch.Tensor,  # 提交长度
    draft_token_num: int,  # 草稿token数
    page_size: int,  # 页大小
) -> torch.Tensor:
    """Compute how many draft slots per request must remain allocated.
    # 计算每个请求需要保留分配的草稿槽位数。

    The allocator frees at page granularity for paged mode, so we can only release
    full pages from the tail after verify.
    # 分页模式下分配器以页粒度释放，因此验证后只能从尾部释放完整页。
    """

    if page_size <= 1:  # 页大小必须大于1
        raise ValueError(f"Expected page_size > 1, got {page_size}.")  # 抛出异常

    seq_dtype = prefix_lens.dtype  # 获取序列长度数据类型
    extended_lens = prefix_lens + int(draft_token_num)  # 扩展后长度 = 前缀 + 草稿token数
    new_lens = prefix_lens + commit_lens.to(seq_dtype)  # 新长度 = 前缀 + 提交长度
    aligned_new_lens = ((new_lens + page_size - 1) // page_size) * page_size  # 按页大小对齐
    keep_lens = torch.minimum(aligned_new_lens, extended_lens)  # 保留长度取对齐后和扩展后的较小值
    keep_slots = (keep_lens - prefix_lens).to(torch.int64)  # 计算需要保留的槽位数
    keep_slots.clamp_(min=0, max=int(draft_token_num))  # 限制在合法范围内
    return keep_slots  # 返回保留槽位数


@dataclass  # 数据类装饰器
class DFlashDraftInput(SpecInput):
    """Per-batch DFlash draft state for spec-v1 (non-overlap) scheduling.
    # 每批次的DFlash草稿状态，用于spec-v1（非重叠）调度。

    This object is stored on `ScheduleBatch.spec_info` between decode iterations.
    # 此对象在解码迭代间存储在ScheduleBatch.spec_info上。
    It is NOT sent to model attention backends; the DFlash worker uses it to run
    the draft model and to track draft-side cache progress.
    # 它不发送给模型注意力后端；DFlash worker用它运行草稿模型并跟踪草稿侧缓存进度。

    When draft windowing is disabled, `draft_seq_lens` matches the committed target
    prefix length already materialized in the draft KV cache. When windowing is
    enabled, `draft_seq_lens` is the logical resident length in the draft worker's
    compact req-to-token mapping. In paged mode this may exceed the requested
    window by up to `page_size - 1` so the local page table remains valid. `ctx_lens`
    tracks newly committed target tokens that still need draft KV materialization.
    # 禁用草稿窗口时，draft_seq_lens匹配已物化到草稿KV缓存中的目标前缀长度。
    # 启用窗口时，draft_seq_lens是草稿worker紧凑req-to-token映射中的逻辑驻留长度。
    # 分页模式下可能超出请求窗口最多page_size-1以保持本地页表有效。
    # ctx_lens跟踪仍需草稿KV物化的新提交目标token。
    """

    # Current token to start the next DFlash block (one per request).
    # 当前token，用于开始下一个DFlash块（每个请求一个）。
    bonus_tokens: torch.Tensor

    # Flattened context features for tokens that need to be appended into the draft cache.
    # 需要追加到草稿缓存中的token的扁平化上下文特征。
    # Shape: [sum(ctx_lens), K * hidden_size], where K is the number of target-layer
    # hidden-state features concatenated per token (len(dflash_config.target_layer_ids),
    # or default K == draft_num_layers for existing checkpoints).
    # 形状：[sum(ctx_lens), K * hidden_size]，K是每个token连接的目标层隐藏状态特征数。
    target_hidden: torch.Tensor

    # Context lengths per request, used to slice `target_hidden`. Device tensor (int32).
    # 每个请求的上下文长度，用于切片target_hidden。设备张量(int32)。
    ctx_lens: torch.Tensor

    # How many committed tokens are visible to the draft worker per request.
    # 每个请求对草稿worker可见的已提交token数。
    draft_seq_lens: torch.Tensor

    def __post_init__(self):  # 后初始化方法
        super().__init__(spec_input_type=SpecInputType.DFLASH_DRAFT)  # 调用父类初始化，设置输入类型

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:  # 获取投机调整token系数
        # Draft state does not change token accounting.
        # 草稿状态不改变token计数。
        return (1, 1)  # 返回(1, 1)表示不调整

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        # 根据新索引过滤批次数据
        old_ctx_lens = self.ctx_lens  # 保存旧的上下文长度
        old_target_hidden = self.target_hidden  # 保存旧的目标隐藏状态

        self.bonus_tokens = self.bonus_tokens[new_indices]  # 过滤bonus_tokens
        self.ctx_lens = old_ctx_lens[new_indices]  # 过滤ctx_lens
        self.draft_seq_lens = self.draft_seq_lens[new_indices]  # 过滤draft_seq_lens

        if old_target_hidden is None or old_target_hidden.numel() == 0:  # 如果目标隐藏状态为空
            self.target_hidden = old_target_hidden  # 保持不变
            return  # 直接返回

        # Rebuild target_hidden for the filtered batch using vectorized indexing.
        # 使用向量化索引为过滤后的批次重建target_hidden。
        old_bs = int(old_ctx_lens.shape[0])  # 旧批次大小
        offsets = torch.zeros(  # 创建偏移量张量
            (old_bs + 1,), dtype=torch.int64, device=old_ctx_lens.device
        )
        offsets[1:].copy_(old_ctx_lens.to(torch.int64).cumsum(0))  # 计算累积偏移

        start = offsets[:-1]  # 每个请求的起始位置
        seg_start = start[new_indices]  # 过滤后的起始位置
        seg_lens = old_ctx_lens[new_indices].to(torch.int64)  # 过滤后的段长度

        max_len = int(seg_lens.max().item()) if seg_lens.numel() > 0 else 0  # 最大段长度
        if max_len <= 0:  # 如果最大长度为0
            self.target_hidden = old_target_hidden[:0]  # 设为空
            return  # 返回

        r = torch.arange(max_len, device=old_ctx_lens.device, dtype=torch.int64)[  # 创建范围张量
            None, :
        ]
        pos2d = seg_start[:, None] + r  # 二维位置索引
        mask = r < seg_lens[:, None]  # 创建有效位置掩码
        flat_pos = pos2d[mask]  # 扁平化有效位置
        self.target_hidden = (  # 重建target_hidden
            old_target_hidden.index_select(0, flat_pos)  # 使用索引选择
            if flat_pos.numel() > 0  # 如果有有效位置
            else old_target_hidden[:0]  # 否则设为空
        )

    def merge_batch(self, spec_info: "DFlashDraftInput"):
        # 合并两个批次的数据
        self.bonus_tokens = torch.cat(  # 合并bonus_tokens
            [self.bonus_tokens, spec_info.bonus_tokens], dim=0
        )
        self.ctx_lens = torch.cat([self.ctx_lens, spec_info.ctx_lens], dim=0)  # 合并ctx_lens
        self.draft_seq_lens = torch.cat(  # 合并draft_seq_lens
            [self.draft_seq_lens, spec_info.draft_seq_lens], dim=0
        )
        if self.target_hidden is None or self.target_hidden.numel() == 0:  # 如果自身target_hidden为空
            self.target_hidden = spec_info.target_hidden  # 直接使用对方的
        elif (  # 如果对方target_hidden不为空
            spec_info.target_hidden is not None and spec_info.target_hidden.numel() > 0
        ):
            self.target_hidden = torch.cat(  # 拼接两边的target_hidden
                [self.target_hidden, spec_info.target_hidden], dim=0
            )


@dataclass  # 数据类装饰器
class DFlashVerifyInput(SpecInput):
    """Inputs for a target-model verify forward in DFlash (spec-v1).
    # DFlash(spec-v1)中目标模型验证前向的输入。

    The verify forward is run with `ForwardMode.TARGET_VERIFY` so that the target
    model returns logits for all tokens in the block, enabling accept-length
    computation.
    # 验证前向以ForwardMode.TARGET_VERIFY模式运行，使目标模型返回块中所有token的logits，
    # 从而计算接受长度。
    """

    draft_token: torch.Tensor  # 草稿token
    positions: torch.Tensor  # 位置编码
    draft_token_num: int  # 草稿token数量
    # Kept for compatibility with attention backends that gate tree metadata by `topk > 1`.
    # DFLASH verify is linear (non-tree), so this is always 1.
    # 为兼容通过topk>1门控树元数据的注意力后端而保留。DFLASH验证是线性的（非树），因此始终为1。
    topk: int = 1
    # Custom attention "allow mask" for TARGET_VERIFY in backends that require it (e.g. triton).
    # Semantics follow SGLang speculative conventions: True means the (q, k) pair is allowed.
    # 需要自定义注意力掩码的后端（如triton）中用于TARGET_VERIFY的自定义注意力"允许掩码"。
    custom_mask: torch.Tensor | None = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL  # 隐藏状态捕获模式

    # Shape info for padding (e.g., DP attention / CUDA graph).
    # 用于填充的形状信息（如DP注意力/CUDA图）。
    num_tokens_per_batch: int = -1

    def __post_init__(self):  # 后初始化方法
        super().__init__(spec_input_type=SpecInputType.DFLASH_VERIFY)  # 设置输入类型为DFLASH验证
        if self.num_tokens_per_batch == -1:  # 如果未指定每批token数
            self.num_tokens_per_batch = int(self.draft_token_num)  # 使用草稿token数

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:  # 获取投机调整token系数
        return self.draft_token_num, self.draft_token_num  # 返回(草稿token数, 草稿token数)

    def prepare_for_verify(
        self,
        batch: ScheduleBatch,  # 调度批次
        page_size: int,  # 页大小
        *,
        build_custom_mask: bool = True,  # 是否构建自定义掩码
    ):
        # 为验证前向准备批次数据
        if batch.forward_mode.is_idle():  # 如果是空闲模式
            return  # 直接返回

        batch.input_ids = self.draft_token  # 设置输入ID为草稿token

        if page_size == 1:  # 非分页模式
            batch.out_cache_loc = alloc_token_slots(  # 分配token槽位
                batch.tree_cache, len(batch.input_ids)
            )
            end_offset = batch.seq_lens + self.draft_token_num  # 计算结束偏移
        else:  # 分页模式
            prefix_lens = batch.seq_lens  # 前缀长度
            prefix_lens_cpu = batch.seq_lens_cpu  # CPU上的前缀长度
            end_offset = prefix_lens + self.draft_token_num  # 结束偏移
            end_offset_cpu = prefix_lens_cpu + self.draft_token_num  # CPU上的结束偏移
            last_loc = get_last_loc(  # 获取最后位置
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                prefix_lens,
            )
            batch.out_cache_loc = alloc_paged_token_slots_extend(  # 分配分页token槽位
                batch.tree_cache,
                prefix_lens,
                prefix_lens_cpu,
                end_offset,
                end_offset_cpu,
                last_loc,
                len(batch.input_ids),
            )

        bs = batch.batch_size()  # 批次大小
        assign_req_to_token_pool_func(  # 分配请求到token池的映射
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )

        if not build_custom_mask:  # 不需要构建自定义掩码
            self.custom_mask = None  # 设为None
            return  # 返回

        if self.draft_token_num <= 0:  # 草稿token数必须为正
            raise ValueError(
                f"DFLASH draft_token_num must be positive, got {self.draft_token_num}."
            )
        mask_chunks: List[torch.Tensor] = []  # 掩码块列表
        q_len = int(self.draft_token_num)  # 查询长度
        q_idx = torch.arange(q_len, device=batch.device, dtype=torch.int32).unsqueeze(1)  # 查询索引
        for prefix_len in batch.seq_lens_cpu.tolist():  # 遍历每个请求的前缀长度
            prefix_len_i = int(prefix_len)  # 当前前缀长度
            kv_len = prefix_len_i + q_len  # KV长度
            k_idx = torch.arange(  # 键索引
                kv_len, device=batch.device, dtype=torch.int32
            ).unsqueeze(0)
            # Allow attending to the full prefix and to tokens up to (and including) the
            # current query position within the verify block (standard causal masking).
            # 允许关注完整前缀以及验证块中当前查询位置及之前的token（标准因果掩码）。
            allow = k_idx <= (prefix_len_i + q_idx)  # 因果掩码条件
            mask_chunks.append(allow.flatten())  # 扁平化并添加到列表
        self.custom_mask = (  # 拼接所有掩码块
            torch.cat(mask_chunks, dim=0)
            if mask_chunks  # 如果有掩码块
            else torch.empty((0,), dtype=torch.bool, device=batch.device)  # 否则为空
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,  # 请求池索引
        paged_kernel_lens: torch.Tensor,  # 分页核长度
        paged_kernel_lens_sum: int,  # 分页核长度总和
        req_to_token: torch.Tensor,  # 请求到token映射
    ):
        # 生成预填充注意力参数
        device = req_pool_indices.device  # 设备
        bs = len(req_pool_indices)  # 批次大小

        qo_indptr = torch.arange(  # 查询输出间接指针
            0,
            (bs + 1) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device=device,
        )

        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)  # 累积KV序列长度
        paged_kernel_lens = paged_kernel_lens + self.draft_token_num  # 加上草稿token数
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)  # 计算累积和

        kv_indices = torch.empty(  # KV索引缓冲区
            paged_kernel_lens_sum + self.draft_token_num * bs,
            dtype=torch.int32,
            device=device,
        )
        create_flashinfer_kv_indices_triton[(bs,)](  # 使用Triton内核创建KV索引
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        mask = self.custom_mask  # 自定义掩码
        if mask is not None:  # 如果有掩码
            mask_numel = (  # 掩码元素数
                paged_kernel_lens_sum * self.draft_token_num
                + (self.draft_token_num**2) * bs
            )
            if mask.numel() < mask_numel:  # 如果掩码太小
                # FIXME(attn): temporary fix for custom mask padding with cuda graph
                # FIXME(attn): CUDA图自定义掩码填充的临时修复
                mask = torch.cat(  # 拼接True值填充
                    [
                        mask,
                        torch.full(
                            (mask_numel - mask.numel(),),
                            True,
                            dtype=torch.bool,
                            device=device,
                        ),
                    ],
                    dim=0,
                )
                self.custom_mask = mask  # 更新掩码
        return kv_indices, cum_kv_seq_len, qo_indptr, mask  # 返回注意力参数

    def verify(
        self,
        *,
        batch: ScheduleBatch,  # 调度批次
        logits_output: LogitsProcessorOutput,  # logits输出
        page_size: int,  # 页大小
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """DFlash verification for greedy and non-greedy sampling.
        # DFlash贪心和非贪心采样的验证。

        Returns:
        # 返回值：
            new_bonus_tokens: int64 tensor [bs] (the new current token per request)
            # 新奖励token：int64张量[bs]（每个请求的新当前token）
            commit_lens: int32 tensor [bs] (how many verify-input tokens are committed)
            # 提交长度：int32张量[bs]（有多少验证输入token被提交）
            next_target_hidden: tensor [sum(commit_lens), feature_dim]
            # 下一个目标隐藏状态：张量[sum(commit_lens), feature_dim]
            num_correct_drafts_per_req_cpu: list[int] (accepted draft tokens per request)
            # 每个请求接受的草稿token数：list[int]
        """
        if batch.forward_mode.is_idle():  # 如果是空闲模式
            empty = torch.empty((0,), dtype=torch.int64, device=batch.device)  # 创建空张量
            return empty, empty.to(torch.int32), empty, []  # 返回空结果

        bs = batch.batch_size()  # 批次大小
        device = logits_output.next_token_logits.device  # 设备

        sampling_info = batch.sampling_info  # 采样信息
        if sampling_info is not None:  # 如果有采样信息
            if len(sampling_info) != bs:  # 采样信息大小不匹配
                raise RuntimeError(
                    "DFLASH verify sampling_info size mismatch: "
                    f"len(sampling_info)={len(sampling_info)}, bs={bs}."
                )

            # Keep speculative verify semantics consistent with normal sampling path.
            # 保持投机验证语义与正常采样路径一致。
            if sampling_info.has_custom_logit_processor:  # 如果有自定义logit处理器
                apply_custom_logit_processor(
                    logits_output.next_token_logits,
                    sampling_info,
                    num_tokens_in_batch=self.draft_token_num,
                )

            if (  # 如果需要应用logits偏置
                sampling_info.penalizer_orchestrator.is_required
                or sampling_info.logit_bias is not None
            ):
                linear_penalty = torch.zeros(  # 创建线性惩罚张量
                    (bs, logits_output.next_token_logits.shape[1]),
                    dtype=torch.float32,
                    device=device,
                )
                sampling_info.apply_logits_bias(linear_penalty)  # 应用logits偏置
                logits_output.next_token_logits.add_(  # 将惩罚加到logits上
                    torch.repeat_interleave(linear_penalty, self.draft_token_num, dim=0)
                )

        candidates = self.draft_token.view(bs, self.draft_token_num)  # 重塑候选token形状
        if (  # 如果支持采样验证且不是全部贪心
            sampling_info is not None
            and not sampling_info.is_all_greedy
            and is_dflash_sampling_verify_available()
        ):
            correct_len, bonus = compute_dflash_sampling_correct_drafts_and_bonus(  # 采样验证
                candidates=candidates,
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
            )
        else:  # 贪心验证
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(  # 目标预测
                bs, self.draft_token_num
            )
            correct_len, bonus = compute_dflash_correct_drafts_and_bonus(  # 贪心验证
                candidates=candidates,
                target_predict=target_predict,
            )

        # Single D2H transfer: candidates[1:] + correct_len + bonus
        # 单次D2H传输：candidates[1:] + correct_len + bonus
        packed = torch.cat(  # 打包数据
            [candidates[:, 1:], correct_len.unsqueeze(1), bonus.unsqueeze(1)], dim=1
        ).cpu()

        max_acc = self.draft_token_num - 1  # 最大接受长度
        num_correct_drafts_per_req_cpu: List[int] = []  # 每个请求的正确草稿数列表
        commit_lens_cpu: List[int] = []  # 每个请求的提交长度列表
        new_bonus_tokens_list: List[int] = []  # 新奖励token列表

        for i, req in enumerate(batch.reqs):  # 遍历每个请求
            acc_len = int(packed[i, max_acc].item())  # 接受长度
            proposed = packed[i, :acc_len].tolist() + [  # 提议的token列表
                int(packed[i, max_acc + 1].item())  # 加上奖励token
            ]

            appended = 0  # 已追加的token数
            for token_id in proposed:  # 遍历提议的token
                token_id = int(token_id)  # 转为整数
                req.output_ids.append(token_id)  # 追加到输出ID
                appended += 1  # 递增追加计数
                req.update_finish_state()  # 更新完成状态
                if req.finished():  # 如果请求已完成
                    break  # 跳出循环
                if req.grammar is not None:  # 如果有语法约束
                    req.grammar.accept_token(token_id)  # 接受token

            if req.output_ids:  # 如果有输出ID
                new_bonus_token = int(req.output_ids[-1])  # 取最后一个输出作为新奖励token
            elif req.origin_input_ids:  # 如果没有输出但有原始输入
                # If no token was appended in this verify step, keep the current token unchanged.
                # 如果在验证步骤中没有追加token，保持当前token不变。
                new_bonus_token = int(req.origin_input_ids[-1])  # 使用最后一个原始输入
            else:  # 两者都为空
                raise RuntimeError(
                    "DFLASH verify cannot determine current token: both output_ids and origin_input_ids are empty."
                )

            commit_lens_cpu.append(appended)  # 记录提交长度
            new_bonus_tokens_list.append(new_bonus_token)  # 记录新奖励token
            num_correct_drafts_per_req_cpu.append(max(0, appended - 1))  # 正确草稿数 = 追加数-1
            req.spec_verify_ct += 1  # 递增验证计数
            req.spec_num_correct_drafts += num_correct_drafts_per_req_cpu[-1]  # 累加正确草稿数

        commit_lens = torch.tensor(commit_lens_cpu, dtype=torch.int32, device=device)  # 提交长度张量
        new_bonus_tokens = torch.tensor(  # 新奖励token张量
            new_bonus_tokens_list, dtype=torch.int64, device=device
        )

        # Free uncommitted KV cache slots and compact out_cache_loc.
        # 释放未提交的KV缓存槽位并压缩out_cache_loc。
        if page_size == 1:  # 非分页模式
            out_cache_loc = batch.out_cache_loc.view(bs, self.draft_token_num)  # 重塑形状
            keep_mask = (  # 保留掩码
                torch.arange(self.draft_token_num, device=device)[None, :]
                < commit_lens[:, None]
            )
            batch.token_to_kv_pool_allocator.free(out_cache_loc[~keep_mask])  # 释放未保留的槽位
            batch.out_cache_loc = out_cache_loc[keep_mask]  # 只保留已提交的
        else:  # 分页模式
            out_cache_loc = batch.out_cache_loc.view(bs, self.draft_token_num)  # 重塑形状
            row_offsets = torch.arange(self.draft_token_num, device=device)[None, :]  # 行偏移
            keep_slots = _compute_paged_keep_slots(  # 计算分页保留槽位
                prefix_lens=batch.seq_lens,
                commit_lens=commit_lens,
                draft_token_num=self.draft_token_num,
                page_size=page_size,
            )
            free_mask = row_offsets >= keep_slots[:, None]  # 需要释放的掩码
            batch.token_to_kv_pool_allocator.free(out_cache_loc[free_mask])  # 释放

            keep_mask = row_offsets < commit_lens[:, None]  # 保留掩码
            batch.out_cache_loc = out_cache_loc[keep_mask]  # 只保留已提交的

        # Update req-level KV cache accounting.
        # 更新请求级KV缓存记账。
        for req, commit_len in zip(batch.reqs, commit_lens_cpu, strict=True):  # 遍历请求和提交长度
            req.kv_committed_len += commit_len  # 更新已提交KV长度
            req.kv_allocated_len = req.kv_committed_len  # 已分配长度等于已提交长度

        # Update req_to_token pool mapping for newly committed tokens.
        # 更新新提交token的req_to_token池映射。
        end_offset = batch.seq_lens + commit_lens.to(batch.seq_lens.dtype)  # 结束偏移
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )

        # Update batch seq lens.
        # 更新批次序列长度。
        batch.seq_lens.add_(commit_lens.to(batch.seq_lens.dtype))  # 更新GPU序列长度
        batch.seq_lens_cpu.add_(  # 更新CPU序列长度
            torch.tensor(commit_lens_cpu, dtype=batch.seq_lens_cpu.dtype)
        )
        # Keep seq_lens_sum in sync; flashinfer indices updaters rely on this for buffer sizing.
        # 保持seq_lens_sum同步；flashinfer索引更新器依赖此值进行缓冲区大小调整。
        batch.seq_lens_sum += sum(commit_lens_cpu)  # 更新序列长度总和

        # Build next-step context features from the committed verify-input tokens.
        # 从已提交的验证输入token构建下一步上下文特征。
        hidden = logits_output.hidden_states  # 获取隐藏状态
        if hidden is None:  # 如果隐藏状态为空
            raise RuntimeError(
                "DFLASH verify requires target hidden states, but got None."
            )
        hidden = hidden.view(bs, self.draft_token_num, -1)  # 重塑隐藏状态形状
        segments: List[torch.Tensor] = []  # 隐藏状态段列表
        for i, ln in enumerate(commit_lens_cpu):  # 遍历每个请求的提交长度
            if ln > 0:  # 如果有提交
                segments.append(hidden[i, :ln, :])  # 添加该请求的隐藏状态段
        next_target_hidden = torch.cat(segments, dim=0) if segments else hidden[:0]  # 拼接所有段

        # Avoid confusing downstream consumers (spec-v1 decode doesn't use this).
        # 避免混淆下游消费者（spec-v1解码不使用此值）。
        logits_output.hidden_states = None  # 清除隐藏状态

        return (  # 返回验证结果
            new_bonus_tokens,
            commit_lens,
            next_target_hidden,
            num_correct_drafts_per_req_cpu,
        )
