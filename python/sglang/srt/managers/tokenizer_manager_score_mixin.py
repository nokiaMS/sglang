# 分词器评分混入类 - 为TokenizerManager提供评分（score）相关功能，支持单条和多条项目的概率评分

import logging  # 导入日志模块
import math  # 导入数学模块
from dataclasses import dataclass  # 导入数据类装饰器
from typing import Any, Dict, List, Optional, Tuple, Union  # 导入类型提示

import torch  # 导入PyTorch

from sglang.srt.configs.model_config import is_cross_encoding_pooler_model  # 导入交叉编码池化模型判断函数
from sglang.srt.managers.embed_types import PositionalEmbeds  # 导入位置嵌入类型
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput  # 导入请求输入结构体
from sglang.srt.server_args import MIS_DELIMITER_TOKEN_ID  # 导入多项目分隔符token ID

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


@dataclass(frozen=True, slots=True)  # 不可变数据类，使用slots优化内存
class ScoreResult:  # 评分结果数据类
    scores: List[List[float]]  # 评分列表，每个项目对应一个分数列表
    prompt_tokens: int = 0  # 提示token数量
    # Per-item pooled hidden states (pre-head transformer output).  每个项目的池化隐藏状态（transformer头之前的输出）
    # CPU tensors when return_pooled_hidden_states=True; kept as tensors so  当return_pooled_hidden_states=True时为CPU张量；保持张量形式
    # in-process consumers (gRPC, engine API) avoid a .tolist() round-trip.  以便进程内消费者（gRPC、引擎API）避免.tolist()往返转换
    # The HTTP path converts to lists in serving_score.py before JSON serialization.  HTTP路径在JSON序列化前在serving_score.py中转换为列表
    # Same layout as scores: one tensor per item (not a single packed 2D tensor).  与scores布局相同：每个项目一个张量（而非单个打包的2D张量）
    pooled_hidden_states: Optional[List[Optional[torch.Tensor]]] = None  # 池化隐藏状态列表


class TokenizerManagerScoreMixin:  # 分词器评分混入类
    async def score_prompts(  # 对提示进行评分
        self,
        prompts: Union[str, List[str], List[List[int]]],  # 提示：字符串、字符串列表或token ID列表
        label_token_ids: List[int],  # 要计算概率的token ID列表
        apply_softmax: bool = False,  # 是否使用softmax归一化
        request: Optional[Any] = None,  # 可选的FastAPI请求对象
    ) -> ScoreResult:  # 返回评分结果
        """
        Score probabilities of specified token IDs after each *full prompt*.
        对每个完整提示后的指定token ID进行概率评分。

        This is a thin wrapper over `score_request` that treats `prompts` as
        already-composed inputs (i.e., no query/item concatenation needed).
        这是 `score_request` 的薄封装，将 `prompts` 视为已组合的输入（即不需要查询/项目拼接）。

        Args:
            prompts: A single prompt string, a list of prompt strings, or a list of
                pre-tokenized prompt token ID sequences.
                单个提示字符串、提示字符串列表或预分词的提示token ID序列列表。
            label_token_ids: Token IDs to compute probabilities for.  要计算概率的token ID。
            apply_softmax: Whether to normalize probabilities using softmax.  是否使用softmax归一化概率。
            request: Optional FastAPI request object.  可选的FastAPI请求对象。

        Returns:
            ScoreResult with:
                scores: List of score lists, one for each prompt, each in the order of label_token_ids.
                    评分列表，每个提示对应一个分数列表，顺序与label_token_ids一致。
                prompt_tokens: The number of prompt tokens processed.  处理的提示token数量。
        """
        # Text prompts  文本提示
        if isinstance(prompts, str) or (
            isinstance(prompts, list) and (not prompts or isinstance(prompts[0], str))
        ):  # 单个字符串或字符串列表
            return await self.score_request(
                query="",  # 查询为空
                items=prompts,  # type: ignore[arg-type]  项目为提示
                label_token_ids=label_token_ids,  # 标签token ID
                apply_softmax=apply_softmax,  # 是否softmax
                item_first=False,  # 项目不在前
                request=request,  # 请求对象
            )

        # Tokenized prompts  已分词的提示
        if isinstance(prompts, list) and (not prompts or isinstance(prompts[0], list)):
            return await self.score_request(
                query=[],  # 查询为空列表
                items=prompts,  # 项目为token ID列表
                label_token_ids=label_token_ids,  # 标签token ID
                apply_softmax=apply_softmax,  # 是否softmax
                item_first=False,  # 项目不在前
                request=request,  # 请求对象
            )

        raise ValueError("Invalid prompts type for score_prompts.")  # 无效的提示类型

    def _build_multi_item_token_sequence(  # 构建多项目评分的token序列
        self, query: List[int], items: List[List[int]], delimiter_token_id: int  # 查询ID、项目ID列表、分隔符token ID
    ) -> Tuple[List[int], List[int]]:  # 返回（合并序列，分隔符位置索引）
        """
        Build a single token sequence for multi-item scoring.
        为多项目评分构建单个token序列。
        Format: query<delimiter>item1<delimiter>item2<delimiter>item3<delimiter>
        格式：query<分隔符>item1<分隔符>item2<分隔符>item3<分隔符>

        Args:
            query: Query token IDs  查询token ID
            items: List of item token ID sequences  项目token ID序列列表
            delimiter_token_id: Token ID to use as delimiter  用作分隔符的token ID

        Returns:
            Tuple of (combined token sequence, delimiter indices)  （合并token序列，分隔符索引）的元组
        """
        combined_sequence = query[:]  # Start with query  以查询开始
        delimiter_indices = []  # 分隔符位置索引列表

        for item in items:  # 遍历每个项目
            delimiter_indices.append(len(combined_sequence))  # 记录分隔符位置
            combined_sequence.append(delimiter_token_id)  # Add delimiter  添加分隔符
            combined_sequence.extend(item)  # Add item tokens  添加项目token

        # Add final delimiter after the last item for logprob extraction  在最后一个项目后添加最终分隔符用于logprob提取
        delimiter_indices.append(len(combined_sequence))  # 记录最终分隔符位置
        combined_sequence.append(delimiter_token_id)  # 添加最终分隔符

        return combined_sequence, delimiter_indices  # 返回合并序列和分隔符索引

    def _batch_tokenize_query_and_items(  # 批量分词查询和项目
        self,
        query: Optional[Union[str, List[int]]],  # 查询文本或token ID
        items: Optional[Union[str, List[str], List[List[int]]]],  # 项目文本或token ID
    ) -> Tuple[List[int], List[List[int]]]:  # 返回（查询ID，项目ID列表）
        """
        Tokenize query and items into token IDs.
        将查询和项目分词为token ID。

        Args:
            query: The query text (str) or pre-tokenized token IDs (List[int]).
                查询文本（str）或预分词的token ID（List[int]）。
            items: Item texts or pre-tokenized token IDs.
                项目文本或预分词的token ID。

        Returns:
            (query_ids, items_ids): query token IDs and list of per-item token IDs.
                （query_ids, items_ids）：查询token ID和每项目的token ID列表。
        """
        if isinstance(query, str):  # 如果查询是字符串
            query_ids = self.tokenizer.encode(query)  # 分词为token ID
        else:  # 如果查询已是token ID
            query_ids = list(query)  # 转为列表

        items_list = [items] if isinstance(items, str) else items  # 统一为列表

        items_ids = []  # 项目token ID列表
        for item in items_list:  # 遍历每个项目
            if isinstance(item, str):  # 如果是字符串
                items_ids.append(self.tokenizer.encode(item))  # 分词
            else:  # 如果已是token ID
                items_ids.append(list(item))  # 转为列表

        return query_ids, items_ids  # 返回查询和项目的token ID

    def _process_multi_item_scoring_results(  # 处理多项目评分结果
        self,
        results: Any,  # 评分请求的结果
        items: List,  # 项目列表
        label_token_ids: Optional[List[int]],  # 标签token ID
        apply_softmax: bool,  # 是否应用softmax
        batch_request=None,  # 原始批请求
        return_pooled_hidden_states: bool = False,  # 是否返回池化隐藏状态
    ) -> ScoreResult:  # 返回评分结果
        """
        Process results from multi-item scoring request.
        处理多项目评分请求的结果。

        Extracts per-delimiter scores from whichever field the scheduler
        populated (input_token_ids_logprobs for generation models,
        embedding for classification models), then uniformly validates,
        skips the query-boundary delimiter, and normalizes.
        从调度器填充的字段中提取每个分隔符的评分（生成模型使用input_token_ids_logprobs，
        分类模型使用embedding），然后统一验证、跳过查询边界分隔符并归一化。

        Args:
            results: Results from generate_request  generate_request的结果
            items: List of items being scored  正在评分的项目列表
            label_token_ids: Token IDs to extract scores for  要提取评分的token ID
            apply_softmax: Whether to apply softmax normalization  是否应用softmax归一化
            batch_request: The original batch request containing input sequence  包含输入序列的原始批请求
            return_pooled_hidden_states: Whether to extract pooled hidden states  是否提取池化隐藏状态
                from the result and include them in the ScoreResult.  从结果中提取并包含在ScoreResult中。

        Returns:
            ScoreResult with per-item scores, prompt token count, and optional
            pooled_hidden_states (when return_pooled_hidden_states=True and the
            model populated the field).
            ScoreResult包含每项目评分、提示token计数和可选的pooled_hidden_states
            （当return_pooled_hidden_states=True且模型填充了该字段时）。
        """
        single_result = results[0] if isinstance(results, list) else results  # 获取单个结果
        meta_info = single_result.get("meta_info", {})  # 获取元信息
        num_items = len(items) if isinstance(items, list) else 1  # 项目数量
        expected_count = num_items + 1  # 预期的分隔符数量（项目数+1）
        request_id = meta_info.get("id", "<unknown>")  # 请求ID
        prompt_tokens = meta_info.get("prompt_tokens", 0)  # 提示token数量

        # Extract per-delimiter scores from whichever field has them  从有数据的字段中提取每个分隔符的评分
        input_logprobs = meta_info.get("input_token_ids_logprobs", [])  # 输入token的logprobs
        embedding = single_result.get("embedding")  # 嵌入/分类logits

        if input_logprobs:  # 如果有输入logprobs
            # Generation model: extract label-token logprobs at each delimiter  生成模型：在每个分隔符位置提取标签token的logprobs
            per_delimiter_scores = []  # 每个分隔符的评分
            for logprobs_data in input_logprobs:  # 遍历每个分隔符位置的logprobs
                logprobs = self._extract_logprobs_for_tokens(  # 提取标签token的logprobs
                    logprobs_data, label_token_ids
                )
                score_list = self._convert_logprobs_to_scores(  # 转换为评分列表
                    logprobs, label_token_ids, apply_softmax
                )
                per_delimiter_scores.append(score_list)  # 添加到列表
        elif embedding is not None:  # 如果有嵌入
            # Classification model: scores are directly in 2D embedding.  分类模型：评分直接在2D嵌入中
            if apply_softmax:  # 如果需要softmax
                scores_tensor = (
                    torch.tensor(embedding)  # 转为张量
                    if isinstance(embedding, list)  # 如果是列表
                    else embedding  # 已经是张量
                )
                scores_tensor = torch.nn.functional.softmax(scores_tensor, dim=-1)  # 应用softmax
                per_delimiter_scores = scores_tensor.tolist()  # 转为列表
            else:  # 不需要softmax
                per_delimiter_scores = (
                    embedding if isinstance(embedding, list) else embedding.tolist()  # 直接转为列表
                )
        else:  # 两者都没有
            raise RuntimeError(
                f"No scoring data found for multi-item scoring request {request_id}. "  # 多项目评分请求未找到评分数据
                "Expected either input_token_ids_logprobs or embedding."  # 期望input_token_ids_logprobs或embedding
            )

        # Validate delimiter count  验证分隔符数量
        if len(per_delimiter_scores) != expected_count:  # 如果数量不匹配
            raise RuntimeError(
                f"Expected {expected_count} delimiter entries for multi-item scoring "  # 多项目评分期望的分隔符条目数
                f"with {num_items} items, but got {len(per_delimiter_scores)}. "  # 实际获取的数量
                f"Request ID: {request_id}"  # 请求ID
            )

        # Skip the first delimiter (query-item boundary)  跳过第一个分隔符（查询-项目边界）
        scores = per_delimiter_scores[1:]  # 取除第一个以外的所有分隔符评分

        phs_list = None  # 池化隐藏状态列表
        if return_pooled_hidden_states:  # 如果需要返回池化隐藏状态
            raw_phs = single_result.get("pooled_hidden_state")  # 获取原始池化隐藏状态
            if raw_phs is not None and len(raw_phs) == expected_count:  # 如果存在且数量匹配
                phs_list = raw_phs[1:]  # 同样跳过第一个

        return ScoreResult(
            scores=scores,  # 评分
            prompt_tokens=prompt_tokens,  # 提示token数量
            pooled_hidden_states=phs_list,  # 池化隐藏状态
        )

    def _process_single_item_scoring_results(  # 处理单项目评分结果
        self,
        results: Any,  # 评分请求的结果
        label_token_ids: Optional[List[int]],  # 标签token ID
        apply_softmax: bool,  # 是否应用softmax
        return_pooled_hidden_states: bool = False,  # 是否返回池化隐藏状态
    ) -> ScoreResult:  # 返回评分结果
        """
        Process results from single-item scoring request.
        处理单项目评分请求的结果。

        For generation (CausalLM) models: reads output_token_ids_logprobs.
        对于生成（CausalLM）模型：读取output_token_ids_logprobs。
        For non-generation (SequenceClassification) models: reads the embedding field
        which contains pooled class logits from the classification head.
        对于非生成（SequenceClassification）模型：读取包含分类头池化类别logits的embedding字段。

        Args:
            results: Results from generate_request  generate_request的结果
            label_token_ids: Token IDs to extract scores for (generation models only)  要提取评分的token ID（仅生成模型）
            apply_softmax: Whether to apply softmax normalization  是否应用softmax归一化
            return_pooled_hidden_states: Whether to extract pooled hidden states  是否提取池化隐藏状态

        Returns:
            ScoreResult with per-item scores, prompt token count, and optional pooled_hidden_states.
            ScoreResult包含每项目评分、提示token计数和可选的pooled_hidden_states。
        """
        scores = []  # 评分列表
        phs_list = []  # 池化隐藏状态列表
        has_phs = False  # 是否有池化隐藏状态
        prompt_tokens = 0  # 提示token数量

        is_generation = self.is_generation  # 是否是生成模型
        if is_generation:  # 如果是生成模型
            for result in results:  # 遍历结果
                # For single-item scoring, logprobs are in output_token_ids_logprobs  单项目评分时，logprobs在output_token_ids_logprobs中
                output_logprobs = result["meta_info"].get(
                    "output_token_ids_logprobs", []  # 输出token的logprobs
                )
                prompt_tokens += result["meta_info"].get("prompt_tokens", 0)  # 累加提示token数

                if not output_logprobs or len(output_logprobs) == 0:  # 如果logprobs为空
                    raise RuntimeError(
                        f"output_logprobs is empty for request "  # output_logprobs为空
                        f"{result['meta_info'].get('id', '<unknown>')}."
                    )

                # Extract logprobs for the first (and only) position  提取第一个（也是唯一一个）位置的logprobs
                logprobs = self._extract_logprobs_for_tokens(
                    output_logprobs[0], label_token_ids  # 提取标签token的logprobs
                )
                score_list = self._convert_logprobs_to_scores(
                    logprobs, label_token_ids, apply_softmax  # 转换为评分
                )
                scores.append(score_list)  # 添加到列表
        else:  # 如果不是生成模型（分类模型）
            for result in results:  # 遍历结果
                embedding = result.get("embedding", None)  # 获取嵌入
                if embedding is None:  # 如果嵌入为空
                    raise ValueError("Embedding not found in the result.")  # 结果中未找到嵌入

                prompt_tokens += result.get("meta_info", {}).get("prompt_tokens", 0)  # 累加提示token数

                if apply_softmax:  # 如果需要softmax
                    embedding = torch.softmax(
                        torch.as_tensor(embedding), dim=-1  # 转为张量并应用softmax
                    ).tolist()

                # The classification head produces per-token logits, which the pooler reduces  分类头生成每token的logits，池化器将其
                # into a single vector per input. That vector is returned in the `.embeddings`  归约为每个输入的单个向量。该向量在`.embeddings`
                # field — not as semantic embeddings, but as pooled classification logits.  字段中返回——不是语义嵌入，而是池化分类logits。
                # The field name is reused for compatibility with the existing  字段名复用是为了兼容现有的
                # EmbeddingPoolerOutput API.  EmbeddingPoolerOutput API。
                scores.append(embedding)  # 添加到评分列表

                if return_pooled_hidden_states:  # 如果需要返回池化隐藏状态
                    phs = result.get("pooled_hidden_state")  # 获取池化隐藏状态
                    phs_list.append(phs)  # 添加到列表
                    if phs is not None:  # 如果不为None
                        has_phs = True  # 标记有池化隐藏状态

        return ScoreResult(
            scores=scores,  # 评分列表
            prompt_tokens=prompt_tokens,  # 提示token数量
            pooled_hidden_states=phs_list if has_phs else None,  # 池化隐藏状态
        )

    # ------------------------------------------------------------------
    # Embed override position resolution  嵌入覆盖位置解析
    # ------------------------------------------------------------------

    def _resolve_overrides_for_sequence(  # 解析token序列中的嵌入覆盖
        self,
        token_ids: List[int],  # 要扫描的token序列
        embeds: Optional[List[torch.Tensor]],  # 放置在占位符位置的嵌入张量（None表示跳过）
        embed_override_token_id: int,  # 占位符token ID
        position_offset: int = 0,  # 位置偏移（用于绝对坐标）
        label: str = "input",  # 错误消息标签
    ) -> Tuple[List[torch.Tensor], List[int]]:  # 返回（嵌入列表，位置列表）
        """Scan token_ids for placeholder occurrences and pair with embeddings.
        扫描token_ids中的占位符出现位置并与嵌入配对。

        Args:
            token_ids: The token sequence to scan.  要扫描的token序列。
            embeds: Embedding tensors to place at placeholder positions (None = skip).
                放置在占位符位置的嵌入张量（None = 跳过）。
            embed_override_token_id: The placeholder token ID.  占位符token ID。
            position_offset: Added to each found position (for absolute coordinates).
                添加到每个找到的位置（用于绝对坐标）。
            label: Label for error messages (e.g. "query", "items[2]").
                错误消息的标签（如 "query"、"items[2]"）。

        Returns:
            (embeds, positions) lists. Empty lists if embeds is None.
                （embeds, positions）列表。如果embeds为None则返回空列表。
        """
        if embeds is None:  # 如果没有嵌入覆盖
            return [], []  # 返回空列表
        positions = [
            idx + position_offset  # 加上位置偏移
            for idx, tok in enumerate(token_ids)  # 遍历token序列
            if tok == embed_override_token_id  # 找到占位符token
        ]
        if len(positions) != len(embeds):  # 如果位置数与嵌入数不匹配
            raise ValueError(
                f"{label} contains {len(positions)} occurrences of "  # 包含的占位符数量
                f"embed_override_token_id={embed_override_token_id}, "  # embed_override_token_id
                f"but {len(embeds)} override embeddings were provided."  # 但提供的覆盖嵌入数量
            )
        return embeds, positions  # 返回嵌入和位置列表

    def _resolve_embed_overrides_for_request(  # 解析单个查询+项目对的嵌入覆盖
        self,
        query: List[int],  # 查询token ID
        item: List[int],  # 项目token ID
        embed_override_token_id: int,  # 占位符token ID
        query_embed_overrides: Optional[List[torch.Tensor]],  # 查询嵌入覆盖
        item_embeds: Optional[List[torch.Tensor]],  # 项目嵌入覆盖
        item_position_offset: int,  # 项目位置偏移
        item_label: str,  # 项目标签
    ) -> Optional[PositionalEmbeds]:  # 返回PositionalEmbeds或None
        """Resolve embed overrides for a single query+item pair.
        解析单个查询+项目对的嵌入覆盖。

        Returns PositionalEmbeds if any overrides exist, None otherwise.
        如果存在覆盖则返回PositionalEmbeds，否则返回None。
        """
        q_embeds, q_positions = self._resolve_overrides_for_sequence(  # 解析查询嵌入覆盖
            query,
            query_embed_overrides,
            embed_override_token_id,
            position_offset=0,  # 查询位置从0开始
            label="query",
        )
        i_embeds, i_positions = self._resolve_overrides_for_sequence(  # 解析项目嵌入覆盖
            item,
            item_embeds,
            embed_override_token_id,
            position_offset=item_position_offset,  # 使用项目位置偏移
            label=item_label,
        )
        all_embeds = q_embeds + i_embeds  # 合并嵌入
        all_positions = q_positions + i_positions  # 合并位置
        if not all_embeds:  # 如果没有嵌入
            return None  # 返回None
        return PositionalEmbeds(embeds=all_embeds, positions=all_positions)  # 返回位置嵌入对象

    # ------------------------------------------------------------------
    # Input preparation (tokenization + input_ids construction)  输入准备（分词 + input_ids构建）
    # ------------------------------------------------------------------

    def _build_token_id_inputs(  # 构建token ID输入和解析嵌入覆盖
        self,
        query: List[int],  # 查询token ID
        items: List[List[int]],  # 项目token ID列表
        item_first: bool,  # 是否项目在前
        use_multi_item_scoring: bool,  # 是否使用多项目评分
        embed_override_token_id: Optional[int],  # 占位符token ID
        query_embed_overrides: Optional[List[torch.Tensor]],  # 查询嵌入覆盖
        item_embed_overrides: Optional[List[Optional[List[torch.Tensor]]]],  # 项目嵌入覆盖
    ) -> Tuple[None, List[List[int]], Optional[list], Optional[List[int]]]:  # 返回（text_prompts, input_ids, positional_embed_overrides, delimiter_indices）
        """Build input_ids and resolve embed overrides for token-ID inputs.
        为token ID输入构建input_ids并解析嵌入覆盖。

        Works identically for multi-item-scoring and single-item modes — the only difference is
        how input_ids are assembled and what position offset each item gets.
        多项目评分和单项目模式工作方式相同——唯一的区别是input_ids的组装方式和每个项目的位置偏移。

        Returns:
            (text_prompts, input_ids, positional_embed_overrides, delimiter_indices)
                （文本提示，输入ID，位置嵌入覆盖，分隔符索引）
        """
        # Both query and items are token IDs  查询和项目都是token ID
        has_embeds = (
            query_embed_overrides is not None or item_embed_overrides is not None  # 是否有任何嵌入覆盖
        )

        # Query placeholder positions are invariant across items — resolve once.  查询占位符位置在所有项目中不变——只需解析一次。
        # (No-op returning ([], []) if has_embeds is False or query_embed_overrides is None.)  （如果has_embeds为False或query_embed_overrides为None则返回([], [])）
        q_embeds, q_positions = self._resolve_overrides_for_sequence(
            query,
            query_embed_overrides,
            embed_override_token_id,
            position_offset=0,
            label="query",
        )

        if use_multi_item_scoring:  # 如果使用多项目评分
            # Multi-item scoring: concatenate with placeholder delimiter token.  多项目评分：使用占位分隔符token拼接。
            # Positions are derived from item lengths (delimiter_indices), not  位置从项目长度（delimiter_indices）派生，而非
            # by scanning for this token — it exists only for FlashInfer compat.  通过扫描此token——它仅为FlashInfer兼容性而存在。
            delimiter_token_id = MIS_DELIMITER_TOKEN_ID  # 多项目分隔符token ID
            combined_input_ids, delimiter_indices = (
                self._build_multi_item_token_sequence(query, items, delimiter_token_id)  # 构建合并序列
            )
            input_ids = [combined_input_ids]  # 包装为列表

            if not has_embeds:  # 如果没有嵌入覆盖
                return None, input_ids, None, delimiter_indices  # 直接返回

            # Resolve embed overrides across the combined multi-item-scoring sequence.  解析合并多项目评分序列中的嵌入覆盖
            all_embeds: List[torch.Tensor] = list(q_embeds)  # 复制查询嵌入
            all_positions: List[int] = list(q_positions)  # 复制查询位置
            current_offset = len(query) + 1  # +1 for first delimiter  +1为第一个分隔符
            for i, item in enumerate(items):  # 遍历每个项目
                item_embs = item_embed_overrides[i] if item_embed_overrides else None  # 获取项目嵌入
                i_embeds, i_positions = self._resolve_overrides_for_sequence(
                    item,
                    item_embs,
                    embed_override_token_id,
                    position_offset=current_offset,  # 使用当前偏移
                    label=f"items[{i}]",
                )
                all_embeds.extend(i_embeds)  # 扩展嵌入列表
                all_positions.extend(i_positions)  # 扩展位置列表
                current_offset += len(item) + 1  # +1 for delimiter  +1为分隔符

            if all_embeds:  # 如果有嵌入
                # PositionalEmbeds.__post_init__ does the single torch.cat stack.  PositionalEmbeds.__post_init__执行单次torch.cat堆叠。
                positional_embed_overrides = [
                    PositionalEmbeds(embeds=all_embeds, positions=all_positions)  # 创建位置嵌入
                ]
            else:  # 没有嵌入
                positional_embed_overrides = None  # 设为None
            return None, input_ids, positional_embed_overrides, delimiter_indices  # 返回结果

        else:  # 单项目评分模式
            # Single-item scoring: process each item separately  单项目评分：分别处理每个项目
            if item_first:  # 项目在前
                input_ids = [item + query for item in items]  # 拼接为item+query
            else:  # 查询在前
                input_ids = [query + item for item in items]  # 拼接为query+item

            if not has_embeds:  # 如果没有嵌入覆盖
                return None, input_ids, None, None  # 直接返回

            positional_embed_overrides = []  # 位置嵌入覆盖列表
            any_overrides = False  # 是否有任何覆盖
            for i, item in enumerate(items):  # 遍历每个项目
                item_embs = item_embed_overrides[i] if item_embed_overrides else None  # 获取项目嵌入
                i_embeds, i_positions = self._resolve_overrides_for_sequence(
                    item,
                    item_embs,
                    embed_override_token_id,
                    position_offset=len(query),  # 偏移为查询长度
                    label=f"items[{i}]",
                )
                combined_embeds = q_embeds + i_embeds  # 合并查询和项目嵌入
                if combined_embeds:  # 如果有嵌入
                    positional_embed_overrides.append(
                        PositionalEmbeds(
                            embeds=combined_embeds,  # 合并的嵌入
                            positions=q_positions + i_positions,  # 合并的位置
                        )
                    )
                    any_overrides = True  # 标记有覆盖
                else:  # 没有嵌入
                    positional_embed_overrides.append(None)  # 添加None

            return (
                None,  # 无文本提示
                input_ids,  # 输入ID
                positional_embed_overrides if any_overrides else None,  # 位置嵌入覆盖
                None,  # 无分隔符索引
            )

    # ------------------------------------------------------------------
    # Main entry point  主入口点
    # ------------------------------------------------------------------

    async def score_request(  # 评分请求主入口
        self,
        query: Optional[Union[str, List[int]]] = None,  # 查询文本或token ID
        items: Optional[Union[str, List[str], List[List[int]]]] = None,  # 项目文本或token ID
        label_token_ids: Optional[List[int]] = None,  # 标签token ID
        apply_softmax: bool = False,  # 是否应用softmax
        item_first: bool = False,  # 是否项目在前
        embed_override_token_id: Optional[int] = None,  # 占位符token ID
        query_embed_overrides: Optional[List[torch.Tensor]] = None,  # 查询嵌入覆盖
        item_embed_overrides: Optional[List[Optional[List[torch.Tensor]]]] = None,  # 项目嵌入覆盖
        request: Optional[Any] = None,  # 可选的FastAPI请求对象
        return_pooled_hidden_states: bool = False,  # 是否返回池化隐藏状态
    ) -> ScoreResult:  # 返回评分结果
        """
        Score the probability of specified token IDs appearing after the given (query + item) pair.
        评分指定token ID出现在给定（query + item）对之后的概率。

        This method supports two scoring approaches:
        本方法支持两种评分方式：
        1. Single-Item scoring (default): Process each query+item pair independently
           单项目评分（默认）：独立处理每个query+item对
        2. Multi-Item scoring: When --enable-mis is set, combine query and
           multiple items into a single sequence using delimiter for efficient processing.
           多项目评分：当设置--enable-mis时，使用分隔符将查询和多个项目合并为单个序列以提高处理效率。
           Note: item_first parameter is ignored in multi-item scoring mode since it uses
           a fixed format: query<delimiter>item1<delimiter>item2<delimiter>item3<delimiter>
           注意：item_first参数在多项目评分模式下被忽略，因为它使用固定格式

           Multi-item scoring works with both text and pre-tokenized inputs:
           多项目评分支持文本和预分词输入：
           - Text: query<delimiter_text>item1<delimiter_text>item2<delimiter_text>item3<delimiter_text>
           - Tokens: query<delimiter_token_id>item1<delimiter_token_id>item2<delimiter_token_id>item3<delimiter_token_id>

        Supports two model types:
        支持两种模型类型：
        - Generation (CausalLM): Requires label_token_ids; returns logprob-based scores.
          生成模型（CausalLM）：需要label_token_ids；返回基于logprob的评分。
        - SequenceClassification: label_token_ids is optional; returns pooled class logits.
          序列分类模型：label_token_ids可选；返回池化类别logits。

        Args:
            query: The query text or pre-tokenized query token IDs  查询文本或预分词的查询token ID
            items: The item text(s) or pre-tokenized item token IDs  项目文本或预分词的项目token ID
            label_token_ids: List of token IDs to compute probabilities for  要计算概率的token ID列表
            apply_softmax: Whether to normalize probabilities using softmax  是否使用softmax归一化概率
            item_first: If True, prepend items to query. Ignored for multi-item scoring.
                如果为True，将项目放在查询前面。在多项目评分模式下被忽略。
            embed_override_token_id: Placeholder token ID for embedding override positions.
                嵌入覆盖位置的占位符token ID。
            query_embed_overrides: Embedding vectors replacing placeholder tokens in query.
                替换查询中占位符token的嵌入向量。
            item_embed_overrides: Per-item embedding vectors replacing placeholder tokens in items.
                每项目替换项目中占位符token的嵌入向量。
            request: Optional FastAPI request object  可选的FastAPI请求对象
            return_pooled_hidden_states: Whether to include the raw pooled transformer
                hidden states (before the task-specific head) in the result. Only
                supported for non-generation models (SequenceClassification,
                RewardModel). Raises ValueError for CausalLM models.
                是否在结果中包含原始池化transformer隐藏状态（任务特定头之前）。
                仅非生成模型（SequenceClassification、RewardModel）支持。
                CausalLM模型会抛出ValueError。

        Returns:
            ScoreResult with:
                scores: List of score lists, one per item.  评分列表，每个项目一个。
                prompt_tokens: The number of prompt tokens processed.  处理的提示token数量。
                pooled_hidden_states: Per-item CPU tensors when
                    return_pooled_hidden_states=True and the model supports it;
                    None otherwise.
                    当return_pooled_hidden_states=True且模型支持时，每项目的CPU张量；
                    否则为None。
        """
        is_generation = self.is_generation  # 是否是生成模型

        if is_generation and label_token_ids is None:  # 生成模型必须提供label_token_ids
            raise ValueError(
                "label_token_ids is required for generation (CausalLM) models."  # 生成模型需要label_token_ids
            )
        if items is None:  # 必须提供items
            raise ValueError("items must be provided")  # 必须提供items
        if not items:  # items为空列表
            return ScoreResult(scores=[], prompt_tokens=0)  # 返回空结果

        has_embeds = (
            query_embed_overrides is not None or item_embed_overrides is not None  # 是否有嵌入覆盖
        )
        if has_embeds and embed_override_token_id is None:  # 有嵌入覆盖但没提供占位符ID
            raise ValueError(
                "embed_override_token_id is required when query_embed_overrides "  # 提供query_embed_overrides
                "or item_embed_overrides are supplied."  # 或item_embed_overrides时需要embed_override_token_id
            )
        if item_first and has_embeds:  # 项目在前模式不支持嵌入覆盖
            raise ValueError("item_first is not supported when embeddings are supplied")  # 提供嵌入时不支持item_first
        if item_embed_overrides is not None and len(item_embed_overrides) != len(items):  # 嵌入覆盖数量不匹配
            raise ValueError(
                f"item_embed_overrides length ({len(item_embed_overrides)}) "  # item_embed_overrides长度
                f"must match items length ({len(items)})."  # 必须与items长度匹配
            )
        if self.tokenizer is not None and label_token_ids is not None:  # 如果分词器存在且有标签token
            vocab_size = self.tokenizer.vocab_size  # 获取词表大小
            for token_id in label_token_ids:  # 遍历标签token
                if token_id >= vocab_size:  # 超出词表范围
                    raise ValueError(
                        f"Token ID {token_id} is out of vocabulary (vocab size: {vocab_size})"  # token ID超出词表范围
                    )

        # Check if multi-item scoring is enabled  检查是否启用多项目评分
        use_multi_item_scoring = self.server_args.enable_mis  # 从服务器参数获取

        input_ids = None  # 输入token ID
        text_prompts = None  # 文本提示
        positional_embed_overrides = None  # 位置嵌入覆盖
        delimiter_indices = None  # 分隔符索引

        use_text_prompts = isinstance(query, str) and not has_embeds  # 是否使用文本提示

        if use_text_prompts:  # 使用文本提示
            # Both query and items are text  查询和项目都是文本
            items_list = [items] if isinstance(items, str) else items  # 统一为列表
            if use_multi_item_scoring:  # 多项目评分
                # Tokenize separately, then combine at token level with placeholder  分别分词，然后在token级别用占位符合并
                # delimiter. Positions come from item lengths (delimiter_indices),  位置从项目长度（delimiter_indices）派生，
                # not from scanning for this token — it's for FlashInfer compat only.  而非扫描此token——它仅为FlashInfer兼容性而存在。
                delimiter_token_id = MIS_DELIMITER_TOKEN_ID  # 分隔符token ID
                query_ids, items_ids = self._batch_tokenize_query_and_items(
                    query, items_list  # 批量分词
                )
                combined_input_ids, delimiter_indices = (
                    self._build_multi_item_token_sequence(
                        query_ids, items_ids, delimiter_token_id  # 构建合并序列
                    )
                )
                input_ids = [combined_input_ids]  # 包装为列表
            else:  # 单项目评分
                # Single-item scoring: create separate prompts for each item  单项目评分：为每个项目创建独立的提示
                if item_first:  # 项目在前
                    text_prompts = [f"{item}{query}" for item in items_list]  # 项目+查询
                else:  # 查询在前
                    text_prompts = [f"{query}{item}" for item in items_list]  # 查询+项目

        elif (
            isinstance(query, list)
            and isinstance(items, list)
            and items
            and isinstance(items[0], list)
        ):  # 查询和项目都是token ID
            # Both query and items are token IDs — tokenize text inputs if needed for embed overrides  查询和项目都是token ID——如果需要嵌入覆盖则分词文本输入
            query_ids, items_ids = query, items  # 直接使用
            _, input_ids, positional_embed_overrides, delimiter_indices = (
                self._build_token_id_inputs(
                    query_ids,
                    items_ids,
                    item_first,
                    use_multi_item_scoring,
                    embed_override_token_id,
                    query_embed_overrides,
                    item_embed_overrides,
                )
            )
        elif has_embeds:  # 有嵌入覆盖的文本输入
            # Text inputs with embed overrides — need to tokenize first to resolve positions  带嵌入覆盖的文本输入——需要先分词以解析位置
            query_ids, items_ids = self._batch_tokenize_query_and_items(query, items)  # 批量分词
            _, input_ids, positional_embed_overrides, delimiter_indices = (
                self._build_token_id_inputs(
                    query_ids,
                    items_ids,
                    item_first,
                    use_multi_item_scoring,
                    embed_override_token_id,
                    query_embed_overrides,
                    item_embed_overrides,
                )
            )
        else:  # 无效的输入组合
            raise ValueError(
                "Invalid combination of query/items types for score_request."  # score_request的query/items类型组合无效
            )

        if return_pooled_hidden_states:  # 如果需要返回池化隐藏状态
            if is_generation:  # 生成模型不支持
                raise ValueError(
                    "return_pooled_hidden_states is not supported for CausalLM models. "  # CausalLM模型不支持return_pooled_hidden_states
                    "It requires a model with a task-specific head "  "需要带有任务特定头的模型"
                    "(e.g. SequenceClassification or RewardModel)."  "（如 SequenceClassification 或 RewardModel）"
                )
            model_config = self.model_config  # 获取模型配置
            if model_config is not None:  # 如果配置存在
                archs = getattr(model_config.hf_config, "architectures", []) or []  # 获取架构列表
                if is_cross_encoding_pooler_model(archs):  # 如果是交叉编码池化模型
                    raise ValueError(
                        f"return_pooled_hidden_states is not supported for "  # 不支持return_pooled_hidden_states
                        f"{archs[0]}. This model uses CrossEncodingPooler which "  # 此模型使用CrossEncodingPooler
                        f"does not expose pre-head hidden states."  # 不暴露头之前的隐藏状态
                    )

        # Create the appropriate request type  创建合适的请求类型
        mis_delimiter_indices = [delimiter_indices] if use_multi_item_scoring else None  # 多项目分隔符索引
        if is_generation:  # 生成模型
            batch_request = GenerateReqInput(
                text=text_prompts,  # 文本提示
                input_ids=input_ids,  # 输入ID
                token_ids_logprob=label_token_ids,  # 要记录logprob的token ID
                return_logprob=True,  # 返回logprob
                # Set logprob_start_len=0 for multi-item scoring since we want logprobs at all delimiter positions  多项目评分设置logprob_start_len=0，因为需要所有分隔符位置的logprobs
                logprob_start_len=0 if use_multi_item_scoring else -1,  # 多项目为0，单项目为-1
                stream=False,  # 非流式
                sampling_params={"max_new_tokens": 0},  # 不生成新token
                positional_embed_overrides=positional_embed_overrides,  # 位置嵌入覆盖
                multi_item_delimiter_indices=mis_delimiter_indices,  # 多项目分隔符索引
            )
        else:  # 非生成模型
            batch_request = EmbeddingReqInput(
                text=text_prompts,  # 文本提示
                input_ids=input_ids,  # 输入ID
                positional_embed_overrides=positional_embed_overrides,  # 位置嵌入覆盖
                return_pooled_hidden_states=return_pooled_hidden_states,  # 是否返回池化隐藏状态
                multi_item_delimiter_indices=mis_delimiter_indices,  # 多项目分隔符索引
            )

        results = await self.generate_request(batch_request, request).__anext__()  # 生成请求并获取结果

        if use_multi_item_scoring:  # 多项目评分
            # Multi-item scoring: extract scores from input_token_ids_logprobs or embedding  多项目评分：从input_token_ids_logprobs或embedding提取评分
            return self._process_multi_item_scoring_results(
                results,
                items,
                label_token_ids,
                apply_softmax,
                batch_request,
                return_pooled_hidden_states,
            )
        else:  # 单项目评分
            # Single-item scoring: process each result separately  单项目评分：分别处理每个结果
            return self._process_single_item_scoring_results(
                results, label_token_ids, apply_softmax, return_pooled_hidden_states
            )

    def _convert_logprobs_to_scores(  # 将logprobs字典转换为有序评分列表
        self,
        logprobs: Dict[int, float],  # token_id到logprob的映射
        label_token_ids: List[int],  # 期望顺序的token ID列表
        apply_softmax: bool,  # 是否应用softmax归一化
    ) -> List[float]:  # 返回与label_token_ids同顺序的评分列表
        """
        Convert logprobs dictionary to ordered score list.
        将logprobs字典转换为有序评分列表。

        Args:
            logprobs: Dictionary mapping token_id to logprob  token_id到logprob的映射字典
            label_token_ids: Token IDs in desired order  期望顺序的token ID
            apply_softmax: Whether to apply softmax normalization  是否应用softmax归一化

        Returns:
            List of scores in the same order as label_token_ids  与label_token_ids同顺序的评分列表
        """
        score_list = [
            logprobs.get(token_id, float("-inf")) for token_id in label_token_ids  # 获取每个标签token的logprob，不存在则为负无穷
        ]

        if apply_softmax:  # 如果应用softmax
            score_list = torch.softmax(torch.tensor(score_list), dim=0).tolist()  # 转为张量并softmax
        else:
            # Convert logprobs to probabilities if not using softmax  如果不使用softmax，将logprobs转换为概率
            score_list = [
                math.exp(x) if x != float("-inf") else 0.0 for x in score_list  # exp(logprob)，负无穷转为0
            ]

        return score_list  # 返回评分列表

    def _extract_logprobs_for_tokens(  # 从logprobs数据中提取指定token ID的logprobs
        self, logprobs_data: List, label_token_ids: List[int]  # logprobs数据列表，目标token ID
    ) -> Dict[int, float]:  # 返回token_id到logprob的映射
        """
        Extract logprobs for specified token IDs from logprobs data.
        从logprobs数据中提取指定token ID的logprobs。

        Args:
            logprobs_data: List of (logprob, token_id, text) tuples  (logprob, token_id, text)元组列表
            label_token_ids: Token IDs to extract logprobs for  要提取logprobs的token ID

        Returns:
            Dictionary mapping token_id to logprob  token_id到logprob的映射字典
        """
        logprobs = {}  # 初始化字典
        if logprobs_data:  # 如果数据不为空
            for logprob, token_id, _ in logprobs_data:  # 遍历每个(logprob, token_id, text)
                if token_id in label_token_ids:  # 如果token_id在目标列表中
                    logprobs[token_id] = logprob  # 添加到字典
        return logprobs  # 返回字典
