# N-gram语料库模块，封装C++后端实现N-gram推测解码。
# 提供了NgramCorpus类，支持批量插入token、状态化匹配查询、
# 外部语料库加载/移除、以及调试结果可视化等功能。
# 基于后缀自动机（SAM）实现高效的前缀匹配。

# -*- coding: utf-8 -*-

import logging  # 导入日志模块
from collections.abc import Iterable, Sequence  # 导入集合抽象基类
from typing import Dict, List, Tuple  # 导入类型提示

import numpy as np  # 导入NumPy

from sglang.jit_kernel.ngram_corpus import get_ngram_corpus_cls  # 导入N-gram语料库C++类

logger = logging.getLogger(__name__)  # 获取日志记录器


class NgramCorpus:
    """N-gram语料库类，封装C++后端实现N-gram推测解码的语料管理。"""

    def __init__(
        self,
        max_trie_depth=18,  # 最大trie深度
        min_bfs_breadth=1,  # 最小BFS广度
        max_bfs_breadth=8,  # 最大BFS广度
        draft_token_num=8,  # 草稿token数
        match_type="BFS",  # 匹配类型
        capacity=1000000,  # 容量
        external_sam_budget=0,  # 外部SAM预算
        external_corpus_max_tokens=10000000,  # 外部语料库最大token数
    ) -> None:
        """初始化N-gram语料库，创建C++后端对象并设置参数。"""
        cls = get_ngram_corpus_cls()  # 获取C++后端类
        self._obj = cls(  # 创建C++后端对象
            capacity=capacity,  # 容量
            max_trie_depth=max_trie_depth,  # 最大trie深度
            min_bfs_breadth=min_bfs_breadth,  # 最小BFS广度
            max_bfs_breadth=max_bfs_breadth,  # 最大BFS广度
            draft_token_num=draft_token_num,  # 草稿token数
            match_type=match_type,  # 匹配类型
            external_sam_budget=external_sam_budget,  # 外部SAM预算
            external_corpus_max_tokens=external_corpus_max_tokens,  # 外部语料库最大token数
        )
        self.draft_token_num = draft_token_num  # 保存草稿token数
        self.external_corpus_max_tokens = external_corpus_max_tokens  # 保存外部语料库最大token数
        self._req_id_to_state_id: Dict[str, int] = {}  # 请求ID到状态ID的映射
        self._next_state_id: int = 0  # 下一个状态ID
        self._corpus_token_counts: Dict[str, int] = {}  # 语料库token计数
        self._total_loaded_tokens: int = 0  # 总加载token数

    def _get_state_id(self, req_id: str) -> int:
        """获取或创建请求对应的状态ID。"""
        sid = self._req_id_to_state_id.get(req_id)  # 查找现有状态ID
        if sid is None:  # 如果不存在
            sid = self._next_state_id  # 分配新ID
            self._next_state_id += 1  # 递增ID计数器
            self._req_id_to_state_id[req_id] = sid  # 保存映射
        return sid  # 返回状态ID

    def batch_put(self, batch_tokens: List[List[int]]):
        """批量插入token到语料库。"""
        self._obj.insert(batch_tokens)  # 调用C++后端插入

    def synchronize(self):
        """同步语料库，确保所有插入操作完成。"""
        self._obj.synchronize()  # type: ignore  # 调用C++后端同步

    @property
    def remaining_token_budget(self) -> int:
        """获取剩余的token预算。"""
        return self.external_corpus_max_tokens - self._total_loaded_tokens  # 最大token数减去已加载数

    def load_external_corpus_named(
        self, corpus_id: str, chunks: Iterable[Sequence[int]]  # 语料库ID, # token块迭代器
    ) -> int:
        """加载命名的外部语料库，返回加载的token数。"""
        if corpus_id in self._corpus_token_counts:  # 如果语料库已存在
            raise ValueError(  # 抛出异常
                f"External corpus '{corpus_id}' already exists. Remove it before "
                f"adding a new corpus with the same id."
            )
        # Note(kpham-sgl): remaining_token_budget is stale (e.g if there are removes
        # during the load), which makes the budget more conservative than it should be.
        # This is acceptable because otherwise load_external_corpus_named would need to check the budget after each chunk,
        # which would be inefficient.
        _, loaded_token_count = self._obj.load_external_corpus_named(  # 调用C++后端加载
            corpus_id, chunks, self.remaining_token_budget
        )
        return loaded_token_count  # 返回加载的token数

    # Commit corpus bookkeeping after successful load. Call only at background thread join.
    # (or after synchronous load_external_corpus_named returns)
    def commit_external_corpus_load(
        self, corpus_id: str, loaded_token_count: int  # 语料库ID, # 加载的token数
    ) -> None:
        """提交外部语料库加载，更新账本信息。"""
        self._corpus_token_counts[corpus_id] = loaded_token_count  # 记录token数
        self._total_loaded_tokens += loaded_token_count  # 累加总token数

    def remove_external_corpus(self, corpus_id: str) -> None:
        """移除命名的外部语料库。"""
        self._obj.remove_corpus(corpus_id)  # 调用C++后端移除
        old_count = self._corpus_token_counts.pop(corpus_id, 0)  # 弹出token计数
        self._total_loaded_tokens -= old_count  # 减少总token数

    def list_external_corpora(self) -> Dict[str, int]:
        """列出所有外部语料库及其token数。"""
        return self._obj.list_corpora()  # 调用C++后端列出

    def reset(self):
        """重置语料库，清除所有状态和数据。"""
        self._obj.reset()  # type: ignore  # 调用C++后端重置
        self._req_id_to_state_id.clear()  # 清除请求ID映射
        self._next_state_id = 0  # 重置状态ID计数器

    def batch_get(
        self,
        req_ids: List[str],  # 请求ID列表
        batch_tokens: List[List[int]],  # 批量token列表
        total_lens: List[int],  # 总长度列表
    ) -> Tuple[np.ndarray, np.ndarray]:
        """批量获取N-gram匹配结果，返回草稿token ID和掩码。"""
        state_ids = [self._get_state_id(rid) for rid in req_ids]  # 获取每个请求的状态ID
        return self._obj.match_stateful(state_ids, batch_tokens, total_lens)  # 调用C++后端匹配

    def erase_match_state(self, req_ids: List[str]):
        """擦除指定请求的匹配状态。"""
        state_ids = []  # 状态ID列表
        for rid in req_ids:  # 遍历请求ID
            sid = self._req_id_to_state_id.pop(rid, None)  # 弹出状态ID
            if sid is not None:  # 如果存在
                state_ids.append(sid)  # 添加到列表
        if state_ids:  # 如果有状态ID
            self._obj.erase_states(state_ids)  # 调用C++后端擦除

    def leaf_paths_from_mask(
        self, tokens: List[int], tree_mask: List[List[int]]  # token列表, # 树掩码
    ) -> List[List[int]]:
        """从二叉树掩码中找出所有叶路径（即不是其他路径前缀的路径）。"""
        """
        Find all leaf paths according to the binary tree_mask (i.e., paths that are not prefixes of any other path).

        Args:
            mask   : List[List[int]]   # nxn binary matrix
            tokens : List[int]         # token list corresponding to columns

        Returns:
            List[List[int]]            # token lists of only the leaf paths, preserving their order of appearance
        """

        row_sets = [  # 计算每行的1索引集合
            (i, {idx for idx, v in enumerate(row) if v == 1})
            for i, row in enumerate(tree_mask)
        ]
        leaf_sets = []  # 叶节点集合
        leaf_rows = []  # 叶节点行号

        for i, cur_set in reversed(row_sets):  # 反向遍历
            if any(cur_set <= kept for kept in leaf_sets):  # 如果是某叶的子集（非叶节点）
                continue  # 跳过
            leaf_sets.append(cur_set)  # 添加到叶集合
            leaf_rows.append(i)  # 添加到叶行号

        leaf_rows.reverse()  # 反转行号列表
        result = []  # 结果列表
        for r in leaf_rows:  # 遍历叶行号
            path = [tokens[col] for col in range(len(tokens)) if tree_mask[r][col] == 1]  # 构建路径
            result.append(path)  # 添加到结果

        return result  # 返回叶路径列表

    def debug_result(
        self, decoding_ids: np.ndarray, decoding_masks: np.ndarray, tokenizer=None  # 解码ID, # 解码掩码, # 分词器，可选
    ):
        """调试结果，打印解码ID和掩码的详细信息。"""
        decoding_ids = decoding_ids.reshape(-1, self.draft_token_num)  # 重塑解码ID
        decoding_masks = decoding_masks.reshape(  # 重塑解码掩码
            -1, self.draft_token_num, self.draft_token_num
        )
        logger.info(f"\n{decoding_ids=}\n{decoding_masks=}")  # 打印原始数据
        for i in range(decoding_ids.shape[0]):  # 遍历每个结果
            leaf_paths = self.leaf_paths_from_mask(  # 获取叶路径
                decoding_ids[i].tolist(), decoding_masks[i].tolist()
            )
            if tokenizer is None:  # 如果没有分词器
                logger.info(f"draft path {i}: {leaf_paths}")  # 打印token ID
            else:
                logger.info(f"result {i}:")  # 打印结果编号
                for leaf_path in leaf_paths:  # 遍历叶路径
                    logger.info(  # 打印解码后的路径
                        f"draft path {i}: {leaf_path} -> {tokenizer.decode(leaf_path, ensure_ascii=False)}"
                    )


# main function
if __name__ == "__main__":  # 主函数入口
    format = f"%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"  # 日志格式
    logging.basicConfig(  # 配置日志
        level=logging.DEBUG,  # 调试级别
        format=format,  # 格式
        datefmt="%Y-%m-%d %H:%M:%S",  # 日期格式
        force=True,  # 强制重新配置
    )

    token_ids = [  # 测试token ID
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 44, 55, 66, 77, 88, 99, 100],
    ]
    corpus = NgramCorpus(max_trie_depth=12, draft_token_num=8)  # 创建语料库
    corpus.batch_put(token_ids)  # 批量插入token

    corpus.synchronize()  # 同步
    queries = [[1, 2, 3], [3, 44], [3, 6, 999]]  # 测试查询
    decoding_ids, decoding_masks = corpus.batch_get(  # 批量查询
        req_ids=[f"query-{i}" for i in range(len(queries))],
        batch_tokens=queries,
        total_lens=[len(q) for q in queries],
    )

    corpus.debug_result(decoding_ids, decoding_masks)  # 调试结果
