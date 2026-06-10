# 本文件用于对分词器(tokenizer)进行补丁(patch)操作，主要为 Kimi TikToken 分词器
# 提供特殊 token 的缓存机制，避免重复计算 all_special_tokens 和 all_special_ids 属性，
# 从而提升分词器在高并发推理场景下的性能。

import logging

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def patch_tokenizer(tokenizer):
    """对分词器应用补丁。如果环境变量 SGLANG_PATCH_TOKENIZER 未启用，则直接返回原分词器；
    如果是 Kimi TikToken 分词器，则应用特殊 token 缓存补丁。"""
    if not envs.SGLANG_PATCH_TOKENIZER.get():
        return tokenizer

    if _is_kimi_tiktoken_tokenizer(tokenizer):
        logger.info(
            f"Applying special tokens cache patch for Kimi tokenizer: {type(tokenizer)}"
        )
        return _SpecialTokensCachePatcher.patch(tokenizer)

    return tokenizer


def unpatch_tokenizer(tokenizer):
    """移除分词器上的特殊 token 缓存补丁，恢复原始属性。"""
    return _SpecialTokensCachePatcher.unpatch(tokenizer)


def _is_kimi_tiktoken_tokenizer(tokenizer):
    """判断给定分词器是否为 Kimi TikToken 分词器。
    通过检查类名为 TikTokenTokenizer 且模块名包含 tokenization_kimi 来判断。"""
    cls = type(tokenizer)
    class_name = cls.__name__
    module_name = cls.__module__ or ""
    return class_name == "TikTokenTokenizer" and "tokenization_kimi" in module_name


def decode_without_hf_kwargs(tokenizer, token_ids, skip_special_tokens):
    """在不传递 HuggingFace 额外参数的情况下解码 token ID 列表。
    如果 skip_special_tokens 为 True，则过滤掉所有特殊 token ID 后再解码。"""
    if skip_special_tokens:
        # 获取所有特殊 token ID 的集合，用于过滤
        special_ids = getattr(tokenizer, "all_special_ids_set", None)
        if special_ids is None:
            special_ids = set(tokenizer.all_special_ids)
        # 过滤掉特殊 token ID
        token_ids = [tid for tid in token_ids if tid not in special_ids]
    return tokenizer.decode(token_ids)


class _SpecialTokensCachePatcher:
    """特殊 token 缓存补丁类，用于将分词器的 all_special_tokens 和 all_special_ids
    属性从每次调用都重新计算的方式改为首次调用后缓存结果的方式，并阻止在补丁生效期间
    修改特殊 token（调用 add_special_tokens 或带 special_tokens=True 的 add_tokens）。"""

    # 标记分词器类是否已被补丁的属性名
    _PATCHED_FLAG = "_sglang_special_tokens_patched"
    # 缓存特殊 token 字符串列表的属性名
    _CACHED_TOKENS_ATTR = "_sglang_cached_special_tokens"
    # 缓存特殊 token ID 列表的属性名
    _CACHED_IDS_ATTR = "_sglang_cached_special_ids"

    @classmethod
    def patch(cls, tokenizer):
        """对分词器类应用缓存补丁：将 all_special_tokens 和 all_special_ids 替换为
        缓存属性，并替换 add_special_tokens 和 add_tokens 方法以防止修改特殊 token。"""
        tokenizer_cls = type(tokenizer)

        # 如果已经打过补丁，直接返回
        if getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer

        # 保存原始属性和方法，以便后续 unpatch 恢复
        tokenizer_cls._original_all_special_tokens = (
            tokenizer_cls.all_special_tokens.fget
        )
        tokenizer_cls._original_all_special_ids = tokenizer_cls.all_special_ids.fget
        tokenizer_cls._original_add_special_tokens = tokenizer_cls.add_special_tokens
        tokenizer_cls._original_add_tokens = tokenizer_cls.add_tokens

        # 创建带缓存的 property，首次访问时计算并缓存结果
        patched_all_special_tokens = _make_cached_property(
            cls._CACHED_TOKENS_ATTR, tokenizer_cls._original_all_special_tokens
        )
        patched_all_special_ids = _make_cached_property(
            cls._CACHED_IDS_ATTR, tokenizer_cls._original_all_special_ids
        )

        # 补丁后的 add_special_tokens：禁止调用
        def patched_add_special_tokens(self, *args, **kwargs):
            assert (
                False
            ), "Cannot modify special tokens after patch. Call unpatch_tokenizer first."

        # 补丁后的 add_tokens：禁止添加特殊 token，普通 token 仍允许
        def patched_add_tokens(self, new_tokens, special_tokens=False):
            assert (
                not special_tokens
            ), "Cannot add special tokens after patch. Call unpatch_tokenizer first."
            return tokenizer_cls._original_add_tokens(
                self, new_tokens, special_tokens=False
            )

        # 替换类属性和方法
        tokenizer_cls.all_special_tokens = patched_all_special_tokens
        tokenizer_cls.all_special_ids = patched_all_special_ids
        tokenizer_cls.add_special_tokens = patched_add_special_tokens
        tokenizer_cls.add_tokens = patched_add_tokens
        # 标记已补丁
        setattr(tokenizer_cls, cls._PATCHED_FLAG, True)

        return tokenizer

    @classmethod
    def unpatch(cls, tokenizer):
        """移除特殊 token 缓存补丁，恢复分词器类的原始属性和方法。"""
        tokenizer_cls = type(tokenizer)

        # 如果没有打过补丁，直接返回
        if not getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer

        # 恢复原始属性和方法
        tokenizer_cls.all_special_tokens = property(
            tokenizer_cls._original_all_special_tokens
        )
        tokenizer_cls.all_special_ids = property(
            tokenizer_cls._original_all_special_ids
        )
        tokenizer_cls.add_special_tokens = tokenizer_cls._original_add_special_tokens
        tokenizer_cls.add_tokens = tokenizer_cls._original_add_tokens

        # 删除保存的原始引用
        del tokenizer_cls._original_all_special_tokens
        del tokenizer_cls._original_all_special_ids
        del tokenizer_cls._original_add_special_tokens
        del tokenizer_cls._original_add_tokens
        delattr(tokenizer_cls, cls._PATCHED_FLAG)

        # 清除实例上的缓存数据
        for attr in [cls._CACHED_TOKENS_ATTR, cls._CACHED_IDS_ATTR]:
            if hasattr(tokenizer, attr):
                delattr(tokenizer, attr)

        logger.info(f"Unpatched special tokens cache for {tokenizer_cls.__name__}")
        return tokenizer


def _make_cached_property(cache_attr, original_fn):
    """创建一个带缓存机制的 property：首次访问时调用 original_fn 计算结果并缓存到
    cache_attr 属性中，后续访问直接返回缓存值。"""
    @property
    def cached_prop(self):
        # 如果缓存不存在，则计算并缓存
        if getattr(self, cache_attr, None) is None:
            setattr(self, cache_attr, original_fn(self))
        return getattr(self, cache_attr)

    return cached_prop
