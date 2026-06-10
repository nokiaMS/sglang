"""Global configurations"""
# 全局配置模块，用于存储和管理框架级别的全局常量与参数

# FIXME: deprecate this file and move all usage to sglang.srt.environ or sglang.__init__.py
# 待办：废弃此文件，将所有用法迁移到 sglang.srt.environ 或 sglang.__init__.py


class GlobalConfig:
    """
    Store some global constants.
    存储一些全局常量。
    """

    def __init__(self):
        # Verbosity level
        # 冗余级别
        # 0: do not output anything
        # 0：不输出任何内容
        # 2: output final text after every run
        # 2：每次运行后输出最终文本
        self.verbosity = 0  # 冗余级别，默认为0（不输出）

        # Default backend of the language
        # 语言的默认后端
        self.default_backend = None  # 默认后端，初始为None

        # Output tokenization configs
        # 输出分词配置
        self.skip_special_tokens_in_output = True  # 输出时是否跳过特殊标记，默认为True
        self.spaces_between_special_tokens_in_out = True  # 输出时特殊标记之间是否加空格，默认为True

        # Language frontend interpreter optimization configs
        # 语言前端解释器优化配置
        self.enable_precache_with_tracing = True  # 是否启用带追踪的预缓存优化，默认为True
        self.enable_parallel_encoding = True  # 是否启用并行编码优化，默认为True


global_config = GlobalConfig()  # 创建全局配置的单例实例
