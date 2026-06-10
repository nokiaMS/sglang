# Whisper语音识别处理器模块
# 本模块为Whisper语音转文本模型提供音频数据处理功能
# 支持语言检测、时间戳生成和ISO 639-1语言代码转换

import logging  # 导入日志模块
from typing import Any, Dict, Optional  # 导入类型提示

from sglang.srt.entrypoints.openai.transcription_adapters.whisper import (  # 导入融合自动检测标志
    FUSED_AUTODETECT_FLAG,
)
from sglang.srt.managers.schedule_batch import (  # 导入调度批次相关类
    Modality,  # 模态枚举
    MultimodalDataItem,  # 多模态数据项
    MultimodalProcessorOutput,  # 多模态处理器输出
)
from sglang.srt.models.whisper import WhisperForConditionalGeneration  # 导入Whisper模型类
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor  # 导入基础多模态处理器
from sglang.srt.utils import load_audio  # 导入音频加载工具

logger = logging.getLogger(__name__)  # 创建日志记录器

# ISO 639-1 supported languages for Whisper
# From https://platform.openai.com/docs/guides/speech-to-text/supported-languages
# Maps ISO 639-1 code -> Full language name
ISO639_1_SUPPORTED_LANGS = {  # ISO 639-1支持的语言映射
    "af": "Afrikaans",  # 南非荷兰语
    "ar": "Arabic",  # 阿拉伯语
    "hy": "Armenian",  # 亚美尼亚语
    "az": "Azerbaijani",  # 阿塞拜疆语
    "be": "Belarusian",  # 白俄罗斯语
    "bs": "Bosnian",  # 波斯尼亚语
    "bg": "Bulgarian",  # 保加利亚语
    "ca": "Catalan",  # 加泰罗尼亚语
    "zh": "Chinese",  # 中文
    "hr": "Croatian",  # 克罗地亚语
    "cs": "Czech",  # 捷克语
    "da": "Danish",  # 丹麦语
    "nl": "Dutch",  # 荷兰语
    "en": "English",  # 英语
    "et": "Estonian",  # 爱沙尼亚语
    "fi": "Finnish",  # 芬兰语
    "fr": "French",  # 法语
    "gl": "Galician",  # 加利西亚语
    "de": "German",  # 德语
    "el": "Greek",  # 希腊语
    "he": "Hebrew",  # 希伯来语
    "hi": "Hindi",  # 印地语
    "hu": "Hungarian",  # 匈牙利语
    "is": "Icelandic",  # 冰岛语
    "id": "Indonesian",  # 印尼语
    "it": "Italian",  # 意大利语
    "ja": "Japanese",  # 日语
    "kn": "Kannada",  # 卡纳达语
    "kk": "Kazakh",  # 哈萨克语
    "ko": "Korean",  # 韩语
    "lv": "Latvian",  # 拉脱维亚语
    "lt": "Lithuanian",  # 立陶宛语
    "mk": "Macedonian",  # 马其顿语
    "ms": "Malay",  # 马来语
    "mr": "Marathi",  # 马拉地语
    "mi": "Maori",  # 毛利语
    "ne": "Nepali",  # 尼泊尔语
    "no": "Norwegian",  # 挪威语
    "fa": "Persian",  # 波斯语
    "pl": "Polish",  # 波兰语
    "pt": "Portuguese",  # 葡萄牙语
    "ro": "Romanian",  # 罗马尼亚语
    "ru": "Russian",  # 俄语
    "sr": "Serbian",  # 塞尔维亚语
    "sk": "Slovak",  # 斯洛伐克语
    "sl": "Slovenian",  # 斯洛文尼亚语
    "es": "Spanish",  # 西班牙语
    "sw": "Swahili",  # 斯瓦希里语
    "sv": "Swedish",  # 瑞典语
    "tl": "Tagalog",  # 他加禄语
    "ta": "Tamil",  # 泰米尔语
    "th": "Thai",  # 泰语
    "tr": "Turkish",  # 土耳其语
    "uk": "Ukrainian",  # 乌克兰语
    "ur": "Urdu",  # 乌尔都语
    "vi": "Vietnamese",  # 越南语
    "cy": "Welsh",  # 威尔士语
}

# Reverse mapping: Full language name (lowercase) -> ISO 639-1 code
LANG_NAME_TO_CODE = {  # 反向映射：语言全名（小写）-> ISO 639-1代码
    name.lower(): code for code, name in ISO639_1_SUPPORTED_LANGS.items()  # 从正向映射生成
}


def normalize_language_to_code(language: Optional[str]) -> Optional[str]:  # 将语言输入（全名或代码）规范化为ISO 639-1代码
    """Convert a language input (full name or code) to ISO 639-1 code.

    Args:
        language: Language as full name (e.g., 'English', 'Spanish') or
                  ISO 639-1 code (e.g., 'en', 'es'). Three-letter Whisper
                  codes the model supports but that aren't in
                  ISO639_1_SUPPORTED_LANGS (e.g., 'yue', 'haw', 'jw') are
                  also accepted so that a code returned by fused autodetect
                  round-trips cleanly when reused as ``language=`` later.

    Returns:
        Whisper language code or None if input is None
    """
    if language is None:  # 如果输入为None
        return None  # 返回None

    language_lower = language.lower().strip()  # 转为小写并去除空白

    # Check if it's already a valid ISO code
    if language_lower in ISO639_1_SUPPORTED_LANGS:  # 如果已经是有效的ISO代码
        return language_lower  # 直接返回

    # Check if it's a full language name
    if language_lower in LANG_NAME_TO_CODE:  # 如果是语言全名
        return LANG_NAME_TO_CODE[language_lower]  # 返回对应的ISO代码

    # Fused autodetect's FSM regex covers the full Whisper language-token
    # vocab (see WHISPER_LANG_TOKEN_CODES), which is wider than the
    # English-name-keyed ISO639_1_SUPPORTED_LANGS dict. Accept any code in
    # that wider set too so that detection -> reuse-as-input round-trips.
    # Lazy import to avoid top-level cycle with the openai entrypoint.
    from sglang.srt.entrypoints.openai.transcription_adapters.whisper import (  # 延迟导入以避免循环依赖
        WHISPER_LANG_TOKEN_CODES,
    )

    if language_lower in WHISPER_LANG_TOKEN_CODES:  # 如果在Whisper语言标记代码中
        return language_lower  # 返回代码

    # Not recognized
    raise ValueError(  # 抛出异常
        f"Language '{language}' not recognized. "  # 未识别的语言
        f"Use full name (e.g., 'English') or ISO 639-1 code (e.g., 'en')."  # 提示使用全名或代码
    )


class WhisperProcessor(BaseMultimodalProcessor):  # Whisper处理器，继承基础多模态处理器
    models = [WhisperForConditionalGeneration]  # 关联的模型列表

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):  # 初始化Whisper处理器
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)  # 调用父类初始化
        # Cache tokenizer for language token lookup
        self._tokenizer = getattr(self._processor, "tokenizer", None)  # 缓存分词器用于语言标记查找

    def _pop_sampling_param(self, request_obj, key: str):  # 从请求对象中弹出采样参数
        sampling_params = getattr(request_obj, "sampling_params", None) or {}  # 获取采样参数
        return sampling_params.pop(key, None)  # 弹出指定键的参数

    def _get_language_token_id(self, language: Optional[str]) -> int:  # 获取语言标记ID
        # Default to English if not specified
        if language is None:  # 如果未指定语言
            language = "en"  # Default to English，默认为英语
        language_token = f"<|{language}|>"  # 构造语言标记
        token_id = self._tokenizer.convert_tokens_to_ids(language_token)  # 转换为标记ID
        # normalize_language_to_code accepts the full Whisper language-token
        # vocab (including yue/haw/jw) so fused autodetect output round-trips.
        # Older checkpoints (v1/v2) don't have every newer token in their
        # vocab, in which case convert_tokens_to_ids returns the unk id.
        # Raise a clean error here instead of silently feeding unk into the
        # decoder and producing garbage.
        unk_id = getattr(self._tokenizer, "unk_token_id", None)  # 获取未知标记ID
        if token_id is None or (unk_id is not None and token_id == unk_id):  # 如果标记ID无效
            raise ValueError(  # 抛出异常
                f"Language '{language}' is not in this Whisper model's vocabulary. "  # 语言不在词表中
                f"The '{language_token}' token may have been added in a later "  # 该标记可能在更新版本中添加
                f"Whisper version than the loaded checkpoint."  # 比加载的检查点更新
            )
        return token_id  # 返回语言标记ID

    async def process_mm_data_async(  # 异步处理多模态数据
        self,
        image_data,  # 图像数据（未使用）
        audio_data,  # 音频数据
        input_text,  # 输入文本
        request_obj,  # 请求对象
        **kwargs,  # 其他关键字参数
    ) -> Optional[Dict[str, Any]]:
        if not audio_data:  # 如果没有音频数据
            return None  # 返回None

        if len(audio_data) != 1:  # 如果音频数量不等于1
            raise ValueError(  # 抛出异常
                f"Whisper expects exactly 1 audio input, got {len(audio_data)}"  # Whisper只接受1个音频输入
            )

        # Check if this is a fused auto-detect request (decoder prompt = [SOT] only,
        # structured generation handles the rest via regex constraint).
        detect_language = self._pop_sampling_param(request_obj, FUSED_AUTODETECT_FLAG)  # 检查是否为融合自动检测请求
        # timestamp_granularities is a transcription-level field; it must be
        # popped in both branches or it leaks into SamplingParams(**kwargs)
        # downstream and TypeErrors. In the fused branch the FSM regex was
        # already picked in build_fused_autodetect_params based on this value,
        # so we only need to keep it here to pick the timestamp_token_id for
        # the explicit-language branch.
        timestamp_granularities = self._pop_sampling_param(  # 弹出时间戳粒度参数
            request_obj, "timestamp_granularities"
        )

        audios = [load_audio(audio) for audio in audio_data]  # 加载所有音频

        # Whisper expects input features padded to max_length (3000 frames = 30 seconds)
        # This is the standard context length for Whisper
        input_features = self._processor.feature_extractor(  # 提取音频特征
            audios[0],  # 第一段音频
            sampling_rate=16000,  # 采样率
            padding="max_length",  # Pad to 3000 frames，填充到3000帧
            return_tensors="pt",  # 返回PyTorch张量
        )["input_features"][0]  # 获取输入特征

        # Whisper is a pure speech-to-text model; text prompts are ignored.
        # The full decoder sequence is:
        #   <|startoftranscript|> <|lang|> <|transcribe|> [<|notimestamps|> | <|0.00|>]
        #
        # When language is known, we build this prefix explicitly below.
        # When auto-detecting (_detect_language=True), we feed only <|startoftranscript|>
        # and let SGLang's structured generation (regex) constrain the model to produce
        # <|lang|><|transcribe|><|notimestamps|> as the first 3 decode tokens — this is
        # equivalent to HuggingFace's forced_decoder_ids but uses SGLang's native API.

        decoder_start_token_id = getattr(  # 获取解码器起始标记ID
            self.hf_config, "decoder_start_token_id", 50258  # 默认为50258
        )

        if detect_language:  # 如果是自动检测语言
            input_ids = [decoder_start_token_id]  # 仅使用起始标记
        else:  # 指定语言
            language = normalize_language_to_code(  # 规范化语言代码
                self._pop_sampling_param(request_obj, "language")  # 从采样参数中获取语言
            )
            language_token_id = self._get_language_token_id(language)  # 获取语言标记ID

            transcribe_token_id = self._tokenizer.convert_tokens_to_ids(  # 获取转录标记ID
                "<|transcribe|>"
            )

            # Use <|0.00|> to enable timestamp generation, or <|notimestamps|> to disable
            if timestamp_granularities:  # 如果启用时间戳
                timestamp_token_id = self._tokenizer.convert_tokens_to_ids("<|0.00|>")  # 获取时间戳起始标记ID
            else:  # 否则
                timestamp_token_id = self._tokenizer.convert_tokens_to_ids(  # 获取非时间戳标记ID
                    "<|notimestamps|>"
                )

            input_ids = [  # 构建输入ID序列
                decoder_start_token_id,  # 解码器起始标记
                language_token_id,  # 语言标记
                transcribe_token_id,  # 转录标记
                timestamp_token_id,  # 时间戳标记
            ]

        return MultimodalProcessorOutput(  # 返回多模态处理器输出
            input_ids=input_ids,  # 输入ID
            mm_items=[  # 多模态数据项
                MultimodalDataItem(  # 创建音频数据项
                    feature=input_features,  # 音频特征
                    modality=Modality.AUDIO,  # 音频模态
                )
            ],
        )
