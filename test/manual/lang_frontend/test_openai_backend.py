# 文件名: test_openai_backend.py - 测试OpenAI后端的各种语言前端功能
import unittest

from sglang import OpenAI, set_default_backend
from sglang.test.test_programs import (
    test_chat_completion_speculative,
    test_completion_speculative,
    test_decode_int,
    test_decode_json,
    test_expert_answer,
    test_few_shot_qa,
    test_image_qa,
    test_mt_bench,
    test_parallel_decoding,
    test_parallel_encoding,
    test_react,
    test_select,
    test_stream,
    test_tool_use,
)
from sglang.test.test_utils import CustomTestCase


class TestOpenAIBackend(CustomTestCase):
    instruct_backend = None
    chat_backend = None
    chat_vision_backend = None

    @classmethod
    # 类级别初始化，启动服务器或设置测试环境
    def setUpClass(cls):
        cls.instruct_backend = OpenAI("gpt-3.5-turbo-instruct")
        cls.chat_backend = OpenAI("gpt-3.5-turbo")
        cls.chat_vision_backend = OpenAI("gpt-4-turbo")

    # 测试few shot qa功能
    def test_few_shot_qa(self):
        set_default_backend(self.instruct_backend)
        test_few_shot_qa()

    # 测试mt bench功能
    def test_mt_bench(self):
        set_default_backend(self.chat_backend)
        test_mt_bench()

    # 测试select功能
    def test_select(self):
        set_default_backend(self.instruct_backend)
        test_select(check_answer=True)

    # 测试decode int功能
    def test_decode_int(self):
        set_default_backend(self.instruct_backend)
        test_decode_int()

    # 测试decode json功能
    def test_decode_json(self):
        set_default_backend(self.instruct_backend)
        test_decode_json()

    # 测试expert answer功能
    def test_expert_answer(self):
        set_default_backend(self.instruct_backend)
        test_expert_answer()

    # 测试tool use功能
    def test_tool_use(self):
        set_default_backend(self.instruct_backend)
        test_tool_use()

    # 测试react功能
    def test_react(self):
        set_default_backend(self.instruct_backend)
        test_react()

    # 测试parallel decoding功能
    def test_parallel_decoding(self):
        set_default_backend(self.instruct_backend)
        test_parallel_decoding()

    # 测试parallel encoding功能
    def test_parallel_encoding(self):
        set_default_backend(self.instruct_backend)
        test_parallel_encoding()

    # 测试image qa功能
    def test_image_qa(self):
        set_default_backend(self.chat_vision_backend)
        test_image_qa()

    # 测试stream功能
    def test_stream(self):
        set_default_backend(self.instruct_backend)
        test_stream()

    # 测试completion speculative功能
    def test_completion_speculative(self):
        set_default_backend(self.instruct_backend)
        test_completion_speculative()

    # 测试chat completion speculative功能
    def test_chat_completion_speculative(self):
        set_default_backend(self.chat_backend)
        test_chat_completion_speculative()


if __name__ == "__main__":
    unittest.main()
