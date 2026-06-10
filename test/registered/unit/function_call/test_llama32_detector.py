# 文件名: test_llama32_detector.py - LLaMA3.2检测器
"""Unit tests for Llama32Detector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.llama32_detector import Llama32Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


# TestLlama32Detector类
class TestLlama32Detector(CustomTestCase):

    # TestLlama32Detector类的测试初始化设置
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Search the web",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            },
                        },
                        "required": ["query"],
                    },
                ),
            ),
        ]
        self.detector = Llama32Detector()

    # ==================== has_tool_call Tests ====================

    def test_has_tool_call_with_python_tag(self):
        text = '<|python_tag|>{"name": "get_weather", "arguments": {"city": "Beijing"}}'
        self.assertTrue(self.detector.has_tool_call(text))  # 断言为真

    # TestLlama32Detector类的测试hastoolcallwithjsonstart
    def test_has_tool_call_with_json_start(self):
        text = '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
        self.assertTrue(self.detector.has_tool_call(text))  # 断言为真

    # TestLlama32Detector类的测试hastoolcallfalse
    def test_has_tool_call_false(self):
        text = "The weather in Beijing is sunny today."
        self.assertFalse(self.detector.has_tool_call(text))  # 断言为假

    # ==================== detect_and_parse Tests ====================

    def test_single_tool_call_with_python_tag(self):
        text = '<|python_tag|>{"name": "get_weather", "arguments": {"city": "Beijing"}}'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")  # 断言相等

    # TestLlama32Detector类的测试singletoolcallwithoutpythontag
    def test_single_tool_call_without_python_tag(self):
        text = '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等

    # TestLlama32Detector类的测试normaltextbeforepythontag
    def test_normal_text_before_python_tag(self):
        text = 'Let me check. <|python_tag|>{"name": "get_weather", "arguments": {"city": "Tokyo"}}'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.normal_text, "Let me check. ")  # 断言相等

    # TestLlama32Detector类的测试notoolcall
    def test_no_tool_call(self):
        text = "The weather is nice today."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)  # 断言相等
        self.assertEqual(result.normal_text, "The weather is nice today.")  # 断言相等

    # TestLlama32Detector类的测试multiplejsonobjects
    def test_multiple_json_objects(self):
        text = '<|python_tag|>{"name": "get_weather", "arguments": {"city": "Beijing"}};{"name": "search", "arguments": {"query": "restaurants"}}'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等
        self.assertEqual(result.calls[1].name, "search")  # 断言相等

    # TestLlama32Detector类的测试toolcallwithmultiplearguments
    def test_tool_call_with_multiple_arguments(self):
        text = '<|python_tag|>{"name": "get_weather", "arguments": {"city": "London", "unit": "celsius"}}'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)  # 断言相等
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "London")  # 断言相等
        self.assertEqual(args["unit"], "celsius")  # 断言相等

    # ==================== Python dict conversion Tests ====================

    def test_convert_python_dict_to_json(self):
        python_dict = "{'name': 'get_weather', 'arguments': {'city': 'Beijing'}}"
        result = self.detector._convert_python_dict_to_json(python_dict)
        parsed = json.loads(result)
        self.assertEqual(parsed["name"], "get_weather")  # 断言相等

    # TestLlama32Detector类的测试convertinvalidstringreturnsoriginal
    def test_convert_invalid_string_returns_original(self):
        invalid = "not a dict at all"
        result = self.detector._convert_python_dict_to_json(invalid)
        self.assertEqual(result, invalid)  # 断言相等

    # ==================== structure_info Tests ====================

    def test_structure_info(self):
        info_func = self.detector.structure_info()
        info = info_func("get_weather")
        self.assertIn("get_weather", info.begin)  # 断言包含
        self.assertIn("<|python_tag|>", info.begin)  # 断言包含
        self.assertEqual(info.trigger, "<|python_tag|>")  # 断言相等

    # ==================== Streaming Tests ====================

    def test_streaming_single_tool_call(self):
        detector = Llama32Detector()
        chunks = [
            "<|python_",
            'tag|>{"name": "get_weather",',
            ' "arguments": {"city": "Beijing"',
            "}}",
        ]
        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)  # 断言相等
        self.assertEqual(func_calls[0].name, "get_weather")  # 断言相等

        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        params = json.loads(full_params)
        self.assertEqual(params["city"], "Beijing")  # 断言相等

    # TestLlama32Detector类的测试streamingnormaltextbeforetool
    def test_streaming_normal_text_before_tool(self):
        detector = Llama32Detector()
        result = detector.parse_streaming_increment(
            "Let me check the weather. ", self.tools
        )
        self.assertEqual(result.normal_text, "Let me check the weather. ")  # 断言相等
        self.assertEqual(len(result.calls), 0)  # 断言相等

    # TestLlama32Detector类的测试streamingtextthentoolcall
    def test_streaming_text_then_tool_call(self):
        detector = Llama32Detector()
        chunks = [
            "I'll look that up. ",
            '<|python_tag|>{"name": "get_weather",',
            ' "arguments": {"city": "Tokyo", "unit": "celsius"',
            "}}",
        ]
        all_calls = []
        all_normal_text = ""
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)
            all_normal_text += result.normal_text

        self.assertEqual(all_normal_text, "I'll look that up. ")  # 断言相等
        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)  # 断言相等
        self.assertEqual(func_calls[0].name, "get_weather")  # 断言相等
        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        params = json.loads(full_params)
        self.assertEqual(params["city"], "Tokyo")  # 断言相等
        self.assertEqual(params["unit"], "celsius")  # 断言相等


if __name__ == "__main__":
    import unittest

    unittest.main()
