# 文件名: test_hermes_detector.py - Hermes检测器
"""Unit tests for HermesDetector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.hermes_detector import HermesDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


# TestHermesDetector类
class TestHermesDetector(CustomTestCase):

    # TestHermesDetector类的测试初始化设置
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
        self.detector = HermesDetector()

    # ==================== has_tool_call Tests ====================

    def test_has_tool_call_true(self):
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Beijing"}}</tool_call>'
        self.assertTrue(self.detector.has_tool_call(text))  # 断言为真

    # TestHermesDetector类的测试hastoolcallfalse
    def test_has_tool_call_false(self):
        text = "The weather in Beijing is sunny today."
        self.assertFalse(self.detector.has_tool_call(text))  # 断言为假

    # ==================== detect_and_parse Tests ====================

    def test_single_tool_call(self):
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Beijing"}}</tool_call>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")  # 断言相等
        self.assertEqual(result.normal_text, "")  # 断言相等

    # TestHermesDetector类的测试multipletoolcalls
    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>{"name": "get_weather", "arguments": {"city": "Beijing"}}</tool_call>'
            '<tool_call>{"name": "search", "arguments": {"query": "restaurants"}}</tool_call>'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等
        self.assertEqual(result.calls[1].name, "search")  # 断言相等

    # TestHermesDetector类的测试toolcallwithleadingtext
    def test_tool_call_with_leading_text(self):
        text = 'I will check the weather for you. <tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等
        self.assertEqual(result.normal_text, "I will check the weather for you.")  # 断言相等

    # TestHermesDetector类的测试notoolcall
    def test_no_tool_call(self):
        text = "The weather is nice today."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)  # 断言相等
        self.assertEqual(result.normal_text, "The weather is nice today.")  # 断言相等

    # TestHermesDetector类的测试toolcallwithmultiplearguments
    def test_tool_call_with_multiple_arguments(self):
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "London", "unit": "celsius"}}</tool_call>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)  # 断言相等
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "London")  # 断言相等
        self.assertEqual(args["unit"], "celsius")  # 断言相等

    # TestHermesDetector类的测试malformedjsonreturnsoriginaltext
    def test_malformed_json_returns_original_text(self):
        text = "<tool_call>not valid json</tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)  # 断言相等
        self.assertEqual(result.normal_text, text)  # 断言相等

    # ==================== structure_info Tests ====================

    def test_structure_info(self):
        info_func = self.detector.structure_info()
        info = info_func("get_weather")
        self.assertIn("get_weather", info.begin)  # 断言包含
        self.assertEqual(info.trigger, "<tool_call>")  # 断言相等
        self.assertEqual(info.end, "}</tool_call>")  # 断言相等

    # ==================== Streaming Tests ====================

    def test_streaming_single_tool_call(self):
        detector = HermesDetector()
        chunks = [
            "<tool_",
            'call>{"name": "get_weather",',
            ' "arguments": {"city": "Beijing"',
            "}}</tool_call>",
        ]
        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        # Verify tool name
        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)  # 断言相等
        self.assertEqual(func_calls[0].name, "get_weather")  # 断言相等

        # Verify parameters
        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        params = json.loads(full_params)
        self.assertEqual(params["city"], "Beijing")  # 断言相等

    # TestHermesDetector类的测试streamingnormaltextbeforetool
    def test_streaming_normal_text_before_tool(self):
        detector = HermesDetector()
        result = detector.parse_streaming_increment("Hello! Let me help. ", self.tools)
        self.assertEqual(result.normal_text, "Hello! Let me help. ")  # 断言相等
        self.assertEqual(len(result.calls), 0)  # 断言相等

    # TestHermesDetector类的测试streamingtextthentoolcall
    def test_streaming_text_then_tool_call(self):
        detector = HermesDetector()
        chunks = [
            "Sure, let me check. ",
            '<tool_call>{"name": "get_weather",',
            ' "arguments": {"city": "Tokyo"',
            "}}</tool_call>",
        ]
        all_calls = []
        all_normal_text = ""
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)
            all_normal_text += result.normal_text

        self.assertEqual(all_normal_text, "Sure, let me check. ")  # 断言相等
        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)  # 断言相等
        self.assertEqual(func_calls[0].name, "get_weather")  # 断言相等
        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        params = json.loads(full_params)
        self.assertEqual(params["city"], "Tokyo")  # 断言相等


if __name__ == "__main__":
    import unittest

    unittest.main()
