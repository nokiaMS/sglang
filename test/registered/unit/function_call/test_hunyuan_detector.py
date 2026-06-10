# 文件名: test_hunyuan_detector.py - 混元检测器
"""Unit tests for HunyuanDetector - no server, no model loading."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.hunyuan_detector import HunyuanDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


# 内部方法_make_tools
def _make_tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="get_current_date",
                description="Get the current date",
                parameters={},
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                description="Get weather information",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "date": {"type": "string", "description": "Date"},
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
                        "count": {
                            "type": "integer",
                            "description": "Number of results",
                        },
                    },
                    "required": ["query"],
                },
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="calculate",
                description="Calculate expression",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                        "precision": {"type": "number"},
                        "verbose": {"type": "boolean"},
                    },
                },
            ),
        ),
    ]


# TestHunyuanDetectorHasToolCall类
class TestHunyuanDetectorHasToolCall(CustomTestCase):

    # TestHunyuanDetectorHasToolCall类的测试初始化设置
    def setUp(self):
        self.detector = HunyuanDetector()

    # TestHunyuanDetectorHasToolCall类的测试hastoolcalltrue
    def test_has_tool_call_true(self):
        text = (
            "<tool_calls><tool_call>get_current_date<tool_sep></tool_call></tool_calls>"
        )
        self.assertTrue(self.detector.has_tool_call(text))  # 断言为真

    # TestHunyuanDetectorHasToolCall类的测试hastoolcallfalse
    def test_has_tool_call_false(self):
        self.assertFalse(  # 断言为假
            self.detector.has_tool_call("The weather in Beijing is sunny today.")
        )

    # TestHunyuanDetectorHasToolCall类的测试hastoolcallpartialtag
    def test_has_tool_call_partial_tag(self):
        self.assertFalse(self.detector.has_tool_call("<tool_call>"))  # 断言为假
        self.assertFalse(self.detector.has_tool_call("<tool_call"))  # 断言为假

    # TestHunyuanDetectorHasToolCall类的测试hastoolcallwithsurroundingtext
    def test_has_tool_call_with_surrounding_text(self):
        self.assertTrue(  # 断言为真
            self.detector.has_tool_call("text before <tool_calls> text after")
        )


# TestHunyuanDetectorDetectAndParse类
class TestHunyuanDetectorDetectAndParse(CustomTestCase):

    # TestHunyuanDetectorDetectAndParse类的测试初始化设置
    def setUp(self):
        self.tools = _make_tools()
        self.detector = HunyuanDetector()

    # TestHunyuanDetectorDetectAndParse类的测试notoolcall
    def test_no_tool_call(self):
        text = "This is a plain response."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)  # 断言相等
        self.assertEqual(result.normal_text, text)  # 断言相等

    # TestHunyuanDetectorDetectAndParse类的测试zeroarginline
    def test_zero_arg_inline(self):
        text = (
            "<tool_calls><tool_call>get_current_date<tool_sep></tool_call></tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_current_date")  # 断言相等
        self.assertEqual(json.loads(result.calls[0].parameters), {})  # 断言相等

    # TestHunyuanDetectorDetectAndParse类的测试zeroargnewline
    def test_zero_arg_newline(self):
        text = (
            "<tool_calls>\n"
            "<tool_call>get_current_date<tool_sep>\n"
            "</tool_call>\n"
            "</tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_current_date")  # 断言相等

    # TestHunyuanDetectorDetectAndParse类的测试singlestringarg
    def test_single_string_arg(self):
        text = (
            "<tool_calls><tool_call>get_weather<tool_sep>"
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>"
            "</tool_call></tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)  # 断言相等
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args, {"city": "Beijing"})  # 断言相等

    # TestHunyuanDetectorDetectAndParse类的测试multipleargssameline
    def test_multiple_args_same_line(self):
        text = (
            "<tool_calls><tool_call>get_weather<tool_sep>"
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>"
            "<arg_key>date</arg_key><arg_value>2026-03-30</arg_value>"
            "</tool_call></tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")  # 断言相等
        self.assertEqual(args["date"], "2026-03-30")  # 断言相等

    # TestHunyuanDetectorDetectAndParse类的测试argswithnewlines
    def test_args_with_newlines(self):
        text = (
            "<tool_calls>\n"
            "<tool_call>get_weather<tool_sep>\n"
            "<arg_key>city</arg_key>\n"
            "<arg_value>Beijing</arg_value>\n"
            "<arg_key>date</arg_key>\n"
            "<arg_value>2026-03-30</arg_value>\n"
            "</tool_call>\n"
            "</tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")  # 断言相等
        self.assertEqual(args["date"], "2026-03-30")  # 断言相等

    # TestHunyuanDetectorDetectAndParse类的测试contentbeforetoolcall
    def test_content_before_tool_call(self):
        text = (
            "Checking."
            "<tool_calls>\n"
            "<tool_call>get_current_date<tool_sep>\n"
            "</tool_call>\n"
            "</tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.normal_text, "Checking.")  # 断言相等

    # TestHunyuanDetectorDetectAndParse类的测试multipletoolcalls
    def test_multiple_tool_calls(self):
        text = (
            "<tool_calls>\n"
            "<tool_call>get_weather<tool_sep>\n"
            "<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n"
            "</tool_call>\n"
            "<tool_call>get_weather<tool_sep>\n"
            "<arg_key>city</arg_key>\n<arg_value>Hangzhou</arg_value>\n"
            "</tool_call>\n"
            "</tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)  # 断言相等
        self.assertEqual(json.loads(result.calls[0].parameters)["city"], "Beijing")  # 断言相等
        self.assertEqual(json.loads(result.calls[1].parameters)["city"], "Hangzhou")  # 断言相等

    # TestHunyuanDetectorDetectAndParse类的测试emptycontentreturnsemptynormaltext
    def test_empty_content_returns_empty_normal_text(self):
        text = "<tool_calls>\n<tool_call>get_current_date<tool_sep>\n</tool_call>\n</tool_calls>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "")  # 断言相等

    # TestHunyuanDetectorDetectAndParse类的测试unknowntoolskipped
    def test_unknown_tool_skipped(self):
        text = (
            "<tool_calls><tool_call>nonexistent_func<tool_sep>"
            "<arg_key>x</arg_key><arg_value>1</arg_value>"
            "</tool_call></tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)  # 断言相等

    # TestHunyuanDetectorDetectAndParse类的测试mixedknownandunknowntools
    def test_mixed_known_and_unknown_tools(self):
        """Known tools should be parsed, unknown ones skipped."""
        text = (
            "<tool_calls>"
            "<tool_call>get_current_date<tool_sep></tool_call>"
            "<tool_call>nonexistent<tool_sep></tool_call>"
            "<tool_call>search<tool_sep>"
            "<arg_key>query</arg_key><arg_value>test</arg_value>"
            "</tool_call>"
            "</tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_current_date")  # 断言相等
        self.assertEqual(result.calls[1].name, "search")  # 断言相等

    # TestHunyuanDetectorDetectAndParse类的测试threeparalleltoolcalls
    def test_three_parallel_tool_calls(self):
        text = (
            "<tool_calls>"
            "<tool_call>get_weather<tool_sep>"
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>"
            "</tool_call>"
            "<tool_call>get_weather<tool_sep>"
            "<arg_key>city</arg_key><arg_value>Tokyo</arg_value>"
            "</tool_call>"
            "<tool_call>get_current_date<tool_sep></tool_call>"
            "</tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 3)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等
        self.assertEqual(result.calls[1].name, "get_weather")  # 断言相等
        self.assertEqual(result.calls[2].name, "get_current_date")  # 断言相等
        # tool_index maps to position in tools list
        self.assertEqual(result.calls[0].tool_index, 1)  # get_weather is index 1  # 断言相等
        self.assertEqual(result.calls[2].tool_index, 0)  # get_current_date is index 0  # 断言相等


# TestHunyuanDetectorArgDeserialization类
class TestHunyuanDetectorArgDeserialization(CustomTestCase):
    """Test type-aware argument deserialization."""

    def setUp(self):
        self.tools = _make_tools()
        self.detector = HunyuanDetector()

    # TestHunyuanDetectorArgDeserialization类的测试integerarg
    def test_integer_arg(self):
        text = (
            "<tool_calls><tool_call>search<tool_sep>"
            "<arg_key>query</arg_key><arg_value>restaurants</arg_value>"
            "<arg_key>count</arg_key><arg_value>5</arg_value>"
            "</tool_call></tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["query"], "restaurants")  # 断言相等
        self.assertEqual(args["count"], 5)  # 断言相等
        self.assertIsInstance(args["count"], int)

    # TestHunyuanDetectorArgDeserialization类的测试floatarg
    def test_float_arg(self):
        text = (
            "<tool_calls><tool_call>calculate<tool_sep>"
            "<arg_key>expression</arg_key><arg_value>1+1</arg_value>"
            "<arg_key>precision</arg_key><arg_value>0.01</arg_value>"
            "</tool_call></tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["expression"], "1+1")  # 断言相等
        self.assertAlmostEqual(args["precision"], 0.01)  # 断言近似相等

    # TestHunyuanDetectorArgDeserialization类的测试booleanarg
    def test_boolean_arg(self):
        text = (
            "<tool_calls><tool_call>calculate<tool_sep>"
            "<arg_key>expression</arg_key><arg_value>2+2</arg_value>"
            "<arg_key>verbose</arg_key><arg_value>true</arg_value>"
            "</tool_call></tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        args = json.loads(result.calls[0].parameters)
        self.assertIs(args["verbose"], True)  # 断言是同一对象

    # TestHunyuanDetectorArgDeserialization类的测试stringargnotdeserialized
    def test_string_arg_not_deserialized(self):
        """String-typed args should stay as strings even if they look like JSON."""
        text = (
            "<tool_calls><tool_call>search<tool_sep>"
            '<arg_key>query</arg_key><arg_value>{"key": "value"}</arg_value>'
            "</tool_call></tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["query"], '{"key": "value"}')  # 断言相等
        self.assertIsInstance(args["query"], str)

    # TestHunyuanDetectorArgDeserialization类的测试nonjsonvaluestaysstring
    def test_non_json_value_stays_string(self):
        """Non-JSON-parseable values for non-string types should fall back to string."""
        text = (
            "<tool_calls><tool_call>search<tool_sep>"
            "<arg_key>query</arg_key><arg_value>hello world</arg_value>"
            "<arg_key>count</arg_key><arg_value>not a number</arg_value>"
            "</tool_call></tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["count"], "not a number")  # 断言相等


# 内部方法_collect_streamed_tool_calls
def _collect_streamed_tool_calls(all_calls):
    """Accumulate streaming ToolCallItems (name + arg-JSON fragments) by tool_index."""
    tools = {}
    for c in all_calls:
        idx = c.tool_index
        if idx not in tools:
            tools[idx] = {"name": c.name or "", "parameters": c.parameters or ""}
        else:
            if c.name:
                tools[idx]["name"] += c.name
            if c.parameters:
                tools[idx]["parameters"] += c.parameters
    return [tools[i] for i in sorted(tools.keys())]


# TestHunyuanDetectorStreaming类
class TestHunyuanDetectorStreaming(CustomTestCase):

    # TestHunyuanDetectorStreaming类的测试初始化设置
    def setUp(self):
        self.tools = _make_tools()

    # TestHunyuanDetectorStreaming类的内部方法_new_detector
    def _new_detector(self):
        return HunyuanDetector()

    # TestHunyuanDetectorStreaming类的测试normaltextonly
    def test_normal_text_only(self):
        detector = self._new_detector()
        result = detector.parse_streaming_increment(
            "Hello, I can help you with that.", self.tools
        )
        self.assertEqual(result.normal_text, "Hello, I can help you with that.")  # 断言相等
        self.assertEqual(len(result.calls), 0)  # 断言相等

    # TestHunyuanDetectorStreaming类的测试completetoolcallsinglechunk
    def test_complete_tool_call_single_chunk(self):
        detector = self._new_detector()
        text = (
            "<tool_calls>"
            "<tool_call>get_current_date<tool_sep></tool_call>"
            "</tool_calls>"
        )
        result = detector.parse_streaming_increment(text, self.tools)
        collected = _collect_streamed_tool_calls(result.calls)
        self.assertEqual(len(collected), 1)  # 断言相等
        self.assertEqual(collected[0]["name"], "get_current_date")  # 断言相等
        self.assertEqual(json.loads(collected[0]["parameters"]), {})  # 断言相等

    # TestHunyuanDetectorStreaming类的测试chunkedtoolcall
    def test_chunked_tool_call(self):
        detector = self._new_detector()
        chunks = [
            "<tool_calls>",
            "<tool_call>get_weather<tool_sep>",
            "<arg_key>city</arg_key>",
            "<arg_value>Tokyo</arg_value>",
            "</tool_call>",
            "</tool_calls>",
        ]
        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        collected = _collect_streamed_tool_calls(all_calls)
        self.assertEqual(len(collected), 1)  # 断言相等
        self.assertEqual(collected[0]["name"], "get_weather")  # 断言相等
        args = json.loads(collected[0]["parameters"])
        self.assertEqual(args["city"], "Tokyo")  # 断言相等

    # TestHunyuanDetectorStreaming类的测试normaltextbeforetool
    def test_normal_text_before_tool(self):
        detector = self._new_detector()
        r1 = detector.parse_streaming_increment("Let me check. ", self.tools)
        self.assertIn("Let me check.", r1.normal_text)  # 断言包含

        r2 = detector.parse_streaming_increment(
            "<tool_calls><tool_call>get_current_date<tool_sep></tool_call></tool_calls>",
            self.tools,
        )
        collected = _collect_streamed_tool_calls(r2.calls)
        self.assertEqual([c["name"] for c in collected], ["get_current_date"])  # 断言相等

    # TestHunyuanDetectorStreaming类的测试multipletoolcallschunked
    def test_multiple_tool_calls_chunked(self):
        detector = self._new_detector()
        chunks = [
            "<tool_calls>\n",
            "<tool_call>get_weather<tool_sep>\n",
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>\n",
            "</tool_call>\n",
            "<tool_call>get_weather<tool_sep>\n",
            "<arg_key>city</arg_key><arg_value>Tokyo</arg_value>\n",
            "</tool_call>\n",
            "</tool_calls>",
        ]
        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        collected = _collect_streamed_tool_calls(all_calls)
        self.assertEqual(len(collected), 2)  # 断言相等
        self.assertEqual(json.loads(collected[0]["parameters"])["city"], "Beijing")  # 断言相等
        self.assertEqual(json.loads(collected[1]["parameters"])["city"], "Tokyo")  # 断言相等

    # TestHunyuanDetectorStreaming类的测试partialbottokenbuffered
    def test_partial_bot_token_buffered(self):
        """Partial <tool_calls> at end of chunk should be buffered, not emitted."""
        detector = self._new_detector()
        r1 = detector.parse_streaming_increment("Hello <tool_", self.tools)
        # "Hello " should be emitted, "<tool_" buffered
        self.assertIn("Hello", r1.normal_text)  # 断言包含
        self.assertNotIn("<tool_", r1.normal_text)  # 断言不包含

    # TestHunyuanDetectorStreaming类的测试charbycharstreaming
    def test_char_by_char_streaming(self):
        """Simulate extreme character-by-character streaming."""
        detector = self._new_detector()
        full = (
            "<tool_calls><tool_call>get_current_date<tool_sep></tool_call></tool_calls>"
        )
        all_calls = []
        for ch in full:
            result = detector.parse_streaming_increment(ch, self.tools)
            all_calls.extend(result.calls)

        collected = _collect_streamed_tool_calls(all_calls)
        self.assertEqual(len(collected), 1)  # 断言相等
        self.assertEqual(collected[0]["name"], "get_current_date")  # 断言相等
        self.assertEqual(json.loads(collected[0]["parameters"]), {})  # 断言相等

    # TestHunyuanDetectorStreaming类的测试streamingwithargscharbychar
    def test_streaming_with_args_char_by_char(self):
        detector = self._new_detector()
        full = (
            "<tool_calls><tool_call>get_weather<tool_sep>"
            "<arg_key>city</arg_key><arg_value>NYC</arg_value>"
            "</tool_call></tool_calls>"
        )
        all_calls = []
        for ch in full:
            result = detector.parse_streaming_increment(ch, self.tools)
            all_calls.extend(result.calls)

        collected = _collect_streamed_tool_calls(all_calls)
        self.assertEqual(len(collected), 1)  # 断言相等
        args = json.loads(collected[0]["parameters"])
        self.assertEqual(args["city"], "NYC")  # 断言相等

    # TestHunyuanDetectorStreaming类的测试streamingthreetoolssequential
    def test_streaming_three_tools_sequential(self):
        """Three different tool calls arriving sequentially."""
        detector = self._new_detector()
        chunks = [
            "<tool_calls>",
            "<tool_call>get_current_date<tool_sep></tool_call>",
            "<tool_call>get_weather<tool_sep><arg_key>city</arg_key><arg_value>SF</arg_value></tool_call>",
            "<tool_call>search<tool_sep><arg_key>query</arg_key><arg_value>test</arg_value></tool_call>",
            "</tool_calls>",
        ]
        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        collected = _collect_streamed_tool_calls(all_calls)
        self.assertEqual(len(collected), 3)  # 断言相等
        self.assertEqual(collected[0]["name"], "get_current_date")  # 断言相等
        self.assertEqual(collected[1]["name"], "get_weather")  # 断言相等
        self.assertEqual(collected[2]["name"], "search")  # 断言相等
        # Streaming uses sequential tool_index (0, 1, 2)
        self.assertEqual(sorted({c.tool_index for c in all_calls}), [0, 1, 2])  # 断言相等

    # TestHunyuanDetectorStreaming类的测试streamingnormaltextnotlost
    def test_streaming_normal_text_not_lost(self):
        """All normal text before tool_calls should be fully emitted."""
        detector = self._new_detector()
        all_normal = ""
        for chunk in ["I will ", "check the ", "date now. "]:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_normal += result.normal_text

        result = detector.parse_streaming_increment(
            "<tool_calls><tool_call>get_current_date<tool_sep></tool_call></tool_calls>",
            self.tools,
        )
        all_normal += result.normal_text
        self.assertIn("I will check the date now.", all_normal)  # 断言包含

    # TestHunyuanDetectorStreaming类的测试streamingnamecomesbeforeargs
    def test_streaming_name_comes_before_args(self):
        """The name delta must arrive before any arg deltas (two-phase contract)."""
        detector = self._new_detector()
        text = (
            "<tool_calls><tool_call>get_weather<tool_sep>"
            "<arg_key>city</arg_key><arg_value>Paris</arg_value>"
            "</tool_call></tool_calls>"
        )
        all_calls = []
        for ch in text:
            all_calls.extend(detector.parse_streaming_increment(ch, self.tools).calls)

        name_indices = [i for i, c in enumerate(all_calls) if c.name]
        param_indices = [i for i, c in enumerate(all_calls) if c.parameters]
        self.assertTrue(name_indices, "expected at least one name delta")  # 断言为真
        self.assertTrue(param_indices, "expected at least one arg delta")  # 断言为真
        self.assertLess(min(name_indices), min(param_indices))  # 断言小于

    # TestHunyuanDetectorStreaming类的测试streamingtypedargscoerced
    def test_streaming_typed_args_coerced(self):
        """Streaming must apply schema-aware type coercion (int/float/bool)."""
        detector = self._new_detector()
        chunks = [
            "<tool_calls>",
            "<tool_call>search<tool_sep>",
            "<arg_key>query</arg_key><arg_value>pizza</arg_value>",
            "<arg_key>count</arg_key><arg_value>7</arg_value>",
            "</tool_call></tool_calls>",
        ]
        all_calls = []
        for chunk in chunks:
            all_calls.extend(
                detector.parse_streaming_increment(chunk, self.tools).calls
            )
        collected = _collect_streamed_tool_calls(all_calls)
        args = json.loads(collected[0]["parameters"])
        self.assertEqual(args["query"], "pizza")  # 断言相等
        self.assertEqual(args["count"], 7)  # 断言相等
        self.assertIsInstance(args["count"], int)

    # TestHunyuanDetectorStreaming类的测试streamingstringargholdsbackpartialendtag
    def test_streaming_string_arg_holds_back_partial_end_tag(self):
        """Char-by-char string streaming must not leak `</arg_value>` into the value."""
        detector = self._new_detector()
        full = (
            "<tool_calls><tool_call>get_weather<tool_sep>"
            "<arg_key>city</arg_key><arg_value>San Francisco</arg_value>"
            "</tool_call></tool_calls>"
        )
        all_calls = []
        for ch in full:
            all_calls.extend(detector.parse_streaming_increment(ch, self.tools).calls)

        collected = _collect_streamed_tool_calls(all_calls)
        args = json.loads(collected[0]["parameters"])
        self.assertEqual(args["city"], "San Francisco")  # 断言相等

    # TestHunyuanDetectorStreaming类的测试streamingallinonedelta
    def test_streaming_all_in_one_delta(self):
        """Entire tool call arriving in a single delta."""
        detector = self._new_detector()
        text = (
            "<tool_calls>\n<tool_call>get_current_date<tool_sep>\n"
            "</tool_call>\n</tool_calls>"
        )
        result = detector.parse_streaming_increment(text, self.tools)
        collected = _collect_streamed_tool_calls(result.calls)
        self.assertEqual(len(collected), 1)  # 断言相等
        self.assertEqual(collected[0]["name"], "get_current_date")  # 断言相等
        self.assertEqual(json.loads(collected[0]["parameters"]), {})  # 断言相等

    # TestHunyuanDetectorStreaming类的测试streamingcontentbefore
    def test_streaming_content_before(self):
        """Normal text preceding a tool call must be surfaced."""
        detector = self._new_detector()
        deltas = [
            "Checking.",
            "<tool_calls>",
            "\n<tool_call>",
            "get_current_date",
            "<tool_sep>",
            "\n</tool_call>",
            "\n</tool_calls>",
        ]
        all_calls = []
        all_normal = ""
        for d in deltas:
            r = detector.parse_streaming_increment(d, self.tools)
            all_calls.extend(r.calls)
            all_normal += r.normal_text
        self.assertIn("Checking.", all_normal)  # 断言包含
        collected = _collect_streamed_tool_calls(all_calls)
        self.assertEqual(len(collected), 1)  # 断言相等
        self.assertEqual(collected[0]["name"], "get_current_date")  # 断言相等


# TestHunyuanDetectorStructureInfo类
class TestHunyuanDetectorStructureInfo(CustomTestCase):

    # TestHunyuanDetectorStructureInfo类的测试初始化设置
    def setUp(self):
        self.detector = HunyuanDetector()

    # TestHunyuanDetectorStructureInfo类的测试structureinfocontent
    def test_structure_info_content(self):
        info_fn = self.detector.structure_info()
        info = info_fn("get_weather")
        self.assertIn("get_weather", info.begin)  # 断言包含
        self.assertIn("<tool_call>", info.begin)  # 断言包含
        self.assertIn("<tool_sep>", info.begin)  # 断言包含
        self.assertIn("</tool_call>", info.end)  # 断言包含
        self.assertEqual(info.trigger, "<tool_calls>")  # 断言相等

    # TestHunyuanDetectorStructureInfo类的测试supportsstructuraltag
    def test_supports_structural_tag(self):
        self.assertFalse(self.detector.supports_structural_tag())  # 断言为假


# TestHunyuanDetectorAccuracy类
class TestHunyuanDetectorAccuracy(CustomTestCase):
    """Accuracy tests for realistic HYV3 output patterns."""

    def setUp(self):
        self.tools = _make_tools()
        self.detector = HunyuanDetector()

    # TestHunyuanDetectorAccuracy类的测试referencezeroarginline
    def test_reference_zero_arg_inline(self):
        out = (
            "<tool_calls><tool_call>get_current_date<tool_sep></tool_call></tool_calls>"
        )
        r = self.detector.detect_and_parse(out, self.tools)
        self.assertEqual(len(r.calls), 1)  # 断言相等
        self.assertEqual(r.calls[0].name, "get_current_date")  # 断言相等
        self.assertEqual(json.loads(r.calls[0].parameters), {})  # 断言相等
        self.assertEqual(r.normal_text, "")  # 断言相等

    # TestHunyuanDetectorAccuracy类的测试referencezeroargnewline
    def test_reference_zero_arg_newline(self):
        out = "<tool_calls>\n<tool_call>get_current_date<tool_sep>\n</tool_call>\n</tool_calls>"
        r = self.detector.detect_and_parse(out, self.tools)
        self.assertEqual(len(r.calls), 1)  # 断言相等
        self.assertEqual(r.calls[0].name, "get_current_date")  # 断言相等

    # TestHunyuanDetectorAccuracy类的测试referenceargssameline
    def test_reference_args_same_line(self):
        out = (
            "<tool_calls><tool_call>get_weather<tool_sep><arg_key>city</arg_key><arg_value>Beijing"
            "</arg_value><arg_key>date</arg_key><arg_value>2026-03-30</arg_value></tool_call></tool_calls>"
        )
        r = self.detector.detect_and_parse(out, self.tools)
        self.assertEqual(len(r.calls), 1)  # 断言相等
        args = json.loads(r.calls[0].parameters)
        self.assertEqual(args, {"city": "Beijing", "date": "2026-03-30"})  # 断言相等

    # TestHunyuanDetectorAccuracy类的测试referenceargswithnewlines
    def test_reference_args_with_newlines(self):
        out = (
            "<tool_calls>\n<tool_call>get_weather<tool_sep>\n<arg_key>city</arg_key>\n<arg_value>Beijing"
            "</arg_value>\n<arg_key>date</arg_key>\n<arg_value>2026-03-30</arg_value>\n</tool_call>\n</tool_calls>"
        )
        r = self.detector.detect_and_parse(out, self.tools)
        self.assertEqual(len(r.calls), 1)  # 断言相等
        args = json.loads(r.calls[0].parameters)
        self.assertEqual(args, {"city": "Beijing", "date": "2026-03-30"})  # 断言相等

    # TestHunyuanDetectorAccuracy类的测试referencecontentbefore
    def test_reference_content_before(self):
        out = "Checking.<tool_calls>\n<tool_call>get_current_date<tool_sep>\n</tool_call>\n</tool_calls>"
        r = self.detector.detect_and_parse(out, self.tools)
        self.assertEqual(len(r.calls), 1)  # 断言相等
        self.assertEqual(r.normal_text, "Checking.")  # 断言相等

    # TestHunyuanDetectorAccuracy类的测试referencemultiple
    def test_reference_multiple(self):
        out = (
            "<tool_calls>\n<tool_call>get_weather<tool_sep>\n<arg_key>city</arg_key>\n<arg_value>Beijing"
            "</arg_value>\n<arg_key>date</arg_key>\n<arg_value>2026-03-30</arg_value>\n</tool_call>\n"
            "<tool_call>get_weather<tool_sep>\n<arg_key>city</arg_key>\n<arg_value>Hangzhou</arg_value>\n"
            "<arg_key>date</arg_key>\n<arg_value>2026-03-30</arg_value>\n</tool_call>\n</tool_calls>"
        )
        r = self.detector.detect_and_parse(out, self.tools)
        self.assertEqual(len(r.calls), 2)  # 断言相等

    # TestHunyuanDetectorAccuracy类的测试referenceemptycontentnone
    def test_reference_empty_content_none(self):
        out = "<tool_calls>\n<tool_call>get_current_date<tool_sep>\n</tool_call>\n</tool_calls>"
        r = self.detector.detect_and_parse(out, self.tools)
        self.assertEqual(r.normal_text, "")  # 断言相等

    # TestHunyuanDetectorAccuracy类的测试referencenotoolcall
    def test_reference_no_tool_call(self):
        out = "This is a plain response."
        r = self.detector.detect_and_parse(out, self.tools)
        self.assertEqual(len(r.calls), 0)  # 断言相等
        self.assertEqual(r.normal_text, out)  # 断言相等


# TestHunyuanDetectorFunctionCallParser类
class TestHunyuanDetectorFunctionCallParser(CustomTestCase):
    """Test through the FunctionCallParser interface."""

    def setUp(self):
        self.tools = _make_tools()

    # TestHunyuanDetectorFunctionCallParser类的测试parserregistry
    def test_parser_registry(self):
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        parser = FunctionCallParser(self.tools, "hunyuan")
        self.assertIsInstance(parser.detector, HunyuanDetector)

    # TestHunyuanDetectorFunctionCallParser类的测试parsenonstream
    def test_parse_non_stream(self):
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        parser = FunctionCallParser(self.tools, "hunyuan")
        text = (
            "Checking.<tool_calls><tool_call>get_weather<tool_sep>"
            "<arg_key>city</arg_key><arg_value>Tokyo</arg_value>"
            "</tool_call></tool_calls>"
        )
        normal, calls = parser.parse_non_stream(text)
        self.assertEqual(normal, "Checking.")  # 断言相等
        self.assertEqual(len(calls), 1)  # 断言相等
        self.assertEqual(calls[0].name, "get_weather")  # 断言相等
        self.assertEqual(json.loads(calls[0].parameters)["city"], "Tokyo")  # 断言相等

    # TestHunyuanDetectorFunctionCallParser类的测试parsestreamchunks
    def test_parse_stream_chunks(self):
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        parser = FunctionCallParser(self.tools, "hunyuan")
        chunks = [
            "<tool_calls>",
            "<tool_call>get_current_date<tool_sep></tool_call>",
            "</tool_calls>",
        ]
        all_calls = []
        for chunk in chunks:
            normal, calls = parser.parse_stream_chunk(chunk)
            all_calls.extend(calls)

        collected = _collect_streamed_tool_calls(all_calls)
        self.assertEqual(len(collected), 1)  # 断言相等
        self.assertEqual(collected[0]["name"], "get_current_date")  # 断言相等
        self.assertEqual(json.loads(collected[0]["parameters"]), {})  # 断言相等

    # TestHunyuanDetectorFunctionCallParser类的测试hastoolcallthroughparser
    def test_has_tool_call_through_parser(self):
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        parser = FunctionCallParser(self.tools, "hunyuan")
        self.assertTrue(parser.has_tool_call("<tool_calls>foo</tool_calls>"))  # 断言为真
        self.assertFalse(parser.has_tool_call("no tools here"))  # 断言为假


if __name__ == "__main__":
    unittest.main()
