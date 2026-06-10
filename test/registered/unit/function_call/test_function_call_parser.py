# 文件名: test_function_call_parser.py - 函数调用解析器
import json
import unittest

from sglang.srt.entrypoints.openai.protocol import (
    Function,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import StreamingParseResult
from sglang.srt.function_call.deepseekv3_detector import DeepSeekV3Detector
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
from sglang.srt.function_call.gemma4_detector import (
    Gemma4Detector,
    _parse_gemma4_args,
    _parse_gemma4_array,
    _parse_gemma4_value,
)
from sglang.srt.function_call.gigachat3_detector import GigaChat3Detector
from sglang.srt.function_call.glm4_moe_detector import Glm4MoeDetector
from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
from sglang.srt.function_call.gpt_oss_detector import GptOssDetector
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.function_call.kimik2_detector import KimiK2Detector
from sglang.srt.function_call.lfm2_detector import Lfm2Detector
from sglang.srt.function_call.llama32_detector import Llama32Detector
from sglang.srt.function_call.mistral_detector import MistralDetector
from sglang.srt.function_call.pythonic_detector import PythonicDetector
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
register_cpu_ci(est_time=61, suite="base-b-test-cpu")


# TestPythonicDetector类
class TestPythonicDetector(unittest.TestCase):

    # TestPythonicDetector类的测试初始化设置
    def setUp(self):
        # Create sample tools for testing
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "Location to get weather for",
                            },
                            "unit": {
                                "type": "string",
                                "description": "Temperature unit",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Search for information",
                    parameters={
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
        self.detector = PythonicDetector()

    # TestPythonicDetector类的测试parsestreamingnobrackets
    def test_parse_streaming_no_brackets(self):
        """Test parsing text with no brackets (no tool calls)."""
        text = "This is just normal text without any tool calls."
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, text)  # 断言相等
        self.assertEqual(result.calls, [])  # 断言相等
        self.assertEqual(self.detector._buffer, "")  # Buffer should be cleared  # 断言相等

    # TestPythonicDetector类的测试parsestreamingcompletetoolcall
    def test_parse_streaming_complete_tool_call(self):
        """Test parsing a complete tool call."""
        text = "Here's a tool call: [get_weather(location='New York', unit='celsius')]"
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "Here's a tool call: ")  # 断言相等
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等
        self.assertEqual(  # 断言相等
            self.detector._buffer, ""
        )  # Buffer should be cleared after processing

        # Check the parameters
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["location"], "New York")  # 断言相等
        self.assertEqual(params["unit"], "celsius")  # 断言相等

    # TestPythonicDetector类的测试parsestreamingtextbeforetoolcall
    def test_parse_streaming_text_before_tool_call(self):
        """Test parsing text that appears before a tool call."""
        text = "This is some text before [get_weather(location='London')]"
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "This is some text before ")  # 断言相等
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等

        # Check the parameters
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["location"], "London")  # 断言相等

    # TestPythonicDetector类的测试parsestreamingpartialtoolcall
    def test_parse_streaming_partial_tool_call(self):
        """Test parsing a partial tool call that spans multiple chunks."""
        # First chunk with opening bracket but no closing bracket
        text1 = "Let me check the weather: [get_weather(location="
        result1 = self.detector.parse_streaming_increment(text1, self.tools)

        self.assertEqual(result1.normal_text, "Let me check the weather: ")  # 断言相等
        self.assertEqual(result1.calls, [])  # 断言相等
        self.assertEqual(  # 断言相等
            self.detector._buffer, "[get_weather(location="
        )  # Partial tool call remains in buffer

        # Second chunk completing the tool call
        text2 = "'Paris')]"
        result2 = self.detector.parse_streaming_increment(text2, self.tools)

        self.assertEqual(result2.normal_text, "")  # 断言相等
        self.assertEqual(len(result2.calls), 1)  # 断言相等
        self.assertEqual(result2.calls[0].name, "get_weather")  # 断言相等

        # Check the parameters
        params = json.loads(result2.calls[0].parameters)
        self.assertEqual(params["location"], "Paris")  # 断言相等
        self.assertEqual(  # 断言相等
            self.detector._buffer, ""
        )  # Buffer should be cleared after processing

    # TestPythonicDetector类的测试parsestreamingbracketwithouttextbefore
    def test_parse_streaming_bracket_without_text_before(self):
        """Test parsing a tool call that starts at the beginning of the text."""
        text = "[search(query='python programming')]"
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "")  # 断言相等
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "search")  # 断言相等

        # Check the parameters
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["query"], "python programming")  # 断言相等

    # TestPythonicDetector类的测试parsestreamingtextaftertoolcall
    def test_parse_streaming_text_after_tool_call(self):
        """Test parsing text that appears after a tool call."""
        # First chunk with complete tool call and some text after
        text = "[get_weather(location='Tokyo')] Here's the forecast:"
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "")  # 断言相等
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等
        self.assertEqual(  # 断言相等
            self.detector._buffer, " Here's the forecast:"
        )  # Text after tool call remains in buffer

        # Process the remaining text in buffer
        result2 = self.detector.parse_streaming_increment("", self.tools)
        self.assertEqual(result2.normal_text, " Here's the forecast:")  # 断言相等
        self.assertEqual(result2.calls, [])  # 断言相等
        self.assertEqual(self.detector._buffer, "")  # Buffer should be cleared  # 断言相等

    # TestPythonicDetector类的测试parsestreamingmultipletoolcalls
    def test_parse_streaming_multiple_tool_calls(self):
        """Test parsing multiple tool calls in sequence."""
        text = "[get_weather(location='Berlin')] and [search(query='restaurants')]"

        # First tool call
        result1 = self.detector.parse_streaming_increment(text, self.tools)
        self.assertEqual(len(result1.calls), 1)  # 断言相等
        self.assertEqual(result1.calls[0].name, "get_weather")  # 断言相等
        self.assertEqual(self.detector._buffer, " and [search(query='restaurants')]")  # 断言相等

        # Second tool call
        result2 = self.detector.parse_streaming_increment("", self.tools)
        self.assertEqual(result2.normal_text, " and ")  # 断言相等
        self.assertEqual(len(result2.calls), 1)  # 断言相等
        self.assertEqual(result2.calls[0].name, "search")  # 断言相等
        self.assertEqual(self.detector._buffer, "")  # 断言相等

    # TestPythonicDetector类的测试parsestreamingopeningbracketonly
    def test_parse_streaming_opening_bracket_only(self):
        """Test parsing text with only an opening bracket but no closing bracket."""
        text = "Let's try this: ["
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "Let's try this: ")  # 断言相等
        self.assertEqual(result.calls, [])  # 断言相等
        self.assertEqual(  # 断言相等
            self.detector._buffer, "["
        )  # Opening bracket remains in buffer

    # TestPythonicDetector类的测试parsestreamingnestedbrackets
    def test_parse_streaming_nested_brackets(self):
        """Test parsing tool calls with nested brackets in arguments."""
        # Test with list argument containing nested brackets
        text = "[get_weather(location='New York', unit='celsius', data=[1, 2, 3])]"
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "")  # 断言相等
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等
        self.assertEqual(self.detector._buffer, "")  # 断言相等

        # Check the parameters
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["location"], "New York")  # 断言相等
        self.assertEqual(params["unit"], "celsius")  # 断言相等
        self.assertEqual(params["data"], [1, 2, 3])  # 断言相等

    # TestPythonicDetector类的测试parsestreamingnestedbracketsdict
    def test_parse_streaming_nested_brackets_dict(self):
        """Test parsing tool calls with nested dictionaries and lists."""
        # Test with nested dict and list arguments
        text = "[search(query='test', config={'options': [1, 2], 'nested': {'key': 'value'}})]"
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "")  # 断言相等
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "search")  # 断言相等
        self.assertEqual(self.detector._buffer, "")  # 断言相等

        # Check the parameters
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["query"], "test")  # 断言相等
        self.assertEqual(params["config"]["options"], [1, 2])  # 断言相等
        self.assertEqual(params["config"]["nested"]["key"], "value")  # 断言相等

    # TestPythonicDetector类的测试parsestreamingmultipletoolswithnestedbrackets
    def test_parse_streaming_multiple_tools_with_nested_brackets(self):
        """Test parsing multiple tool calls with nested brackets."""
        text = "[get_weather(location='Paris', data=[10, 20]), search(query='test', filters=['a', 'b'])]"
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "")  # 断言相等
        self.assertEqual(len(result.calls), 2)  # 断言相等
        self.assertEqual(self.detector._buffer, "")  # 断言相等

        # Check first tool call
        params1 = json.loads(result.calls[0].parameters)
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等
        self.assertEqual(params1["location"], "Paris")  # 断言相等
        self.assertEqual(params1["data"], [10, 20])  # 断言相等

        # Check second tool call
        params2 = json.loads(result.calls[1].parameters)
        self.assertEqual(result.calls[1].name, "search")  # 断言相等
        self.assertEqual(params2["query"], "test")  # 断言相等
        self.assertEqual(params2["filters"], ["a", "b"])  # 断言相等

    # TestPythonicDetector类的测试parsestreamingpartialnestedbrackets
    def test_parse_streaming_partial_nested_brackets(self):
        """Test parsing partial tool calls with nested brackets across chunks."""
        # First chunk with nested brackets but incomplete
        text1 = "Here's a call: [get_weather(location='Tokyo', data=[1, 2"
        result1 = self.detector.parse_streaming_increment(text1, self.tools)

        self.assertEqual(result1.normal_text, "Here's a call: ")  # 断言相等
        self.assertEqual(result1.calls, [])  # 断言相等
        self.assertEqual(  # 断言相等
            self.detector._buffer, "[get_weather(location='Tokyo', data=[1, 2"
        )

        # Second chunk completing the nested brackets
        text2 = ", 3])]"
        result2 = self.detector.parse_streaming_increment(text2, self.tools)

        self.assertEqual(result2.normal_text, "")  # 断言相等
        self.assertEqual(len(result2.calls), 1)  # 断言相等
        self.assertEqual(result2.calls[0].name, "get_weather")  # 断言相等
        self.assertEqual(self.detector._buffer, "")  # 断言相等

        # Check the parameters
        params = json.loads(result2.calls[0].parameters)
        self.assertEqual(params["location"], "Tokyo")  # 断言相等
        self.assertEqual(params["data"], [1, 2, 3])  # 断言相等

    # TestPythonicDetector类的测试parsestreamingwithpythonstartandendtoken
    def test_parse_streaming_with_python_start_and_end_token(self):
        """Test parsing a message that starts with <|python_start|> and <|python_end|> across chunks."""
        chunks = [
            "Here's a call: ",
            "<|python_",
            "start|>[get_weather(location=",
            "'Tokyo', data=[1, 2",
            ", 3])]<|python_end|>",
        ]

        normal_text = ""
        call_name = ""
        parameters = ""
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            if result.normal_text:
                normal_text += result.normal_text
            if result.calls:
                call_name += result.calls[0].name
                parameters += result.calls[0].parameters

        self.assertEqual(normal_text, "Here's a call: ")  # 断言相等
        self.assertEqual(call_name, "get_weather")  # 断言相等
        self.assertEqual(self.detector._buffer, "")  # 断言相等
        self.assertEqual(  # 断言相等
            result.normal_text, "", "Final result should have no normal text"
        )

        # Check the parameters
        params = json.loads(parameters)
        self.assertEqual(params["location"], "Tokyo")  # 断言相等
        self.assertEqual(params["data"], [1, 2, 3])  # 断言相等

        chunks = [
            "Here's a call: <|python_start|>[get_weather(location='Tokyo', data=[1, 2, 3])]<|python_end|>"
        ]

        normal_text = ""
        call_name = ""
        parameters = ""
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            if result.normal_text:
                normal_text += result.normal_text
            if result.calls:
                call_name += result.calls[0].name
                parameters += result.calls[0].parameters

        self.assertEqual(normal_text, "Here's a call: ")  # 断言相等
        self.assertEqual(call_name, "get_weather")  # 断言相等
        self.assertEqual(self.detector._buffer, "")  # 断言相等

        # Check the parameters
        params = json.loads(parameters)
        self.assertEqual(params["location"], "Tokyo")  # 断言相等
        self.assertEqual(params["data"], [1, 2, 3])  # 断言相等

    # TestPythonicDetector类的测试detectandparsewithpythonstartandendtoken
    def test_detect_and_parse_with_python_start_and_end_token(self):
        """Test parsing a message that starts with <|python_start|> and contains a valid tool call."""
        text = "User wants to get the weather in Mars. <|python_start|>[get_weather(location='Mars', unit='celsius')]<|python_end|> In this way we will get the weather in Mars."
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(  # 断言相等
            result.normal_text,
            "User wants to get the weather in Mars.  In this way we will get the weather in Mars.",
        )
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等
        self.assertEqual(self.detector._buffer, "")  # 断言相等

        # Check the parameters
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["location"], "Mars")  # 断言相等
        self.assertEqual(params["unit"], "celsius")  # 断言相等


# TestMistralDetector类
class TestMistralDetector(unittest.TestCase):

    # TestMistralDetector类的测试初始化设置
    def setUp(self):
        """Set up test tools and detector for Mistral format testing."""
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="make_next_step_decision",
                    description="Test function for decision making",
                    parameters={
                        "type": "object",
                        "properties": {
                            "decision": {
                                "type": "string",
                                "description": "The next step to take",
                            },
                            "content": {
                                "type": "string",
                                "description": "The content of the next step",
                            },
                        },
                        "required": ["decision", "content"],
                    },
                ),
            ),
        ]
        self.detector = MistralDetector()

    # TestMistralDetector类的测试detectandparsewithnestedbracketsincontent
    def test_detect_and_parse_with_nested_brackets_in_content(self):
        """Test parsing Mistral format with nested brackets in JSON content.

        This test case specifically addresses the issue where the regex pattern
        was incorrectly truncating JSON when it contained nested brackets like [City Name].
        """
        # This is the exact problematic text from the original test failure
        test_text = '[TOOL_CALLS] [{"name":"make_next_step_decision", "arguments":{"decision":"","content":"```\\nTOOL: Access a weather API or service\\nOBSERVATION: Retrieve the current weather data for the top 5 populated cities in the US\\nANSWER: The weather in the top 5 populated cities in the US is as follows: [City Name] - [Weather Conditions] - [Temperature]\\n```"}}]'

        result = self.detector.detect_and_parse(test_text, self.tools)

        # Verify that the parsing was successful
        self.assertEqual(len(result.calls), 1, "Should detect exactly one tool call")  # 断言相等

        call = result.calls[0]
        self.assertEqual(  # 断言相等
            call.name,
            "make_next_step_decision",
            "Should detect the correct function name",
        )

        # Verify that the parameters are valid JSON and contain the expected content
        params = json.loads(call.parameters)
        self.assertEqual(  # 断言相等
            params["decision"], "", "Decision parameter should be empty string"
        )

        # The content should contain the full text including the nested brackets [City Name]
        expected_content = "```\nTOOL: Access a weather API or service\nOBSERVATION: Retrieve the current weather data for the top 5 populated cities in the US\nANSWER: The weather in the top 5 populated cities in the US is as follows: [City Name] - [Weather Conditions] - [Temperature]\n```"
        self.assertEqual(  # 断言相等
            params["content"],
            expected_content,
            "Content should include nested brackets without truncation",
        )

        # Verify that normal text is empty (since the entire input is a tool call)
        self.assertEqual(  # 断言相等
            result.normal_text, "", "Normal text should be empty for pure tool call"
        )

    # TestMistralDetector类的测试detectandparsesimplecase
    def test_detect_and_parse_simple_case(self):
        """Test parsing a simple Mistral format tool call without nested brackets."""
        test_text = '[TOOL_CALLS] [{"name":"make_next_step_decision", "arguments":{"decision":"TOOL", "content":"Use weather API"}}]'

        result = self.detector.detect_and_parse(test_text, self.tools)

        self.assertEqual(len(result.calls), 1)  # 断言相等
        call = result.calls[0]
        self.assertEqual(call.name, "make_next_step_decision")  # 断言相等

        params = json.loads(call.parameters)
        self.assertEqual(params["decision"], "TOOL")  # 断言相等
        self.assertEqual(params["content"], "Use weather API")  # 断言相等

    # TestMistralDetector类的测试detectandparsenotoolcalls
    def test_detect_and_parse_no_tool_calls(self):
        """Test parsing text without any tool calls."""
        test_text = "This is just normal text without any tool calls."

        result = self.detector.detect_and_parse(test_text, self.tools)

        self.assertEqual(len(result.calls), 0, "Should detect no tool calls")  # 断言相等
        self.assertEqual(  # 断言相等
            result.normal_text,
            test_text,
            "Should return the original text as normal text",
        )

    # TestMistralDetector类的测试detectandparsewithtextbeforetoolcall
    def test_detect_and_parse_with_text_before_tool_call(self):
        """Test parsing text that has content before the tool call."""
        test_text = 'Here is some text before the tool call: [TOOL_CALLS] [{"name":"make_next_step_decision", "arguments":{"decision":"ANSWER", "content":"The answer is 42"}}]'

        result = self.detector.detect_and_parse(test_text, self.tools)

        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.normal_text, "Here is some text before the tool call:")  # 断言相等

        call = result.calls[0]
        self.assertEqual(call.name, "make_next_step_decision")  # 断言相等

        params = json.loads(call.parameters)
        self.assertEqual(params["decision"], "ANSWER")  # 断言相等
        self.assertEqual(params["content"], "The answer is 42")  # 断言相等

    # TestMistralDetector类的测试detectandparsecompactargsformat
    def test_detect_and_parse_compact_args_format(self):
        """Test parsing compact format: [TOOL_CALLS]name[ARGS]{...}."""
        test_text = '[TOOL_CALLS]make_next_step_decision[ARGS]{"decision":"TOOL", "content":"Use weather API"}'

        result = self.detector.detect_and_parse(test_text, self.tools)
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "make_next_step_decision")  # 断言相等
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["decision"], "TOOL")  # 断言相等
        self.assertEqual(params["content"], "Use weather API")  # 断言相等

    # TestMistralDetector类的测试streamingcompactargsformatemitstoolcalls
    def test_streaming_compact_args_format_emits_tool_calls(self):
        """Test streaming chunks for compact format produce tool_calls items."""
        chunks = [
            "[TOOL_CALLS]make_next_step_decision[ARGS]",
            '{"decision":"TOOL", ',
            '"content":"Use weather API"}',
        ]

        emitted = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            if result.calls:
                emitted.extend(result.calls)

        # Expect two items: name chunk + full args chunk
        self.assertEqual(len(emitted), 2)  # 断言相等
        self.assertEqual(emitted[0].name, "make_next_step_decision")  # 断言相等
        self.assertEqual(emitted[0].parameters, "")  # 断言相等
        self.assertIsNone(emitted[1].name)  # 断言为None
        params = json.loads(emitted[1].parameters)
        self.assertEqual(params["decision"], "TOOL")  # 断言相等
        self.assertEqual(params["content"], "Use weather API")  # 断言相等


# TestBaseFormatDetector类
class TestBaseFormatDetector(unittest.TestCase):
    """Test buffer management and sequential tool index assignment in BaseFormatDetector."""

    def setUp(self):
        """Set up test detector and tools."""

        # Create a concrete implementation of BaseFormatDetector for testing
        class TestFormatDetector(BaseFormatDetector):

            # TestFormatDetector类的初始化
            def __init__(self):
                super().__init__()
                self.bot_token = "<tool_call>"
                self.eot_token = "</tool_call>"

            # TestFormatDetector类的detect_and_parse
            def detect_and_parse(self, text, tools):
                # Not used in streaming tests
                pass

            # TestFormatDetector类的has_tool_call
            def has_tool_call(self, text):
                return "<tool_call>" in text

            # TestFormatDetector类的structure_info
            def structure_info(self):
                # Not used in streaming tests
                pass

        self.detector = TestFormatDetector()
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_tourist_attractions",
                    description="Get tourist attractions",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["city"],
                    },
                ),
            ),
        ]

    # TestBaseFormatDetector类的测试sequentialtoolindexassignment
    def test_sequential_tool_index_assignment(self):
        """Test that multiple tool calls get sequential tool_index values (0, 1, 2, ...)."""
        # Simulate streaming chunks for two consecutive tool calls
        chunks = [
            "<tool_call>",
            '{"name": "get_weather", ',
            '"arguments": {"city": "Paris"}}',
            ", ",
            '{"name": "get_tourist_attractions", ',
            '"arguments": {"city": "London"}}',
            "</tool_call>",
        ]

        tool_indices_seen = []

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)

            if result.calls:
                for call in result.calls:
                    if call.tool_index is not None:
                        tool_indices_seen.append(call.tool_index)

        # Verify we got sequential tool indices
        unique_indices = sorted(set(tool_indices_seen))
        self.assertEqual(  # 断言相等
            unique_indices,
            [0, 1],
            f"Expected sequential tool indices [0, 1], got {unique_indices}",
        )

    # TestBaseFormatDetector类的测试buffercontentpreservation
    def test_buffer_content_preservation(self):
        """Test that buffer correctly preserves unprocessed content when tool completes."""
        # Test simpler scenario: tool completion followed by new tool start
        chunks = [
            "<tool_call>",
            '{"name": "get_weather", ',
            '"arguments": {"city": "Paris"}}',
            ", ",
            '{"name": "get_tourist_attractions", ',
            '"arguments": {"city": "London"}} </tool_call>',
        ]

        tool_calls_seen = []

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            if result.calls:
                for call in result.calls:
                    if (
                        call.name
                    ):  # Only count calls with names (not just parameter updates)
                        tool_calls_seen.append(call.name)

        # Should see both tool names
        self.assertIn("get_weather", tool_calls_seen, "Should process first tool")  # 断言包含
        self.assertIn(  # 断言包含
            "get_tourist_attractions", tool_calls_seen, "Should process second tool"
        )

    # TestBaseFormatDetector类的测试currenttoolidincrementoncompletion
    def test_current_tool_id_increment_on_completion(self):
        """Test that current_tool_id increments when a tool completes."""
        # Initial state
        self.assertEqual(  # 断言相等
            self.detector.current_tool_id, -1, "Should start with current_tool_id=-1"
        )

        # Process first tool completely
        chunks = [
            "<tool_call>",
            '{"name": "get_weather", ',
        ]

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)

        self.assertEqual(  # 断言相等
            self.detector.current_tool_id, 0, "current_tool_id should be 0"
        )
        self.assertEqual(  # 断言相等
            result.calls[0].name, "get_weather", "The first tool should be get_weather"
        )
        self.assertEqual(  # 断言相等
            result.calls[0].tool_index, 0, "The first tool index should be 0"
        )

        # Complete second tool name - this should show that current_tool_id is now 1
        result = self.detector.parse_streaming_increment(
            '"arguments": {"city": "Paris"}}, {"name": "get_', self.tools
        )
        self.assertEqual(result.calls[0].parameters, '{"city": "Paris"}')  # 断言相等

        self.assertEqual(  # 断言相等
            self.detector.current_tool_id,
            1,
            "current_tool_id should be 1 after first tool completes and second tool starts",
        )

        result = self.detector.parse_streaming_increment(
            'tourist_attractions", ', self.tools
        )

        # Second tool should have tool_index=1
        tourist_calls = [
            call for call in result.calls if call.name == "get_tourist_attractions"
        ]
        self.assertEqual(  # 断言相等
            tourist_calls[0].tool_index, 1, "Second tool should have tool_index=1"
        )

    # TestBaseFormatDetector类的测试toolnamestreamingwithcorrectindex
    def test_tool_name_streaming_with_correct_index(self):
        """Test that tool names are streamed with correct tool_index values."""
        # Process first tool
        self.detector.parse_streaming_increment("<tool_call>", self.tools)
        result1 = self.detector.parse_streaming_increment(
            '{"name": "get_weather", ', self.tools
        )

        # First tool name should have tool_index=0
        weather_calls = [call for call in result1.calls if call.name == "get_weather"]
        self.assertEqual(len(weather_calls), 1, "Should have one weather call")  # 断言相等
        self.assertEqual(  # 断言相等
            weather_calls[0].tool_index, 0, "First tool should have tool_index=0"
        )

        # Complete first tool
        self.detector.parse_streaming_increment(
            '"arguments": {"city": "Paris"}}', self.tools
        )

        # Start second tool
        self.detector.parse_streaming_increment(", ", self.tools)
        result2 = self.detector.parse_streaming_increment(
            '{"name": "get_tourist_attractions", ', self.tools
        )

        # Second tool name should have tool_index=1
        tourist_calls = [
            call for call in result2.calls if call.name == "get_tourist_attractions"
        ]
        self.assertEqual(  # 断言相等
            len(tourist_calls), 1, "Should have one tourist attractions call"
        )
        self.assertEqual(  # 断言相等
            tourist_calls[0].tool_index, 1, "Second tool should have tool_index=1"
        )

    # TestBaseFormatDetector类的测试bufferresetoninvalidtool
    def test_buffer_reset_on_invalid_tool(self):
        """Test that buffer and state are reset when an invalid tool name is encountered."""
        # Start fresh with an invalid tool name from the beginning
        result = self.detector.parse_streaming_increment(
            '<tool_call>{"name": "invalid_tool", ', self.tools
        )

        # Should return empty result and reset state
        self.assertEqual(result.calls, [], "Should return no calls for invalid tool")  # 断言相等
        self.assertEqual(  # 断言相等
            self.detector.current_tool_id,
            -1,
            "current_tool_id should remain -1 for invalid tool",
        )
        self.assertEqual(  # 断言相等
            self.detector._buffer, "", "Buffer should be cleared for invalid tool"
        )

    # TestBaseFormatDetector类的测试chinesecharactersnotdoubleescaped
    def test_chinese_characters_not_double_escaped(self):
        """Test that Chinese characters in tool call parameters are not double-escaped."""
        # Test with Chinese city name "杭州" (Hangzhou)
        chunks = [
            "<tool_call>",
            '{"name": "get_weather", ',
            '"arguments": {"city": "杭州"}}',
            "</tool_call>",
        ]

        accumulated_parameters = {}
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            if result.calls:
                for call in result.calls:
                    if call.parameters:
                        tool_idx = call.tool_index if call.tool_index is not None else 0
                        if tool_idx not in accumulated_parameters:
                            accumulated_parameters[tool_idx] = ""
                        accumulated_parameters[tool_idx] += call.parameters

        # Verify that Chinese characters are preserved (not escaped as \uXXXX)
        self.assertGreater(  # 断言大于
            len(accumulated_parameters), 0, "Should have parsed parameters"
        )
        final_params_str = accumulated_parameters[0]

        # The parameters string should contain the actual Chinese characters, not escaped Unicode
        self.assertIn(  # 断言包含
            "杭州", final_params_str, "Should contain actual Chinese characters"
        )
        self.assertNotIn(  # 断言不包含
            "\\u676d", final_params_str, "Should not contain escaped Unicode sequences"
        )
        self.assertNotIn(  # 断言不包含
            "\\u5dde", final_params_str, "Should not contain escaped Unicode sequences"
        )

        # Verify the JSON can be parsed and contains the correct value
        params = json.loads(final_params_str)
        self.assertEqual(  # 断言相等
            params["city"], "杭州", "Should correctly parse Chinese city name"
        )

    # TestBaseFormatDetector类的测试chinesecharactersincrementalstreaming
    def test_chinese_characters_incremental_streaming(self):
        """Test that Chinese characters work correctly with incremental streaming."""
        # Test incremental streaming with Chinese characters
        chunks = [
            "<tool_call>",
            '{"name": "get_weather", ',
            '"arguments": {"city": "',
            "杭州",
            '"}}',
            "</tool_call>",
        ]

        accumulated_parameters = {}
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            if result.calls:
                for call in result.calls:
                    if call.parameters:
                        tool_idx = call.tool_index if call.tool_index is not None else 0
                        if tool_idx not in accumulated_parameters:
                            accumulated_parameters[tool_idx] = ""
                        accumulated_parameters[tool_idx] += call.parameters

        # Verify Chinese characters are preserved throughout streaming
        self.assertGreater(  # 断言大于
            len(accumulated_parameters), 0, "Should have parsed parameters"
        )
        final_params_str = accumulated_parameters[0]

        # Should contain actual Chinese characters, not escaped
        self.assertIn(  # 断言包含
            "杭州", final_params_str, "Should contain actual Chinese characters"
        )

        # Parse and verify
        params = json.loads(final_params_str)
        self.assertEqual(  # 断言相等
            params["city"], "杭州", "Should correctly parse Chinese city name"
        )

    # TestBaseFormatDetector类的测试multiplechineseparameters
    def test_multiple_chinese_parameters(self):
        """Test multiple tool calls with Chinese parameters."""
        # Test with multiple tool calls containing Chinese characters
        chunks = [
            "<tool_call>",
            '{"name": "get_weather", "arguments": {"city": "北京"}}, ',
            '{"name": "get_tourist_attractions", "arguments": {"city": "上海"}}',
            "</tool_call>",
        ]

        accumulated_parameters = {}
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            if result.calls:
                for call in result.calls:
                    if call.parameters:
                        tool_idx = call.tool_index if call.tool_index is not None else 0
                        if tool_idx not in accumulated_parameters:
                            accumulated_parameters[tool_idx] = ""
                        accumulated_parameters[tool_idx] += call.parameters

        # Verify both tool calls have correct Chinese characters
        self.assertGreaterEqual(
            len(accumulated_parameters), 1, "Should have parsed parameters"
        )

        # Check first tool call (北京 - Beijing)
        if 0 in accumulated_parameters:
            params0 = json.loads(accumulated_parameters[0])
            self.assertIn(  # 断言包含
                "北京",
                accumulated_parameters[0],
                "Should contain actual Chinese characters",
            )
            self.assertEqual(  # 断言相等
                params0["city"], "北京", "Should correctly parse first Chinese city"
            )

        # Check second tool call (上海 - Shanghai) if present
        if 1 in accumulated_parameters:
            params1 = json.loads(accumulated_parameters[1])
            self.assertIn(  # 断言包含
                "上海",
                accumulated_parameters[1],
                "Should contain actual Chinese characters",
            )
            self.assertEqual(  # 断言相等
                params1["city"], "上海", "Should correctly parse second Chinese city"
            )


# TestLlama32Detector类
class TestLlama32Detector(unittest.TestCase):

    # TestLlama32Detector类的测试初始化设置
    def setUp(self):
        """Set up test tools and detector for Mistral format testing."""
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_tourist_attractions",
                    description="Get tourist attractions",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["city"],
                    },
                ),
            ),
        ]
        self.detector = Llama32Detector()

    # TestLlama32Detector类的测试singlejson
    def test_single_json(self):
        text = '{"name": "get_weather", "parameters": {"city": "Paris"}}'
        result = self.detector.detect_and_parse(text, self.tools)
        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert result.normal_text == ""

    # TestLlama32Detector类的测试multiplejsonwithseparator
    def test_multiple_json_with_separator(self):
        text = (
            '<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}};'
            '{"name": "get_tourist_attractions", "parameters": {"city": "Paris"}}'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)  # 断言相等
        self.assertEqual(result.calls[1].name, "get_tourist_attractions")  # 断言相等
        self.assertEqual(result.normal_text, "")  # 断言相等

    # TestLlama32Detector类的测试multiplejsonwithseparatorcustomized
    def test_multiple_json_with_separator_customized(self):
        text = (
            '<|python_tag|>{"name": "get_weather", "parameters": {}}'
            '<|python_tag|>{"name": "get_tourist_attractions", "parameters": {}}'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)  # 断言相等
        self.assertEqual(result.calls[1].name, "get_tourist_attractions")  # 断言相等
        self.assertEqual(result.normal_text, "")  # 断言相等

    # TestLlama32Detector类的测试jsonwithtrailingtext
    def test_json_with_trailing_text(self):
        text = '{"name": "get_weather", "parameters": {}} Some follow-up text'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertIn("follow-up", result.normal_text)  # 断言包含

    # TestLlama32Detector类的测试invalidthenvalidjson
    def test_invalid_then_valid_json(self):
        text = (
            '{"name": "get_weather", "parameters": {'  # malformed
            '{"name": "get_weather", "parameters": {}}'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等

    # TestLlama32Detector类的测试plaintextonly
    def test_plain_text_only(self):
        text = "This is just plain explanation text."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.calls, [])  # 断言相等
        self.assertEqual(result.normal_text, text)  # 断言相等

    # TestLlama32Detector类的测试withpythontagprefix
    def test_with_python_tag_prefix(self):
        text = 'Some intro. <|python_tag|>{"name": "get_weather", "parameters": {}}'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertTrue(result.normal_text.strip().startswith("Some intro."))  # 断言为真


# TestKimiK2Detector类
class TestKimiK2Detector(unittest.TestCase):

    # TestKimiK2Detector类的测试初始化设置
    def setUp(self):
        """Set up test tools and detector."""
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_tourist_attractions",
                    description="Get tourist attractions",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["city"],
                    },
                ),
            ),
        ]
        self.detector = KimiK2Detector()

    # TestKimiK2Detector类的测试singletoolcall
    def test_single_tool_call(self):
        """Test parsing a single tool call in a complete text."""
        text = '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "Paris"}<|tool_call_end|><|tool_calls_section_end|>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等
        self.assertEqual(result.calls[0].parameters, '{"city": "Paris"}')  # 断言相等
        self.assertEqual(result.normal_text, "")  # 断言相等

    # TestKimiK2Detector类的测试multipletoolcalls
    def test_multiple_tool_calls(self):
        """Test parsing multiple tool calls in a complete text."""
        text = '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "Paris"}<|tool_call_end|><|tool_call_begin|>functions.get_tourist_attractions:1<|tool_call_argument_begin|>{"city": "London"}<|tool_call_end|><|tool_calls_section_end|>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等
        self.assertEqual(result.calls[0].parameters, '{"city": "Paris"}')  # 断言相等
        self.assertEqual(result.calls[1].name, "get_tourist_attractions")  # 断言相等
        self.assertEqual(result.calls[1].parameters, '{"city": "London"}')  # 断言相等
        self.assertEqual(result.normal_text, "")  # 断言相等

    # TestKimiK2Detector类的测试streamingtoolcall
    def test_streaming_tool_call(self):
        """Test streaming incremental parsing of a tool call."""
        chunks = [
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{",
            '"city": "Paris"',
            "}",
            "<|tool_call_end|><|tool_calls_section_end|>",
        ]

        tool_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for tool_call_chunk in result.calls:
                if tool_call_chunk.tool_index is not None:

                    while len(tool_calls) <= tool_call_chunk.tool_index:
                        tool_calls.append({"name": "", "parameters": ""})

                    tc = tool_calls[tool_call_chunk.tool_index]

                    if tool_call_chunk.name:
                        tc["name"] += tool_call_chunk.name
                    if tool_call_chunk.parameters:
                        tc["parameters"] += tool_call_chunk.parameters

        self.assertEqual(len(tool_calls), 1)  # 断言相等
        self.assertEqual(tool_calls[0]["name"], "get_weather")  # 断言相等
        self.assertEqual(tool_calls[0]["parameters"], '{"city": "Paris"}')  # 断言相等

    # TestKimiK2Detector类的测试streamingmultipletoolcalls
    def test_streaming_multiple_tool_calls(self):
        """Test streaming incremental parsing of multiple tool calls."""
        chunks = [
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{",
            '"city": "Paris"',
            "}<|tool_call_end|>",
            "<|tool_call_begin|>functions.get_tourist_attractions:1<|tool_call_argument_begin|>{",
            '"city": "London"',
            "}<|tool_call_end|>",
            "<|tool_calls_section_end|>",
        ]

        tool_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for tool_call_chunk in result.calls:
                if tool_call_chunk.tool_index is not None:

                    while len(tool_calls) <= tool_call_chunk.tool_index:
                        tool_calls.append({"name": "", "parameters": ""})

                    tc = tool_calls[tool_call_chunk.tool_index]

                    if tool_call_chunk.name:
                        tc["name"] += tool_call_chunk.name
                    if tool_call_chunk.parameters:
                        tc["parameters"] += tool_call_chunk.parameters

        self.assertEqual(len(tool_calls), 2)  # 断言相等
        self.assertEqual(tool_calls[0]["name"], "get_weather")  # 断言相等
        self.assertEqual(tool_calls[0]["parameters"], '{"city": "Paris"}')  # 断言相等
        self.assertEqual(tool_calls[1]["name"], "get_tourist_attractions")  # 断言相等
        self.assertEqual(tool_calls[1]["parameters"], '{"city": "London"}')  # 断言相等

    # TestKimiK2Detector类的测试toolcallcompletion
    def test_tool_call_completion(self):
        """Test that the buffer and state are reset after a tool call is completed."""
        chunks = [
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{",
            '"city": "Paris"',
            "}",
            "<|tool_call_end|>",
            "<|tool_calls_section_end|>",
        ]

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)

        # After processing all chunks, the buffer should be empty and current_tool_id should be reset
        self.assertEqual(self.detector._buffer, "")  # 断言相等
        self.assertEqual(self.detector.current_tool_id, 1)  # 断言相等

    # TestKimiK2Detector类的测试toolnamestreaming
    def test_tool_name_streaming(self):
        """Test that tool names are streamed correctly with the right index."""
        chunks = [
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{",
            '"city": "Paris"',
            "}",
            "<|tool_call_end|>",
            "<|tool_call_begin|>functions.get_tourist_attractions:1<|tool_call_argument_begin|>{",
        ]

        tool_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for tool_call_chunk in result.calls:
                if tool_call_chunk.tool_index is not None:

                    while len(tool_calls) <= tool_call_chunk.tool_index:
                        tool_calls.append({"name": "", "parameters": ""})

                    tc = tool_calls[tool_call_chunk.tool_index]

                    if tool_call_chunk.name:
                        tc["name"] += tool_call_chunk.name
                    if tool_call_chunk.parameters:
                        tc["parameters"] += tool_call_chunk.parameters

        self.assertEqual(len(tool_calls), 2)  # 断言相等
        self.assertEqual(tool_calls[0]["name"], "get_weather")  # 断言相等
        self.assertEqual(tool_calls[0]["parameters"], '{"city": "Paris"}')  # 断言相等
        self.assertEqual(tool_calls[1]["name"], "get_tourist_attractions")  # 断言相等

    # TestKimiK2Detector类的测试invalidtoolcall
    def test_invalid_tool_call(self):
        """Test that invalid tool calls are handled correctly."""
        text = 'invalid_tool:0<|tool_call_argument_begin|>{"city": "Paris"}<|tool_call_end|><|tool_calls_section_end|>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)  # 断言相等
        self.assertEqual(result.normal_text, text)  # 断言相等

    # TestKimiK2Detector类的测试partialtoolcall
    def test_partial_tool_call(self):
        """Test that partial tool calls are handled correctly in streaming mode."""
        chunks = [
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{",
            '"city": "Paris"',
        ]

        tool_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for tool_call_chunk in result.calls:
                if tool_call_chunk.tool_index is not None:

                    while len(tool_calls) <= tool_call_chunk.tool_index:
                        tool_calls.append({"name": "", "parameters": ""})

                    tc = tool_calls[tool_call_chunk.tool_index]

                    if tool_call_chunk.name:
                        tc["name"] += tool_call_chunk.name
                    if tool_call_chunk.parameters:
                        tc["parameters"] += tool_call_chunk.parameters

        self.assertEqual(len(tool_calls), 1)  # 断言相等
        self.assertEqual(tool_calls[0]["name"], "get_weather")  # 断言相等
        self.assertEqual(tool_calls[0]["parameters"], '{"city": "Paris"')  # 断言相等


# TestDeepSeekV3Detector类
class TestDeepSeekV3Detector(unittest.TestCase):

    # TestDeepSeekV3Detector类的测试初始化设置
    def setUp(self):
        """Set up test tools and detector for DeepSeekV3 format testing."""
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_tourist_attractions",
                    description="Get tourist attractions",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["city"],
                    },
                ),
            ),
        ]
        self.detector = DeepSeekV3Detector()

    # TestDeepSeekV3Detector类的测试parsestreamingmultipletoolcallswithmultitokenchunk
    def test_parse_streaming_multiple_tool_calls_with_multi_token_chunk(self):
        """Test parsing multiple tool calls when streaming chunks contains multi-tokens (e.g. DeepSeekV3 enable MTP)"""
        # Simulate streaming chunks with multi-tokens for two consecutive tool calls
        chunks = [
            "<｜tool▁calls▁begin｜>",
            "<｜tool▁call▁begin｜>function",
            "<｜tool▁sep｜>get",
            "_weather\n",
            "```json\n",
            '{"city":',
            '"Shanghai',
            '"}\n```<｜tool▁call▁end｜>',
            "\n<｜tool▁call▁begin｜>",
            "function<｜tool▁sep｜>",
            "get_tour",
            "ist_att",
            "ractions\n```" 'json\n{"',
            'city": "',
            'Beijing"}\n',
            "```<｜tool▁call▁end｜>",
            "<｜tool▁calls▁end｜>",
        ]

        tool_calls_seen = []
        tool_calls_parameters = []

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            if result.calls:
                for call in result.calls:
                    if call.name:
                        tool_calls_seen.append(call.name)
                    if call.parameters:
                        tool_calls_parameters.append(call.parameters)

        # Should see both tool names
        self.assertIn("get_weather", tool_calls_seen, "Should process first tool")  # 断言包含
        self.assertIn(  # 断言包含
            "get_tourist_attractions", tool_calls_seen, "Should process second tool"
        )

        # Verify that the parameters are valid JSON and contain the expected content
        params1 = json.loads(tool_calls_parameters[0])
        params2 = json.loads(tool_calls_parameters[1])
        self.assertEqual(params1["city"], "Shanghai")  # 断言相等
        self.assertEqual(params2["city"], "Beijing")  # 断言相等


# TestDeepSeekV32Detector类
class TestDeepSeekV32Detector(unittest.TestCase):

    # TestDeepSeekV32Detector类的测试初始化设置
    def setUp(self):
        """Set up test tools and detector for DeepSeekV32 format testing."""
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Searches for information related to query and displays topn results.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query string",
                            },
                            "topn": {
                                "type": "integer",
                                "description": "Number of top results to display",
                                "default": 10,
                            },
                            "source": {
                                "type": "string",
                                "description": "Source to search within",
                                "enum": ["web", "news"],
                                "default": "web",
                            },
                        },
                        "required": ["query"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_favorite_tourist_spot",
                    description="Return the favorite tourist spot for a given city.",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            ),
        ]
        self.detector = DeepSeekV32Detector()
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer

        self.tokenizer = get_tokenizer("deepseek-ai/DeepSeek-V3.2")
        self.interval = 1

    # TestDeepSeekV32Detector类的测试detectandparsexmlformat
    def test_detect_and_parse_xml_format(self):
        """Test parsing standard XML format (DSML)"""
        text = """I'll help you with information about San Francisco and get its favorite tourist spot for you.\n\n
        <｜DSML｜function_calls>\n
            <｜DSML｜invoke name="get_favorite_tourist_spot">\n
                <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>\n
            </｜DSML｜invoke>\n
            <｜DSML｜invoke name="search">
                <｜DSML｜parameter name="query" string="true">WebNav benchmark</｜DSML｜parameter>
                <｜DSML｜parameter name="topn" string="false">10</｜DSML｜parameter>
                <｜DSML｜parameter name="source" string="true">web</｜DSML｜parameter>
            </｜DSML｜invoke>
        </｜DSML｜function_calls>
        """
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertIn("I'll help you with information", result.normal_text)  # 断言包含
        self.assertEqual(len(result.calls), 2)  # 断言相等

        # Check first call
        call1 = result.calls[0]
        self.assertEqual(call1.name, "get_favorite_tourist_spot")  # 断言相等
        params1 = json.loads(call1.parameters)
        self.assertEqual(params1["city"], "San Francisco")  # 断言相等

        # Check second call
        call2 = result.calls[1]
        self.assertEqual(call2.name, "search")  # 断言相等
        params2 = json.loads(call2.parameters)
        self.assertEqual(params2["query"], "WebNav benchmark")  # 断言相等
        self.assertEqual(params2["topn"], 10)  # 断言相等
        self.assertEqual(params2["source"], "web")  # 断言相等

    # TestDeepSeekV32Detector类的测试detectandparsejsonformat
    def test_detect_and_parse_json_format(self):
        """Test parsing JSON format inside invoke tags"""
        text = """I'll help you with information about San Francisco and get its favorite tourist spot for you.

        <｜DSML｜function_calls>
            <｜DSML｜invoke name="get_favorite_tourist_spot">
            {
                "city": "San Francisco"
            }
        </｜DSML｜invoke>
            <｜DSML｜invoke name="search">
            {
                "query": "WebNav benchmark",
                "topn": 10,
                "source": "web"
            }
        </｜DSML｜invoke>
        </｜DSML｜function_calls>
        """
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertIn("I'll help you with information", result.normal_text)  # 断言包含
        self.assertEqual(len(result.calls), 2)  # 断言相等

        # Check first call
        call1 = result.calls[0]
        self.assertEqual(call1.name, "get_favorite_tourist_spot")  # 断言相等
        params1 = json.loads(call1.parameters)
        self.assertEqual(params1["city"], "San Francisco")  # 断言相等

        # Check second call
        call2 = result.calls[1]
        self.assertEqual(call2.name, "search")  # 断言相等
        params2 = json.loads(call2.parameters)
        self.assertEqual(params2["query"], "WebNav benchmark")  # 断言相等
        self.assertEqual(params2["topn"], 10)  # 断言相等
        self.assertEqual(params2["source"], "web")  # 断言相等

    # TestDeepSeekV32Detector类的测试streamingxmlformat
    def test_streaming_xml_format(self):
        """Test streaming parsing of XML format"""
        text = """<｜DSML｜function_calls>
            <｜DSML｜invoke name="get_favorite_tourist_spot">
                <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
                <｜DSML｜parameter name="another_city" string="true">London</｜DSML｜parameter>
                <｜DSML｜parameter name="topn" string="false">10</｜DSML｜parameter>
                <｜DSML｜parameter name="obj" string="false">{"name": "John", "age": 30}</｜DSML｜parameter>
            </｜DSML｜invoke>
        </｜DSML｜function_calls>"""

        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunk_ids = [
            input_ids[i : i + self.interval]
            for i in range(0, len(input_ids), self.interval)
        ]
        chunks = [self.tokenizer.decode(chunk_id) for chunk_id in chunk_ids]

        tool_calls_by_index = {}

        num_tool_call_chunks = 0
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for call in result.calls:
                num_tool_call_chunks += 1
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        self.assertGreater(num_tool_call_chunks, 8)  # 断言大于

        self.assertEqual(len(tool_calls_by_index), 1)  # 断言相等
        self.assertEqual(tool_calls_by_index[0]["name"], "get_favorite_tourist_spot")  # 断言相等
        params = json.loads(tool_calls_by_index[0]["parameters"])
        self.assertEqual(params["city"], "San Francisco")  # 断言相等
        self.assertEqual(params["another_city"], "London")  # 断言相等
        self.assertEqual(params["topn"], 10)  # 断言相等
        self.assertEqual(params["obj"]["name"], "John")  # 断言相等
        self.assertEqual(params["obj"]["age"], 30)  # 断言相等

    # TestDeepSeekV32Detector类的测试streamingjsonformat
    def test_streaming_json_format(self):
        """Test streaming parsing of JSON format"""
        text = """<｜DSML｜function_calls>
            <｜DSML｜invoke name="get_favorite_tourist_spot">
            {
                "city": "San Francisco",
                "another_city": "London",
                "topn": 10,
                "obj": {
                    "name": "John",
                    "age": 30
                }
            }
            </｜DSML｜invoke>
        </｜DSML｜function_calls>"""

        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunk_ids = [
            input_ids[i : i + self.interval]
            for i in range(0, len(input_ids), self.interval)
        ]
        chunks = [self.tokenizer.decode(chunk_id) for chunk_id in chunk_ids]

        tool_calls_by_index = {}

        num_tool_call_chunks = 0
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for call in result.calls:
                num_tool_call_chunks += 1
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        self.assertGreater(num_tool_call_chunks, 8)  # 断言大于
        self.assertEqual(len(tool_calls_by_index), 1)  # 断言相等
        self.assertEqual(tool_calls_by_index[0]["name"], "get_favorite_tourist_spot")  # 断言相等

        # Clean up parameters string if needed (trim whitespace)
        params_str = tool_calls_by_index[0]["parameters"].strip()
        params = json.loads(params_str)
        self.assertEqual(params["city"], "San Francisco")  # 断言相等

    # TestDeepSeekV32Detector类的测试detectandparsenoparameters
    def test_detect_and_parse_no_parameters(self):
        """Test parsing function calls with no parameters (non-streaming)"""
        # Add a no-parameter tool
        tools_with_no_param = self.tools + [
            Tool(
                type="function",
                function=Function(
                    name="get_date",
                    description="Get the current date.",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
        ]

        text = """Let me get the current date for you.

<｜DSML｜function_calls>
<｜DSML｜invoke name="get_date">
</｜DSML｜invoke>
</｜DSML｜function_calls>"""

        result = self.detector.detect_and_parse(text, tools_with_no_param)

        self.assertIn("Let me get the current date", result.normal_text)  # 断言包含
        self.assertEqual(len(result.calls), 1)  # 断言相等

        call = result.calls[0]
        self.assertEqual(call.name, "get_date")  # 断言相等
        params = json.loads(call.parameters)
        self.assertEqual(params, {})  # 断言相等

    # TestDeepSeekV32Detector类的测试streamingnoparameters
    def test_streaming_no_parameters(self):
        """Test streaming parsing of function calls with no parameters.

        This test verifies the fix for the bug where functions with no parameters
        were being silently skipped in streaming mode.
        """
        # Add a no-parameter tool
        tools_with_no_param = self.tools + [
            Tool(
                type="function",
                function=Function(
                    name="get_date",
                    description="Get the current date.",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
        ]

        text = """<｜DSML｜function_calls>
<｜DSML｜invoke name="get_date">
</｜DSML｜invoke>
</｜DSML｜function_calls>"""

        # Reset detector state
        self.detector = DeepSeekV32Detector()

        # Simulate streaming by splitting into small chunks
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunk_ids = [
            input_ids[i : i + self.interval]
            for i in range(0, len(input_ids), self.interval)
        ]
        chunks = [self.tokenizer.decode(chunk_id) for chunk_id in chunk_ids]

        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, tools_with_no_param)
            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        # Verify that the no-parameter function was correctly parsed
        self.assertEqual(  # 断言相等
            len(tool_calls_by_index), 1, "Should have exactly one tool call"
        )
        self.assertEqual(tool_calls_by_index[0]["name"], "get_date")  # 断言相等

        # Parameters should be empty JSON object
        params_str = tool_calls_by_index[0]["parameters"].strip()
        params = json.loads(params_str)
        self.assertEqual(params, {})  # 断言相等

    # TestDeepSeekV32Detector类的测试streamingnoparameterswithwhitespace
    def test_streaming_no_parameters_with_whitespace(self):
        """Test streaming parsing when invoke content has only whitespace (newlines)."""
        tools_with_no_param = self.tools + [
            Tool(
                type="function",
                function=Function(
                    name="get_date",
                    description="Get the current date.",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
        ]

        # This format has newlines inside the invoke tag (common model output)
        text = """<｜DSML｜function_calls>
<｜DSML｜invoke name="get_date">

</｜DSML｜invoke>
</｜DSML｜function_calls>"""

        # Reset detector state
        self.detector = DeepSeekV32Detector()

        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunk_ids = [
            input_ids[i : i + self.interval]
            for i in range(0, len(input_ids), self.interval)
        ]
        chunks = [self.tokenizer.decode(chunk_id) for chunk_id in chunk_ids]

        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, tools_with_no_param)
            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        # Should still parse correctly even with whitespace-only content
        self.assertEqual(  # 断言相等
            len(tool_calls_by_index), 1, "Should have exactly one tool call"
        )
        self.assertEqual(tool_calls_by_index[0]["name"], "get_date")  # 断言相等
        params = json.loads(tool_calls_by_index[0]["parameters"])
        self.assertEqual(params, {})  # 断言相等

    # TestDeepSeekV32Detector类的测试getmodelstructuraltag
    def test_get_model_structural_tag(self):
        import xgrammar as xgr

        self.assertEqual(self.detector.get_structural_tag_name(), "deepseek_v3_2")  # 断言相等

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=True
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)
        serialized = structural_tag.model_dump_json()
        self.assertIn("</｜DSML｜invoke>\\n", serialized)  # 断言包含
        self.assertNotIn("</｜DSML｜invoke>\\n\\n", serialized)  # 断言不包含

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=False
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=True, tool_choice="required"
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=False, tool_choice="required"
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        tool_choice_name = ToolChoiceFuncName(name="search")
        tool_choice = ToolChoice(function=tool_choice_name)
        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=True, tool_choice=tool_choice
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=False, tool_choice=tool_choice
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

    # TestDeepSeekV32Detector类的测试selfclosingzeroarginvoke
    def test_self_closing_zero_arg_invoke(self):
        """V32 inherits the same regex; verify self-closing parses to empty
        params here too (V32 model rarely emits this shape, but the parser
        must agree with V4 since V4 inherits from V32)."""
        submit_tool = Tool(
            type="function",
            function=Function(
                name="submit",
                parameters={"type": "object", "properties": {}},
            ),
        )
        text = (
            '<｜DSML｜function_calls>\n<｜DSML｜invoke name="submit"/>\n'
            "</｜DSML｜function_calls>"
        )
        result = self.detector.detect_and_parse(text, [submit_tool])
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "submit")  # 断言相等
        self.assertEqual(json.loads(result.calls[0].parameters), {})  # 断言相等


# TestDeepSeekV4Detector类
class TestDeepSeekV4Detector(unittest.TestCase):

    # TestDeepSeekV4Detector类的测试初始化设置
    def setUp(self):
        """Set up test tools and detector for DeepSeekV4 format testing."""
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Searches for information related to query and displays topn results.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query string",
                            },
                            "topn": {
                                "type": "integer",
                                "description": "Number of top results to display",
                                "default": 10,
                            },
                            "source": {
                                "type": "string",
                                "description": "Source to search within",
                                "enum": ["web", "news"],
                                "default": "web",
                            },
                        },
                        "required": ["query"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_favorite_tourist_spot",
                    description="Return the favorite tourist spot for a given city.",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            ),
        ]
        self.detector = DeepSeekV4Detector()
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer

        self.tokenizer = get_tokenizer("deepseek-ai/DeepSeek-V3.2")
        self.interval = 1

    # TestDeepSeekV4Detector类的测试detectandparsexmlformat
    def test_detect_and_parse_xml_format(self):
        """Test parsing standard XML format (DSML)"""
        text = """I'll help you with information about San Francisco and get its favorite tourist spot for you.\n\n
        <｜DSML｜tool_calls>\n
            <｜DSML｜invoke name="get_favorite_tourist_spot">\n
                <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>\n
            </｜DSML｜invoke>\n
            <｜DSML｜invoke name="search">
                <｜DSML｜parameter name="query" string="true">WebNav benchmark</｜DSML｜parameter>
                <｜DSML｜parameter name="topn" string="false">10</｜DSML｜parameter>
                <｜DSML｜parameter name="source" string="true">web</｜DSML｜parameter>
            </｜DSML｜invoke>
        </｜DSML｜tool_calls>
        """
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertIn("I'll help you with information", result.normal_text)  # 断言包含
        self.assertEqual(len(result.calls), 2)  # 断言相等

        # Check first call
        call1 = result.calls[0]
        self.assertEqual(call1.name, "get_favorite_tourist_spot")  # 断言相等
        params1 = json.loads(call1.parameters)
        self.assertEqual(params1["city"], "San Francisco")  # 断言相等

        # Check second call
        call2 = result.calls[1]
        self.assertEqual(call2.name, "search")  # 断言相等
        params2 = json.loads(call2.parameters)
        self.assertEqual(params2["query"], "WebNav benchmark")  # 断言相等
        self.assertEqual(params2["topn"], 10)  # 断言相等
        self.assertEqual(params2["source"], "web")  # 断言相等

    # TestDeepSeekV4Detector类的测试detectandparsejsonformat
    def test_detect_and_parse_json_format(self):
        """Test parsing JSON format inside invoke tags"""
        text = """I'll help you with information about San Francisco and get its favorite tourist spot for you.

        <｜DSML｜tool_calls>
            <｜DSML｜invoke name="get_favorite_tourist_spot">
            {
                "city": "San Francisco"
            }
        </｜DSML｜invoke>
            <｜DSML｜invoke name="search">
            {
                "query": "WebNav benchmark",
                "topn": 10,
                "source": "web"
            }
        </｜DSML｜invoke>
        </｜DSML｜tool_calls>
        """
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertIn("I'll help you with information", result.normal_text)  # 断言包含
        self.assertEqual(len(result.calls), 2)  # 断言相等

        # Check first call
        call1 = result.calls[0]
        self.assertEqual(call1.name, "get_favorite_tourist_spot")  # 断言相等
        params1 = json.loads(call1.parameters)
        self.assertEqual(params1["city"], "San Francisco")  # 断言相等

        # Check second call
        call2 = result.calls[1]
        self.assertEqual(call2.name, "search")  # 断言相等
        params2 = json.loads(call2.parameters)
        self.assertEqual(params2["query"], "WebNav benchmark")  # 断言相等
        self.assertEqual(params2["topn"], 10)  # 断言相等
        self.assertEqual(params2["source"], "web")  # 断言相等

    # TestDeepSeekV4Detector类的测试streamingxmlformat
    def test_streaming_xml_format(self):
        """Test streaming parsing of XML format"""
        text = """<｜DSML｜tool_calls>
            <｜DSML｜invoke name="get_favorite_tourist_spot">
                <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
                <｜DSML｜parameter name="another_city" string="true">London</｜DSML｜parameter>
                <｜DSML｜parameter name="topn" string="false">10</｜DSML｜parameter>
                <｜DSML｜parameter name="obj" string="false">{"name": "John", "age": 30}</｜DSML｜parameter>
            </｜DSML｜invoke>
        </｜DSML｜tool_calls>"""

        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunk_ids = [
            input_ids[i : i + self.interval]
            for i in range(0, len(input_ids), self.interval)
        ]
        chunks = [self.tokenizer.decode(chunk_id) for chunk_id in chunk_ids]

        tool_calls_by_index = {}

        num_tool_call_chunks = 0
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for call in result.calls:
                num_tool_call_chunks += 1
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        self.assertGreater(num_tool_call_chunks, 8)  # 断言大于

        self.assertEqual(len(tool_calls_by_index), 1)  # 断言相等
        self.assertEqual(tool_calls_by_index[0]["name"], "get_favorite_tourist_spot")  # 断言相等
        params = json.loads(tool_calls_by_index[0]["parameters"])
        self.assertEqual(params["city"], "San Francisco")  # 断言相等
        self.assertEqual(params["another_city"], "London")  # 断言相等
        self.assertEqual(params["topn"], 10)  # 断言相等
        self.assertEqual(params["obj"]["name"], "John")  # 断言相等
        self.assertEqual(params["obj"]["age"], 30)  # 断言相等

    # TestDeepSeekV4Detector类的测试streamingjsonformat
    def test_streaming_json_format(self):
        """Test streaming parsing of JSON format"""
        text = """<｜DSML｜tool_calls>
            <｜DSML｜invoke name="get_favorite_tourist_spot">
            {
                "city": "San Francisco",
                "another_city": "London",
                "topn": 10,
                "obj": {
                    "name": "John",
                    "age": 30
                }
            }
            </｜DSML｜invoke>
        </｜DSML｜tool_calls>"""

        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunk_ids = [
            input_ids[i : i + self.interval]
            for i in range(0, len(input_ids), self.interval)
        ]
        chunks = [self.tokenizer.decode(chunk_id) for chunk_id in chunk_ids]

        tool_calls_by_index = {}

        num_tool_call_chunks = 0
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for call in result.calls:
                num_tool_call_chunks += 1
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        self.assertGreater(num_tool_call_chunks, 8)  # 断言大于
        self.assertEqual(len(tool_calls_by_index), 1)  # 断言相等
        self.assertEqual(tool_calls_by_index[0]["name"], "get_favorite_tourist_spot")  # 断言相等

        # Clean up parameters string if needed (trim whitespace)
        params_str = tool_calls_by_index[0]["parameters"].strip()
        params = json.loads(params_str)
        self.assertEqual(params["city"], "San Francisco")  # 断言相等

    # TestDeepSeekV4Detector类的测试detectandparsenoparameters
    def test_detect_and_parse_no_parameters(self):
        """Test parsing function calls with no parameters (non-streaming)"""
        # Add a no-parameter tool
        tools_with_no_param = self.tools + [
            Tool(
                type="function",
                function=Function(
                    name="get_date",
                    description="Get the current date.",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
        ]

        text = """Let me get the current date for you.

<｜DSML｜tool_calls>
<｜DSML｜invoke name="get_date">
</｜DSML｜invoke>
</｜DSML｜tool_calls>"""

        result = self.detector.detect_and_parse(text, tools_with_no_param)

        self.assertIn("Let me get the current date", result.normal_text)  # 断言包含
        self.assertEqual(len(result.calls), 1)  # 断言相等

        call = result.calls[0]
        self.assertEqual(call.name, "get_date")  # 断言相等
        params = json.loads(call.parameters)
        self.assertEqual(params, {})  # 断言相等

    # TestDeepSeekV4Detector类的测试streamingnoparameters
    def test_streaming_no_parameters(self):
        """Test streaming parsing of function calls with no parameters.

        This test verifies the fix for the bug where functions with no parameters
        were being silently skipped in streaming mode.
        """
        # Add a no-parameter tool
        tools_with_no_param = self.tools + [
            Tool(
                type="function",
                function=Function(
                    name="get_date",
                    description="Get the current date.",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
        ]

        text = """<｜DSML｜tool_calls>
<｜DSML｜invoke name="get_date">
</｜DSML｜invoke>
</｜DSML｜tool_calls>"""

        # Reset detector state
        self.detector = DeepSeekV4Detector()

        # Simulate streaming by splitting into small chunks
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunk_ids = [
            input_ids[i : i + self.interval]
            for i in range(0, len(input_ids), self.interval)
        ]
        chunks = [self.tokenizer.decode(chunk_id) for chunk_id in chunk_ids]

        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, tools_with_no_param)
            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        # Verify that the no-parameter function was correctly parsed
        self.assertEqual(  # 断言相等
            len(tool_calls_by_index), 1, "Should have exactly one tool call"
        )
        self.assertEqual(tool_calls_by_index[0]["name"], "get_date")  # 断言相等

        # Parameters should be empty JSON object
        params_str = tool_calls_by_index[0]["parameters"].strip()
        params = json.loads(params_str)
        self.assertEqual(params, {})  # 断言相等

    # TestDeepSeekV4Detector类的测试streamingnoparameterswithwhitespace
    def test_streaming_no_parameters_with_whitespace(self):
        """Test streaming parsing when invoke content has only whitespace (newlines)."""
        tools_with_no_param = self.tools + [
            Tool(
                type="function",
                function=Function(
                    name="get_date",
                    description="Get the current date.",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
        ]

        # This format has newlines inside the invoke tag (common model output)
        text = """<｜DSML｜tool_calls>
<｜DSML｜invoke name="get_date">

</｜DSML｜invoke>
</｜DSML｜tool_calls>"""

        # Reset detector state
        self.detector = DeepSeekV4Detector()

        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunk_ids = [
            input_ids[i : i + self.interval]
            for i in range(0, len(input_ids), self.interval)
        ]
        chunks = [self.tokenizer.decode(chunk_id) for chunk_id in chunk_ids]

        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, tools_with_no_param)
            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        # Should still parse correctly even with whitespace-only content
        self.assertEqual(  # 断言相等
            len(tool_calls_by_index), 1, "Should have exactly one tool call"
        )
        self.assertEqual(tool_calls_by_index[0]["name"], "get_date")  # 断言相等
        params = json.loads(tool_calls_by_index[0]["parameters"])
        self.assertEqual(params, {})  # 断言相等

    # TestDeepSeekV4Detector类的测试getmodelstructuraltag
    def test_get_model_structural_tag(self):
        import xgrammar as xgr

        self.assertEqual(self.detector.get_structural_tag_name(), "deepseek_v4")  # 断言相等

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=True
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)
        serialized = structural_tag.model_dump_json()
        self.assertIn("</｜DSML｜invoke>\\n", serialized)  # 断言包含
        self.assertNotIn("</｜DSML｜invoke>\\n\\n", serialized)  # 断言不包含

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=False
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=True, tool_choice="required"
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=False, tool_choice="required"
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        tool_choice_name = ToolChoiceFuncName(name="search")
        tool_choice = ToolChoice(function=tool_choice_name)
        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=True, tool_choice=tool_choice
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=False, tool_choice=tool_choice
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

    # TestDeepSeekV4Detector类的测试selfclosingzeroarginvoke
    def test_self_closing_zero_arg_invoke(self):
        """V4 emits `<｜DSML｜invoke name="x"/>` for zero-arg tools; the
        detector must parse it as a complete tool call with empty params
        instead of leaking the raw markup back into normal_text."""
        submit_tool = Tool(
            type="function",
            function=Function(
                name="submit",
                description="Submit the final answer.",
                parameters={"type": "object", "properties": {}},
            ),
        )

        text = (
            "Final answer.\n"
            '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="submit"/>\n'
            "</｜DSML｜tool_calls>"
        )
        result = self.detector.detect_and_parse(text, [submit_tool])
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "submit")  # 断言相等
        self.assertEqual(json.loads(result.calls[0].parameters), {})  # 断言相等
        self.assertNotIn("DSML", result.normal_text)  # 断言不包含

    # TestDeepSeekV4Detector类的测试selfclosingmixedwithlongform
    def test_self_closing_mixed_with_long_form(self):
        """Mix of long-form (with params) and self-closing tags in one block."""
        submit_tool = Tool(
            type="function",
            function=Function(
                name="submit",
                parameters={"type": "object", "properties": {}},
            ),
        )
        text = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_favorite_tourist_spot">\n'
            '<｜DSML｜parameter name="city" string="true">SF</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            '<｜DSML｜invoke name="submit"/>\n'
            "</｜DSML｜tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools + [submit_tool])
        self.assertEqual(len(result.calls), 2)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_favorite_tourist_spot")  # 断言相等
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "SF"})  # 断言相等
        self.assertEqual(result.calls[1].name, "submit")  # 断言相等
        self.assertEqual(json.loads(result.calls[1].parameters), {})  # 断言相等

    # TestDeepSeekV4Detector类的测试streamingselfclosinginvoke
    def test_streaming_self_closing_invoke(self):
        """Self-closing invoke must terminate cleanly even when `/>` arrives
        after the `name=` attribute crosses chunk boundaries."""
        submit_tool = Tool(
            type="function",
            function=Function(
                name="submit",
                parameters={"type": "object", "properties": {}},
            ),
        )
        # Build the prompt and feed it through the tokenizer to exercise the
        # same chunk shapes the runtime sees.
        text = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="submit"/>\n'
            "</｜DSML｜tool_calls>"
        )
        self.detector = DeepSeekV4Detector()
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = [
            self.tokenizer.decode(input_ids[i : i + self.interval])
            for i in range(0, len(input_ids), self.interval)
        ]

        tool_calls_by_index = {}
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, [submit_tool])
            for call in result.calls:
                if call.tool_index is None:
                    continue
                slot = tool_calls_by_index.setdefault(
                    call.tool_index, {"name": "", "parameters": ""}
                )
                if call.name:
                    slot["name"] = call.name
                if call.parameters:
                    slot["parameters"] += call.parameters

        self.assertEqual(len(tool_calls_by_index), 1)  # 断言相等
        self.assertEqual(tool_calls_by_index[0]["name"], "submit")  # 断言相等
        self.assertEqual(json.loads(tool_calls_by_index[0]["parameters"]), {})  # 断言相等


# TestQwen3CoderDetector类
class TestQwen3CoderDetector(unittest.TestCase):
    """Test suite for Qwen3CoderDetector."""

    def setUp(self):
        """Initialize test fixtures before each test method."""
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_current_weather",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                            "days": {"type": "integer"},
                        },
                        "required": ["location"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="sql_interpreter",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "dry_run": {"type": "boolean"},
                        },
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="TodoWrite",
                    parameters={
                        "type": "object",
                        "properties": {
                            "todos": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "content": {"type": "string"},
                                        "status": {"type": "string"},
                                    },
                                    "required": ["content", "status"],
                                },
                            },
                        },
                    },
                ),
            ),
        ]
        self.detector = Qwen3CoderDetector()

    # ==================== Basic Functionality Tests ====================

    def test_plain_text_only(self):
        """
        Test parsing of plain text without any tool calls.

        Scenario: Input contains only plain text, no tool call markers.
        Purpose: Verify that plain text is correctly identified and no false tool calls are detected.
        """
        text = "This is plain text without any tool calls."
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, text)  # 断言相等
        self.assertEqual(len(result.calls), 0)  # 断言相等

    # TestQwen3CoderDetector类的测试singletoolcall
    def test_single_tool_call(self):
        """
        Test parsing of a single tool call.

        Scenario: Input contains one complete tool call with parameters.
        Purpose: Verify correct extraction of tool name and parameters.
        """
        text = """<tool_call>
<function=get_current_weather>
<parameter=location>Boston</parameter>
<parameter=unit>celsius</parameter>
<parameter=days>3</parameter>
</function>
</tool_call>"""
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_current_weather")  # 断言相等

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["location"], "Boston")  # 断言相等
        self.assertEqual(params["unit"], "celsius")  # 断言相等
        self.assertEqual(params["days"], 3)  # 断言相等

    # TestQwen3CoderDetector类的测试singletoolcallwithtextprefix
    def test_single_tool_call_with_text_prefix(self):
        """
        Test parsing of tool call with preceding text.

        Scenario: Input has plain text followed by a tool call.
        Purpose: Verify correct separation of text and tool call.
        """
        text = """Let me check the weather for you.

<tool_call>
<function=get_current_weather>
<parameter=location>New York</parameter>
</function>
</tool_call>"""
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertTrue(result.normal_text.startswith("Let me check"))  # 断言为真
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_current_weather")  # 断言相等

    # TestQwen3CoderDetector类的测试multipletoolcalls
    def test_multiple_tool_calls(self):
        """
        Test parsing of multiple consecutive tool calls.

        Scenario: Input contains two tool calls one after another.
        Purpose: Verify that multiple tool calls are correctly identified and parsed.
        """
        text = """<tool_call>
<function=get_current_weather>
<parameter=location>New York</parameter>
</function>
</tool_call>
<tool_call>
<function=sql_interpreter>
<parameter=query>SELECT * FROM users</parameter>
<parameter=dry_run>True</parameter>
</function>
</tool_call>"""
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 2)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_current_weather")  # 断言相等
        self.assertEqual(result.calls[1].name, "sql_interpreter")  # 断言相等

        params1 = json.loads(result.calls[0].parameters)
        self.assertEqual(params1["location"], "New York")  # 断言相等

        params2 = json.loads(result.calls[1].parameters)
        self.assertEqual(params2["query"], "SELECT * FROM users")  # 断言相等
        self.assertEqual(params2["dry_run"], True)  # 断言相等

    # ==================== Streaming Tests ====================

    def test_streaming_single_tool_call(self):
        """
        Test streaming parsing of a single tool call.

        Scenario: Tool call is fed incrementally in chunks.
        Purpose: Verify streaming parser correctly assembles tool call from chunks.
        """
        chunks = [
            "<tool_call>",
            "<function=get_current_weather>",
            "<parameter=location>",
            "Boston",
            "</parameter>",
            "<parameter=unit>celsius</parameter>",
            "</function>",
            "</tool_call>",
        ]

        detector = Qwen3CoderDetector()
        all_calls = []
        collected_params = ""

        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)
            for call in result.calls:
                if call.parameters:
                    collected_params += call.parameters

        # Verify we got the tool call
        self.assertGreater(len(all_calls), 0)  # 断言大于

        # Verify parameters were collected
        if collected_params:
            params = json.loads(collected_params)
            self.assertEqual(params["location"], "Boston")  # 断言相等
            self.assertEqual(params["unit"], "celsius")  # 断言相等

    # TestQwen3CoderDetector类的测试streamingwithtextandtool
    def test_streaming_with_text_and_tool(self):
        """
        Test streaming parsing with mixed text and tool call.

        Scenario: Stream contains plain text followed by a tool call.
        Purpose: Verify correct separation in streaming mode.
        """
        chunks = [
            "Let me ",
            "help you.\n\n",
            "<tool_call>",
            "<function=get_current_weather>",
            "<parameter=location>Paris</parameter>",
            "</function>",
            "</tool_call>",
        ]

        detector = Qwen3CoderDetector()
        full_text = ""
        all_calls = []

        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            if result.normal_text:
                full_text += result.normal_text
            all_calls.extend(result.calls)

        self.assertTrue(full_text.startswith("Let me"))  # 断言为真
        self.assertGreater(len(all_calls), 0)  # 断言大于

    # ==================== Parameter Type Tests ====================

    def test_integer_parameter_conversion(self):
        """
        Test correct type conversion for integer parameters.

        Scenario: Tool call with integer parameter.
        Purpose: Verify integer values are correctly parsed and typed.
        """
        text = """<tool_call>
<function=get_current_weather>
<parameter=location>Tokyo</parameter>
<parameter=days>5</parameter>
</function>
</tool_call>"""
        result = self.detector.detect_and_parse(text, self.tools)

        params = json.loads(result.calls[0].parameters)
        self.assertIsInstance(params["days"], int)
        self.assertEqual(params["days"], 5)  # 断言相等

    # TestQwen3CoderDetector类的测试booleanparameterconversion
    def test_boolean_parameter_conversion(self):
        """
        Test correct type conversion for boolean parameters.

        Scenario: Tool call with boolean parameter.
        Purpose: Verify boolean values are correctly parsed.
        """
        text = """<tool_call>
<function=sql_interpreter>
<parameter=query>SELECT 1</parameter>
<parameter=dry_run>True</parameter>
</function>
</tool_call>"""
        result = self.detector.detect_and_parse(text, self.tools)

        params = json.loads(result.calls[0].parameters)
        self.assertIsInstance(params["dry_run"], bool)
        self.assertEqual(params["dry_run"], True)  # 断言相等

    # TestQwen3CoderDetector类的测试complexarrayparameter
    def test_complex_array_parameter(self):
        """
        Test parsing of complex array parameters.

        Scenario: Tool call with array of objects as parameter.
        Purpose: Verify complex nested structures are correctly parsed.
        """
        text = """<tool_call>
<function=TodoWrite>
<parameter=todos>
[
  {"content": "Buy groceries", "status": "pending"},
  {"content": "Finish report", "status": "completed"}
]
</parameter>
</function>
</tool_call>"""
        result = self.detector.detect_and_parse(text, self.tools)

        params = json.loads(result.calls[0].parameters)
        self.assertIsInstance(params["todos"], list)
        self.assertEqual(len(params["todos"]), 2)  # 断言相等
        self.assertEqual(params["todos"][0]["content"], "Buy groceries")  # 断言相等
        self.assertEqual(params["todos"][1]["status"], "completed")  # 断言相等

    # ==================== Edge Cases ====================

    def test_empty_parameter_value(self):
        """
        Test handling of empty parameter values.

        Scenario: Tool call with empty parameter value.
        Purpose: Verify empty values are handled gracefully.
        """
        text = """<tool_call>
<function=get_current_weather>
<parameter=location></parameter>
</function>
</tool_call>"""
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)  # 断言相等
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["location"], "")  # 断言相等

    # TestQwen3CoderDetector类的测试parameterwithspecialcharacters
    def test_parameter_with_special_characters(self):
        """
        Test handling of parameters with special characters.

        Scenario: Parameter value contains special characters like quotes, newlines.
        Purpose: Verify special characters are correctly preserved.
        """
        text = """<tool_call>
<function=sql_interpreter>
<parameter=query>SELECT * FROM users WHERE name = 'John "Doe"'</parameter>
</function>
</tool_call>"""
        result = self.detector.detect_and_parse(text, self.tools)

        params = json.loads(result.calls[0].parameters)
        self.assertIn("John", params["query"])  # 断言包含
        self.assertIn("Doe", params["query"])  # 断言包含

    # TestQwen3CoderDetector类的测试incompletetoolcall
    def test_incomplete_tool_call(self):
        """
        Test handling of incomplete tool call at end of stream.

        Scenario: Stream ends with an incomplete tool call (missing closing tag).
        Purpose: Verify detector handles incomplete input gracefully without crashing.
        """
        text = """<tool_call>
<function=get_current_weather>
<parameter=location>London"""

        # Should not crash
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertIsInstance(result, StreamingParseResult)

    # TestQwen3CoderDetector类的测试hastoolcalldetection
    def test_has_tool_call_detection(self):
        """
        Test the has_tool_call method for detecting tool call markers.

        Scenario: Various inputs with and without tool call markers.
        Purpose: Verify correct detection of tool call presence.
        """
        self.assertTrue(self.detector.has_tool_call("<tool_call>"))  # 断言为真
        self.assertTrue(self.detector.has_tool_call("text <tool_call> more"))  # 断言为真
        self.assertFalse(self.detector.has_tool_call("plain text only"))  # 断言为假
        self.assertFalse(self.detector.has_tool_call(""))  # 断言为假

    # ==================== Structural tag (xgrammar builtin) ====================
    # Qwen3 Coder uses the new builtin structural tag path. supports_structural_tag()
    # is True so required/named tool_choice routes through FunctionCallParser
    # instead of JsonArrayParser.

    def test_supports_structural_tag(self):
        self.assertTrue(self.detector.supports_structural_tag())  # 断言为真

    # TestQwen3CoderDetector类的测试getmodelstructuraltag
    def test_get_model_structural_tag(self):
        import xgrammar as xgr

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=True
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=False
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=True, tool_choice="required"
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=False, tool_choice="required"
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        tool_choice_name = ToolChoiceFuncName(name="get_current_weather")
        tool_choice = ToolChoice(function=tool_choice_name)
        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=True, tool_choice=tool_choice
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=False, tool_choice=tool_choice
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)


# TestGptOssDetector类
class TestGptOssDetector(unittest.TestCase):

    # TestGptOssDetector类的测试初始化设置
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Searches for information.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "topn": {"type": "integer"},
                        },
                        "required": ["query"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information for a city.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["city"],
                    },
                ),
            ),
        ]
        self.detector = GptOssDetector()

    # TestGptOssDetector类的测试getmodelstructuraltag
    def test_get_model_structural_tag(self):
        import xgrammar as xgr

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=True
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=False
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=True, tool_choice="required"
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=False, tool_choice="required"
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        tool_choice_name = ToolChoiceFuncName(name="search")
        tool_choice = ToolChoice(function=tool_choice_name)
        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=True, tool_choice=tool_choice
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=False, tool_choice=tool_choice
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)


# TestGlm4MoeDetector类
class TestGlm4MoeDetector(unittest.TestCase):

    # TestGlm4MoeDetector类的测试初始化设置
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
                            "date": {"type": "string", "description": "Date"},
                        },
                        "required": ["city", "date"],
                    },
                ),
            ),
        ]
        self.detector = Glm4MoeDetector()

    # TestGlm4MoeDetector类的测试singletoolcall
    def test_single_tool_call(self):
        text = (
            "<tool_call>get_weather\n"
            "<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n"
            "<arg_key>date</arg_key>\n<arg_value>2024-06-27</arg_value>\n"
            "</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等
        self.assertEqual(  # 断言相等
            result.calls[0].parameters, '{"city": "Beijing", "date": "2024-06-27"}'
        )
        self.assertEqual(result.normal_text, "")  # 断言相等

    # TestGlm4MoeDetector类的测试multipletoolcalls
    def test_multiple_tool_calls(self):
        text = (
            "<tool_call>get_weather\n"
            "<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n"
            "<arg_key>date</arg_key>\n<arg_value>2024-06-27</arg_value>\n"
            "</tool_call>"
            "<tool_call>get_weather\n"
            "<arg_key>city</arg_key>\n<arg_value>Shanghai</arg_value>\n"
            "<arg_key>date</arg_key>\n<arg_value>2024-06-28</arg_value>\n"
            "</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等
        self.assertEqual(  # 断言相等
            result.calls[0].parameters, '{"city": "Beijing", "date": "2024-06-27"}'
        )
        self.assertEqual(result.calls[1].name, "get_weather")  # 断言相等
        self.assertEqual(  # 断言相等
            result.calls[1].parameters, '{"city": "Shanghai", "date": "2024-06-28"}'
        )
        self.assertEqual(result.normal_text, "")  # 断言相等

    # TestGlm4MoeDetector类的测试streamingtoolcall
    def test_streaming_tool_call(self):
        """Test streaming incremental parsing of a tool call."""
        chunks = [
            "<tool_call>get_weather\n",
            "<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n",
            "<arg_key>date</arg_key>\n<arg_value>2024-06-27</arg_value>\n",
            "</tool_call>",
        ]
        tool_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for tool_call_chunk in result.calls:
                if (
                    hasattr(tool_call_chunk, "tool_index")
                    and tool_call_chunk.tool_index is not None
                ):
                    while len(tool_calls) <= tool_call_chunk.tool_index:
                        tool_calls.append({"name": "", "parameters": ""})
                    tc = tool_calls[tool_call_chunk.tool_index]
                    if tool_call_chunk.name:
                        tc["name"] = tool_call_chunk.name
                    if tool_call_chunk.parameters:
                        tc["parameters"] += tool_call_chunk.parameters
        self.assertEqual(len(tool_calls), 1)  # 断言相等
        self.assertEqual(tool_calls[0]["name"], "get_weather")  # 断言相等
        self.assertEqual(  # 断言相等
            tool_calls[0]["parameters"], '{"city": "Beijing", "date": "2024-06-27"}'
        )

    # TestGlm4MoeDetector类的测试streamingmultipletoolcalls
    def test_streaming_multiple_tool_calls(self):
        """Test streaming incremental parsing of multiple tool calls."""
        chunks = [
            "<tool_call>get_weather\n",
            "<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n",
            "<arg_key>date</arg_key>\n<arg_value>2024-06-27</arg_value>\n",
            "</tool_call><tool_call>get_weather\n",
            "<arg_key>city</arg_key>\n<arg_value>Shanghai</arg_value>\n",
            "<arg_key>date</arg_key>\n<arg_value>2024-06-28</arg_value>\n",
            "</tool_call>",
        ]
        tool_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for tool_call_chunk in result.calls:
                if (
                    hasattr(tool_call_chunk, "tool_index")
                    and tool_call_chunk.tool_index is not None
                ):
                    while len(tool_calls) <= tool_call_chunk.tool_index:
                        tool_calls.append({"name": "", "parameters": ""})
                    tc = tool_calls[tool_call_chunk.tool_index]
                    if tool_call_chunk.name:
                        tc["name"] = tool_call_chunk.name
                    if tool_call_chunk.parameters:
                        tc["parameters"] += tool_call_chunk.parameters
        self.assertEqual(len(tool_calls), 2)  # 断言相等
        self.assertEqual(tool_calls[0]["name"], "get_weather")  # 断言相等
        self.assertEqual(  # 断言相等
            tool_calls[0]["parameters"], '{"city": "Beijing", "date": "2024-06-27"}'
        )
        self.assertEqual(tool_calls[1]["name"], "get_weather")  # 断言相等
        self.assertEqual(  # 断言相等
            tool_calls[1]["parameters"], '{"city": "Shanghai", "date": "2024-06-28"}'
        )

    # TestGlm4MoeDetector类的测试toolcallid
    def test_tool_call_id(self):
        """Test that the buffer and state are reset after a tool call is completed."""
        chunks = [
            "<tool_call>get_weather\n",
            "<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n",
            "<arg_key>date</arg_key>\n<arg_value>2024-06-27</arg_value>\n",
            "</tool_call>",
        ]
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
        self.assertEqual(self.detector.current_tool_id, 1)  # 断言相等

    # TestGlm4MoeDetector类的测试invalidtoolcall
    def test_invalid_tool_call(self):
        """Test that invalid tool calls are handled correctly."""
        text = "<tool_call>invalid_func\n<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n</tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)  # 断言相等

    # TestGlm4MoeDetector类的测试partialtoolcall
    def test_partial_tool_call(self):
        """Test parsing a partial tool call that spans multiple chunks."""
        chunks = [
            "<tool_call>get_weather\n",
            "<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n",
            "<arg_key>date</arg_key>\n<arg_value>2024-06-27</arg_value>\n</tool_call>",
        ]

        tool_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for tool_call_chunk in result.calls:
                if (
                    hasattr(tool_call_chunk, "tool_index")
                    and tool_call_chunk.tool_index is not None
                ):
                    while len(tool_calls) <= tool_call_chunk.tool_index:
                        tool_calls.append({"name": "", "parameters": ""})
                    tc = tool_calls[tool_call_chunk.tool_index]
                    if tool_call_chunk.name:
                        tc["name"] = tool_call_chunk.name
                    if tool_call_chunk.parameters:
                        tc["parameters"] += tool_call_chunk.parameters

        self.assertEqual(len(tool_calls), 1)  # 断言相等
        self.assertEqual(tool_calls[0]["name"], "get_weather")  # 断言相等
        self.assertEqual(  # 断言相等
            tool_calls[0]["parameters"], '{"city": "Beijing", "date": "2024-06-27"}'
        )

    # TestGlm4MoeDetector类的测试arrayargumentwithescapedjson
    def test_array_argument_with_escaped_json(self):
        """Test that array arguments with escaped JSON are properly handled without double-escaping."""
        # Add a tool with array parameter
        tools_with_array = [
            Tool(
                type="function",
                function=Function(
                    name="todo_write",
                    description="Write todos",
                    parameters={
                        "type": "object",
                        "properties": {
                            "todos": {
                                "type": "array",
                                "description": "The updated todo list",
                            }
                        },
                        "required": ["todos"],
                    },
                ),
            ),
        ]

        # check_params
        def check_params(result):
            self.assertEqual(1, len(result.calls))  # 断言相等
            self.assertEqual("todo_write", result.calls[0].name)  # 断言相等
            params = json.loads(result.calls[0].parameters)
            self.assertIsInstance(params["todos"], list)
            self.assertEqual(4, len(params["todos"]))  # 断言相等
            self.assertEqual("1", params["todos"][0]["id"])  # 断言相等
            self.assertEqual(  # 断言相等
                "Check for hard-coded issues in the backend code",
                params["todos"][0]["task"],
            )
            self.assertEqual("in_progress", params["todos"][0]["status"])  # 断言相等
            self.assertEqual("2", params["todos"][1]["id"])  # 断言相等
            self.assertEqual(  # 断言相等
                "Check for hard-coded issues in the frontend code",
                params["todos"][1]["task"],
            )
            self.assertEqual("pending", params["todos"][1]["status"])  # 断言相等
            self.assertEqual("3", params["todos"][2]["id"])  # 断言相等
            self.assertEqual(  # 断言相等
                "Check for code violating the Single Responsibility Principle",
                params["todos"][2]["task"],
            )
            self.assertEqual("pending", params["todos"][2]["status"])  # 断言相等
            self.assertEqual("4", params["todos"][3]["id"])  # 断言相等
            self.assertEqual(  # 断言相等
                "Generate a rectification proposal report", params["todos"][3]["task"]
            )
            self.assertEqual("pending", params["todos"][3]["status"])  # 断言相等

        # Simulate the raw response from GLM-4.6 model with normal and escaped JSON in XML
        result = self.detector.detect_and_parse(
            """<tool_call>todo_write\n<arg_key>todos</arg_key>\n<arg_value>[{\"id\": \"1\", \"task\": \"Check for hard-coded issues in the backend code\", \"status\": \"in_progress\"}, {\"id\": \"2\", \"task\": \"Check for hard-coded issues in the frontend code\", \"status\": \"pending\"}, {\"id\": \"3\", \"task\": \"Check for code violating the Single Responsibility Principle\", \"status\": \"pending\"}, {\"id\": \"4\", \"task\": \"Generate a rectification proposal report\", \"status\": \"pending\"}]</arg_value>
</tool_call>""",
            tools_with_array,
        )
        check_params(result)
        result = self.detector.detect_and_parse(
            r"""<tool_call>todo_write\n<arg_key>todos</arg_key>\n<arg_value>[{\"id\": \"1\", \"task\": \"Check for hard-coded issues in the backend code\", \"status\": \"in_progress\"}, {\"id\": \"2\", \"task\": \"Check for hard-coded issues in the frontend code\", \"status\": \"pending\"}, {\"id\": \"3\", \"task\": \"Check for code violating the Single Responsibility Principle\", \"status\": \"pending\"}, {\"id\": \"4\", \"task\": \"Generate a rectification proposal report\", \"status\": \"pending\"}]</arg_value>
</tool_call>""",
            tools_with_array,
        )
        check_params(result)

        # check_single_todos
        def check_single_todos(tool_result, expected):
            self.assertEqual(1, len(tool_result.calls))  # 断言相等
            self.assertEqual("todo_write", tool_result.calls[0].name)  # 断言相等
            params = json.loads(tool_result.calls[0].parameters)
            self.assertIsInstance(params["todos"], list)
            self.assertEqual(1, len(params["todos"]))  # 断言相等
            self.assertEqual("1", params["todos"][0]["id"])  # 断言相等
            self.assertEqual(expected, params["todos"][0]["task"])  # 断言相等
            self.assertEqual("pending", params["todos"][0]["status"])  # 断言相等

        # Test with escaped JSON containing backslashes in content (e.g., Windows paths)
        expected_path = r"Check file at C:\Users\test.txt"
        result = self.detector.detect_and_parse(
            """<tool_call>todo_write\n<arg_key>todos</arg_key>\n<arg_value>[{\"id\": \"1\", \"task\": \"Check file at C:\\\\Users\\\\test.txt\", \"status\": \"pending\"}]</arg_value></tool_call>""",
            tools_with_array,
        )
        check_single_todos(result, expected_path)
        result = self.detector.detect_and_parse(
            r"""<tool_call>todo_write\n<arg_key>todos</arg_key>\n<arg_value>[{\"id\": \"1\", \"task\": \"Check file at C:\\\\Users\\\\test.txt\", \"status\": \"pending\"}]</arg_value></tool_call>""",
            tools_with_array,
        )
        check_single_todos(result, expected_path)

        # Should contain literal \n, not actual newline
        expected_output = r"Print \n to see newline"
        result = self.detector.detect_and_parse(
            """<tool_call>todo_write\n<arg_key>todos</arg_key>\n<arg_value>[{\"id\": \"1\", \"task\": \"Print \\\\n to see newline\",\"status\": \"pending\"}]</arg_value></tool_call>""",
            tools_with_array,
        )
        check_single_todos(result, expected_output)
        result = self.detector.detect_and_parse(
            r"""<tool_call>todo_write\n<arg_key>todos</arg_key>\n<arg_value>[{\"id\": \"1\", \"task\": \"Print \\\\n to see newline\",\"status\": \"pending\"}]</arg_value></tool_call>""",
            tools_with_array,
        )
        check_single_todos(result, expected_output)

    # TestGlm4MoeDetector类的测试emptyfunctionnamehandling
    def test_empty_function_name_handling(self):
        """Test that empty function name is handled gracefully without assertion error."""
        # This test simulates the issue where the model outputs only the start token without a function name
        chunks = [
            "<tool_call>",  # Start token only, no function name yet
            "\n",  # More content without function name
        ]

        for chunk in chunks:
            # Should not raise AssertionError: func_name should not be empty
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            # Should return empty calls without error
            self.assertIsInstance(result, StreamingParseResult)
            self.assertEqual(result.calls, [])  # 断言相等

    # TestGlm4MoeDetector类的测试whitespacepreservedinargvalues
    def test_whitespace_preserved_in_arg_values(self):
        """Test that leading/trailing whitespace in arg values is not stripped."""
        tools_with_string = [
            Tool(
                type="function",
                function=Function(
                    name="apply_diff",
                    description="Apply a diff",
                    parameters={
                        "type": "object",
                        "properties": {
                            "old_string": {"type": "string"},
                            "new_string": {"type": "string"},
                        },
                        "required": ["old_string", "new_string"],
                    },
                ),
            )
        ]
        text = (
            "<tool_call>apply_diff\n"
            "<arg_key>old_string</arg_key>\n"
            "<arg_value>    indented code</arg_value>\n"
            "<arg_key>new_string</arg_key>\n"
            "<arg_value>        also indented</arg_value>\n"
            "</tool_call>"
        )
        result = self.detector.detect_and_parse(text, tools_with_string)
        self.assertEqual(len(result.calls), 1)  # 断言相等
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["old_string"], "    indented code")  # 断言相等
        self.assertEqual(params["new_string"], "        also indented")  # 断言相等


# TestGlm47MoeDetector类
class TestGlm47MoeDetector(unittest.TestCase):

    # TestGlm47MoeDetector类的测试初始化设置
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
                            "date": {"type": "string", "description": "Date"},
                        },
                        "required": ["city", "date"],
                    },
                ),
            ),
        ]
        self.detector = Glm47MoeDetector()

    # TestGlm47MoeDetector类的测试singletoolcall
    def test_single_tool_call(self):
        text = (
            "<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>"
            "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value>"
            "</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等
        self.assertEqual(  # 断言相等
            result.calls[0].parameters, '{"city": "Beijing", "date": "2024-06-27"}'
        )
        self.assertEqual(result.normal_text, "")  # 断言相等

    # TestGlm47MoeDetector类的测试multipletoolcalls
    def test_multiple_tool_calls(self):
        text = (
            "<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>"
            "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value>"
            "</tool_call>"
            "<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>Shanghai</arg_value>"
            "<arg_key>date</arg_key><arg_value>2024-06-28</arg_value>"
            "</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等
        self.assertEqual(  # 断言相等
            result.calls[0].parameters, '{"city": "Beijing", "date": "2024-06-27"}'
        )
        self.assertEqual(result.calls[1].name, "get_weather")  # 断言相等
        self.assertEqual(  # 断言相等
            result.calls[1].parameters, '{"city": "Shanghai", "date": "2024-06-28"}'
        )
        self.assertEqual(result.normal_text, "")  # 断言相等

    # TestGlm47MoeDetector类的测试streamingtoolcall
    def test_streaming_tool_call(self):
        """Test streaming incremental parsing of a tool call."""
        chunks = [
            "<tool_call>get_weather",
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>",
            "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value>",
            "</tool_call>",
        ]
        tool_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for tool_call_chunk in result.calls:
                if (
                    hasattr(tool_call_chunk, "tool_index")
                    and tool_call_chunk.tool_index is not None
                ):
                    while len(tool_calls) <= tool_call_chunk.tool_index:
                        tool_calls.append({"name": "", "parameters": ""})
                    tc = tool_calls[tool_call_chunk.tool_index]
                    if tool_call_chunk.name:
                        tc["name"] = tool_call_chunk.name
                    if tool_call_chunk.parameters:
                        tc["parameters"] += tool_call_chunk.parameters
        self.assertEqual(len(tool_calls), 1)  # 断言相等
        self.assertEqual(tool_calls[0]["name"], "get_weather")  # 断言相等
        self.assertEqual(  # 断言相等
            tool_calls[0]["parameters"], '{"city": "Beijing", "date": "2024-06-27"}'
        )

    # TestGlm47MoeDetector类的测试streamingmultipletoolcalls
    def test_streaming_multiple_tool_calls(self):
        """Test streaming incremental parsing of multiple tool calls."""
        chunks = [
            "<tool_call>get_weather",
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>",
            "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value>",
            "</tool_call><tool_call>get_weather",
            "<arg_key>city</arg_key><arg_value>Shanghai</arg_value>",
            "<arg_key>date</arg_key><arg_value>2024-06-28</arg_value>",
            "</tool_call>",
        ]
        tool_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for tool_call_chunk in result.calls:
                if (
                    hasattr(tool_call_chunk, "tool_index")
                    and tool_call_chunk.tool_index is not None
                ):
                    while len(tool_calls) <= tool_call_chunk.tool_index:
                        tool_calls.append({"name": "", "parameters": ""})
                    tc = tool_calls[tool_call_chunk.tool_index]
                    if tool_call_chunk.name:
                        tc["name"] = tool_call_chunk.name
                    if tool_call_chunk.parameters:
                        tc["parameters"] += tool_call_chunk.parameters
        self.assertEqual(len(tool_calls), 2)  # 断言相等
        self.assertEqual(tool_calls[0]["name"], "get_weather")  # 断言相等
        self.assertEqual(  # 断言相等
            tool_calls[0]["parameters"], '{"city": "Beijing", "date": "2024-06-27"}'
        )
        self.assertEqual(tool_calls[1]["name"], "get_weather")  # 断言相等
        self.assertEqual(  # 断言相等
            tool_calls[1]["parameters"], '{"city": "Shanghai", "date": "2024-06-28"}'
        )

    # TestGlm47MoeDetector类的测试toolcallid
    def test_tool_call_id(self):
        """Test that the buffer and state are reset after a tool call is completed."""
        chunks = [
            "<tool_call>get_weather",
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>",
            "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value>",
            "</tool_call>",
        ]
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
        self.assertEqual(self.detector.current_tool_id, 1)  # 断言相等

    # TestGlm47MoeDetector类的测试invalidtoolcall
    def test_invalid_tool_call(self):
        """Test that invalid tool calls are handled correctly."""
        text = "<tool_call>invalid_func<arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)  # 断言相等

    # TestGlm47MoeDetector类的测试partialtoolcall
    def test_partial_tool_call(self):
        """Test parsing a partial tool call that spans multiple chunks."""
        chunks = [
            "<tool_call>get_weather",
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>",
            "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value></tool_call>",
        ]

        tool_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for tool_call_chunk in result.calls:
                if (
                    hasattr(tool_call_chunk, "tool_index")
                    and tool_call_chunk.tool_index is not None
                ):
                    while len(tool_calls) <= tool_call_chunk.tool_index:
                        tool_calls.append({"name": "", "parameters": ""})
                    tc = tool_calls[tool_call_chunk.tool_index]
                    if tool_call_chunk.name:
                        tc["name"] = tool_call_chunk.name
                    if tool_call_chunk.parameters:
                        tc["parameters"] += tool_call_chunk.parameters

        self.assertEqual(len(tool_calls), 1)  # 断言相等
        self.assertEqual(tool_calls[0]["name"], "get_weather")  # 断言相等
        self.assertEqual(  # 断言相等
            tool_calls[0]["parameters"], '{"city": "Beijing", "date": "2024-06-27"}'
        )

    # TestGlm47MoeDetector类的测试arrayargumentwithescapedjson
    def test_array_argument_with_escaped_json(self):
        """Test that array arguments with escaped JSON are properly handled without double-escaping."""
        # Add a tool with array parameter
        tools_with_array = [
            Tool(
                type="function",
                function=Function(
                    name="todo_write",
                    description="Write todos",
                    parameters={
                        "type": "object",
                        "properties": {
                            "todos": {
                                "type": "array",
                                "description": "The updated todo list",
                            }
                        },
                        "required": ["todos"],
                    },
                ),
            ),
        ]

        # check_params
        def check_params(result):
            self.assertEqual(1, len(result.calls))  # 断言相等
            self.assertEqual("todo_write", result.calls[0].name)  # 断言相等
            params = json.loads(result.calls[0].parameters)
            self.assertIsInstance(params["todos"], list)
            self.assertEqual(4, len(params["todos"]))  # 断言相等
            self.assertEqual("1", params["todos"][0]["id"])  # 断言相等
            self.assertEqual(  # 断言相等
                "Check for hard-coded issues in the backend code",
                params["todos"][0]["task"],
            )
            self.assertEqual("in_progress", params["todos"][0]["status"])  # 断言相等
            self.assertEqual("2", params["todos"][1]["id"])  # 断言相等
            self.assertEqual(  # 断言相等
                "Check for hard-coded issues in the frontend code",
                params["todos"][1]["task"],
            )
            self.assertEqual("pending", params["todos"][1]["status"])  # 断言相等
            self.assertEqual("3", params["todos"][2]["id"])  # 断言相等
            self.assertEqual(  # 断言相等
                "Check for code violating the Single Responsibility Principle",
                params["todos"][2]["task"],
            )
            self.assertEqual("pending", params["todos"][2]["status"])  # 断言相等
            self.assertEqual("4", params["todos"][3]["id"])  # 断言相等
            self.assertEqual(  # 断言相等
                "Generate a rectification proposal report", params["todos"][3]["task"]
            )
            self.assertEqual("pending", params["todos"][3]["status"])  # 断言相等

        # Simulate the raw response from GLM-4.6 model with normal and escaped JSON in XML
        result = self.detector.detect_and_parse(
            """<tool_call>todo_write<arg_key>todos</arg_key><arg_value>[{\"id\": \"1\", \"task\": \"Check for hard-coded issues in the backend code\", \"status\": \"in_progress\"}, {\"id\": \"2\", \"task\": \"Check for hard-coded issues in the frontend code\", \"status\": \"pending\"}, {\"id\": \"3\", \"task\": \"Check for code violating the Single Responsibility Principle\", \"status\": \"pending\"}, {\"id\": \"4\", \"task\": \"Generate a rectification proposal report\", \"status\": \"pending\"}]</arg_value>
</tool_call>""",
            tools_with_array,
        )
        check_params(result)
        result = self.detector.detect_and_parse(
            r"""<tool_call>todo_write<arg_key>todos</arg_key><arg_value>[{\"id\": \"1\", \"task\": \"Check for hard-coded issues in the backend code\", \"status\": \"in_progress\"}, {\"id\": \"2\", \"task\": \"Check for hard-coded issues in the frontend code\", \"status\": \"pending\"}, {\"id\": \"3\", \"task\": \"Check for code violating the Single Responsibility Principle\", \"status\": \"pending\"}, {\"id\": \"4\", \"task\": \"Generate a rectification proposal report\", \"status\": \"pending\"}]</arg_value>
</tool_call>""",
            tools_with_array,
        )
        check_params(result)

        # check_single_todos
        def check_single_todos(tool_result, expected):
            self.assertEqual(1, len(tool_result.calls))  # 断言相等
            self.assertEqual("todo_write", tool_result.calls[0].name)  # 断言相等
            params = json.loads(tool_result.calls[0].parameters)
            self.assertIsInstance(params["todos"], list)
            self.assertEqual(1, len(params["todos"]))  # 断言相等
            self.assertEqual("1", params["todos"][0]["id"])  # 断言相等
            self.assertEqual(expected, params["todos"][0]["task"])  # 断言相等
            self.assertEqual("pending", params["todos"][0]["status"])  # 断言相等

        # Test with escaped JSON containing backslashes in content (e.g., Windows paths)
        expected_path = r"Check file at C:\Users\test.txt"
        result = self.detector.detect_and_parse(
            """<tool_call>todo_write<arg_key>todos</arg_key><arg_value>[{\"id\": \"1\", \"task\": \"Check file at C:\\\\Users\\\\test.txt\", \"status\": \"pending\"}]</arg_value></tool_call>""",
            tools_with_array,
        )
        check_single_todos(result, expected_path)
        result = self.detector.detect_and_parse(
            r"""<tool_call>todo_write<arg_key>todos</arg_key><arg_value>[{\"id\": \"1\", \"task\": \"Check file at C:\\\\Users\\\\test.txt\", \"status\": \"pending\"}]</arg_value></tool_call>""",
            tools_with_array,
        )
        check_single_todos(result, expected_path)

        # Should contain literal \n, not actual newline
        expected_output = r"Print \n to see newline"
        result = self.detector.detect_and_parse(
            """<tool_call>todo_write<arg_key>todos</arg_key><arg_value>[{\"id\": \"1\", \"task\": \"Print \\\\n to see newline\",\"status\": \"pending\"}]</arg_value></tool_call>""",
            tools_with_array,
        )
        check_single_todos(result, expected_output)
        result = self.detector.detect_and_parse(
            r"""<tool_call>todo_write<arg_key>todos</arg_key><arg_value>[{\"id\": \"1\", \"task\": \"Print \\\\n to see newline\",\"status\": \"pending\"}]</arg_value></tool_call>""",
            tools_with_array,
        )
        check_single_todos(result, expected_output)

    # TestGlm47MoeDetector类的测试whitespacepreservedinargvalues
    def test_whitespace_preserved_in_arg_values(self):
        """Test that leading/trailing whitespace in arg values is not stripped."""
        tools_with_string = [
            Tool(
                type="function",
                function=Function(
                    name="apply_diff",
                    description="Apply a diff",
                    parameters={
                        "type": "object",
                        "properties": {
                            "old_string": {"type": "string"},
                            "new_string": {"type": "string"},
                        },
                        "required": ["old_string", "new_string"],
                    },
                ),
            )
        ]
        text = (
            "<tool_call>apply_diff"
            "<arg_key>old_string</arg_key>"
            "<arg_value>    indented code</arg_value>"
            "<arg_key>new_string</arg_key>"
            "<arg_value>        also indented</arg_value>"
            "</tool_call>"
        )
        result = self.detector.detect_and_parse(text, tools_with_string)
        self.assertEqual(len(result.calls), 1)  # 断言相等
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["old_string"], "    indented code")  # 断言相等
        self.assertEqual(params["new_string"], "        also indented")  # 断言相等


# TestJsonArrayParser类
class TestJsonArrayParser(unittest.TestCase):

    # TestJsonArrayParser类的测试初始化设置
    def setUp(self):
        # Create sample tools for testing
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "Location to get weather for",
                            },
                            "unit": {
                                "type": "string",
                                "description": "Temperature unit",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Search for information",
                    parameters={
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
        self.detector = JsonArrayParser()

    # TestJsonArrayParser类的测试jsondetectorhasnoebnf
    def test_json_detector_has_no_ebnf(self):
        """JsonArrayParser no longer exposes EBNF generation helpers."""
        self.assertFalse(  # 断言为假
            hasattr(self.detector, "build_ebnf"),
            "JsonArrayParser should not expose EBNF helpers after cleanup",
        )

    # TestJsonArrayParser类的测试parsestreamingincrementmalformedjson
    def test_parse_streaming_increment_malformed_json(self):
        """Test parsing with malformed JSON"""
        # Test with malformed JSON
        text = '[{"name": "get_weather", "parameters": {"location": "Tokyo"'
        result = self.detector.parse_streaming_increment(text, self.tools)

        # Should not crash and return a valid result
        self.assertIsInstance(result, StreamingParseResult)

        text = "[{}}}]"
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertIsInstance(result, StreamingParseResult)

    # TestJsonArrayParser类的测试parsestreamingincrementemptyinput
    def test_parse_streaming_increment_empty_input(self):
        """Test parsing with empty input"""
        result = self.detector.parse_streaming_increment("", self.tools)
        self.assertEqual(len(result.calls), 0)  # 断言相等
        self.assertEqual(result.normal_text, "")  # 断言相等

    # TestJsonArrayParser类的测试parsestreamingincrementwhitespacehandling
    def test_parse_streaming_increment_whitespace_handling(self):
        """Test parsing with various whitespace scenarios"""
        # Test with leading/trailing whitespace split across chunks
        chunk1 = '  [{"name": "get_weather", "parameters": '
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)
        chunk2 = '{"location": "Tokyo"}}]  '
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        # The base class should handle this
        self.assertIsInstance(result2, StreamingParseResult)

    # TestJsonArrayParser类的测试parsestreamingincrementnestedobjects
    def test_parse_streaming_increment_nested_objects(self):
        """Test parsing with nested JSON objects"""
        chunk1 = '[{"name": "get_weather", "parameters": {"location": "Tokyo", '
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)
        chunk2 = '"nested": {"key": "value"}}}]'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        # The base class should handle this
        self.assertIsInstance(result2, StreamingParseResult)

    # TestJsonArrayParser类的测试jsonparsingwithcommas
    def test_json_parsing_with_commas(self):
        """Test that JSON parsing works correctly with comma separators"""
        # Stream two complete objects, at least 2 chunks per tool call
        chunk1 = '[{"name": "get_weather", "parameters": {"location": "Tok'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)
        chunk2 = 'yo"}},'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)
        self.assertIsInstance(result2, StreamingParseResult)

        chunk3 = '{"name": "get_weather", "parameters": {"location": "Par'
        result3 = self.detector.parse_streaming_increment(chunk3, self.tools)
        self.assertIsInstance(result3, StreamingParseResult)
        chunk4 = 'is"}}]'
        result4 = self.detector.parse_streaming_increment(chunk4, self.tools)
        self.assertIsInstance(result4, StreamingParseResult)
        self.assertGreater(  # 断言大于
            len(result4.calls), 0, "Should parse tool calls from text with separators"
        )

    # TestJsonArrayParser类的测试bracesinstrings
    def test_braces_in_strings(self):
        """Test that JSON with } characters inside strings works correctly"""
        # Test case: JSON array with } inside string values - streamed across chunks
        chunk1 = '[{"name": "get_weather", "parameters": {"location": "has } inside"'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)
        chunk2 = "}}"
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)
        self.assertIsInstance(result2, StreamingParseResult)
        self.assertGreater(  # 断言大于
            len(result2.calls), 0, "Should parse tool call with } in string"
        )

        # Test with separator (streaming in progress)
        chunk3 = '[{"name": "get_weather", "parameters": {"location": "has } inside"}'
        result3 = self.detector.parse_streaming_increment(chunk3, self.tools)
        self.assertIsInstance(result3, StreamingParseResult)
        chunk4 = "},"
        result4 = self.detector.parse_streaming_increment(chunk4, self.tools)
        self.assertIsInstance(result4, StreamingParseResult)
        chunk5 = '{"name": "get_weather"'
        result5 = self.detector.parse_streaming_increment(chunk5, self.tools)
        self.assertIsInstance(result5, StreamingParseResult)
        self.assertGreater(  # 断言大于
            len(result5.calls),
            0,
            "Should parse tool calls with separator and } in string",
        )

    # TestJsonArrayParser类的测试separatorinsamechunk
    def test_separator_in_same_chunk(self):
        """Test that separator already present in chunk works correctly"""
        # Test case: separator already in the chunk (streaming in progress) with 2+ chunks per tool call
        chunk1 = '[{"name": "get_weather", "parameters": {"location": "Tokyo"'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)
        chunk2 = '}},{"name": "get_weather"'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)
        self.assertIsInstance(result2, StreamingParseResult)
        self.assertGreater(  # 断言大于
            len(result2.calls),
            0,
            "Should parse tool calls with separator in same chunk",
        )

    # TestJsonArrayParser类的测试separatorinseparatechunk
    def test_separator_in_separate_chunk(self):
        """Test that separator in separate chunk works correctly"""
        # Test case: separator in separate chunk - this tests streaming behavior
        chunk1 = '[{"name": "get_weather", "parameters": {"location": "Tokyo"}}'
        chunk2 = ","
        chunk3 = '{"name": "get_weather", "parameters": {"location": "Paris"}}'

        # Process first chunk
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)

        # Process separator chunk
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)
        self.assertIsInstance(result2, StreamingParseResult)

        # Process second chunk (streaming in progress)
        result3 = self.detector.parse_streaming_increment(chunk3, self.tools)
        self.assertIsInstance(result3, StreamingParseResult)

    # TestJsonArrayParser类的测试incompletejsonacrosschunks
    def test_incomplete_json_across_chunks(self):
        """Test that incomplete JSON across chunks works correctly"""
        # Test case: incomplete JSON across chunks - this tests streaming behavior
        chunk1 = '[{"name": "get_weather", "parameters": {"location": "Tokyo"'
        chunk2 = '}},{"name": "get_weather"'

        # Process first chunk (incomplete)
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)

        # Process second chunk (completes first object and starts second, streaming in progress)
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)
        self.assertIsInstance(result2, StreamingParseResult)

    # TestJsonArrayParser类的测试malformedjsonrecovery
    def test_malformed_json_recovery(self):
        """Test that malformed JSON recovers gracefully"""
        # Test with malformed JSON - should handle gracefully
        malformed_text = (
            '[{"name": "get_weather", "parameters": {"location": "unclosed string'
        )

        result1 = self.detector.parse_streaming_increment(malformed_text, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)

        # Test valid JSON after malformed - streamed across 2 chunks (streaming in progress)
        valid_chunk1 = '[{"name": "get_weather", "parameters": {"location": "Tok'
        result2 = self.detector.parse_streaming_increment(valid_chunk1, self.tools)
        self.assertIsInstance(result2, StreamingParseResult)
        valid_chunk2 = 'yo"}}'
        result3 = self.detector.parse_streaming_increment(valid_chunk2, self.tools)
        self.assertIsInstance(result3, StreamingParseResult)

    # TestJsonArrayParser类的测试nestedobjectswithcommas
    def test_nested_objects_with_commas(self):
        """Test that nested objects with commas inside work correctly"""
        # Test with nested objects that have commas - should work with json.loads()
        chunk1 = '[{"name": "get_weather", "parameters": {"location": "Tok'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)
        chunk2 = 'yo", "unit": "celsius"}}'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)
        self.assertIsInstance(result2, StreamingParseResult)
        self.assertGreater(  # 断言大于
            len(result2.calls), 0, "Should parse tool call with nested objects"
        )

    # TestJsonArrayParser类的测试emptyobjects
    def test_empty_objects(self):
        """Test that empty objects work correctly"""
        # Test with empty objects - should work with json.loads()
        chunk1 = '[{"name": "get_weather", "parameters": '
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)
        chunk2 = "{}}"
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)
        self.assertIsInstance(result2, StreamingParseResult)

    # TestJsonArrayParser类的测试whitespacehandling
    def test_whitespace_handling(self):
        """Test that various whitespace scenarios work correctly"""
        # Test with various whitespace patterns - should work with json.loads()
        chunk1 = ' \n\n [{"name": "get_weather", "parameters": '
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)
        chunk2 = '{"location": "Tokyo"}}'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)
        self.assertIsInstance(result2, StreamingParseResult)

    # TestJsonArrayParser类的测试multiplecommasinchunk
    def test_multiple_commas_in_chunk(self):
        """Test that multiple commas in a single chunk work correctly"""
        # Stream multiple tool calls ensuring at least 2 chunks per complete tool call
        chunk1 = '[{"name": "get_weather", "parameters": {"location": "To'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)
        chunk2 = 'kyo"}},'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)
        self.assertIsInstance(result2, StreamingParseResult)

        chunk3 = '{"name": "get_weather", "parameters": {"location": "Pa'
        result3 = self.detector.parse_streaming_increment(chunk3, self.tools)
        self.assertIsInstance(result3, StreamingParseResult)
        chunk4 = 'ris"}},'
        result4 = self.detector.parse_streaming_increment(chunk4, self.tools)
        self.assertIsInstance(result4, StreamingParseResult)

        chunk5 = '{"name": "get_weather"'
        result5 = self.detector.parse_streaming_increment(chunk5, self.tools)
        self.assertIsInstance(result5, StreamingParseResult)
        self.assertGreater(  # 断言大于
            len(result5.calls), 0, "Should parse tool calls with multiple commas"
        )

    # TestJsonArrayParser类的测试completetoolcallwithtrailingcomma
    def test_complete_tool_call_with_trailing_comma(self):
        """Test that complete tool call with trailing comma parses correctly"""
        # Test case: complete tool call followed by comma at end of chunk (split across 2 chunks)
        chunk1 = '[{"name": "get_weather", "parameters": {"location": "Tokyo"}'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)
        chunk2 = "}, "
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)
        self.assertIsInstance(result2, StreamingParseResult)
        self.assertGreater(len(result2.calls), 0, "Should parse complete tool call")  # 断言大于

        # Test that next chunk with opening brace gets the separator prepended
        next_chunk = '{"name": "get_weather", "parameters": {"location": "Paris"}}'
        result_next = self.detector.parse_streaming_increment(next_chunk, self.tools)
        self.assertIsInstance(result_next, StreamingParseResult)
        self.assertGreater(  # 断言大于
            len(result_next.calls), 0, "Should parse subsequent tool call"
        )

    # TestJsonArrayParser类的测试threetoolcallsseparatechunkswithcommas
    def test_three_tool_calls_separate_chunks_with_commas(self):
        """Test parsing 3 tool calls in separate chunks with commas at the end"""
        # First tool call: 2 chunks
        chunk1_1 = '[{"name": "get_weather", "parameters": '
        result1_1 = self.detector.parse_streaming_increment(chunk1_1, self.tools)
        chunk1_2 = '{"location": "Tokyo"}},'
        result1_2 = self.detector.parse_streaming_increment(chunk1_2, self.tools)
        self.assertIsInstance(result1_2, StreamingParseResult)
        self.assertGreater(len(result1_2.calls), 0, "Should parse first tool call")  # 断言大于

        # Second tool call: 2 chunks
        chunk2_1 = '{"name": "search", "parameters": '
        result2_1 = self.detector.parse_streaming_increment(chunk2_1, self.tools)
        chunk2_2 = '{"query": "restaurants"}},'
        result2_2 = self.detector.parse_streaming_increment(chunk2_2, self.tools)
        self.assertIsInstance(result2_2, StreamingParseResult)
        self.assertGreater(len(result2_2.calls), 0, "Should parse second tool call")  # 断言大于

        # Third tool call: 2 chunks
        chunk3_1 = '{"name": "get_weather", "parameters": '
        result3_1 = self.detector.parse_streaming_increment(chunk3_1, self.tools)
        chunk3_2 = '{"location": "Paris"}}]'
        result3_2 = self.detector.parse_streaming_increment(chunk3_2, self.tools)
        self.assertIsInstance(result3_2, StreamingParseResult)
        self.assertGreater(len(result3_2.calls), 0, "Should parse third tool call")  # 断言大于
        # Verify all tool calls were parsed correctly
        total_calls = len(result1_2.calls) + len(result2_2.calls) + len(result3_2.calls)
        self.assertEqual(total_calls, 3, "Should have parsed exactly 3 tool calls")  # 断言相等


# TestLfm2Detector类
class TestLfm2Detector(unittest.TestCase):
    """Tests for LFM2 (Liquid Foundation Model 2) function call detector."""

    def setUp(self):
        """Set up test tools and detector."""
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            },
                            "unit": {
                                "type": "string",
                                "description": "Temperature unit",
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
                    description="Search for information",
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
            Tool(
                type="function",
                function=Function(
                    name="calculator",
                    description="Perform calculations",
                    parameters={
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Math expression",
                            },
                        },
                        "required": ["expression"],
                    },
                ),
            ),
        ]
        self.detector = Lfm2Detector()

    # ==================== has_tool_call tests ====================

    def test_has_tool_call_true(self):
        """Test detection of tool call markers."""
        text = '<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>'
        self.assertTrue(self.detector.has_tool_call(text))  # 断言为真

    # TestLfm2Detector类的测试hastoolcallfalse
    def test_has_tool_call_false(self):
        """Test no false positives for regular text."""
        text = "The weather in Paris is nice today."
        self.assertFalse(self.detector.has_tool_call(text))  # 断言为假

    # TestLfm2Detector类的测试hastoolcallpartialmarker
    def test_has_tool_call_partial_marker(self):
        """Test that partial markers are detected (start token present)."""
        text = '<|tool_call_start|>[get_weather(city="Paris")'
        self.assertTrue(self.detector.has_tool_call(text))  # 断言为真

    # ==================== detect_and_parse tests (Pythonic format) ====================

    def test_detect_and_parse_pythonic_simple(self):
        """Test parsing a simple Pythonic format tool call."""
        text = '<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等
        self.assertEqual(result.calls[0].tool_index, 0)  # 断言相等

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "Paris")  # 断言相等

    # TestLfm2Detector类的测试detectandparsepythonicmultipleargs
    def test_detect_and_parse_pythonic_multiple_args(self):
        """Test parsing with multiple arguments."""
        text = '<|tool_call_start|>[get_weather(city="London", unit="celsius")]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "London")  # 断言相等
        self.assertEqual(params["unit"], "celsius")  # 断言相等

    # TestLfm2Detector类的测试detectandparsepythonicnoargs
    def test_detect_and_parse_pythonic_no_args(self):
        """Test parsing function with no arguments."""
        # Add a no-arg tool for this test
        tools_with_noarg = self.tools + [
            Tool(
                type="function",
                function=Function(
                    name="get_time",
                    description="Get current time",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
        ]
        text = "<|tool_call_start|>[get_time()]<|tool_call_end|>"
        result = self.detector.detect_and_parse(text, tools_with_noarg)

        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_time")  # 断言相等

    # TestLfm2Detector类的测试detectandparsepythonicmultiplecalls
    def test_detect_and_parse_pythonic_multiple_calls(self):
        """Test parsing multiple tool calls in one block."""
        text = '<|tool_call_start|>[get_weather(city="Paris"), search(query="restaurants")]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 2)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等
        self.assertEqual(result.calls[1].name, "search")  # 断言相等

        params1 = json.loads(result.calls[0].parameters)
        params2 = json.loads(result.calls[1].parameters)
        self.assertEqual(params1["city"], "Paris")  # 断言相等
        self.assertEqual(params2["query"], "restaurants")  # 断言相等

    # TestLfm2Detector类的测试detectandparsewithnormaltextbefore
    def test_detect_and_parse_with_normal_text_before(self):
        """Test parsing with normal text before the tool call."""
        text = 'Let me check the weather for you. <|tool_call_start|>[get_weather(city="Tokyo")]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "Let me check the weather for you.")  # 断言相等
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等

    # TestLfm2Detector类的测试detectandparsespecialcharactersinvalue
    def test_detect_and_parse_special_characters_in_value(self):
        """Test parsing with special characters in argument values."""
        text = (
            '<|tool_call_start|>[search(query="what\'s the weather?")]<|tool_call_end|>'
        )
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)  # 断言相等
        params = json.loads(result.calls[0].parameters)
        self.assertIn("weather", params["query"])  # 断言包含

    # TestLfm2Detector类的测试detectandparsenumericvalues
    def test_detect_and_parse_numeric_values(self):
        """Test parsing with numeric argument values."""
        text = '<|tool_call_start|>[calculator(expression="5 * 7")]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "calculator")  # 断言相等

    # ==================== detect_and_parse tests (JSON format) ====================

    def test_detect_and_parse_json_simple(self):
        """Test parsing JSON format tool call."""
        text = '<|tool_call_start|>[{"name": "get_weather", "arguments": {"city": "Berlin"}}]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "Berlin")  # 断言相等

    # TestLfm2Detector类的测试detectandparsejsonmultiplecalls
    def test_detect_and_parse_json_multiple_calls(self):
        """Test parsing multiple JSON format tool calls."""
        text = '<|tool_call_start|>[{"name": "get_weather", "arguments": {"city": "Paris"}}, {"name": "search", "arguments": {"query": "hotels"}}]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 2)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等
        self.assertEqual(result.calls[1].name, "search")  # 断言相等

    # TestLfm2Detector类的测试detectandparsejsonwithparameterskey
    def test_detect_and_parse_json_with_parameters_key(self):
        """Test parsing JSON format with 'parameters' key instead of 'arguments'."""
        text = '<|tool_call_start|>[{"name": "get_weather", "parameters": {"city": "Madrid"}}]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)  # 断言相等
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "Madrid")  # 断言相等

    # ==================== Edge cases ====================

    def test_detect_and_parse_no_tool_call(self):
        """Test parsing text with no tool calls."""
        text = "This is just regular text without any tool calls."
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, text)  # 断言相等
        self.assertEqual(result.calls, [])  # 断言相等

    # TestLfm2Detector类的测试detectandparseunknownfunction
    def test_detect_and_parse_unknown_function(self):
        """Test parsing with unknown function name - skipped by default (SGLANG_FORWARD_UNKNOWN_TOOLS=false)."""
        text = '<|tool_call_start|>[unknown_function(arg="value")]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        # By default, unknown functions are skipped (consistent with other detectors)
        self.assertEqual(len(result.calls), 0)  # 断言相等

    # TestLfm2Detector类的测试detectandparseemptycontent
    def test_detect_and_parse_empty_content(self):
        """Test parsing with empty content between markers."""
        text = "<|tool_call_start|><|tool_call_end|>"
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.calls, [])  # 断言相等

    # TestLfm2Detector类的测试detectandparsemultipleblocks
    def test_detect_and_parse_multiple_blocks(self):
        """Test parsing multiple separate tool call blocks."""
        text = '<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|> Some text <|tool_call_start|>[search(query="food")]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 2)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等
        self.assertEqual(result.calls[1].name, "search")  # 断言相等

    # ==================== Streaming tests ====================
    # The LFM2 detector buffers until it sees complete <|tool_call_start|>...<|tool_call_end|>
    # blocks, then parses the complete block. This allows proper handling of both
    # JSON and Pythonic formats.

    def test_streaming_json_complete_in_one_chunk(self):
        """Test streaming with complete JSON tool call in one chunk."""
        text = '<|tool_call_start|>{"name": "get_weather", "arguments": {"city": "Rome"}}<|tool_call_end|>'
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等

    # TestLfm2Detector类的测试streamingjsonsplitacrosschunks
    def test_streaming_json_split_across_chunks(self):
        """Test streaming with JSON tool call split across multiple chunks - waits for complete block."""
        # Reset detector state
        self.detector = Lfm2Detector()

        # First chunk: start marker and partial JSON (no end token)
        chunk1 = '<|tool_call_start|>{"name": "get_weather", "arguments": {"city": '
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)

        # Should buffer and not emit calls yet (waiting for complete block)
        self.assertEqual(len(result1.calls), 0)  # 断言相等
        self.assertEqual(result1.normal_text, "")  # 断言相等

        # Second chunk: complete the JSON and end token
        chunk2 = '"Vienna"}}<|tool_call_end|>'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        # Now should have the complete tool call
        self.assertEqual(len(result2.calls), 1)  # 断言相等
        self.assertEqual(result2.calls[0].name, "get_weather")  # 断言相等

    # TestLfm2Detector类的测试streamingjsonnormaltextbeforetoolcall
    def test_streaming_json_normal_text_before_tool_call(self):
        """Test streaming with normal text before JSON tool call."""
        # Reset detector state
        self.detector = Lfm2Detector()

        chunk1 = "I'll check the weather. "
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)

        # Normal text should be returned
        self.assertIn("check the weather", result1.normal_text)  # 断言包含

        chunk2 = '<|tool_call_start|>{"name": "get_weather", "arguments": {"city": "Amsterdam"}}<|tool_call_end|>'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        self.assertEqual(len(result2.calls), 1)  # 断言相等

    # TestLfm2Detector类的测试streamingeottokenfiltering
    def test_streaming_eot_token_filtering(self):
        """Test that end-of-turn token is filtered from normal text."""
        # Reset detector state
        self.detector = Lfm2Detector()

        # Send text that ends with tool call end token (JSON format)
        text = '<|tool_call_start|>{"name": "get_weather", "arguments": {"city": "Oslo"}}<|tool_call_end|>'
        result = self.detector.parse_streaming_increment(text, self.tools)

        # The normal_text should not contain the eot_token
        self.assertNotIn("<|tool_call_end|>", result.normal_text)  # 断言不包含

    # ==================== Pythonic streaming tests ====================

    def test_streaming_pythonic_complete_in_one_chunk(self):
        """Test streaming with complete Pythonic tool call in one chunk."""
        self.detector = Lfm2Detector()
        text = '<|tool_call_start|>[get_weather(city="Berlin")]<|tool_call_end|>'
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Berlin"})  # 断言相等

    # TestLfm2Detector类的测试streamingpythonicsplitacrosschunks
    def test_streaming_pythonic_split_across_chunks(self):
        """Test streaming with Pythonic tool call split across multiple chunks."""
        self.detector = Lfm2Detector()

        # First chunk: start marker and partial call
        chunk1 = '<|tool_call_start|>[get_weather(city="'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)

        # Should buffer and not emit calls yet
        self.assertEqual(len(result1.calls), 0)  # 断言相等

        # Second chunk: complete the call
        chunk2 = 'Munich")]<|tool_call_end|>'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        # Now should have the complete tool call
        self.assertEqual(len(result2.calls), 1)  # 断言相等
        self.assertEqual(result2.calls[0].name, "get_weather")  # 断言相等
        self.assertEqual(json.loads(result2.calls[0].parameters), {"city": "Munich"})  # 断言相等

    # TestLfm2Detector类的测试streamingpythonicmultiplecalls
    def test_streaming_pythonic_multiple_calls(self):
        """Test streaming with multiple Pythonic tool calls."""
        self.detector = Lfm2Detector()

        text = '<|tool_call_start|>[get_weather(city="Paris"), search(query="hotels")]<|tool_call_end|>'
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(len(result.calls), 2)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等
        self.assertEqual(result.calls[1].name, "search")  # 断言相等

    # ==================== structure_info tests ====================

    def test_supports_structural_tag(self):
        """Test that LFM2 does not support structural tags (Pythonic format)."""
        # LFM2 uses Pythonic format which is not JSON-compatible,
        # so structural_tag constrained generation cannot be used
        self.assertFalse(self.detector.supports_structural_tag())  # 断言为假

    # TestLfm2Detector类的测试structureinfo
    def test_structure_info(self):
        """Test structure info for constrained generation."""
        info_func = self.detector.structure_info()
        info = info_func("get_weather")

        self.assertEqual(info.begin, "<|tool_call_start|>[get_weather(")  # 断言相等
        self.assertEqual(info.end, ")]<|tool_call_end|>")  # 断言相等
        self.assertEqual(info.trigger, "<|tool_call_start|>")  # 断言相等


# TestGigaChat3Detector类
class TestGigaChat3Detector(unittest.TestCase):

    # TestGigaChat3Detector类的测试初始化设置
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="manage_user_memory",
                    description="Create, update, or delete a user memory entry.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "content": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "default": None,
                            },
                            "action": {
                                "type": "string",
                                "enum": ["create", "update", "delete"],
                                "default": "create",
                            },
                            "id": {
                                "anyOf": [
                                    {"type": "string", "format": "uuid"},
                                    {"type": "null"},
                                ],
                                "default": None,
                            },
                        },
                    },
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
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["city"],
                    },
                ),
            ),
        ]
        self.detector = GigaChat3Detector()

    # TestGigaChat3Detector类的测试hastoolcall
    def test_has_tool_call(self):
        """Test detection of tool call markers."""
        self.assertTrue(self.detector.has_tool_call("function call<|role_sep|>\n{}"))  # 断言为真
        self.assertTrue(self.detector.has_tool_call("<|function_call|>{}"))  # 断言为真
        self.assertFalse(self.detector.has_tool_call("No tool call here"))  # 断言为假

    # TestGigaChat3Detector类的测试detectandparsenotoolcall
    def test_detect_and_parse_no_tool_call(self):
        """Test parsing text without tool calls."""
        text = "How can I help you today?"
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, text)  # 断言相等
        self.assertEqual(len(result.calls), 0)  # 断言相等

    # TestGigaChat3Detector类的测试detectandparsesimpletoolcall
    def test_detect_and_parse_simple_tool_call(self):
        """Test parsing a simple tool call without content."""
        text = '<|message_sep|>\n\nfunction call<|role_sep|>\n{"name": "manage_user_memory", "arguments": {"action": "create", "id": "preferences"}}'
        result = self.detector.detect_and_parse(text, self.tools)

        # No content before tool call
        self.assertEqual(result.normal_text, "")  # 断言相等
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "manage_user_memory")  # 断言相等

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["action"], "create")  # 断言相等
        self.assertEqual(params["id"], "preferences")  # 断言相等

    # TestGigaChat3Detector类的测试detectandparseparameterlesstoolcall
    def test_detect_and_parse_parameterless_tool_call(self):
        """Test parsing a tool call with empty arguments."""
        text = '<|message_sep|>\n\nfunction call<|role_sep|>\n{"name": "manage_user_memory", "arguments": {}}'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "")  # 断言相等
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "manage_user_memory")  # 断言相等

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params, {})  # 断言相等

    # TestGigaChat3Detector类的测试detectandparsecomplextoolcall
    def test_detect_and_parse_complex_tool_call(self):
        """Test parsing a tool call with nested objects."""
        text = """<|message_sep|>

function call<|role_sep|>
{"name": "manage_user_memory", "arguments": {"action": "create", "id": "preferences", "content": {"short_answers": true, "hate_emojis": true, "english_ui": false, "russian_math_explanations": true}}}"""

        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "")  # 断言相等
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "manage_user_memory")  # 断言相等

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["action"], "create")  # 断言相等
        self.assertEqual(params["id"], "preferences")  # 断言相等
        self.assertIsInstance(params["content"], dict)
        self.assertEqual(params["content"]["short_answers"], True)  # 断言相等
        self.assertEqual(params["content"]["hate_emojis"], True)  # 断言相等

    # TestGigaChat3Detector类的测试detectandparsewithcontentbefore
    def test_detect_and_parse_with_content_before(self):
        """Test parsing tool call with text content before it."""
        text = 'I\'ll check that for you.<|message_sep|>\n\nfunction call<|role_sep|>\n{"name": "manage_user_memory", "arguments": {"action": "create", "id": "preferences"}}'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "I'll check that for you.")  # 断言相等
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "manage_user_memory")  # 断言相等

    # TestGigaChat3Detector类的测试detectandparsewitheostoken
    def test_detect_and_parse_with_eos_token(self):
        """Test parsing tool call with EOS token at the end."""
        text = '<|message_sep|>\n\nfunction call<|role_sep|>\n{"name": "manage_user_memory", "arguments": {"action": "create", "id": "preferences"}}</s>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "")  # 断言相等
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "manage_user_memory")  # 断言相等

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["action"], "create")  # 断言相等
        self.assertEqual(params["id"], "preferences")  # 断言相等

    # TestGigaChat3Detector类的测试detectandparsewithcontentandeos
    def test_detect_and_parse_with_content_and_eos(self):
        """Test parsing tool call with content and EOS token."""
        text = 'I\'ll remember that.<|message_sep|>\n\nfunction call<|role_sep|>\n{"name": "manage_user_memory", "arguments": {"action": "create", "id": "test"}}</s>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "I'll remember that.")  # 断言相等
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "manage_user_memory")  # 断言相等

    # TestGigaChat3Detector类的测试detectandparseinvalidjson
    def test_detect_and_parse_invalid_json(self):
        """Test parsing with invalid JSON in function call."""
        text = '<|message_sep|>\n\nfunction call<|role_sep|>\n{"name": "manage_user_memory", "arguments": {invalid json}}'
        result = self.detector.detect_and_parse(text, self.tools)

        # Should return the full text as content when JSON parsing fails
        self.assertIn("function call", result.normal_text)  # 断言包含
        self.assertEqual(len(result.calls), 0)  # 断言相等

    # TestGigaChat3Detector类的测试detectandparsemissingname
    def test_detect_and_parse_missing_name(self):
        """Test parsing with missing function name."""
        text = '<|message_sep|>\n\nfunction call<|role_sep|>\n{"arguments": {"action": "create"}}'
        result = self.detector.detect_and_parse(text, self.tools)

        # Should not extract tool call if name is missing
        self.assertEqual(len(result.calls), 0)  # 断言相等

    # TestGigaChat3Detector类的测试detectandparsemissingarguments
    def test_detect_and_parse_missing_arguments(self):
        """Test parsing with missing arguments field."""
        text = '<|message_sep|>\n\nfunction call<|role_sep|>\n{"name": "manage_user_memory"}'
        result = self.detector.detect_and_parse(text, self.tools)

        # Should not extract tool call if arguments is missing
        self.assertEqual(len(result.calls), 0)  # 断言相等

    # TestGigaChat3Detector类的测试detectandparseargumentsnotdict
    def test_detect_and_parse_arguments_not_dict(self):
        """Test parsing with arguments that is not a dict."""
        text = '<|message_sep|>\n\nfunction call<|role_sep|>\n{"name": "manage_user_memory", "arguments": "string_args"}'
        result = self.detector.detect_and_parse(text, self.tools)

        # Should not extract tool call if arguments is not a dict
        self.assertEqual(len(result.calls), 0)  # 断言相等

    # TestGigaChat3Detector类的测试streamingnotoolcall
    def test_streaming_no_tool_call(self):
        """Test streaming text without tool calls."""
        chunks = ["How ", "can ", "I ", "help ", "you?"]

        accumulated_text = ""
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            accumulated_text += result.normal_text

        self.assertEqual(accumulated_text, "How can I help you?")  # 断言相等
        self.assertEqual(len(result.calls), 0)  # 断言相等

    # TestGigaChat3Detector类的测试streamingsimpletoolcall
    def test_streaming_simple_tool_call(self):
        """Test streaming a simple tool call."""
        chunks = [
            "<|message_sep|>\n\n",
            "function call",
            "<|role_sep|>\n",
            '{"name": "manage_user_memory", ',
            '"arguments": {"action": "create"',
            ', "id": "preferences"}}',
        ]

        tool_calls_by_index = {}
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)

            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        self.assertEqual(len(tool_calls_by_index), 1)  # 断言相等
        self.assertEqual(tool_calls_by_index[0]["name"], "manage_user_memory")  # 断言相等

        params = json.loads(tool_calls_by_index[0]["parameters"])
        self.assertEqual(params["action"], "create")  # 断言相等
        self.assertEqual(params["id"], "preferences")  # 断言相等

    # TestGigaChat3Detector类的测试streamingwithcontentbefore
    def test_streaming_with_content_before(self):
        """Test streaming with content before tool call."""
        chunks = [
            "I'll ",
            "help ",
            "you.",
            "<|message_sep|>\n\n",
            "function call",
            "<|role_sep|>\n",
            '{"name": "get_weather", ',
            '"arguments": {"city": "Tokyo"}}',
        ]

        accumulated_text = ""
        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            accumulated_text += result.normal_text

            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        self.assertEqual(accumulated_text, "I'll help you.")  # 断言相等
        self.assertEqual(len(tool_calls_by_index), 1)  # 断言相等
        self.assertEqual(tool_calls_by_index[0]["name"], "get_weather")  # 断言相等

        params = json.loads(tool_calls_by_index[0]["parameters"])
        self.assertEqual(params["city"], "Tokyo")  # 断言相等

    # TestGigaChat3Detector类的测试streamingcomplexarguments
    def test_streaming_complex_arguments(self):
        """Test streaming with complex nested arguments."""
        chunks = [
            "<|message_sep|>\n\n",
            "functi",
            "on call<|role_sep|>\n",
            '{"name": "manage_user_memory", "arguments": ',
            '{"action": "create", "id": "prefs", ',
            '"content": {"likes": ["short", "clear"], ',
            '"dislikes": ["emojis", "verbose"]}',
            "}}",
        ]

        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)

            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        self.assertEqual(len(tool_calls_by_index), 1)  # 断言相等
        self.assertEqual(tool_calls_by_index[0]["name"], "manage_user_memory")  # 断言相等

        params = json.loads(tool_calls_by_index[0]["parameters"])
        self.assertEqual(params["action"], "create")  # 断言相等
        self.assertEqual(params["content"]["likes"], ["short", "clear"])  # 断言相等
        self.assertEqual(params["content"]["dislikes"], ["emojis", "verbose"])  # 断言相等

    # TestGigaChat3Detector类的测试streamingwitheostoken
    def test_streaming_with_eos_token(self):
        """Test streaming with EOS token at the end."""
        chunks = [
            "<|message_sep|>\n\n",
            "function c",
            "all<|role_sep|>\n",
            '{"name": "get_weather", ',
            '"arguments": {"city": "Paris"}}',
            "</s>",
        ]

        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)

            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        self.assertEqual(len(tool_calls_by_index), 1)  # 断言相等
        self.assertEqual(tool_calls_by_index[0]["name"], "get_weather")  # 断言相等

        params = json.loads(tool_calls_by_index[0]["parameters"])
        self.assertEqual(params["city"], "Paris")  # 断言相等

    # TestGigaChat3Detector类的测试streamingincompletejson
    def test_streaming_incomplete_json(self):
        """Test streaming with incomplete JSON (no closing brace)."""
        chunks = [
            "<|message_sep|>\n\n",
            "fun",
            "ction call<|role_sep|>\n",
            '{"name": "get_weather", ',
            '"arguments": {"city": "London"',
            # Missing closing braces
        ]

        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)

            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        # Should have name but incomplete parameters
        self.assertEqual(len(tool_calls_by_index), 1)  # 断言相等
        self.assertEqual(tool_calls_by_index[0]["name"], "get_weather")  # 断言相等
        self.assertTrue(tool_calls_by_index[0]["parameters"].startswith('{"city":'))  # 断言为真

    # TestGigaChat3Detector类的测试streaminglargesteps
    def test_streaming_large_steps(self):
        """Test streaming with large chunks that complete in fewer steps."""
        chunks = [
            "I'll remember that.",
            "<|message_sep|>\n\nfuncti",
            "on call<|role_sep|>\n",
            '{"name": "manage_user_memory", "arguments": {"action": "create", "id": "preferences", "content": {"short_answers": true, "hate_emojis": true, ',
            '"english_ui": false, "russian_math_explanations": true}',
            "}}",
        ]

        accumulated_text = ""
        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            accumulated_text += result.normal_text

            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        self.assertEqual(accumulated_text, "I'll remember that.")  # 断言相等
        self.assertEqual(len(tool_calls_by_index), 1)  # 断言相等
        self.assertEqual(tool_calls_by_index[0]["name"], "manage_user_memory")  # 断言相等

        params = json.loads(tool_calls_by_index[0]["parameters"])
        self.assertEqual(params["action"], "create")  # 断言相等
        self.assertEqual(params["content"]["short_answers"], True)  # 断言相等
        self.assertEqual(params["content"]["russian_math_explanations"], True)  # 断言相等

    # TestGigaChat3Detector类的测试streamingverysmallchunks
    def test_streaming_very_small_chunks(self):
        """Test streaming with very small chunks (character by character)."""
        text = '{"name": "get_weather", "arguments": {"city": "NYC"}}'

        # Split into very small chunks (every 5 characters)
        chunk_size = 5
        chunked_text = [
            text[i : i + chunk_size] for i in range(0, len(text), chunk_size)
        ]
        chunks = [
            "<|message_sep|>\n\n",
            "func",
            "tion call",
            "<|role_sep|>\n",
            *chunked_text,
        ]
        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)

            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        self.assertEqual(len(tool_calls_by_index), 1)  # 断言相等
        self.assertEqual(tool_calls_by_index[0]["name"], "get_weather")  # 断言相等

        params = json.loads(tool_calls_by_index[0]["parameters"])
        self.assertEqual(params["city"], "NYC")  # 断言相等

    # TestGigaChat3Detector类的测试streamingjsonsplitatquotes
    def test_streaming_json_split_at_quotes(self):
        """Test streaming when JSON is split at quote boundaries."""
        chunks = [
            "<|message_sep|>\n\nfunction call<|role_sep|>\n",
            '{"name',
            '": "',
            "get_weather",
            '", "arguments',
            '": {"city',
            '": "',
            "Rome",
            '"}}',
        ]

        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)

            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        self.assertEqual(len(tool_calls_by_index), 1)  # 断言相等
        self.assertEqual(tool_calls_by_index[0]["name"], "get_weather")  # 断言相等

        params = json.loads(tool_calls_by_index[0]["parameters"])
        self.assertEqual(params["city"], "Rome")  # 断言相等

    # TestGigaChat3Detector类的测试detectandparsefunctioncallmarkersimpletoolcall
    def test_detect_and_parse_function_call_marker_simple_tool_call(self):
        """Test parsing a simple <|function_call|> tool call (GigaChat3.1-style)."""
        text = '<|function_call|>{"name": "manage_user_memory", "arguments": {"action": "create", "id": "preferences"}}'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "")  # 断言相等
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "manage_user_memory")  # 断言相等

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["action"], "create")  # 断言相等
        self.assertEqual(params["id"], "preferences")  # 断言相等

    # TestGigaChat3Detector类的测试detectandparsefunctioncallmarkerwithcontentbefore
    def test_detect_and_parse_function_call_marker_with_content_before(self):
        """Test parsing <|function_call|> tool call with prefix content."""
        text = (
            'I\'ll check that for you.<|function_call|>{"name": "get_weather", '
            '"arguments": {"city": "Tokyo"}}'
        )
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "I'll check that for you.")  # 断言相等
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "Tokyo")  # 断言相等

    # TestGigaChat3Detector类的测试detectandparsefunctioncallmarkerwitheostoken
    def test_detect_and_parse_function_call_marker_with_eos_token(self):
        """Test parsing <|function_call|> tool call with EOS token at the end."""
        text = '<|function_call|>{"name": "manage_user_memory", "arguments": {"action": "create", "id": "preferences"}}</s>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "")  # 断言相等
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "manage_user_memory")  # 断言相等

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["action"], "create")  # 断言相等
        self.assertEqual(params["id"], "preferences")  # 断言相等

    # TestGigaChat3Detector类的测试detectandparsefunctioncallmarkerinvalidjson
    def test_detect_and_parse_function_call_marker_invalid_json(self):
        """Test parsing invalid JSON after <|function_call|> marker."""
        text = '<|function_call|>{"name": "manage_user_memory", "arguments": {invalid json}}'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertIn("<|function_call|>", result.normal_text)  # 断言包含
        self.assertEqual(len(result.calls), 0)  # 断言相等

    # TestGigaChat3Detector类的测试streamingfunctioncallmarkersimpletoolcall
    def test_streaming_function_call_marker_simple_tool_call(self):
        """Test streaming parsing of the <|function_call|> marker form."""
        chunks = [
            "I'll help you.",
            "<|function_call|>",
            '{"name": "manage_user_memory", "arguments": ',
            '{"action": "create", "id": "prefs"}}',
        ]

        accumulated_text = ""
        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            accumulated_text += result.normal_text

            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        self.assertEqual(accumulated_text, "I'll help you.")  # 断言相等
        self.assertEqual(len(tool_calls_by_index), 1)  # 断言相等
        self.assertEqual(tool_calls_by_index[0]["name"], "manage_user_memory")  # 断言相等

        params = json.loads(tool_calls_by_index[0]["parameters"])
        self.assertEqual(params["action"], "create")  # 断言相等
        self.assertEqual(params["id"], "prefs")  # 断言相等

    # TestGigaChat3Detector类的测试streamingfunctioncallmarkerjsonsplitatquotes
    def test_streaming_function_call_marker_json_split_at_quotes(self):
        """Test streaming when JSON is split at quote boundaries (<|function_call|>)."""
        chunks = [
            "<|function_call|>",
            '{"name',
            '": "',
            "get_weather",
            '", "arguments',
            '": {"city',
            '": "',
            "Rome",
            '"}}',
        ]

        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)

            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        self.assertEqual(len(tool_calls_by_index), 1)  # 断言相等
        self.assertEqual(tool_calls_by_index[0]["name"], "get_weather")  # 断言相等

        params = json.loads(tool_calls_by_index[0]["parameters"])
        self.assertEqual(params["city"], "Rome")  # 断言相等


# TestGetStructureConstraint类
class TestGetStructureConstraint(unittest.TestCase):
    """Tests for FunctionCallParser.get_structure_constraint() logic.

    Verifies that detectors supporting structural_tag use it for required/named
    tool_choice, and that the generic json_schema fallback is used otherwise.
    """

    def _make_tools(self, strict=False):
        return [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                    strict=strict,
                ),
            ),
        ]

    # TestGetStructureConstraint类的内部方法_make_parser
    def _make_parser(self, parser_name, strict=False):
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        return FunctionCallParser(self._make_tools(strict=strict), parser_name)

    # TestGetStructureConstraint类的内部方法_constraint_json
    def _constraint_json(self, result):
        return result[1].model_dump_json()

    # --- structural_tag detectors (kimi_k2, deepseekv3, qwen25, etc.) ---

    def test_kimi_required_strict_returns_structural_tag(self):
        import xgrammar as xgr

        parser = self._make_parser("kimi_k2", strict=True)
        result = parser.get_structure_constraint("required")
        self.assertIsNotNone(result)  # 断言不为None
        self.assertEqual(result[0], "structural_tag")  # 断言相等
        self.assertIsInstance(result[1], xgr.StructuralTag)
        self.assertIn("<|tool_calls_section_begin|>", self._constraint_json(result))  # 断言包含

    # TestGetStructureConstraint类的测试kimirequirednostrictreturnsstructuraltag
    def test_kimi_required_no_strict_returns_structural_tag(self):
        """required should use structural_tag even without strict, to preserve native format."""
        import xgrammar as xgr

        parser = self._make_parser("kimi_k2", strict=False)
        result = parser.get_structure_constraint("required")
        self.assertIsNotNone(result)  # 断言不为None
        self.assertEqual(result[0], "structural_tag")  # 断言相等
        self.assertIsInstance(result[1], xgr.StructuralTag)
        self.assertIn("<|tool_calls_section_begin|>", self._constraint_json(result))  # 断言包含

    # TestGetStructureConstraint类的测试kimiautostrictreturnsstructuraltag
    def test_kimi_auto_strict_returns_structural_tag(self):
        import xgrammar as xgr

        parser = self._make_parser("kimi_k2", strict=True)
        result = parser.get_structure_constraint("auto")
        self.assertIsNotNone(result)  # 断言不为None
        self.assertEqual(result[0], "structural_tag")  # 断言相等
        self.assertIsInstance(result[1], xgr.StructuralTag)
        serialized = self._constraint_json(result)
        self.assertIn('"type":"triggered_tags"', serialized)  # 断言包含
        self.assertIn("<|tool_calls_section_begin|>", serialized)  # 断言包含

    # TestGetStructureConstraint类的测试kimiroutesthroughnativewithsectionmarkers
    def test_kimi_routes_through_native_with_section_markers(self):
        """xgrammar 0.2.1's Kimi builtin keeps auto tool calls section-wrapped."""
        import xgrammar as xgr

        parser = self._make_parser("kimi_k2", strict=True)
        result = parser.get_structure_constraint("auto")
        self.assertIsInstance(result[1], xgr.StructuralTag)
        serialized = self._constraint_json(result)
        self.assertIn("<|tool_calls_section_begin|>", serialized)  # 断言包含
        self.assertIn("<|tool_calls_section_end|>", serialized)  # 断言包含

    # TestGetStructureConstraint类的测试kimiautonostrictreturnsnone
    def test_kimi_auto_no_strict_returns_none(self):
        """auto without strict should not constrain."""
        parser = self._make_parser("kimi_k2", strict=False)
        result = parser.get_structure_constraint("auto")
        self.assertIsNone(result)  # 断言为None

    # TestGetStructureConstraint类的测试kiminamedtoolchoicereturnsstructuraltag
    def test_kimi_named_tool_choice_returns_structural_tag(self):
        from sglang.srt.entrypoints.openai.protocol import (
            ToolChoice,
            ToolChoiceFuncName,
        )

        parser = self._make_parser("kimi_k2", strict=False)
        tool_choice = ToolChoice(function=ToolChoiceFuncName(name="get_weather"))
        result = parser.get_structure_constraint(tool_choice)
        self.assertIsNotNone(result)  # 断言不为None
        self.assertEqual(result[0], "structural_tag")  # 断言相等

    # TestGetStructureConstraint类的测试deepseekv3requirednostrictreturnsstructuraltag
    def test_deepseekv3_required_no_strict_returns_structural_tag(self):
        parser = self._make_parser("deepseekv3", strict=False)
        result = parser.get_structure_constraint("required")
        self.assertIsNotNone(result)  # 断言不为None
        self.assertEqual(result[0], "structural_tag")  # 断言相等

    # TestGetStructureConstraint类的测试qwen25requirednostrictreturnsstructuraltag
    def test_qwen25_required_no_strict_returns_structural_tag(self):
        parser = self._make_parser("qwen25", strict=False)
        result = parser.get_structure_constraint("required")
        self.assertIsNotNone(result)  # 断言不为None
        self.assertEqual(result[0], "structural_tag")  # 断言相等

    # --- structural_tag content verification ---

    def test_kimi_structural_tag_has_kimi_tokens(self):
        """Verify structural_tag contains kimi-specific special tokens."""
        parser = self._make_parser("kimi_k2", strict=True)
        result = parser.get_structure_constraint("required")
        serialized = self._constraint_json(result)
        self.assertIn("<|tool_calls_section_begin|>", serialized)  # 断言包含
        self.assertIn("functions.get_weather:", serialized)  # 断言包含
        self.assertIn('"pattern":"\\\\d+"', serialized)  # 断言包含
        self.assertIn("<|tool_call_end|>", serialized)  # 断言包含
        self.assertIn("<|tool_calls_section_end|>", serialized)  # 断言包含

    # TestGetStructureConstraint类的测试kimirequirednostrictuseslooseobjectschema
    def test_kimi_required_no_strict_uses_loose_object_schema(self):
        """Kimi required calls keep non-strict arguments object-shaped but loose."""
        parser = self._make_parser("kimi_k2", strict=False)
        result = parser.get_structure_constraint("required")
        serialized = self._constraint_json(result)
        self.assertIn('"json_schema":{"type":"object"}', serialized)  # 断言包含
        self.assertNotIn('"additionalProperties":false', serialized)  # 断言不包含
        self.assertNotIn('"properties"', serialized)  # 断言不包含

    # TestGetStructureConstraint类的测试kimirequiredstrictusestoolschema
    def test_kimi_required_strict_uses_tool_schema(self):
        """With strict, native xgrammar should include the tool's parameter schema."""
        parser = self._make_parser("kimi_k2", strict=True)
        result = parser.get_structure_constraint("required")
        serialized = self._constraint_json(result)
        self.assertIn('"properties"', serialized)  # 断言包含
        self.assertIn('"city"', serialized)  # 断言包含

    # --- reasoning-prefix ownership ---

    def test_default_thinking_mode_is_false(self):
        """Default must be False so callers don't silently get a reasoning
        prefix added to their grammar (only relevant for detectors routed
        through the xgrammar builtin)."""
        import inspect

        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        sig = inspect.signature(FunctionCallParser.get_structure_constraint)
        self.assertIs(sig.parameters["thinking_mode"].default, False)  # 断言是同一对象


# TestQwen25Detector类
class TestQwen25Detector(unittest.TestCase):
    """Test Qwen25Detector streaming and non-streaming multi-tool-call parsing."""

    def setUp(self):
        from sglang.srt.function_call.qwen25_detector import Qwen25Detector

        self.detector = Qwen25Detector()
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_current_weather",
                    description="Get the current weather in a given location",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The city name",
                            },
                            "state": {
                                "type": "string",
                                "description": "Two-letter state abbreviation",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["city", "state", "unit"],
                    },
                ),
            ),
        ]

    # -- Non-streaming tests --

    def test_detect_and_parse_single_tool_call(self):
        text = '<tool_call>\n{"name": "get_current_weather", "arguments": {"city": "NYC", "state": "NY", "unit": "fahrenheit"}}\n</tool_call>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_current_weather")  # 断言相等
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "NYC")  # 断言相等

    # TestQwen25Detector类的测试detectandparsemultipletoolcalls
    def test_detect_and_parse_multiple_tool_calls(self):
        text = (
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"city": "NYC", "state": "NY", "unit": "fahrenheit"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"city": "Baltimore", "state": "MD", "unit": "fahrenheit"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"city": "Minneapolis", "state": "MN", "unit": "fahrenheit"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"city": "Los Angeles", "state": "CA", "unit": "fahrenheit"}}\n</tool_call>'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 4)  # 断言相等
        cities = [json.loads(c.parameters)["city"] for c in result.calls]
        self.assertEqual(cities, ["NYC", "Baltimore", "Minneapolis", "Los Angeles"])  # 断言相等

    # TestQwen25Detector类的测试detectandparsewithnormaltextprefix
    def test_detect_and_parse_with_normal_text_prefix(self):
        text = (
            "Sure, let me check the weather.\n"
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"city": "NYC", "state": "NY", "unit": "celsius"}}\n</tool_call>'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertIn("let me check", result.normal_text)  # 断言包含

    # -- Streaming tests --

    def _collect_streaming_tool_calls(self, chunks):
        """Helper: feed chunks through streaming parser and collect tool calls by index."""
        tool_calls_by_index = {}
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }
                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters
        return tool_calls_by_index

    # TestQwen25Detector类的测试streamingsingletoolcall
    def test_streaming_single_tool_call(self):
        chunks = [
            "<tool_call>\n",
            '{"name": "get_current_weather",',
            ' "arguments": {"city": "NYC",',
            ' "state": "NY",',
            ' "unit": "fahrenheit"}}',
            "\n</tool_call>",
        ]
        result = self._collect_streaming_tool_calls(chunks)
        self.assertEqual(len(result), 1)  # 断言相等
        self.assertEqual(result[0]["name"], "get_current_weather")  # 断言相等
        params = json.loads(result[0]["parameters"])
        self.assertEqual(params["city"], "NYC")  # 断言相等

    # TestQwen25Detector类的测试streamingmultipletoolcalls
    def test_streaming_multiple_tool_calls(self):
        """Core regression test: multiple tool calls must all be parsed in streaming mode."""
        chunks = [
            "<tool_call>\n",
            '{"name": "get_current_weather",',
            ' "arguments": {"city": "NYC", "state": "NY", "unit": "fahrenheit"}}',
            "\n</tool_call>\n",
            "<tool_call>\n",
            '{"name": "get_current_weather",',
            ' "arguments": {"city": "Baltimore", "state": "MD", "unit": "fahrenheit"}}',
            "\n</tool_call>\n",
            "<tool_call>\n",
            '{"name": "get_current_weather",',
            ' "arguments": {"city": "LA", "state": "CA", "unit": "fahrenheit"}}',
            "\n</tool_call>",
        ]
        result = self._collect_streaming_tool_calls(chunks)
        self.assertEqual(len(result), 3, f"Expected 3 tool calls, got {len(result)}")  # 断言相等
        cities = [json.loads(result[i]["parameters"])["city"] for i in sorted(result)]
        self.assertEqual(cities, ["NYC", "Baltimore", "LA"])  # 断言相等

    # TestQwen25Detector类的测试streamingmultipletoolcallsfusedchunks
    def test_streaming_multiple_tool_calls_fused_chunks(self):
        """Test when separator and next bot_token arrive in a single chunk."""
        chunks = [
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"city": "NYC", "state": "NY", "unit": "fahrenheit"}}',
            '\n</tool_call>\n<tool_call>\n{"name": "get_current_weather",',
            ' "arguments": {"city": "LA", "state": "CA", "unit": "fahrenheit"}}',
            "\n</tool_call>",
        ]
        result = self._collect_streaming_tool_calls(chunks)
        self.assertEqual(len(result), 2, f"Expected 2 tool calls, got {len(result)}")  # 断言相等
        cities = [json.loads(result[i]["parameters"])["city"] for i in sorted(result)]
        self.assertEqual(cities, ["NYC", "LA"])  # 断言相等

    # TestQwen25Detector类的测试streamingmultipletoolcallscharbycharseparator
    def test_streaming_multiple_tool_calls_char_by_char_separator(self):
        """Test when the separator between tool calls arrives character by character."""
        call1 = '{"name": "get_current_weather", "arguments": {"city": "NYC", "state": "NY", "unit": "fahrenheit"}}'
        call2 = '{"name": "get_current_weather", "arguments": {"city": "LA", "state": "CA", "unit": "celsius"}}'
        separator = "\n</tool_call>\n<tool_call>\n"

        chunks = ["<tool_call>\n", call1]
        for ch in separator:
            chunks.append(ch)
        chunks.append(call2)
        chunks.append("\n</tool_call>")

        result = self._collect_streaming_tool_calls(chunks)
        self.assertEqual(len(result), 2, f"Expected 2 tool calls, got {len(result)}")  # 断言相等
        cities = [json.loads(result[i]["parameters"])["city"] for i in sorted(result)]
        self.assertEqual(cities, ["NYC", "LA"])  # 断言相等


# TestGemma4Detector类
class TestGemma4Detector(unittest.TestCase):

    # TestGemma4Detector类的测试初始化设置
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
                            "location": {"type": "string"},
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                ),
            )
        ]
        self.detector = Gemma4Detector()

    # TestGemma4Detector类的测试detectandparse
    def test_detect_and_parse(self):
        text = 'Some text before <|tool_call>call:get_weather{location:<|"|>Tokyo<|"|>}<tool_call|>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "Some text before ")  # 断言相等
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["location"], "Tokyo")  # 断言相等

    # TestGemma4Detector类的测试parsestreamingincrement
    def test_parse_streaming_increment(self):
        chunks = [
            "Some text ",
            "before <|tool",
            "_call>call:get_we",
            "ather{location:<|",  # codespell:ignore
            '"|>Tokyo<|"|>}<tool_',
            "call|> after",
        ]

        all_results = []
        for chunk in chunks:
            res = self.detector.parse_streaming_increment(chunk, self.tools)
            all_results.append(res)

        combined_normal_text = "".join(r.normal_text for r in all_results)
        self.assertEqual(combined_normal_text, "Some text before  after")  # 断言相等

        found_name = False
        found_params = False
        for res in all_results:
            for call in res.calls:
                if call.name == "get_weather":
                    found_name = True
                if call.parameters:
                    params = json.loads(call.parameters)
                    if params == {"location": "Tokyo"}:
                        found_params = True

        self.assertTrue(found_name)  # 断言为真
        self.assertTrue(found_params)  # 断言为真

    # TestGemma4Detector类的测试nestedarraystreaming
    def test_nested_array_streaming(self):
        # Additional coverage for complex structure
        chunks = [
            '<|tool_call>call:get_weather{location:<|"',
            '|>New York<|"|>,nested:[1, 2, {inner:<|"|>',
            'val<|"|>}]}<tool_call|>',
        ]

        all_results = []
        for chunk in chunks:
            res = self.detector.parse_streaming_increment(chunk, self.tools)
            all_results.append(res)

        found_params = False
        for res in all_results:
            for call in res.calls:
                if call.parameters:
                    params = json.loads(call.parameters)
                    if "location" in params and params["location"] == "New York":
                        if "nested" in params and params["nested"] == [
                            1,
                            2,
                            {"inner": "val"},
                        ]:
                            found_params = True

        self.assertTrue(found_params)  # 断言为真

    # TestGemma4Detector类的测试hastoolcall
    def test_has_tool_call(self):
        self.assertTrue(  # 断言为真
            self.detector.has_tool_call(
                '<|tool_call>call:get_weather{location:<|"|>Tokyo<|"|>}<tool_call|>'
            )
        )
        self.assertFalse(self.detector.has_tool_call("no tool call here"))  # 断言为假

    # TestGemma4Detector类的测试detectandparsenotoolcall
    def test_detect_and_parse_no_tool_call(self):
        text = "This is plain text without any tool calls."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, text)  # 断言相等
        self.assertEqual(len(result.calls), 0)  # 断言相等

    # TestGemma4Detector类的测试detectandparsetoolindex
    def test_detect_and_parse_tool_index(self):
        text = '<|tool_call>call:get_weather{location:<|"|>Tokyo<|"|>}<tool_call|>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].tool_index, 0)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等

    # TestGemma4Detector类的测试detectandparseunknowntoolindex
    def test_detect_and_parse_unknown_tool_index(self):
        text = '<|tool_call>call:unknown_func{arg:<|"|>val<|"|>}<tool_call|>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)  # 断言相等
        self.assertEqual(result.calls[0].tool_index, -1)  # 断言相等

    # TestGemma4Detector类的测试detectandparsenestedobject
    def test_detect_and_parse_nested_object(self):
        text = '<|tool_call>call:get_weather{location:<|"|>Tokyo<|"|>,details:{temp:25,unit:<|"|>celsius<|"|>}}<tool_call|>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)  # 断言相等
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["location"], "Tokyo")  # 断言相等
        self.assertIsInstance(params["details"], dict)
        self.assertEqual(params["details"]["temp"], 25)  # 断言相等
        self.assertEqual(params["details"]["unit"], "celsius")  # 断言相等

    # TestGemma4Detector类的测试detectandparsemultiplecalls
    def test_detect_and_parse_multiple_calls(self):
        extra_tools = self.tools + [
            Tool(
                type="function",
                function=Function(
                    name="get_time",
                    description="Get current time",
                    parameters={
                        "type": "object",
                        "properties": {"timezone": {"type": "string"}},
                    },
                ),
            )
        ]
        text = (
            'Some text <|tool_call>call:get_weather{location:<|"|>Tokyo<|"|>}<tool_call|>'
            ' more text <|tool_call>call:get_time{timezone:<|"|>UTC<|"|>}<tool_call|>'
        )
        result = self.detector.detect_and_parse(text, extra_tools)
        self.assertEqual(len(result.calls), 2)  # 断言相等
        self.assertEqual(result.calls[0].name, "get_weather")  # 断言相等
        self.assertEqual(result.calls[1].name, "get_time")  # 断言相等
        self.assertEqual(result.normal_text, "Some text ")  # 断言相等

    # TestGemma4Detector类的测试parsegemma4argsempty
    def test_parse_gemma4_args_empty(self):
        self.assertEqual(_parse_gemma4_args(""), {})  # 断言相等
        self.assertEqual(_parse_gemma4_args("   "), {})  # 断言相等

    # TestGemma4Detector类的测试parsegemma4argsbooleans
    def test_parse_gemma4_args_booleans(self):
        result = _parse_gemma4_args("flag:true,other:false")
        self.assertIs(result["flag"], True)  # 断言是同一对象
        self.assertIs(result["other"], False)  # 断言是同一对象

    # TestGemma4Detector类的测试parsegemma4argsnumbers
    def test_parse_gemma4_args_numbers(self):
        result = _parse_gemma4_args("count:42,ratio:3.14")
        self.assertEqual(result["count"], 42)  # 断言相等
        self.assertAlmostEqual(result["ratio"], 3.14)  # 断言近似相等

    # TestGemma4Detector类的测试parsegemma4argsstringwithcolon
    def test_parse_gemma4_args_string_with_colon(self):
        result = _parse_gemma4_args('url:<|"|>http://example.com<|"|>')
        self.assertEqual(result["url"], "http://example.com")  # 断言相等

    # TestGemma4Detector类的测试parsegemma4argsnestedobject
    def test_parse_gemma4_args_nested_object(self):
        result = _parse_gemma4_args('outer:{inner:<|"|>val<|"|>,num:5}')
        self.assertIsInstance(result["outer"], dict)
        self.assertEqual(result["outer"]["inner"], "val")  # 断言相等
        self.assertEqual(result["outer"]["num"], 5)  # 断言相等

    # TestGemma4Detector类的测试parsegemma4arraymixedtypes
    def test_parse_gemma4_array_mixed_types(self):
        result = _parse_gemma4_array('<|"|>hello<|"|>, 42, true, {key:<|"|>val<|"|>}')
        self.assertEqual(result[0], "hello")  # 断言相等
        self.assertEqual(result[1], 42)  # 断言相等
        self.assertIs(result[2], True)  # 断言是同一对象
        self.assertIsInstance(result[3], dict)
        self.assertEqual(result[3]["key"], "val")  # 断言相等

    # TestGemma4Detector类的测试parsegemma4valuetypes
    def test_parse_gemma4_value_types(self):
        self.assertIs(_parse_gemma4_value("true"), True)  # 断言是同一对象
        self.assertIs(_parse_gemma4_value("false"), False)  # 断言是同一对象
        self.assertEqual(_parse_gemma4_value("42"), 42)  # 断言相等
        self.assertAlmostEqual(_parse_gemma4_value("3.14"), 3.14)  # 断言近似相等
        self.assertEqual(_parse_gemma4_value("hello"), "hello")  # 断言相等
        self.assertEqual(_parse_gemma4_value(""), "")  # 断言相等

    # TestGemma4Detector类的内部方法_collect_streaming
    def _collect_streaming(self, chunks):
        """Helper: feed chunks and collect normal text + tool calls by index."""
        normal_text = ""
        tool_calls_by_index = {}
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            normal_text += result.normal_text
            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }
                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters
        return normal_text, tool_calls_by_index

    # TestGemma4Detector类的测试streamingmultipletoolcalls
    def test_streaming_multiple_tool_calls(self):
        """Test streaming with two consecutive tool calls."""
        extra_tools = self.tools + [
            Tool(
                type="function",
                function=Function(
                    name="get_time",
                    description="Get current time",
                    parameters={
                        "type": "object",
                        "properties": {"timezone": {"type": "string"}},
                    },
                ),
            )
        ]
        chunks = [
            '<|tool_call>call:get_weather{location:<|"|>',
            'Tokyo<|"|>}<tool_call|>',
            ' <|tool_call>call:get_time{timezone:<|"|>',
            'UTC<|"|>}<tool_call|>',
        ]
        normal_text = ""
        tool_calls_by_index = {}
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, extra_tools)
            normal_text += result.normal_text
            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }
                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        self.assertEqual(len(tool_calls_by_index), 2)  # 断言相等
        self.assertEqual(tool_calls_by_index[0]["name"], "get_weather")  # 断言相等
        self.assertEqual(tool_calls_by_index[1]["name"], "get_time")  # 断言相等
        params0 = json.loads(tool_calls_by_index[0]["parameters"])
        params1 = json.loads(tool_calls_by_index[1]["parameters"])
        self.assertEqual(params0["location"], "Tokyo")  # 断言相等
        self.assertEqual(params1["timezone"], "UTC")  # 断言相等

    # TestGemma4Detector类的测试streamingverysmallchunks
    def test_streaming_very_small_chunks(self):
        """Test streaming with character-by-character chunks."""
        full_text = '<|tool_call>call:get_weather{location:<|"|>Rome<|"|>}<tool_call|>'
        chunks = list(full_text)

        normal_text, tool_calls = self._collect_streaming(chunks)

        self.assertEqual(len(tool_calls), 1)  # 断言相等
        self.assertEqual(tool_calls[0]["name"], "get_weather")  # 断言相等
        params = json.loads(tool_calls[0]["parameters"])
        self.assertEqual(params["location"], "Rome")  # 断言相等

    # TestGemma4Detector类的测试streamingemptyargs
    def test_streaming_empty_args(self):
        """Test streaming a tool call with no arguments."""
        chunks = ["<|tool_call>call:get_weather{}", "<tool_call|>"]
        normal_text, tool_calls = self._collect_streaming(chunks)
        self.assertEqual(len(tool_calls), 1)  # 断言相等
        self.assertEqual(tool_calls[0]["name"], "get_weather")  # 断言相等

    # TestGemma4Detector类的测试streamingtextbetweentoolcalls
    def test_streaming_text_between_tool_calls(self):
        """Test streaming with normal text interleaved between two different tool calls."""
        extra_tools = self.tools + [
            Tool(
                type="function",
                function=Function(
                    name="get_time",
                    description="Get current time",
                    parameters={
                        "type": "object",
                        "properties": {"timezone": {"type": "string"}},
                    },
                ),
            )
        ]
        chunks = [
            "Hello! ",
            '<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|>',
            " Let me also check ",
            '<|tool_call>call:get_time{timezone:<|"|>UTC<|"|>}<tool_call|>',
        ]
        normal_text = ""
        tool_calls_by_index = {}
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, extra_tools)
            normal_text += result.normal_text
            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }
                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters
        self.assertIn("Hello!", normal_text)  # 断言包含
        self.assertIn("Let me also check", normal_text)  # 断言包含
        self.assertEqual(len(tool_calls_by_index), 2)  # 断言相等
        self.assertEqual(tool_calls_by_index[0]["name"], "get_weather")  # 断言相等
        self.assertEqual(tool_calls_by_index[1]["name"], "get_time")  # 断言相等
        params0 = json.loads(tool_calls_by_index[0]["parameters"])
        params1 = json.loads(tool_calls_by_index[1]["parameters"])
        self.assertEqual(params0["location"], "Paris")  # 断言相等
        self.assertEqual(params1["timezone"], "UTC")  # 断言相等


if __name__ == "__main__":
    unittest.main()
