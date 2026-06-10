# 文件名: test_json_schema_constraint.py - JSON Schema约束
"""
Tests for JSON schema constraint functionality used by JsonArrayParser
"""

import unittest

import jsonschema

from sglang.srt.entrypoints.openai.protocol import (
    Function,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.function_call.utils import (
    _get_tool_schema_defs,
    get_json_schema_constraint,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(5, "base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-b-test-cpu")


# TestJsonSchemaConstraint类
class TestJsonSchemaConstraint(unittest.TestCase):
    """Test JSON schema constraint generation for tool choices"""

    def setUp(self):
        """Set up test tools"""
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "Location to get weather for",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "Temperature unit",
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

    # TestJsonSchemaConstraint类的测试requiredtoolchoiceschema
    def test_required_tool_choice_schema(self):
        """Test schema generation for tool_choice='required'"""
        schema = get_json_schema_constraint(self.tools, "required")

        self.assertIsNotNone(schema)  # 断言不为None
        jsonschema.Draft202012Validator.check_schema(schema)

        self.assertEqual(schema["type"], "array")  # 断言相等
        self.assertEqual(schema["minItems"], 1)  # 断言相等
        self.assertIn("items", schema)  # 断言包含
        self.assertIn("anyOf", schema["items"])  # 断言包含

        # Should have schemas for both tools
        self.assertEqual(len(schema["items"]["anyOf"]), 2)  # 断言相等

        # Check that each tool schema is present
        tool_names = [
            item["properties"]["name"]["enum"][0] for item in schema["items"]["anyOf"]
        ]
        self.assertIn("get_weather", tool_names)  # 断言包含
        self.assertIn("search", tool_names)  # 断言包含

    # TestJsonSchemaConstraint类的测试specifictoolchoiceschema
    def test_specific_tool_choice_schema(self):
        """Test schema generation for specific tool choice"""
        tool_choice = ToolChoice(
            type="function", function=ToolChoiceFuncName(name="get_weather")
        )
        schema = get_json_schema_constraint(self.tools, tool_choice)

        self.assertIsNotNone(schema)  # 断言不为None
        jsonschema.Draft202012Validator.check_schema(schema)

        self.assertEqual(schema["type"], "array")  # 断言相等
        self.assertEqual(schema["minItems"], 1)  # 断言相等
        self.assertNotIn("maxItems", schema)  # 断言不包含

        # Should only have schema for the specific tool
        item_schema = schema["items"]
        self.assertEqual(item_schema["properties"]["name"]["enum"], ["get_weather"])  # 断言相等
        self.assertIn("parameters", item_schema["properties"])  # 断言包含

    # TestJsonSchemaConstraint类的测试specifictoolchoicedictschema
    def test_specific_tool_choice_dict_schema(self):
        """Test schema generation for specific tool choice as ToolChoice object"""
        tool_choice = ToolChoice(
            type="function", function=ToolChoiceFuncName(name="search")
        )
        schema = get_json_schema_constraint(self.tools, tool_choice)

        self.assertIsNotNone(schema)  # 断言不为None
        jsonschema.Draft202012Validator.check_schema(schema)

        self.assertEqual(schema["type"], "array")  # 断言相等
        self.assertEqual(schema["minItems"], 1)  # 断言相等
        self.assertNotIn("maxItems", schema)  # 断言不包含

        # Should only have schema for the specific tool
        item_schema = schema["items"]
        self.assertEqual(item_schema["properties"]["name"]["enum"], ["search"])  # 断言相等
        self.assertIn("parameters", item_schema["properties"])  # 断言包含

    # TestJsonSchemaConstraint类的测试specifictoolchoiceallowsmultiplecalls
    def test_specific_tool_choice_allows_multiple_calls(self):
        """Test that specific tool choice schema allows multiple calls.

        Regression test for https://github.com/sgl-project/sglang/issues/17998:
        maxItems: 1 caused the model to stall on whitespace when the prompt
        implied multiple calls to the same function.
        """
        tool_choice = ToolChoice(
            type="function", function=ToolChoiceFuncName(name="get_weather")
        )
        schema = get_json_schema_constraint(self.tools, tool_choice)

        single_call = [
            {"name": "get_weather", "parameters": {"location": "NYC"}},
        ]
        multi_call = [
            {"name": "get_weather", "parameters": {"location": "NYC"}},
            {"name": "get_weather", "parameters": {"location": "LA"}},
            {"name": "get_weather", "parameters": {"location": "Chicago"}},
        ]

        validator = jsonschema.Draft202012Validator(schema)
        validator.validate(single_call)
        validator.validate(multi_call)

    # TestJsonSchemaConstraint类的测试specifictoolchoicenoparallel
    def test_specific_tool_choice_no_parallel(self):
        """Test that parallel_tool_calls=False sets maxItems=1"""
        tool_choice = ToolChoice(
            type="function", function=ToolChoiceFuncName(name="get_weather")
        )
        schema = get_json_schema_constraint(
            self.tools, tool_choice, parallel_tool_calls=False
        )

        self.assertIsNotNone(schema)  # 断言不为None
        self.assertEqual(schema["maxItems"], 1)  # 断言相等

        single_call = [
            {"name": "get_weather", "parameters": {"location": "NYC"}},
        ]
        multi_call = [
            {"name": "get_weather", "parameters": {"location": "NYC"}},
            {"name": "get_weather", "parameters": {"location": "LA"}},
        ]

        validator = jsonschema.Draft202012Validator(schema)
        validator.validate(single_call)
        with self.assertRaises(jsonschema.ValidationError):  # 断言抛出异常
            validator.validate(multi_call)

    # TestJsonSchemaConstraint类的测试requiredtoolchoicenoparallel
    def test_required_tool_choice_no_parallel(self):
        """Test that required + parallel_tool_calls=False sets maxItems=1"""
        schema = get_json_schema_constraint(
            self.tools, "required", parallel_tool_calls=False
        )

        self.assertIsNotNone(schema)  # 断言不为None
        self.assertEqual(schema["maxItems"], 1)  # 断言相等

    # TestJsonSchemaConstraint类的测试nonexistenttoolchoice
    def test_nonexistent_tool_choice(self):
        """Test schema generation for nonexistent tool"""
        tool_choice = ToolChoice(
            type="function", function=ToolChoiceFuncName(name="nonexistent")
        )
        schema = get_json_schema_constraint(self.tools, tool_choice)

        self.assertIsNone(schema)  # 断言为None

    # TestJsonSchemaConstraint类的测试nonexistenttoolchoicedict
    def test_nonexistent_tool_choice_dict(self):
        """Test schema generation for nonexistent tool as dict"""
        tool_choice = {"type": "function", "function": {"name": "nonexistent"}}
        schema = get_json_schema_constraint(self.tools, tool_choice)

        self.assertIsNone(schema)  # 断言为None

    # TestJsonSchemaConstraint类的测试autotoolchoiceschema
    def test_auto_tool_choice_schema(self):
        """Test schema generation for tool_choice='auto'"""
        schema = get_json_schema_constraint(self.tools, "auto")

        self.assertIsNone(schema)  # 断言为None

    # TestJsonSchemaConstraint类的测试nonetoolchoiceschema
    def test_none_tool_choice_schema(self):
        """Test schema generation for tool_choice=None"""
        schema = get_json_schema_constraint(self.tools, None)

        self.assertIsNone(schema)  # 断言为None

    # TestJsonSchemaConstraint类的测试toolswithdefs
    def test_tools_with_defs(self):
        """Test schema generation with tools that have $defs"""
        tools_with_defs = [
            Tool(
                type="function",
                function=Function(
                    name="complex_tool",
                    description="Tool with complex schema",
                    parameters={
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "object",
                                "properties": {
                                    "nested": {"$ref": "#/$defs/NestedType"},
                                },
                            },
                        },
                        "$defs": {
                            "NestedType": {
                                "type": "object",
                                "properties": {
                                    "value": {"type": "string"},
                                },
                            },
                        },
                    },
                ),
            ),
        ]

        try:
            _get_tool_schema_defs(tools_with_defs)
        except ValueError as e:
            self.fail(f"Should not raise ValueError, but got: {e}")  # 抛出异常

        schema = get_json_schema_constraint(tools_with_defs, "required")

        self.assertIsNotNone(schema)  # 断言不为None
        jsonschema.Draft202012Validator.check_schema(schema)

        self.assertIn("$defs", schema)  # 断言包含
        self.assertIn("NestedType", schema["$defs"])  # 断言包含

    # TestJsonSchemaConstraint类的测试toolswithoutparameters
    def test_tools_without_parameters(self):
        """Test schema generation with tools that have no parameters"""
        tools_without_params = [
            Tool(
                type="function",
                function=Function(
                    name="simple_tool",
                    description="Tool without parameters",
                    parameters=None,
                ),
            ),
        ]

        schema = get_json_schema_constraint(tools_without_params, "required")

        self.assertIsNotNone(schema)  # 断言不为None
        jsonschema.Draft202012Validator.check_schema(schema)

        item_schema = schema["items"]["anyOf"][0]
        self.assertEqual(  # 断言相等
            item_schema["properties"]["parameters"],
            {"type": "object", "properties": {}},
        )

    # TestJsonSchemaConstraint类的测试conflictingdefsraisesvalueerror
    def test_conflicting_defs_raises_valueerror(self):
        """Test that conflicting tool definitions raise ValueError with proper message"""
        tools_with_conflicting_defs = [
            Tool(
                type="function",
                function=Function(
                    name="tool1",
                    description="Tool 1",
                    parameters={
                        "type": "object",
                        "properties": {},
                        "$defs": {
                            "ConflictingType": {
                                "type": "object",
                                "properties": {"value": {"type": "string"}},
                            },
                        },
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="tool2",
                    description="Tool 2",
                    parameters={
                        "type": "object",
                        "properties": {},
                        "$defs": {
                            "ConflictingType": {
                                "type": "object",
                                "properties": {"value": {"type": "number"}},
                            },
                        },
                    },
                ),
            ),
        ]

        with self.assertRaises(ValueError) as context:  # 断言抛出异常
            _get_tool_schema_defs(tools_with_conflicting_defs)

        self.assertIn(  # 断言包含
            "Tool definition 'ConflictingType' has multiple schemas",
            str(context.exception),
        )
        self.assertIn("which is not supported", str(context.exception))  # 断言包含

    # TestJsonSchemaConstraint类的测试toolswithemptydefs
    def test_tools_with_empty_defs(self):
        """Test tools with empty $defs objects"""
        tools_with_empty_defs = [
            Tool(
                type="function",
                function=Function(
                    name="empty_defs_tool",
                    description="Tool with empty $defs",
                    parameters={
                        "type": "object",
                        "properties": {
                            "data": {"type": "string"},
                        },
                        "required": ["data"],
                        "$defs": {},
                    },
                ),
            ),
        ]

        try:
            _get_tool_schema_defs(tools_with_empty_defs)
        except ValueError as e:
            self.fail(f"Should not raise ValueError, but got: {e}")  # 抛出异常

        schema = get_json_schema_constraint(tools_with_empty_defs, "required")
        self.assertIsNotNone(schema)  # 断言不为None
        jsonschema.Draft202012Validator.check_schema(schema)

        # Should not have $defs section when empty
        self.assertNotIn("$defs", schema)  # 断言不包含

    # TestJsonSchemaConstraint类的测试toolswithidenticaldefs
    def test_tools_with_identical_defs(self):
        """Test different tools with same $defs names but identical schemas (should not raise exception)"""
        tools_with_identical_defs = [
            Tool(
                type="function",
                function=Function(
                    name="weather_tool",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {"$ref": "#/$defs/Location"},
                        },
                        "required": ["location"],
                        "$defs": {
                            "Location": {
                                "type": "object",
                                "properties": {
                                    "lat": {"type": "number"},
                                    "lon": {"type": "number"},
                                },
                                "required": ["lat", "lon"],
                            },
                        },
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="address_tool",
                    description="Get address information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "address": {"$ref": "#/$defs/Location"},
                        },
                        "required": ["address"],
                        "$defs": {
                            "Location": {
                                "type": "object",
                                "properties": {
                                    "lat": {"type": "number"},
                                    "lon": {"type": "number"},
                                },
                                "required": ["lat", "lon"],
                            },
                        },
                    },
                ),
            ),
        ]

        try:
            _get_tool_schema_defs(tools_with_identical_defs)
        except ValueError as e:
            self.fail(
                f"Should not raise ValueError for identical schemas, but got: {e}"  # 抛出异常
            )

        # Also test that schema generation works
        schema = get_json_schema_constraint(tools_with_identical_defs, "required")
        self.assertIsNotNone(schema)  # 断言不为None
        jsonschema.Draft202012Validator.check_schema(schema)

        # Verify both tools are present
        tool_names = [
            item["properties"]["name"]["enum"][0] for item in schema["items"]["anyOf"]
        ]
        self.assertIn("weather_tool", tool_names)  # 断言包含
        self.assertIn("address_tool", tool_names)  # 断言包含

        # Should have $defs with Location
        self.assertIn("$defs", schema)  # 断言包含
        self.assertIn("Location", schema["$defs"])  # 断言包含

    # TestJsonSchemaConstraint类的测试toolswithnesteddefs
    def test_tools_with_nested_defs(self):
        """Test tools with nested $defs"""
        tools_with_nested_defs = [
            Tool(
                type="function",
                function=Function(
                    name="complex_tool",
                    description="Tool with nested $defs",
                    parameters={
                        "type": "object",
                        "properties": {
                            "user": {"$ref": "#/$defs/User"},
                            "settings": {"$ref": "#/$defs/Settings"},
                        },
                        "required": ["user"],
                        "$defs": {
                            "User": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "profile": {"$ref": "#/$defs/Profile"},
                                },
                                "required": ["id"],
                            },
                            "Profile": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "email": {"type": "string", "format": "email"},
                                },
                                "required": ["name"],
                            },
                            "Settings": {
                                "type": "object",
                                "properties": {
                                    "theme": {
                                        "type": "string",
                                        "enum": ["light", "dark"],
                                    },
                                    "notifications": {"type": "boolean"},
                                },
                            },
                        },
                    },
                ),
            ),
        ]

        try:
            _get_tool_schema_defs(tools_with_nested_defs)
        except ValueError as e:
            self.fail(f"Should not raise ValueError, but got: {e}")  # 抛出异常

        schema = get_json_schema_constraint(tools_with_nested_defs, "required")
        self.assertIsNotNone(schema)  # 断言不为None
        jsonschema.Draft202012Validator.check_schema(schema)

        # Verify all $defs are properly included
        self.assertIn("$defs", schema)  # 断言包含
        self.assertIn("User", schema["$defs"])  # 断言包含
        self.assertIn("Profile", schema["$defs"])  # 断言包含
        self.assertIn("Settings", schema["$defs"])  # 断言包含

    # TestJsonSchemaConstraint类的测试mixedtoolswithandwithoutdefs
    def test_mixed_tools_with_and_without_defs(self):
        """Test mixed tools with and without $defs"""
        mixed_tools = [
            Tool(
                type="function",
                function=Function(
                    name="simple_tool",
                    description="Simple tool without $defs",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="complex_tool",
                    description="Complex tool with $defs",
                    parameters={
                        "type": "object",
                        "properties": {
                            "data": {"$ref": "#/$defs/DataType"},
                        },
                        "required": ["data"],
                        "$defs": {
                            "DataType": {
                                "type": "object",
                                "properties": {
                                    "value": {"type": "string"},
                                    "metadata": {"type": "object"},
                                },
                                "required": ["value"],
                            },
                        },
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="another_simple_tool",
                    description="Another simple tool",
                    parameters={
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                        },
                        "required": ["id"],
                    },
                ),
            ),
        ]

        try:
            _get_tool_schema_defs(mixed_tools)
        except ValueError as e:
            self.fail(f"Should not raise ValueError, but got: {e}")  # 抛出异常

        schema = get_json_schema_constraint(mixed_tools, "required")
        self.assertIsNotNone(schema)  # 断言不为None
        jsonschema.Draft202012Validator.check_schema(schema)

        # Should have $defs from the complex tool
        self.assertIn("$defs", schema)  # 断言包含
        self.assertIn("DataType", schema["$defs"])  # 断言包含

        # Should have all three tools
        tool_names = [
            item["properties"]["name"]["enum"][0] for item in schema["items"]["anyOf"]
        ]
        self.assertEqual(len(tool_names), 3)  # 断言相等
        self.assertIn("simple_tool", tool_names)  # 断言包含
        self.assertIn("complex_tool", tool_names)  # 断言包含
        self.assertIn("another_simple_tool", tool_names)  # 断言包含

    # TestJsonSchemaConstraint类的测试toolswithdefsbutnorefs
    def test_tools_with_defs_but_no_refs(self):
        """Test tools with $defs but no $ref usage"""
        tools_with_unused_defs = [
            Tool(
                type="function",
                function=Function(
                    name="unused_defs_tool",
                    description="Tool with $defs but no $ref usage",
                    parameters={
                        "type": "object",
                        "properties": {
                            "data": {"type": "string"},
                        },
                        "required": ["data"],
                        "$defs": {
                            "UnusedType": {
                                "type": "object",
                                "properties": {
                                    "value": {"type": "string"},
                                },
                            },
                        },
                    },
                ),
            ),
        ]

        try:
            _get_tool_schema_defs(tools_with_unused_defs)
        except ValueError as e:
            self.fail(f"Should not raise ValueError, but got: {e}")  # 抛出异常

        schema = get_json_schema_constraint(tools_with_unused_defs, "required")
        self.assertIsNotNone(schema)  # 断言不为None
        jsonschema.Draft202012Validator.check_schema(schema)

        # Should still include $defs even if not referenced
        self.assertIn("$defs", schema)  # 断言包含
        self.assertIn("UnusedType", schema["$defs"])  # 断言包含


if __name__ == "__main__":
    unittest.main()
