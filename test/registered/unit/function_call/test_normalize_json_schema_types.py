# 文件名: test_normalize_json_schema_types.py - JSON Schema类型规范化
"""Unit tests for tool-parameter schema alias normalization."""

import json
import unittest

from jsonschema import Draft202012Validator, SchemaError

from sglang.srt.function_call.utils import normalize_json_schema_types
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


# TestNormalizeJsonSchemaTypes类
class TestNormalizeJsonSchemaTypes(CustomTestCase):

    # TestNormalizeJsonSchemaTypes类的内部方法_assert_accepts
    def _assert_accepts(self, schema: dict) -> None:
        Draft202012Validator.check_schema(schema)

    # TestNormalizeJsonSchemaTypes类的测试enumaliasbecomesstring
    def test_enum_alias_becomes_string(self):
        schema = {
            "type": "object",
            "properties": {"color": {"type": "enum", "enum": ["red", "green", "blue"]}},
        }
        normalize_json_schema_types(schema)
        self.assertEqual(schema["properties"]["color"]["type"], "string")  # 断言相等
        self.assertEqual(  # 断言相等
            schema["properties"]["color"]["enum"], ["red", "green", "blue"]
        )
        self._assert_accepts(schema)

    # TestNormalizeJsonSchemaTypes类的测试varcharaliasbecomesstring
    def test_varchar_alias_becomes_string(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "varchar"},
                "short_name": {"type": "VARCHAR(255)"},
            },
        }
        normalize_json_schema_types(schema)
        self.assertEqual(schema["properties"]["name"]["type"], "string")  # 断言相等
        self.assertEqual(schema["properties"]["short_name"]["type"], "string")  # 断言相等
        self._assert_accepts(schema)

    # TestNormalizeJsonSchemaTypes类的测试numericaliases
    def test_numeric_aliases(self):
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "int"},
                "big": {"type": "bigint"},
                "price": {"type": "decimal(10,2)"},
                "ratio": {"type": "float"},
            },
        }
        normalize_json_schema_types(schema)
        props = schema["properties"]
        self.assertEqual(props["age"]["type"], "integer")  # 断言相等
        self.assertEqual(props["big"]["type"], "integer")  # 断言相等
        self.assertEqual(props["price"]["type"], "number")  # 断言相等
        self.assertEqual(props["ratio"]["type"], "number")  # 断言相等
        self._assert_accepts(schema)

    # TestNormalizeJsonSchemaTypes类的测试prefixmatchednumerictypes
    def test_prefix_matched_numeric_types(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "int32"},
                "b": {"type": "int64"},
                "c": {"type": "uint"},
                "d": {"type": "unsigned"},
                "e": {"type": "long"},
                "f": {"type": "short"},
                "g": {"type": "float32"},
                "h": {"type": "float64"},
                "i": {"type": "num"},
                "j": {"type": "numeric"},
            },
        }
        normalize_json_schema_types(schema)
        p = schema["properties"]
        for k in ("a", "b", "c", "d", "e", "f"):
            self.assertEqual(p[k]["type"], "integer")  # 断言相等
        for k in ("g", "h", "i", "j"):
            self.assertEqual(p[k]["type"], "number")  # 断言相等
        self._assert_accepts(schema)

    # TestNormalizeJsonSchemaTypes类的测试prefixmatchedcompoundtypes
    def test_prefix_matched_compound_types(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "list[str]"},
                "b": {"type": "list<int>"},
                "c": {"type": "dict"},
                "d": {"type": "dict[str, int]"},
                "e": {"type": "long long"},
            },
        }
        normalize_json_schema_types(schema)
        p = schema["properties"]
        self.assertEqual(p["a"]["type"], "array")  # 断言相等
        self.assertEqual(p["b"]["type"], "array")  # 断言相等
        self.assertEqual(p["c"]["type"], "object")  # 断言相等
        self.assertEqual(p["d"]["type"], "object")  # 断言相等
        self.assertEqual(p["e"]["type"], "integer")  # 断言相等
        self._assert_accepts(schema)

    # TestNormalizeJsonSchemaTypes类的测试wordboundarypreventsfalsepositives
    def test_word_boundary_prevents_false_positives(self):
        """Prefixes must end at a non-identifier char, so custom type names
        that merely start with a known prefix are left alone."""
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "internal"},
                "b": {"type": "list_price"},
                "c": {"type": "integer_enum"},
                "d": {"type": "dictionary_entry"},
                "e": {"type": "floating"},
            },
        }
        normalize_json_schema_types(schema)
        p = schema["properties"]
        for key, expected in [
            ("a", "internal"),
            ("b", "list_price"),
            ("c", "integer_enum"),
            ("d", "dictionary_entry"),
            ("e", "floating"),
        ]:
            self.assertEqual(p[key]["type"], expected)  # 断言相等

    # TestNormalizeJsonSchemaTypes类的测试recursesintodraft202012keywords
    def test_recurses_into_draft_2020_12_keywords(self):
        schema = {
            "type": "object",
            "properties": {
                "row": {
                    "dependentSchemas": {
                        "kind": {
                            "properties": {"sku": {"type": "varchar"}},
                        },
                    },
                    "propertyNames": {"type": "str"},
                    "unevaluatedProperties": {"type": "bigint"},
                },
                "rows": {
                    "type": "arr",
                    "unevaluatedItems": {"type": "int32"},
                },
            },
        }
        normalize_json_schema_types(schema)
        row = schema["properties"]["row"]
        rows = schema["properties"]["rows"]
        self.assertEqual(  # 断言相等
            row["dependentSchemas"]["kind"]["properties"]["sku"]["type"], "string"
        )
        self.assertEqual(row["propertyNames"]["type"], "string")  # 断言相等
        self.assertEqual(row["unevaluatedProperties"]["type"], "integer")  # 断言相等
        self.assertEqual(rows["type"], "array")  # 断言相等
        self.assertEqual(rows["unevaluatedItems"]["type"], "integer")  # 断言相等
        self._assert_accepts(schema)

    # TestNormalizeJsonSchemaTypes类的测试bytesandarraliases
    def test_bytes_and_arr_aliases(self):
        schema = {
            "type": "object",
            "properties": {
                "payload": {"type": "binary"},
                "raw": {"type": "bytea"},
                "items": {"type": "arr"},
            },
        }
        normalize_json_schema_types(schema)
        self.assertEqual(schema["properties"]["payload"]["type"], "string")  # 断言相等
        self.assertEqual(schema["properties"]["raw"]["type"], "string")  # 断言相等
        self.assertEqual(schema["properties"]["items"]["type"], "array")  # 断言相等
        self._assert_accepts(schema)

    # TestNormalizeJsonSchemaTypes类的测试boolaliasbecomesboolean
    def test_bool_alias_becomes_boolean(self):
        schema = {"type": "object", "properties": {"flag": {"type": "bool"}}}
        normalize_json_schema_types(schema)
        self.assertEqual(schema["properties"]["flag"]["type"], "boolean")  # 断言相等
        self._assert_accepts(schema)

    # TestNormalizeJsonSchemaTypes类的测试caseinsensitive
    def test_case_insensitive(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "VARCHAR"},
                "b": {"type": "INT"},
                "c": {"type": "String"},
            },
        }
        normalize_json_schema_types(schema)
        p = schema["properties"]
        self.assertEqual(p["a"]["type"], "string")  # 断言相等
        self.assertEqual(p["b"]["type"], "integer")  # 断言相等
        self.assertEqual(p["c"]["type"], "string")  # 断言相等
        self._assert_accepts(schema)

    # TestNormalizeJsonSchemaTypes类的测试arrayandobjectaliases
    def test_array_and_object_aliases(self):
        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "list", "items": {"type": "str"}},
                "meta": {"type": "dict"},
            },
        }
        normalize_json_schema_types(schema)
        self.assertEqual(schema["properties"]["tags"]["type"], "array")  # 断言相等
        self.assertEqual(schema["properties"]["tags"]["items"]["type"], "string")  # 断言相等
        self.assertEqual(schema["properties"]["meta"]["type"], "object")  # 断言相等
        self._assert_accepts(schema)

    # TestNormalizeJsonSchemaTypes类的测试nestedanyofanddefs
    def test_nested_anyof_and_defs(self):
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "int"},
                        {"type": "varchar"},
                    ]
                }
            },
            "$defs": {
                "Row": {"type": "object", "properties": {"id": {"type": "bigint"}}}
            },
        }
        normalize_json_schema_types(schema)
        any_of = schema["properties"]["value"]["anyOf"]
        self.assertEqual(any_of[0]["type"], "integer")  # 断言相等
        self.assertEqual(any_of[1]["type"], "string")  # 断言相等
        self.assertEqual(schema["$defs"]["Row"]["properties"]["id"]["type"], "integer")  # 断言相等
        self._assert_accepts(schema)

    # TestNormalizeJsonSchemaTypes类的测试typelistmembernormalized
    def test_type_list_member_normalized(self):
        schema = {"type": ["varchar", "null"]}
        normalize_json_schema_types(schema)
        self.assertEqual(schema["type"], ["string", "null"])  # 断言相等
        self._assert_accepts(schema)

    # TestNormalizeJsonSchemaTypes类的测试typelistdeduplicatesafternormalization
    def test_type_list_deduplicates_after_normalization(self):
        schema = {"type": ["varchar", "string", "null", "int", "integer"]}
        normalize_json_schema_types(schema)
        self.assertEqual(schema["type"], ["string", "null", "integer"])  # 断言相等
        self._assert_accepts(schema)

    # TestNormalizeJsonSchemaTypes类的测试standardtypesuntouched
    def test_standard_types_untouched(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
                "c": {"type": "boolean"},
            },
        }
        normalize_json_schema_types(schema)
        self.assertEqual(schema["properties"]["a"]["type"], "string")  # 断言相等
        self.assertEqual(schema["properties"]["b"]["type"], "integer")  # 断言相等
        self.assertEqual(schema["properties"]["c"]["type"], "boolean")  # 断言相等

    # TestNormalizeJsonSchemaTypes类的测试unknowntypeleftalone
    def test_unknown_type_left_alone(self):
        schema = {"type": "object", "properties": {"x": {"type": "geometry"}}}
        normalize_json_schema_types(schema)
        self.assertEqual(schema["properties"]["x"]["type"], "geometry")  # 断言相等
        with self.assertRaises(SchemaError):  # 断言抛出异常
            self._assert_accepts(schema)

    # TestNormalizeJsonSchemaTypes类的测试commondbormtypenamesaccepted
    def test_common_db_orm_type_names_accepted(self):
        """Common non-standard DB/ORM type names all survive validation."""
        recognized = [
            # string family
            "string",
            "str",
            "text",
            "varchar",
            "char",
            "enum",
            # integer via prefix
            "int",
            "int32",
            "int64",
            "uint",
            "uint8",
            "long",
            "long long",
            "short",
            "unsigned",
            # number via prefix
            "num",
            "numeric",
            "float",
            "float32",
            "float64",
            # boolean
            "boolean",
            "bool",
            "binary",
            "bytea",
            "bytes",
            "blob",
            "varbinary",
            # compound
            "object",
            "array",
            "arr",
            "dict",
            "dict[str, int]",
            "list",
            "list[str]",
        ]
        for t in recognized:
            schema = {"type": "object", "properties": {"x": {"type": t}}}
            normalize_json_schema_types(schema)
            try:
                self._assert_accepts(schema)
            except SchemaError as e:
                self.fail(f"type {t!r} -> {schema['properties']['x']['type']!r}: {e}")

    # TestNormalizeJsonSchemaTypes类的测试preexisting400schemanowaccepted
    def test_pre_existing_400_schema_now_accepted(self):
        schema = {
            "type": "object",
            "properties": {
                "sql": {"type": "varchar"},
                "mode": {"type": "enum", "enum": ["read", "write"]},
            },
            "required": ["sql", "mode"],
        }
        normalize_json_schema_types(schema)
        self._assert_accepts(schema)

    # TestNormalizeJsonSchemaTypes类的测试idempotent
    def test_idempotent(self):
        """Running normalize twice produces the same result as running it once."""
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "varchar"},
                "b": {"type": "int32"},
                "c": {"type": ["bigint", "null"]},
                "d": {
                    "anyOf": [{"type": "enum"}, {"type": "decimal(10,2)"}],
                },
            },
        }
        normalize_json_schema_types(schema)
        once = json.loads(json.dumps(schema))
        normalize_json_schema_types(schema)
        self.assertEqual(schema, once)  # 断言相等

    # TestNormalizeJsonSchemaTypes类的测试nonstringtypevaluespassthrough
    def test_non_string_type_values_pass_through(self):
        """``type`` that isn't str/list is left for the real validator to reject."""
        for bad in (None, 42, {"$ref": "#/$defs/Foo"}, ["string", 1, None]):
            schema = {"properties": {"x": {"type": bad}}}
            normalize_json_schema_types(schema)
            self.assertEqual(schema["properties"]["x"]["type"], bad)  # 断言相等

    # TestNormalizeJsonSchemaTypes类的测试recursesintoallwalkedkeywords
    def test_recurses_into_all_walked_keywords(self):
        """Every keyword the walker recurses into must actually rewrite nested aliases."""
        schema = {
            "patternProperties": {"^x_": {"type": "varchar"}},
            "definitions": {"Row": {"type": "bigint"}},
            "prefixItems": [{"type": "int32"}, {"type": "float64"}],
            "if": {"type": "bool"},
            "then": {"type": "str"},
            "else": {"type": "decimal"},
            "not": {"type": "uuid"},
            "contains": {"type": "enum"},
            "additionalProperties": {"type": "bool"},
        }
        normalize_json_schema_types(schema)
        self.assertEqual(schema["patternProperties"]["^x_"]["type"], "string")  # 断言相等
        self.assertEqual(schema["definitions"]["Row"]["type"], "integer")  # 断言相等
        self.assertEqual(schema["prefixItems"][0]["type"], "integer")  # 断言相等
        self.assertEqual(schema["prefixItems"][1]["type"], "number")  # 断言相等
        self.assertEqual(schema["if"]["type"], "boolean")  # 断言相等
        self.assertEqual(schema["then"]["type"], "string")  # 断言相等
        self.assertEqual(schema["else"]["type"], "number")  # 断言相等
        self.assertEqual(schema["not"]["type"], "string")  # 断言相等
        self.assertEqual(schema["contains"]["type"], "string")  # 断言相等
        self.assertEqual(schema["additionalProperties"]["type"], "boolean")  # 断言相等

    # TestNormalizeJsonSchemaTypes类的测试cyclicschemaraisesrecursionerror
    def test_cyclic_schema_raises_recursion_error(self):
        """A pathological cyclic schema surfaces as RecursionError; caller
        (``_validate_request``) converts it to a 400, not a 500."""
        schema = {"type": "object"}
        schema["items"] = schema
        with self.assertRaises(RecursionError):  # 断言抛出异常
            normalize_json_schema_types(schema)

    # TestNormalizeJsonSchemaTypes类的测试booleansubschemadoesnotcrash
    def test_boolean_subschema_does_not_crash(self):
        """``additionalProperties: True`` and ``items: false`` are valid 2020-12
        forms; the walker must pass through without raising."""
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "varchar"},
                "b": {"type": "array", "items": False},
            },
            "additionalProperties": True,
            "unevaluatedProperties": False,
        }
        normalize_json_schema_types(schema)
        self.assertEqual(schema["properties"]["a"]["type"], "string")  # 断言相等
        self.assertIs(schema["properties"]["b"]["items"], False)  # 断言是同一对象
        self.assertIs(schema["additionalProperties"], True)  # 断言是同一对象
        self.assertIs(schema["unevaluatedProperties"], False)  # 断言是同一对象
        self._assert_accepts(schema)


if __name__ == "__main__":
    unittest.main()
