# 文件名: test_structural_tag.py - 测试结构化标签功能（常量字符串与JSON Schema约束）
"""
python3 -m unittest test.srt.openai_server.features.test_structural_tag
"""

import json
import unittest
from typing import Any

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


# 设置测试类（启动服务器）
def setup_class(cls, backend: str):
    cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    cls.base_url = DEFAULT_URL_FOR_TEST

    other_args = [
        "--max-running-requests",
        "10",
        "--grammar-backend",
        backend,
    ]

    cls.process = popen_launch_server(  # 启动推理服务器
        cls.model,
        cls.base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
    )


class TestStructuralTagXGrammarBackend(CustomTestCase):
    model: str
    base_url: str
    process: Any

    @classmethod
    # 类级别初始化，启动服务器或设置测试环境
    def setUpClass(cls):
        setup_class(cls, backend="xgrammar")

    @classmethod
    # 类级别清理，关闭服务器或清理资源
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)  # 终止服务器进程

    # 测试stag constant str openai功能
    def test_stag_constant_str_openai(self):
        client = openai.Client(api_key="EMPTY", base_url=f"{self.base_url}/v1")

        # even when the answer is ridiculous, the model should follow the instruction
        answer = "The capital of France is Berlin."

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {
                    "role": "user",
                    "content": "Introduce the capital of France. Return in a JSON format.",
                },
            ],
            temperature=0,
            max_tokens=128,
            response_format={
                "type": "structural_tag",
                "format": {
                    "type": "const_string",
                    "value": answer,
                },
            },
        )

        text = response.choices[0].message.content
        self.assertEqual(text, answer)  # 断言值相等

    # 测试stag json schema openai功能
    def test_stag_json_schema_openai(self):
        client = openai.Client(api_key="EMPTY", base_url=f"{self.base_url}/v1")
        json_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "pattern": "^[\\w]+$"},
                "population": {"type": "integer"},
            },
            "required": ["name", "population"],
            "additionalProperties": False,
        }

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {
                    "role": "user",
                    "content": "Introduce the capital of France. Return in a JSON format.",
                },
            ],
            temperature=0,
            max_tokens=128,
            response_format={
                "type": "structural_tag",
                "format": {
                    "type": "json_schema",
                    "json_schema": json_schema,
                },
            },
        )

        text = response.choices[0].message.content
        try:
            js_obj = json.loads(text)
        except (TypeError, json.decoder.JSONDecodeError):
            print("JSONDecodeError", text)
            raise

        self.assertIsInstance(js_obj["name"], str)
        self.assertIsInstance(js_obj["population"], int)


if __name__ == "__main__":
    unittest.main()
