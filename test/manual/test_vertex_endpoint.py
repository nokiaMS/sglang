# 文件名: test_vertex_endpoint.py - Vertex端点测试 - 验证Vertex AI兼容的generate端点
"""
python3 -m unittest test_vertex_endpoint.TestVertexEndpoint.test_vertex_generate
"""

import unittest
from http import HTTPStatus

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestVertexEndpoint(CustomTestCase):
    @classmethod
    # setUpClass
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--cuda-graph-max-bs", 2],
        )

    @classmethod
    # tearDownClass
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    # 运行生成 - 发送生成请求并返回结果
    def run_generate(self, parameters):
        data = {
            "instances": [
                {"text": "The capital of France is"},
                {"text": "The capital of China is"},
            ],
            "parameters": parameters,
        }
        response = requests.post(self.base_url + "/vertex_generate", json=data)
        response_json = response.json()
        assert len(response_json["predictions"]) == len(data["instances"])
        return response_json

    # 测试vertex generate
    def test_vertex_generate(self):
        for parameters in [None, {"sampling_params": {"max_new_tokens": 4}}]:
            self.run_generate(parameters)

    # 测试vertex generate fail
    def test_vertex_generate_fail(self):
        data = {
            "instances": [
                {"prompt": "The capital of France is"},
            ],
        }
        response = requests.post(self.base_url + "/vertex_generate", json=data)
        assert response.status_code == HTTPStatus.BAD_REQUEST


if __name__ == "__main__":
    unittest.main()
