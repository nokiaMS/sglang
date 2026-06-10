# 文件名: test_json_response.py - JSON响应
import unittest

import numpy as np
import orjson

from sglang.srt.utils.json_response import (
    SGLangORJSONResponse,
    dumps_json,
    orjson_response,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-b-test-cpu")


# TestJSONResponseUtils类
class TestJSONResponseUtils(unittest.TestCase):

    # TestJSONResponseUtils类的测试dumpsjsonmapsnonfinitevaluestonull
    def test_dumps_json_maps_non_finite_values_to_null(self):
        payload = {
            "neg_inf": float("-inf"),
            "pos_inf": float("inf"),
            "nan": float("nan"),
        }
        parsed = orjson.loads(dumps_json(payload))

        self.assertIsNone(parsed["neg_inf"])  # 断言为None
        self.assertIsNone(parsed["pos_inf"])  # 断言为None
        self.assertIsNone(parsed["nan"])  # 断言为None

    # TestJSONResponseUtils类的测试dumpsjsonsupportsnumpyandnonstringkeys
    def test_dumps_json_supports_numpy_and_non_string_keys(self):
        payload = {
            1: np.array([1, 2, 3], dtype=np.int64),
            "scalar": np.float32(1.5),
        }
        parsed = orjson.loads(dumps_json(payload))

        self.assertEqual(parsed["1"], [1, 2, 3])  # 断言相等
        self.assertAlmostEqual(parsed["scalar"], 1.5)  # 断言近似相等

    # TestJSONResponseUtils类的测试orjsonresponseusesexpectedmediatype
    def test_orjson_response_uses_expected_media_type(self):
        response = orjson_response({"value": float("-inf")}, status_code=201)
        parsed = orjson.loads(response.body)

        self.assertEqual(response.status_code, 201)  # 断言相等
        self.assertEqual(response.media_type, "application/json")  # 断言相等
        self.assertIsNone(parsed["value"])  # 断言为None

    # TestJSONResponseUtils类的测试sglangorjsonresponseserializeswithsharedoptions
    def test_sglang_orjson_response_serializes_with_shared_options(self):
        response = SGLangORJSONResponse(content={"value": float("-inf")})
        parsed = orjson.loads(response.body)

        self.assertIsNone(parsed["value"])  # 断言为None


if __name__ == "__main__":
    unittest.main()
