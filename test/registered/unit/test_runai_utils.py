# 文件名: test_runai_utils.py - RunAI工具
import unittest
from pathlib import Path

from sglang.srt.configs.load_config import LoadFormat
from sglang.srt.utils.runai_utils import ObjectStorageModel, is_runai_obj_uri
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-b-test-cpu")


# TestRunaiUtils类
class TestRunaiUtils(CustomTestCase):

    # TestRunaiUtils类的测试isrunaiobjuris3
    def test_is_runai_obj_uri_s3(self):
        self.assertTrue(is_runai_obj_uri("s3://bucket/model/"))  # 断言为真
        self.assertTrue(is_runai_obj_uri("S3://Bucket/Model/"))  # 断言为真

    # TestRunaiUtils类的测试isrunaiobjurigs
    def test_is_runai_obj_uri_gs(self):
        self.assertTrue(is_runai_obj_uri("gs://bucket/model/"))  # 断言为真
        self.assertTrue(is_runai_obj_uri("GS://Bucket/Model/"))  # 断言为真

    # TestRunaiUtils类的测试isrunaiobjuriaz
    def test_is_runai_obj_uri_az(self):
        self.assertTrue(is_runai_obj_uri("az://container/model/"))  # 断言为真
        self.assertTrue(is_runai_obj_uri("AZ://Container/Model/"))  # 断言为真

    # TestRunaiUtils类的测试isrunaiobjurilocalpaths
    def test_is_runai_obj_uri_local_paths(self):
        self.assertFalse(is_runai_obj_uri("/path/to/model"))  # 断言为假
        self.assertFalse(is_runai_obj_uri("./relative/path"))  # 断言为假
        self.assertFalse(is_runai_obj_uri("meta-llama/Llama-3.2-1B"))  # 断言为假

    # TestRunaiUtils类的测试isrunaiobjuriotherschemes
    def test_is_runai_obj_uri_other_schemes(self):
        self.assertFalse(is_runai_obj_uri("http://example.com/model"))  # 断言为假
        self.assertFalse(is_runai_obj_uri("https://example.com/model"))  # 断言为假
        self.assertFalse(is_runai_obj_uri("ftp://example.com/model"))  # 断言为假

    # TestRunaiUtils类的测试isrunaiobjuripathlib
    def test_is_runai_obj_uri_pathlib(self):
        self.assertFalse(is_runai_obj_uri(Path("/local/model")))  # 断言为假

    # TestRunaiUtils类的测试getpathdeterministic
    def test_get_path_deterministic(self):
        path1 = ObjectStorageModel.get_path("s3://bucket/model/")
        path2 = ObjectStorageModel.get_path("s3://bucket/model/")
        self.assertEqual(path1, path2)  # 断言相等

    # TestRunaiUtils类的测试getpathdifferenturis
    def test_get_path_different_uris(self):
        path1 = ObjectStorageModel.get_path("s3://bucket/model-a/")
        path2 = ObjectStorageModel.get_path("s3://bucket/model-b/")
        self.assertNotEqual(path1, path2)  # 断言不相等

    # TestRunaiUtils类的测试getpathcontainsmodelstreamer
    def test_get_path_contains_model_streamer(self):
        path = ObjectStorageModel.get_path("s3://bucket/model/")
        self.assertIn("model_streamer", path)  # 断言包含

    # TestRunaiUtils类的测试loadformatenum
    def test_load_format_enum(self):
        self.assertEqual(LoadFormat.RUNAI_STREAMER.value, "runai_streamer")  # 断言相等


if __name__ == "__main__":
    unittest.main()
