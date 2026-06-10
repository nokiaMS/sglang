# 文件名: test_bind_cache.py - 测试SGLang函数绑定与缓存功能
import unittest

import sglang as sgl
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST, CustomTestCase


class TestBind(CustomTestCase):
    backend = None

    @classmethod
    # 类级别初始化，启动服务器或设置测试环境
    def setUpClass(cls):
        cls.backend = sgl.Runtime(model_path=DEFAULT_MODEL_NAME_FOR_TEST)
        sgl.set_default_backend(cls.backend)

    @classmethod
    # 类级别清理，关闭服务器或清理资源
    def tearDownClass(cls):
        cls.backend.shutdown()

    # 测试bind功能
    def test_bind(self):
        @sgl.function
        # 执行few_shot_qa
        def few_shot_qa(s, prompt, question):
            s += prompt
            s += "Q: What is the capital of France?\n"
            s += "A: Paris\n"
            s += "Q: " + question + "\n"
            s += "A:" + sgl.gen("answer", stop="\n")

        few_shot_qa_2 = few_shot_qa.bind(
            prompt="The following are questions with answers.\n\n"
        )

        tracer = few_shot_qa_2.trace()
        print(tracer.last_node.print_graph_dfs() + "\n")

    # 测试cache功能
    def test_cache(self):
        @sgl.function
        # 执行few_shot_qa
        def few_shot_qa(s, prompt, question):
            s += prompt
            s += "Q: What is the capital of France?\n"
            s += "A: Paris\n"
            s += "Q: " + question + "\n"
            s += "A:" + sgl.gen("answer", stop="\n")

        few_shot_qa_2 = few_shot_qa.bind(
            prompt="Answer the following questions as if you were a 5-year-old kid.\n\n"
        )
        few_shot_qa_2.cache(self.backend)


if __name__ == "__main__":
    unittest.main()
