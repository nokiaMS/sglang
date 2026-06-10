# 文件名: test_continuous_usage_stats.py - 测试连续使用统计功能（流式token使用量统计）
import asyncio
import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestContinuousUsageStats(CustomTestCase):
    @classmethod
    # 类级别初始化，启动服务器或设置测试环境
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(cls.model, cls.base_url, timeout=300)  # 启动推理服务器
        cls.client = openai.Client(api_key="EMPTY", base_url=f"{cls.base_url}/v1")
        cls.aclient = openai.AsyncClient(api_key="EMPTY", base_url=f"{cls.base_url}/v1")

    @classmethod
    # 类级别清理，关闭服务器或清理资源
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)  # 终止服务器进程

    # 测试continuous usage stats enabled功能
    def test_continuous_usage_stats_enabled(self):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "What is machine learning?"}],
            stream=True,
            max_tokens=30,
            temperature=0,
            stream_options={"include_usage": True, "continuous_usage_stats": True},
        )

        chunks_with_usage = 0
        chunks_with_content = 0
        last_usage = None

        for chunk in stream:
            has_content = len(chunk.choices) > 0 and chunk.choices[0].delta.content
            if chunk.usage:
                chunks_with_usage += 1
                last_usage = chunk.usage
            if has_content:
                chunks_with_content += 1

        assert chunks_with_content > 0
        assert chunks_with_usage >= chunks_with_content
        assert last_usage.prompt_tokens > 0
        assert last_usage.completion_tokens > 0
        assert (
            last_usage.total_tokens
            == last_usage.prompt_tokens + last_usage.completion_tokens
        )

    # 测试continuous usage stats async功能
    async def test_continuous_usage_stats_async(self):
        stream = await self.aclient.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "What is deep learning?"}],
            stream=True,
            max_tokens=30,
            temperature=0,
            stream_options={"include_usage": True, "continuous_usage_stats": True},
        )

        chunks_with_usage = 0
        chunks_with_content = 0

        async for chunk in stream:
            has_content = len(chunk.choices) > 0 and chunk.choices[0].delta.content
            if chunk.usage:
                chunks_with_usage += 1
            if has_content:
                chunks_with_content += 1

        assert chunks_with_content > 0
        assert chunks_with_usage >= chunks_with_content

    # 测试continuous usage stats disabled功能
    def test_continuous_usage_stats_disabled(self):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "What is AI?"}],
            stream=True,
            max_tokens=30,
            temperature=0,
            stream_options={"include_usage": True, "continuous_usage_stats": False},
        )

        usage_chunks = []
        for chunk in stream:
            if chunk.usage:
                usage_chunks.append(chunk)

        assert len(usage_chunks) == 1
        assert len(usage_chunks[0].choices) == 0

    # 测试async runner功能
    def test_async_runner(self):
        asyncio.run(self.test_continuous_usage_stats_async())


if __name__ == "__main__":
    unittest.main()
