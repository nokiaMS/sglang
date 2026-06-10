# 文件名: test_pp_with_hicache.py - 测试流水线并行与分层缓存(HiCache)的集成精度
"""
Usage:
python3 -m unittest test_pp_with_hicache.TestPPWithHiCache.test_eval_accuracy
"""

import os
import subprocess
import time
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    find_available_port,
    popen_launch_server,
)


class TestPPWithHiCache(unittest.TestCase):
    @classmethod
    # 类级别初始化，启动服务器或设置测试环境
    def setUpClass(cls):
        cls.base_url = f"http://127.0.0.1:{find_available_port(23337)}"  # 查找可用端口
        parsed_url = urlparse(cls.base_url)
        cls.base_host = parsed_url.hostname
        cls.base_port = str(parsed_url.port)
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST

        cls._start_mooncake_services()

        server_args_dict = {
            "--enable-hierarchical-cache": True,
            "--mem-fraction-static": 0.6,
            "--hicache-ratio": 1.2,
            "--page-size": 64,
            "--enable-cache-report": True,
            "--hicache-storage-prefetch-policy": "wait_complete",
            "--hicache-storage-backend": "mooncake",
            "--tp-size": 2,
            "--pp-size": 2,
            "--chunked-prefill-size": 256,
            "--hicache-mem-layout": "page_first",
        }

        final_server_args = []
        for key, value in server_args_dict.items():
            final_server_args.append(str(key))
            if value is not True:
                final_server_args.append(str(value))

        env_vars = {**os.environ, **cls._mooncake_env()}

        try:
            cls.process = popen_launch_server(  # 启动推理服务器
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=final_server_args,
                env=env_vars,
            )
        except Exception:
            cls._stop_mooncake_services()
            raise

    @classmethod
    # 类级别清理，关闭服务器或清理资源
    def tearDownClass(cls):
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid)  # 终止服务器进程
        cls._stop_mooncake_services()

    @classmethod
    # 启动Mooncake元数据与主控服务
    def _start_mooncake_services(cls):
        try:
            import mooncake.http_metadata_server  # type: ignore  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(  # 跳过测试
                f"Mooncake metadata server module unavailable: {exc}"
            ) from exc

        cls._mooncake_master_port = find_available_port(50051)  # 查找可用端口
        cls._mooncake_metadata_port = find_available_port(8080)  # 查找可用端口

        try:
            cls._mooncake_metadata_process = subprocess.Popen(  # 启动子进程
                [
                    "python3",
                    "-m",
                    "mooncake.http_metadata_server",
                    "--port",
                    str(cls._mooncake_metadata_port),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
            )
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            cls._stop_mooncake_services()
            raise unittest.SkipTest(  # 跳过测试
                f"Could not start Mooncake metadata service: {exc}"
            ) from exc

        try:
            cls._mooncake_master_process = subprocess.Popen(  # 启动子进程
                ["mooncake_master", "--port", str(cls._mooncake_master_port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
            )
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            cls._stop_mooncake_services()
            raise unittest.SkipTest(f"Could not start mooncake_master: {exc}") from exc  # 跳过测试

        if not cls._wait_for_mooncake_ready():
            cls._stop_mooncake_services()
            raise unittest.SkipTest("Mooncake services did not become ready in time")  # 跳过测试

    @classmethod
    # 停止Mooncake服务进程
    def _stop_mooncake_services(cls):
        for attr in ("_mooncake_metadata_process", "_mooncake_master_process"):
            proc = getattr(cls, attr, None)
            if proc:
                try:
                    os.killpg(os.getpgid(proc.pid), 9)
                    proc.wait(timeout=5)
                except Exception:
                    pass
        cls._mooncake_metadata_process = None
        cls._mooncake_master_process = None

    @classmethod
    # 构建Mooncake环境变量字典
    def _mooncake_env(cls):
        return {
            "MOONCAKE_MASTER": f"127.0.0.1:{cls._mooncake_master_port}",
            "MOONCAKE_PROTOCOL": "tcp",
            "MC_MS_AUTO_DISC": "0",
            "MOONCAKE_DEVICE": "",
            "MOONCAKE_TE_META_DATA_SERVER": f"http://127.0.0.1:{cls._mooncake_metadata_port}/metadata",
            "MOONCAKE_GLOBAL_SEGMENT_SIZE": "4294967296",
            "SGLANG_ENABLE_DETERMINISTIC_INFERENCE": "1",
        }

    @classmethod
    # 等待Mooncake服务就绪
    def _wait_for_mooncake_ready(cls, timeout: int = 30) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            metadata_ready = False
            master_ready = False

            if (
                getattr(cls, "_mooncake_metadata_process", None)
                and cls._mooncake_metadata_process.poll() is None
            ):
                try:
                    resp = requests.get(  # 发送GET请求
                        f"http://127.0.0.1:{cls._mooncake_metadata_port}/metadata",
                        timeout=2,
                    )
                    print(resp)
                    metadata_ready = True
                except requests.RequestException:
                    metadata_ready = False

            if (
                getattr(cls, "_mooncake_master_process", None)
                and cls._mooncake_master_process.poll() is None
            ):
                if time.time() - start_time > 3:
                    master_ready = True

            if metadata_ready and master_ready:
                return True

            time.sleep(1.5)

        return False

    # 刷新服务器缓存
    def flush_cache(self):
        res = requests.post(  # 发送POST请求
            f"{self.base_url}/flush_cache",
            params={"timeout": 30},
            timeout=40,
        )
        res.raise_for_status()

    # 测试eval accuracy功能
    def test_eval_accuracy(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=40,
            num_threads=24,
        )

        metrics_initial = run_eval(args)  # 运行评估
        self.assertGreater(metrics_initial["score"], 0.6)  # 断言精度大于阈值

        self.flush_cache()

        metrics_cached = run_eval(args)  # 运行评估
        self.assertGreater(metrics_cached["score"], 0.6)  # 断言精度大于阈值

        accuracy_diff = abs(metrics_initial["score"] - metrics_cached["score"])
        self.assertLess(accuracy_diff, 0.05)  # 断言值小于阈值


if __name__ == "__main__":
    unittest.main()
