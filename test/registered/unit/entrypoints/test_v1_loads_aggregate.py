# 文件名: test_v1_loads_aggregate.py - V1负载聚合
"""Unit tests for /v1/loads load snapshot response behavior."""

import asyncio
import os
import tempfile
import unittest
from types import SimpleNamespace

import msgspec.msgpack

from sglang.srt.entrypoints.v1_loads import get_loads
from sglang.srt.managers.load_snapshot import (
    HEADER_STRUCT,
    MAGIC,
    SLOT_LEN_STRUCT,
    SLOT_SIZE,
    VERSION,
    LoadSnapshot,
    ShmLoadSnapshotReader,
    ShmLoadSnapshotWriter,
    slot_offset,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()


register_cpu_ci(est_time=10, suite="base-a-test-cpu")


# 内部方法_temp_path
def _temp_path() -> str:
    fd, path = tempfile.mkstemp()
    os.close(fd)
    os.unlink(path)
    return path


# _FakeTokenizerManager类
class _FakeTokenizerManager(TokenizerControlMixin):

    # _FakeTokenizerManager类的初始化
    def __init__(self, reader, dp_size: int):
        self.load_snapshot_reader = reader
        self.server_args = SimpleNamespace(
            dp_size=dp_size,
            enable_dp_attention=False,
            nnodes=1,
        )

    # _FakeTokenizerManager类的auto_create_handle_loop
    def auto_create_handle_loop(self):
        pass


# _FakeHttpTokenizerManager类
class _FakeHttpTokenizerManager:
    metrics_collector = None

    # _FakeHttpTokenizerManager类的初始化
    def __init__(self, loads):
        self.loads = loads

    async def get_loads(self, include=None, dp_rank=None):
        results = []
        for load in self.loads:
            if dp_rank is not None and load.dp_rank != dp_rank:
                continue
            results.append(load)
        return results


# TestLoadsResponse类
class TestLoadsResponse(CustomTestCase):

    # TestLoadsResponse类的测试responseomitsserversideaggregateandredundantfields
    def test_response_omits_server_side_aggregate_and_redundant_fields(self):
        manager = _FakeHttpTokenizerManager(
            [
                LoadSnapshot(
                    dp_rank=0,
                    num_running_reqs=3,
                    num_waiting_reqs=2,
                    num_total_tokens=256,
                )
            ]
        )

        response = asyncio.run(get_loads(tokenizer_manager=manager))

        self.assertNotIn("dp_rank_count", response)  # 断言不包含
        self.assertNotIn("aggregate", response)  # 断言不包含
        self.assertEqual(len(response["loads"]), 1)  # 断言相等
        self.assertNotIn("num_total_reqs", response["loads"][0])  # 断言不包含
        self.assertEqual(response["loads"][0]["num_running_reqs"], 3)  # 断言相等
        self.assertEqual(response["loads"][0]["num_waiting_reqs"], 2)  # 断言相等


# TestGetLoads类
class TestGetLoads(CustomTestCase):

    # TestGetLoads类的测试loadsnapshotwireformatismsgpackslots
    def test_load_snapshot_wire_format_is_msgpack_slots(self):
        path = _temp_path()
        writer = ShmLoadSnapshotWriter(path, dp_size=2, dp_rank=1)
        try:
            writer.write(
                LoadSnapshot(
                    dp_rank=1,
                    num_running_reqs=3,
                    num_waiting_reqs=2,
                    token_usage=0.25,
                )
            )

            with open(path, "rb") as f:
                data = f.read()

            self.assertEqual(len(data), HEADER_STRUCT.size + 2 * SLOT_SIZE)  # 断言相等
            magic, version, dp_size, slot_size = HEADER_STRUCT.unpack_from(data, 0)
            self.assertEqual(magic, MAGIC)  # 断言相等
            self.assertEqual(version, VERSION)  # 断言相等
            self.assertEqual(dp_size, 2)  # 断言相等
            self.assertEqual(slot_size, SLOT_SIZE)  # 断言相等

            offset = slot_offset(1, slot_size)
            (payload_len,) = SLOT_LEN_STRUCT.unpack_from(data, offset)
            payload_start = offset + SLOT_LEN_STRUCT.size
            payload = data[payload_start : payload_start + payload_len]
            decoded = msgspec.msgpack.decode(payload)

            self.assertEqual(decoded["dp_rank"], 1)  # 断言相等
            self.assertEqual(decoded["num_running_reqs"], 3)  # 断言相等
            self.assertEqual(decoded["num_waiting_reqs"], 2)  # 断言相等
            self.assertEqual(decoded["token_usage"], 0.25)  # 断言相等
        finally:
            writer.close()
            if os.path.exists(path):
                os.unlink(path)

    # TestGetLoads类的测试readssnapshotandfilterssections
    def test_reads_snapshot_and_filters_sections(self):
        path = _temp_path()
        writer = ShmLoadSnapshotWriter(path, dp_size=1, dp_rank=0)
        reader = ShmLoadSnapshotReader(path, dp_size=1)
        try:
            initial_load = reader.read(0)
            self.assertIsNotNone(initial_load)  # 断言不为None
            self.assertEqual(initial_load.num_total_tokens, 0)  # 断言相等

            writer.write(
                LoadSnapshot(
                    dp_rank=0,
                    timestamp=1.25,
                    num_running_reqs=3,
                    num_waiting_reqs=2,
                    num_used_tokens=128,
                    num_total_tokens=256,
                    max_total_num_tokens=4096,
                    token_usage=0.125,
                    gen_throughput=99.5,
                    cache_hit_rate=0.75,
                    utilization=0.5,
                    max_running_requests=128,
                    has_disaggregation=1,
                    disagg_mode=2,
                    decode_transfer_queue_reqs=4,
                    has_queues=1,
                    queue_waiting=2,
                    queue_grammar=1,
                    queue_paused=0,
                    queue_retracted=3,
                )
            )

            manager = _FakeTokenizerManager(reader, dp_size=1)
            loads = asyncio.run(manager.get_loads(include=["core"], dp_rank=0))

            self.assertEqual(len(loads), 1)  # 断言相等
            self.assertEqual(loads[0].num_total_tokens, 256)  # 断言相等

            d = loads[0].to_dict({"core"})
            self.assertNotIn("disaggregation", d)  # 断言不包含
            self.assertNotIn("queues", d)  # 断言不包含

            loads_all = asyncio.run(manager.get_loads(include=["all"], dp_rank=0))
            d_all = loads_all[0].to_dict()
            self.assertIn("disaggregation", d_all)  # 断言包含
            self.assertIn("queues", d_all)  # 断言包含
        finally:
            reader.close()
            writer.close()
            if os.path.exists(path):
                os.unlink(path)


if __name__ == "__main__":
    unittest.main()
