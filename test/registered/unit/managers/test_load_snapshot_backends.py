# 文件名: test_load_snapshot_backends.py - 快照加载后端
"""Unit tests for LoadSnapshot SHM and ZMQ backends."""

import os
import tempfile
import time
import unittest
from types import SimpleNamespace

from sglang.srt.managers.load_snapshot import (
    LoadSnapshot,
    ShmLoadSnapshotReader,
    ShmLoadSnapshotWriter,
    ZmqLoadSnapshotWriter,
    ZmqShmLoadSnapshotReader,
    _zmq_addr_for,
    create_load_snapshot_reader,
    create_load_snapshot_writer,
    should_use_zmq,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()


register_cpu_ci(est_time=15, suite="base-a-test-cpu")


# 内部方法_temp_path
def _temp_path() -> str:
    fd, path = tempfile.mkstemp()
    os.close(fd)
    os.unlink(path)
    return path


# 内部方法_ipc_addr
def _ipc_addr() -> str:
    fd, path = tempfile.mkstemp(prefix="sglang_test_zmq_", suffix=".sock")
    os.close(fd)
    os.unlink(path)
    return f"ipc://{path}"


# 内部方法_warmup_zmq
def _warmup_zmq(writers, reader, attempts=20, interval=0.05):
    """Send warmup messages until the reader receives from all writers."""
    expected = {w.dp_rank for w in writers}
    received = set()
    for _ in range(attempts):
        for w in writers:
            w.write(LoadSnapshot(dp_rank=w.dp_rank, timestamp=-1.0, num_running_reqs=0))
        time.sleep(interval)
        for rank in expected:
            load = reader.read(rank)
            if load is not None:
                received.add(rank)
        if received >= expected:
            return
    raise RuntimeError(f"warmup failed: expected {expected}, received {received}")  # 抛出异常


# TestShmRoundTrip类
class TestShmRoundTrip(CustomTestCase):

    # TestShmRoundTrip类的测试singlerankwriteread
    def test_single_rank_write_read(self):
        path = _temp_path()
        writer = ShmLoadSnapshotWriter(path, dp_size=1, dp_rank=0)
        reader = ShmLoadSnapshotReader(path, dp_size=1)
        try:
            writer.write(LoadSnapshot(dp_rank=0, num_running_reqs=5, timestamp=1.0))
            load = reader.read(0)
            self.assertIsNotNone(load)  # 断言不为None
            self.assertEqual(load.num_running_reqs, 5)  # 断言相等
            self.assertEqual(load.timestamp, 1.0)  # 断言相等
        finally:
            reader.close()
            writer.close()
            if os.path.exists(path):
                os.unlink(path)

    # TestShmRoundTrip类的测试multirankwritereadall
    def test_multi_rank_write_read_all(self):
        path = _temp_path()
        writers = []
        try:
            for rank in range(4):
                w = ShmLoadSnapshotWriter(path, dp_size=4, dp_rank=rank)
                w.write(
                    LoadSnapshot(
                        dp_rank=rank,
                        num_running_reqs=rank * 10,
                        timestamp=1.0,
                    )
                )
                writers.append(w)

            reader = ShmLoadSnapshotReader(path, dp_size=4)
            loads = reader.read_all()
            self.assertEqual(len(loads), 4)  # 断言相等
            for i, load in enumerate(loads):
                self.assertEqual(load.dp_rank, i)  # 断言相等
                self.assertEqual(load.num_running_reqs, i * 10)  # 断言相等
            reader.close()
        finally:
            for w in writers:
                w.close()
            if os.path.exists(path):
                os.unlink(path)

    # TestShmRoundTrip类的测试readeremptybeforewriter
    def test_reader_empty_before_writer(self):
        path = _temp_path()
        reader = ShmLoadSnapshotReader(path, dp_size=2)
        self.assertEqual(reader.read_all(), [])  # 断言相等
        self.assertIsNone(reader.read(0))  # 断言为None
        reader.close()


# TestZmqRoundTrip类
class TestZmqRoundTrip(CustomTestCase):

    # TestZmqRoundTrip类的测试singlerankzmqtoshm
    def test_single_rank_zmq_to_shm(self):
        shm_path = _temp_path()
        addr = _ipc_addr()
        reader = ZmqShmLoadSnapshotReader(addr, shm_path, dp_size=2)
        writer = ZmqLoadSnapshotWriter(addr, dp_size=2, dp_rank=0)
        try:
            _warmup_zmq([writer], reader)

            writer.write(LoadSnapshot(dp_rank=0, num_running_reqs=7, timestamp=2.0))
            time.sleep(0.05)

            load = reader.read(0)
            self.assertIsNotNone(load)  # 断言不为None
            self.assertEqual(load.num_running_reqs, 7)  # 断言相等
            self.assertEqual(load.timestamp, 2.0)  # 断言相等
        finally:
            writer.close()
            reader.close()
            if os.path.exists(shm_path):
                os.unlink(shm_path)

    # TestZmqRoundTrip类的测试multirankzmq
    def test_multi_rank_zmq(self):
        shm_path = _temp_path()
        addr = _ipc_addr()
        dp_size = 4
        reader = ZmqShmLoadSnapshotReader(addr, shm_path, dp_size)
        writers = []
        try:
            for rank in range(dp_size):
                w = ZmqLoadSnapshotWriter(addr, dp_size, dp_rank=rank)
                writers.append(w)

            _warmup_zmq(writers, reader)

            for rank, w in enumerate(writers):
                w.write(
                    LoadSnapshot(dp_rank=rank, num_running_reqs=rank + 1, timestamp=3.0)
                )
            time.sleep(0.05)

            loads = reader.read_all()
            self.assertEqual(len(loads), dp_size)  # 断言相等
            for load in loads:
                self.assertEqual(load.num_running_reqs, load.dp_rank + 1)  # 断言相等
        finally:
            for w in writers:
                w.close()
            reader.close()
            if os.path.exists(shm_path):
                os.unlink(shm_path)

    # TestZmqRoundTrip类的测试readreturnslatest
    def test_read_returns_latest(self):
        shm_path = _temp_path()
        addr = _ipc_addr()
        reader = ZmqShmLoadSnapshotReader(addr, shm_path, dp_size=1)
        writer = ZmqLoadSnapshotWriter(addr, dp_size=1, dp_rank=0)
        try:
            _warmup_zmq([writer], reader)

            for i in range(10):
                writer.write(
                    LoadSnapshot(dp_rank=0, num_running_reqs=i, timestamp=float(i))
                )
            time.sleep(0.05)

            load = reader.read(0)
            self.assertIsNotNone(load)  # 断言不为None
            self.assertEqual(load.num_running_reqs, 9)  # 断言相等
            self.assertEqual(load.timestamp, 9.0)  # 断言相等
        finally:
            writer.close()
            reader.close()
            if os.path.exists(shm_path):
                os.unlink(shm_path)

    # TestZmqRoundTrip类的测试zmqwriternoblockwithoutreader
    def test_zmq_writer_noblock_without_reader(self):
        addr = _ipc_addr()
        writer = ZmqLoadSnapshotWriter(addr, dp_size=1, dp_rank=0)
        try:
            writer.write(LoadSnapshot(dp_rank=0, num_running_reqs=1, timestamp=1.0))
        finally:
            writer.close()
            ipc_path = addr[len("ipc://") :]
            if os.path.exists(ipc_path):
                os.unlink(ipc_path)

    # TestZmqRoundTrip类的测试readeripccleanup
    def test_reader_ipc_cleanup(self):
        addr = _ipc_addr()
        shm_path = _temp_path()
        ipc_path = addr[len("ipc://") :]
        reader = ZmqShmLoadSnapshotReader(addr, shm_path, dp_size=1)
        self.assertTrue(os.path.exists(ipc_path))  # 断言为真
        reader.close()
        self.assertFalse(os.path.exists(ipc_path))  # 断言为假
        if os.path.exists(shm_path):
            os.unlink(shm_path)


# TestFactoryFunctions类
class TestFactoryFunctions(CustomTestCase):

    # TestFactoryFunctions类的测试shmmode
    def test_shm_mode(self):
        server_args = SimpleNamespace(
            enable_dp_attention=False,
            nnodes=1,
            dp_size=1,
            load_balance_method="round_robin",
            node_rank=0,
        )
        port_args = SimpleNamespace(instance_id="test_shm_factory")
        writer = create_load_snapshot_writer(
            server_args, port_args, dp_size=1, dp_rank=0
        )
        self.assertIsInstance(writer, ShmLoadSnapshotWriter)
        reader = create_load_snapshot_reader(server_args, port_args, caller="tokenizer")
        self.assertIsInstance(reader, ShmLoadSnapshotReader)
        reader.close()
        writer.close()
        from sglang.srt.managers.load_snapshot import shm_path_for

        path = shm_path_for("test_shm_factory")
        if os.path.exists(path):
            os.unlink(path)

    # TestFactoryFunctions类的测试zmqmodeviaenv
    def test_zmq_mode_via_env(self):
        server_args = SimpleNamespace(
            enable_dp_attention=False,
            nnodes=1,
            dp_size=1,
            load_balance_method="round_robin",
            node_rank=0,
        )
        port_args = SimpleNamespace(instance_id="test_zmq_factory")
        os.environ["SGLANG_LOAD_SNAPSHOT_USE_ZMQ"] = "1"
        try:
            writer = create_load_snapshot_writer(
                server_args, port_args, dp_size=1, dp_rank=0
            )
            self.assertIsInstance(writer, ZmqLoadSnapshotWriter)
            reader = create_load_snapshot_reader(
                server_args, port_args, caller="tokenizer"
            )
            self.assertIsInstance(reader, ZmqShmLoadSnapshotReader)
            reader.close()
            writer.close()
        finally:
            del os.environ["SGLANG_LOAD_SNAPSHOT_USE_ZMQ"]

    # TestFactoryFunctions类的测试shouldusezmqmultinodedpattention
    def test_should_use_zmq_multinode_dp_attention(self):
        args = SimpleNamespace(enable_dp_attention=True, nnodes=2)
        self.assertTrue(should_use_zmq(args))  # 断言为真

    # TestFactoryFunctions类的测试shouldusezmqsinglenode
    def test_should_use_zmq_single_node(self):
        args = SimpleNamespace(enable_dp_attention=False, nnodes=1)
        self.assertFalse(should_use_zmq(args))  # 断言为假

    # TestFactoryFunctions类的测试shouldusezmqdpattentionsinglenode
    def test_should_use_zmq_dp_attention_single_node(self):
        args = SimpleNamespace(enable_dp_attention=True, nnodes=1)
        self.assertFalse(should_use_zmq(args))  # 断言为假


# TestZmqAddr类
class TestZmqAddr(CustomTestCase):

    # TestZmqAddr类的测试ipcforsinglenode
    def test_ipc_for_single_node(self):
        port_args = SimpleNamespace(instance_id="myinstance")
        addr = _zmq_addr_for(port_args)
        self.assertTrue(addr.startswith("ipc://"))  # 断言为真
        self.assertIn("myinstance", addr)  # 断言包含

    # TestZmqAddr类的测试tcpfromportargs
    def test_tcp_from_port_args(self):
        from sglang.srt.utils.network import NetworkAddress

        port_args = SimpleNamespace(
            instance_id="myinstance",
            load_collector_ipc_name=NetworkAddress("10.0.0.1", 29506).to_tcp(),
        )
        addr = _zmq_addr_for(port_args)
        self.assertTrue(addr.startswith("tcp://"))  # 断言为真
        self.assertIn("10.0.0.1", addr)  # 断言包含


# TestEndToEndZmqSimulation类
class TestEndToEndZmqSimulation(CustomTestCase):
    """Simulate multi-node DP attention on single machine using IPC."""

    def test_full_flow_dp_size_2(self):
        shm_path = _temp_path()
        addr = _ipc_addr()
        dp_size = 2

        reader = ZmqShmLoadSnapshotReader(addr, shm_path, dp_size)

        writers = []
        for rank in range(dp_size):
            w = ZmqLoadSnapshotWriter(addr, dp_size, dp_rank=rank)
            writers.append(w)

        try:
            _warmup_zmq(writers, reader)

            for rank, w in enumerate(writers):
                w.write(
                    LoadSnapshot(
                        dp_rank=rank,
                        timestamp=1.0,
                        num_running_reqs=10 + rank,
                        num_waiting_reqs=5 + rank,
                        num_total_tokens=100 + rank * 50,
                    )
                )
            time.sleep(0.05)

            loads = reader.read_all()
            self.assertEqual(len(loads), dp_size)  # 断言相等
            self.assertEqual(loads[0].num_running_reqs, 10)  # 断言相等
            self.assertEqual(loads[1].num_running_reqs, 11)  # 断言相等
            self.assertEqual(loads[0].num_total_tokens, 100)  # 断言相等
            self.assertEqual(loads[1].num_total_tokens, 150)  # 断言相等

            for rank, w in enumerate(writers):
                w.write(
                    LoadSnapshot(
                        dp_rank=rank,
                        timestamp=2.0,
                        num_running_reqs=20 + rank,
                        num_waiting_reqs=0,
                        num_total_tokens=200 + rank * 50,
                    )
                )
            time.sleep(0.05)

            loads = reader.read_all()
            self.assertEqual(loads[0].num_running_reqs, 20)  # 断言相等
            self.assertEqual(loads[1].num_running_reqs, 21)  # 断言相等
            self.assertEqual(loads[0].num_total_tokens, 200)  # 断言相等
            self.assertEqual(loads[1].num_total_tokens, 250)  # 断言相等
        finally:
            for w in writers:
                w.close()
            reader.close()
            if os.path.exists(shm_path):
                os.unlink(shm_path)


if __name__ == "__main__":
    unittest.main()
