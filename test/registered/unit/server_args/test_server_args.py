# 文件名: test_server_args.py - 服务器参数
import importlib
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import sglang.srt.server_args as server_args_module
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.server_args import PortArgs, ServerArgs, prepare_server_args
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    CustomTestCase,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cpu_ci(est_time=12, suite="base-b-test-cpu")

# Mock get_device() so all tests run on CPU-only CI runners
_mock_device = patch("sglang.srt.server_args.get_device", return_value="cuda")
_mock_device.start()


# TestPrepareServerArgs类
class TestPrepareServerArgs(CustomTestCase):

    # TestPrepareServerArgs类的测试prepareserverargs
    def test_prepare_server_args(self):
        server_args = prepare_server_args(
            [
                "--model-path",
                DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
                "--json-model-override-args",
                '{"rope_scaling": {"factor": 2.0, "rope_type": "linear"}}',
            ]
        )
        self.assertEqual(server_args.model_path, DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN)  # 断言相等
        self.assertEqual(  # 断言相等
            json.loads(server_args.json_model_override_args),
            {"rope_scaling": {"factor": 2.0, "rope_type": "linear"}},
        )


# TestLoadBalanceMethod类
class TestLoadBalanceMethod(unittest.TestCase):

    # TestLoadBalanceMethod类的测试nonpddefaultstoroundrobin
    def test_non_pd_defaults_to_round_robin(self):
        server_args = ServerArgs(model_path="dummy", disaggregation_mode="null")
        self.assertEqual(server_args.load_balance_method, "round_robin")  # 断言相等

    # TestLoadBalanceMethod类的测试pdprefilldefaultstofollowbootstraproom
    def test_pd_prefill_defaults_to_follow_bootstrap_room(self):
        server_args = ServerArgs(model_path="dummy", disaggregation_mode="prefill")
        self.assertEqual(server_args.load_balance_method, "follow_bootstrap_room")  # 断言相等

    # TestLoadBalanceMethod类的测试pddecodedefaultstoroundrobin
    def test_pd_decode_defaults_to_round_robin(self):
        server_args = ServerArgs(model_path="dummy", disaggregation_mode="decode")
        self.assertEqual(server_args.load_balance_method, "round_robin")  # 断言相等

    # TestLoadBalanceMethod类的测试pddecoderadixcacherejectshisparse
    def test_pd_decode_radix_cache_rejects_hisparse(self):
        with self.assertRaises(ValueError) as context:  # 断言抛出异常
            ServerArgs(
                model_path="dummy",
                disaggregation_mode="decode",
                disaggregation_decode_enable_radix_cache=True,
                disaggregation_transfer_backend="nixl",
                enable_hisparse=True,
            )

        self.assertIn(  # 断言包含
            "--disaggregation-decode-enable-radix-cache is incompatible with "
            "--enable-hisparse",
            str(context.exception),
        )

    # TestLoadBalanceMethod类的测试pddecoderadixcacheallowsmooncake
    def test_pd_decode_radix_cache_allows_mooncake(self):
        server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            disaggregation_decode_enable_radix_cache=True,
            disaggregation_transfer_backend="mooncake",
        )

        self.assertFalse(server_args.disable_radix_cache)  # 断言为假

    # TestLoadBalanceMethod类的测试pddecoderadixcacherejectsunknownbackend
    def test_pd_decode_radix_cache_rejects_unknown_backend(self):
        with self.assertRaises(ValueError) as context:  # 断言抛出异常
            ServerArgs(
                model_path="dummy",
                disaggregation_mode="decode",
                disaggregation_decode_enable_radix_cache=True,
                disaggregation_transfer_backend="fake",
            )

        self.assertIn("('nixl', 'mooncake')", str(context.exception))  # 断言包含
        self.assertIn("'fake'", str(context.exception))  # 断言包含


# TestPortArgs类
class TestPortArgs(unittest.TestCase):
    @patch("sglang.srt.server_args.get_free_port")
    @patch("sglang.srt.server_args.tempfile.NamedTemporaryFile")

    # TestPortArgs类的测试initnewwithncclportnone
    def test_init_new_with_nccl_port_none(self, mock_temp_file, mock_get_free_port):
        """Test that get_free_port() is called when nccl_port is None"""
        mock_temp_file.return_value.name = "temp_file"
        mock_get_free_port.return_value = 45678  # Mock ephemeral port

        # Use MagicMock here to verify get_free_port is called
        server_args = MagicMock()
        server_args.nccl_port = None
        server_args.enable_dp_attention = False
        server_args.tokenizer_worker_num = 1

        port_args = PortArgs.init_new(server_args)

        # Verify get_free_port was called
        mock_get_free_port.assert_called_once()

        # Verify the returned port is used
        self.assertEqual(port_args.nccl_port, 45678)  # 断言相等

    @patch("sglang.srt.server_args.tempfile.NamedTemporaryFile")

    # TestPortArgs类的测试initnewstandardcase
    def test_init_new_standard_case(self, mock_temp_file):
        mock_temp_file.return_value.name = "temp_file"

        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = False

        port_args = PortArgs.init_new(server_args)

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("ipc://"))  # 断言为真
        self.assertTrue(port_args.scheduler_input_ipc_name.startswith("ipc://"))  # 断言为真
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("ipc://"))  # 断言为真
        self.assertIsInstance(port_args.nccl_port, int)

    # TestPortArgs类的测试initnewwithsinglenodedpattention
    def test_init_new_with_single_node_dp_attention(self):

        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 1
        server_args.dist_init_addr = None

        port_args = PortArgs.init_new(server_args)

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://127.0.0.1:"))  # 断言为真
        self.assertTrue(  # 断言为真
            port_args.scheduler_input_ipc_name.startswith("tcp://127.0.0.1:")
        )
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("tcp://127.0.0.1:"))  # 断言为真
        self.assertIsInstance(port_args.nccl_port, int)

    # TestPortArgs类的测试initnewwithdprank
    def test_init_new_with_dp_rank(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 1
        server_args.dist_init_addr = "192.168.1.1:25000"

        worker_ports = [25006, 25007, 25008, 25009]
        port_args = PortArgs.init_new(server_args, dp_rank=2, worker_ports=worker_ports)

        self.assertTrue(port_args.scheduler_input_ipc_name.endswith(":25008"))  # 断言为真

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://192.168.1.1:"))  # 断言为真
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("tcp://192.168.1.1:"))  # 断言为真
        self.assertIsInstance(port_args.nccl_port, int)

    # TestPortArgs类的测试initnewwithipv4address
    def test_init_new_with_ipv4_address(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "192.168.1.1:25000"

        port_args = PortArgs.init_new(server_args)

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://192.168.1.1:"))  # 断言为真
        self.assertTrue(  # 断言为真
            port_args.scheduler_input_ipc_name.startswith("tcp://192.168.1.1:")
        )
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("tcp://192.168.1.1:"))  # 断言为真
        self.assertIsInstance(port_args.nccl_port, int)

    # TestPortArgs类的测试initnewwithmalformedipv4address
    def test_init_new_with_malformed_ipv4_address(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "192.168.1.1"

        with self.assertRaises(ValueError) as context:  # 断言抛出异常
            PortArgs.init_new(server_args)

        self.assertIn("Missing port", str(context.exception))  # 断言包含

    # TestPortArgs类的测试initnewwithmalformedipv4addressinvalidport
    def test_init_new_with_malformed_ipv4_address_invalid_port(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "192.168.1.1:abc"

        with self.assertRaises(ValueError):  # 断言抛出异常
            PortArgs.init_new(server_args)


# TestSSLArgs类
class TestSSLArgs(unittest.TestCase):

    # TestSSLArgs类的测试defaultsslfieldsarenone
    def test_default_ssl_fields_are_none(self):
        server_args = ServerArgs(model_path="dummy")
        self.assertIsNone(server_args.ssl_keyfile)  # 断言为None
        self.assertIsNone(server_args.ssl_certfile)  # 断言为None
        self.assertIsNone(server_args.ssl_ca_certs)  # 断言为None
        self.assertIsNone(server_args.ssl_keyfile_password)  # 断言为None

    # TestSSLArgs类的测试sslkeyfilewithoutcertfileraises
    def test_ssl_keyfile_without_certfile_raises(self):
        with self.assertRaises(ValueError) as context:  # 断言抛出异常
            ServerArgs(model_path="dummy", ssl_keyfile="key.pem")
        self.assertIn("--ssl-certfile", str(context.exception))  # 断言包含

    # TestSSLArgs类的测试sslcertfilewithoutkeyfileraises
    def test_ssl_certfile_without_keyfile_raises(self):
        with self.assertRaises(ValueError) as context:  # 断言抛出异常
            ServerArgs(model_path="dummy", ssl_certfile="cert.pem")
        self.assertIn("--ssl-keyfile", str(context.exception))  # 断言包含

    @patch("os.path.isfile", return_value=True)

    # TestSSLArgs类的测试sslbothkeyfileandcertfileaccepted
    def test_ssl_both_keyfile_and_certfile_accepted(self, _mock_isfile):
        server_args = ServerArgs(
            model_path="dummy", ssl_keyfile="key.pem", ssl_certfile="cert.pem"
        )
        self.assertEqual(server_args.ssl_keyfile, "key.pem")  # 断言相等
        self.assertEqual(server_args.ssl_certfile, "cert.pem")  # 断言相等

    # TestSSLArgs类的测试urlreturnshttpwithoutssl
    def test_url_returns_http_without_ssl(self):
        server_args = ServerArgs(model_path="dummy")
        self.assertTrue(server_args.url().startswith("http://"))  # 断言为真

    # TestSSLArgs类的测试urlrewritesallinterfacestoloopback
    def test_url_rewrites_all_interfaces_to_loopback(self):
        server_args = ServerArgs(model_path="dummy", host="0.0.0.0")
        self.assertEqual(server_args.url(), "http://127.0.0.1:30000")  # 断言相等

    # TestSSLArgs类的测试urlrewritesemptyhosttoloopback
    def test_url_rewrites_empty_host_to_loopback(self):
        server_args = ServerArgs(model_path="dummy", host="")
        self.assertEqual(server_args.url(), "http://127.0.0.1:30000")  # 断言相等

    @patch("os.path.isfile", return_value=True)

    # TestSSLArgs类的测试urlreturnshttpswithssl
    def test_url_returns_https_with_ssl(self, _mock_isfile):
        server_args = ServerArgs(
            model_path="dummy", ssl_keyfile="key.pem", ssl_certfile="cert.pem"
        )
        self.assertTrue(server_args.url().startswith("https://"))  # 断言为真

    @patch("os.path.isfile", return_value=True)

    # TestSSLArgs类的测试sslcliargsparsed
    def test_ssl_cli_args_parsed(self, _mock_isfile):
        server_args = prepare_server_args(
            [
                "--model-path",
                "dummy",
                "--ssl-keyfile",
                "key.pem",
                "--ssl-certfile",
                "cert.pem",
                "--ssl-ca-certs",
                "ca.pem",
                "--ssl-keyfile-password",
                "secret",
            ]
        )
        self.assertEqual(server_args.ssl_keyfile, "key.pem")  # 断言相等
        self.assertEqual(server_args.ssl_certfile, "cert.pem")  # 断言相等
        self.assertEqual(server_args.ssl_ca_certs, "ca.pem")  # 断言相等
        self.assertEqual(server_args.ssl_keyfile_password, "secret")  # 断言相等

    # TestSSLArgs类的测试sslverifywithoutssl
    def test_ssl_verify_without_ssl(self):
        server_args = ServerArgs(model_path="dummy")
        self.assertIs(server_args.ssl_verify(), True)  # 断言是同一对象

    @patch("os.path.isfile", return_value=True)

    # TestSSLArgs类的测试sslverifywithsslnoca
    def test_ssl_verify_with_ssl_no_ca(self, _mock_isfile):
        server_args = ServerArgs(
            model_path="dummy", ssl_keyfile="key.pem", ssl_certfile="cert.pem"
        )
        self.assertIs(server_args.ssl_verify(), False)  # 断言是同一对象

    @patch("os.path.isfile", return_value=True)

    # TestSSLArgs类的测试sslverifywithsslandca
    def test_ssl_verify_with_ssl_and_ca(self, _mock_isfile):
        server_args = ServerArgs(
            model_path="dummy",
            ssl_keyfile="key.pem",
            ssl_certfile="cert.pem",
            ssl_ca_certs="ca.pem",
        )
        self.assertEqual(server_args.ssl_verify(), "ca.pem")  # 断言相等

    # TestSSLArgs类的测试sslcacertswithoutcertfileraises
    def test_ssl_ca_certs_without_certfile_raises(self):
        with self.assertRaises(ValueError) as context:  # 断言抛出异常
            ServerArgs(model_path="dummy", ssl_ca_certs="ca.pem")
        self.assertIn("--ssl-ca-certs", str(context.exception))  # 断言包含

    # TestSSLArgs类的测试sslkeyfilepasswordwithoutcertfileraises
    def test_ssl_keyfile_password_without_certfile_raises(self):
        with self.assertRaises(ValueError) as context:  # 断言抛出异常
            ServerArgs(model_path="dummy", ssl_keyfile_password="secret")
        self.assertIn("--ssl-keyfile-password", str(context.exception))  # 断言包含

    # TestSSLArgs类的测试sslkeyfilenotfoundraises
    def test_ssl_keyfile_not_found_raises(self):
        with self.assertRaises(ValueError) as context:  # 断言抛出异常
            ServerArgs(
                model_path="dummy",
                ssl_keyfile="/nonexistent/key.pem",
                ssl_certfile="/nonexistent/cert.pem",
            )
        self.assertIn("not found", str(context.exception))  # 断言包含

    # TestSSLArgs类的测试sslcertfilenotfoundraises
    def test_ssl_certfile_not_found_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".pem") as keyfile:
            with self.assertRaises(ValueError) as context:  # 断言抛出异常
                ServerArgs(
                    model_path="dummy",
                    ssl_keyfile=keyfile.name,
                    ssl_certfile="/nonexistent/cert.pem",
                )
            self.assertIn("SSL certificate file not found", str(context.exception))  # 断言包含

    # TestSSLArgs类的测试sslcacertsnotfoundraises
    def test_ssl_ca_certs_not_found_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".pem") as keyfile:
            with tempfile.NamedTemporaryFile(suffix=".pem") as certfile:
                with self.assertRaises(ValueError) as context:  # 断言抛出异常
                    ServerArgs(
                        model_path="dummy",
                        ssl_keyfile=keyfile.name,
                        ssl_certfile=certfile.name,
                        ssl_ca_certs="/nonexistent/ca.pem",
                    )
                self.assertIn(  # 断言包含
                    "SSL CA certificates file not found", str(context.exception)
                )

    # TestSSLArgs类的测试enablesslrefreshdefaultfalse
    def test_enable_ssl_refresh_default_false(self):
        server_args = ServerArgs(model_path="dummy")
        self.assertFalse(server_args.enable_ssl_refresh)  # 断言为假

    # TestSSLArgs类的测试enablesslrefreshwithoutsslraises
    def test_enable_ssl_refresh_without_ssl_raises(self):
        with self.assertRaises(ValueError) as context:  # 断言抛出异常
            ServerArgs(model_path="dummy", enable_ssl_refresh=True)
        self.assertIn("--enable-ssl-refresh", str(context.exception))  # 断言包含
        self.assertIn("--ssl-certfile", str(context.exception))  # 断言包含

    @patch("os.path.isfile", return_value=True)

    # TestSSLArgs类的测试enablesslrefreshwithsslaccepted
    def test_enable_ssl_refresh_with_ssl_accepted(self, _mock_isfile):
        server_args = ServerArgs(
            model_path="dummy",
            ssl_keyfile="key.pem",
            ssl_certfile="cert.pem",
            enable_ssl_refresh=True,
        )
        self.assertTrue(server_args.enable_ssl_refresh)  # 断言为真

    @patch("os.path.isfile", return_value=True)

    # TestSSLArgs类的测试enablesslrefreshcliflag
    def test_enable_ssl_refresh_cli_flag(self, _mock_isfile):
        server_args = prepare_server_args(
            [
                "--model-path",
                "dummy",
                "--ssl-keyfile",
                "key.pem",
                "--ssl-certfile",
                "cert.pem",
                "--enable-ssl-refresh",
            ]
        )
        self.assertTrue(server_args.enable_ssl_refresh)  # 断言为真


# TestHiCacheArgs类
class TestHiCacheArgs(unittest.TestCase):

    # TestHiCacheArgs类的内部方法_make_args
    def _make_args(self, **overrides) -> ServerArgs:
        args = ServerArgs(model_path="dummy")
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    # TestHiCacheArgs类的内部方法_assert_hicache_fields
    def _assert_hicache_fields(
        self,
        args: ServerArgs,
        *,
        expected_io_backend: str,
        expected_mem_layout: str,
        expected_decode_backend: str | None = None,
    ):
        self.assertEqual(args.hicache_io_backend, expected_io_backend)  # 断言相等
        self.assertEqual(args.hicache_mem_layout, expected_mem_layout)  # 断言相等
        if expected_decode_backend is not None:
            self.assertEqual(args.decode_attention_backend, expected_decode_backend)  # 断言相等

    # TestHiCacheArgs类的测试hicacheiobackendandmemlayoutcompatibility
    def test_hicache_io_backend_and_mem_layout_compatibility(self):
        cases = [
            {
                "name": "kernel_with_page_first_direct",
                "overrides": {
                    "enable_hierarchical_cache": True,
                    "hicache_io_backend": "kernel",
                    "hicache_mem_layout": "page_first_direct",
                },
                "expected_io_backend": "direct",
                "expected_mem_layout": "page_first_direct",
            },
            {
                "name": "direct_with_page_first",
                "overrides": {
                    "enable_hierarchical_cache": True,
                    "hicache_io_backend": "direct",
                    "hicache_mem_layout": "page_first",
                },
                "expected_io_backend": "direct",
                "expected_mem_layout": "page_first_direct",
            },
            {
                "name": "mooncake_with_layer_first",
                "overrides": {
                    "enable_hierarchical_cache": True,
                    "hicache_storage_backend": "mooncake",
                    "hicache_io_backend": "direct",
                    "hicache_mem_layout": "layer_first",
                },
                "expected_io_backend": "direct",
                "expected_mem_layout": "page_first_direct",
            },
            {
                "name": "fa3_kernel_with_explicit_decode_backend",
                "overrides": {
                    "enable_hierarchical_cache": True,
                    "hicache_io_backend": "kernel",
                    "hicache_mem_layout": "page_first",
                    "attention_backend": "triton",
                    "decode_attention_backend": "fa3",
                },
                "expected_io_backend": "direct",
                "expected_mem_layout": "page_first_direct",
            },
        ]

        for case in cases:
            with self.subTest(case=case["name"]):
                args = self._make_args(**case["overrides"])
                args._handle_hicache()
                self._assert_hicache_fields(
                    args,
                    expected_io_backend=case["expected_io_backend"],
                    expected_mem_layout=case["expected_mem_layout"],
                )

    @patch.object(ServerArgs, "use_mla_backend", return_value=False)
    @patch("sglang.srt.server_args.is_flashinfer_available", return_value=False)

    # TestHiCacheArgs类的测试decodeattentionbackendwithimplicitfa3
    def test_decode_attention_backend_with_implicit_fa3(
        self, _mock_flashinfer, _mock_use_mla_backend
    ):
        args = self._make_args(
            enable_hierarchical_cache=True,
            hicache_io_backend="kernel",
            attention_backend="fa3",
            decode_attention_backend=None,
        )

        args._handle_hicache()

        self.assertEqual(args.decode_attention_backend, "triton")  # 断言相等


# TestNgramExternalSamArgs类
class TestNgramExternalSamArgs(CustomTestCase):

    # TestNgramExternalSamArgs类的测试prepareserverargsparsesexternalsamargs
    def test_prepare_server_args_parses_external_sam_args(self):
        server_args = prepare_server_args(
            [
                "--model-path",
                "dummy",
                "--speculative-algorithm",
                "NGRAM",
                "--speculative-ngram-external-corpus-path",
                "/tmp/ngram-corpus.jsonl",
                "--speculative-ngram-external-sam-budget",
                "4",
                "--speculative-ngram-external-corpus-max-tokens",
                "128",
            ]
        )
        self.assertEqual(  # 断言相等
            server_args.speculative_ngram_external_corpus_path,
            "/tmp/ngram-corpus.jsonl",
        )
        self.assertEqual(server_args.speculative_ngram_external_sam_budget, 4)  # 断言相等
        self.assertEqual(server_args.speculative_ngram_external_corpus_max_tokens, 128)  # 断言相等

    # TestNgramExternalSamArgs类的内部方法_make_dummy_ngram_args
    def _make_dummy_ngram_args(self, **overrides):
        args = ServerArgs(model_path="dummy")
        args.speculative_algorithm = "NGRAM"
        args.speculative_num_draft_tokens = 12
        args.device = "cuda"
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    # TestNgramExternalSamArgs类的测试externalsambudgetmustfitdraftbudget
    def test_external_sam_budget_must_fit_draft_budget(self):
        args = self._make_dummy_ngram_args(
            speculative_num_draft_tokens=4,
            speculative_ngram_external_corpus_path="/tmp/ngram-corpus.jsonl",
            speculative_ngram_external_sam_budget=4,
        )
        with self.assertRaises(ValueError) as context:  # 断言抛出异常
            handle_speculative_decoding(args)
        self.assertIn("speculative_num_draft_tokens - 1", str(context.exception))  # 断言包含

    # TestNgramExternalSamArgs类的测试externalcorpusmaxtokensmustbepositive
    def test_external_corpus_max_tokens_must_be_positive(self):
        args = self._make_dummy_ngram_args(
            speculative_ngram_external_corpus_path="/tmp/ngram-corpus.jsonl",
            speculative_ngram_external_sam_budget=2,
            speculative_ngram_external_corpus_max_tokens=0,
        )
        with self.assertRaises(ValueError) as context:  # 断言抛出异常
            handle_speculative_decoding(args)
        self.assertIn("external-corpus-max-tokens", str(context.exception))  # 断言包含


# TestDeepEPWaterfillArgs类
class TestDeepEPWaterfillArgs(CustomTestCase):

    # TestDeepEPWaterfillArgs类的测试waterfillenforcessharedexpertsfusion
    def test_waterfill_enforces_shared_experts_fusion(self):
        server_args = ServerArgs(
            model_path="dummy",
            moe_a2a_backend="deepep",
            enable_deepep_waterfill=True,
            disable_shared_experts_fusion=True,
        )
        # dummy-model path short-circuits __post_init__; invoke the handler directly.
        server_args._handle_a2a_moe()

        self.assertFalse(server_args.disable_shared_experts_fusion)  # 断言为假
        self.assertTrue(server_args.enforce_shared_experts_fusion)  # 断言为真

    # TestDeepEPWaterfillArgs类的测试waterfilloverridesmoea2abackendtodeepep
    def test_waterfill_overrides_moe_a2a_backend_to_deepep(self):
        server_args = ServerArgs(
            model_path="dummy",
            moe_a2a_backend="none",
            enable_deepep_waterfill=True,
        )
        # dummy-model path short-circuits __post_init__; invoke the handler directly.
        server_args._handle_a2a_moe()

        self.assertEqual(server_args.moe_a2a_backend, "deepep")  # 断言相等
        self.assertTrue(server_args.enforce_shared_experts_fusion)  # 断言为真

    # TestDeepEPWaterfillArgs类的测试waterfillsupportsdeepeplowlatencymode
    def test_waterfill_supports_deepep_low_latency_mode(self):
        server_args = ServerArgs(
            model_path="dummy",
            moe_a2a_backend="deepep",
            enable_deepep_waterfill=True,
            deepep_mode="low_latency",
        )
        # dummy-model path short-circuits __post_init__; invoke the handler directly.
        server_args._handle_a2a_moe()

        self.assertEqual(server_args.deepep_mode, "low_latency")  # 断言相等
        self.assertFalse(server_args.disable_cuda_graph)  # 断言为假
        self.assertTrue(server_args.enforce_shared_experts_fusion)  # 断言为真


# TestPrefillOnlyDisableKvCache类
class TestPrefillOnlyDisableKvCache(unittest.TestCase):
    """Validation for --prefill-only-disable-kv-cache.

    The flag wires NoOpMHATokenToKVPool, which is only safe when:
      - the engine is in embedding mode (fa_skip_kv_cache active in FA backend),
      - chunked_prefill_size == -1 (no inter-chunk K/V reuse),
      - disable_radix_cache (radix cache otherwise indexes empty pool slots),
      - no context-parallel attention (CP writes to the pool via set_kv_buffer),
      - no HiSparse (uses a different pool family),
      - kv_cache_dtype != fp4_e2m1 (FP4 pool is a separate allocation path).
    All other configurations must be rejected at __post_init__ time so users
    get a clear error before model load.
    """

    def _base_kwargs(self, **overrides):
        kwargs = dict(
            model_path="dummy",
            is_embedding=True,
            chunked_prefill_size=-1,
            disable_radix_cache=True,
            prefill_only_disable_kv_cache=True,
        )
        kwargs.update(overrides)
        return kwargs

    # TestPrefillOnlyDisableKvCache类的测试validminimalconfigconstructs
    def test_valid_minimal_config_constructs(self):
        sa = ServerArgs(**self._base_kwargs())
        self.assertTrue(sa.prefill_only_disable_kv_cache)  # 断言为真

    # TestPrefillOnlyDisableKvCache类的测试rejectswhennotembedding
    def test_rejects_when_not_embedding(self):
        with self.assertRaisesRegex(ValueError, "requires --is-embedding"):
            ServerArgs(**self._base_kwargs(is_embedding=False))

    # TestPrefillOnlyDisableKvCache类的测试rejectswhenchunkedprefillsizenotminusone
    def test_rejects_when_chunked_prefill_size_not_minus_one(self):
        with self.assertRaisesRegex(ValueError, "--chunked-prefill-size=-1"):
            ServerArgs(**self._base_kwargs(chunked_prefill_size=8192))

    # TestPrefillOnlyDisableKvCache类的测试rejectswhenradixcacheenabled
    def test_rejects_when_radix_cache_enabled(self):
        with self.assertRaisesRegex(ValueError, "--disable-radix-cache"):
            ServerArgs(**self._base_kwargs(disable_radix_cache=False))

    # TestPrefillOnlyDisableKvCache类的测试rejectsattncpsizegreaterthanone
    def test_rejects_attn_cp_size_greater_than_one(self):
        with self.assertRaisesRegex(ValueError, "--attn-cp-size"):
            ServerArgs(**self._base_kwargs(attn_cp_size=2, tp_size=2))

    # TestPrefillOnlyDisableKvCache类的测试rejectsprefillcontextparallel
    def test_rejects_prefill_context_parallel(self):
        with self.assertRaisesRegex(ValueError, "--enable-prefill-context-parallel"):
            ServerArgs(**self._base_kwargs(enable_prefill_context_parallel=True))

    # TestPrefillOnlyDisableKvCache类的测试rejectshisparse
    def test_rejects_hisparse(self):
        with self.assertRaisesRegex(ValueError, "--enable-hisparse"):
            ServerArgs(**self._base_kwargs(enable_hisparse=True))

    # TestPrefillOnlyDisableKvCache类的测试rejectsfp4kvcache
    def test_rejects_fp4_kv_cache(self):
        with self.assertRaisesRegex(ValueError, "fp4_e2m1"):
            ServerArgs(**self._base_kwargs(kv_cache_dtype="fp4_e2m1"))


# TestCutedslMoeMaxNumTokens类
class TestCutedslMoeMaxNumTokens(unittest.TestCase):
    """The shared CuteDSL MoE per-forward token bound. Fields are set directly
    to exercise the math independently of __post_init__ resolution."""

    # TestCutedslMoeMaxNumTokens类的内部方法_args
    def _args(self, **overrides):
        server_args = ServerArgs(model_path="dummy")
        fields = dict(
            speculative_algorithm=None,
            speculative_num_draft_tokens=None,
            max_prefill_tokens=16384,
            disable_piecewise_cuda_graph=False,
            piecewise_cuda_graph_max_tokens=2048,
            cuda_graph_max_bs=512,
        )
        fields.update(overrides)
        for key, value in fields.items():
            setattr(server_args, key, value)
        return server_args

    # TestCutedslMoeMaxNumTokens类的测试prefilldominatesindefaultconfig
    def test_prefill_dominates_in_default_config(self):
        self.assertEqual(self._args().cutedsl_moe_max_num_tokens(), 16384)  # 断言相等

    # TestCutedslMoeMaxNumTokens类的测试speculativedecodingscalesdecodebound
    def test_speculative_decoding_scales_decode_bound(self):
        # decode bound 512 * 8 dominates the small prefill/piecewise bounds
        args = self._args(
            max_prefill_tokens=512,
            piecewise_cuda_graph_max_tokens=512,
            speculative_algorithm="EAGLE",
            speculative_num_draft_tokens=8,
        )
        self.assertEqual(args.cutedsl_moe_max_num_tokens(), 4096)  # 断言相等

    # TestCutedslMoeMaxNumTokens类的测试piecewiseboundexcludedwhendisabled
    def test_piecewise_bound_excluded_when_disabled(self):
        args = self._args(
            max_prefill_tokens=512,
            disable_piecewise_cuda_graph=True,
            cuda_graph_max_bs=64,
        )
        self.assertEqual(args.cutedsl_moe_max_num_tokens(), 512)  # 断言相等


# TestSamplingBackendTokenOracleEnvGate类
class TestSamplingBackendTokenOracleEnvGate(CustomTestCase):
    """The 'token_oracle' choice is gated on SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE.

    The choice set is built once at server_args.py import time, so each subtest
    reloads the module with the env var set to the desired value.
    """

    def _reload_server_args_with_env(self, *, enabled: bool):
        previous = os.environ.get("SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE")
        os.environ["SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE"] = "1" if enabled else "0"
        try:
            return importlib.reload(server_args_module)
        finally:
            if previous is None:
                os.environ.pop("SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE", None)
            else:
                os.environ["SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE"] = previous

    # TestSamplingBackendTokenOracleEnvGate类的测试tokenoraclerejectedwhenenvdisabled
    def test_token_oracle_rejected_when_env_disabled(self):
        reloaded = self._reload_server_args_with_env(enabled=False)
        self.assertNotIn("token_oracle", reloaded.SAMPLING_BACKEND_CHOICES)  # 断言不包含

        with self.assertRaises(SystemExit):  # 断言抛出异常
            reloaded.prepare_server_args(
                [
                    "--model-path",
                    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
                    "--sampling-backend",
                    "token_oracle",
                ]
            )

    # TestSamplingBackendTokenOracleEnvGate类的测试tokenoracleacceptedwhenenvenabled
    def test_token_oracle_accepted_when_env_enabled(self):
        reloaded = self._reload_server_args_with_env(enabled=True)
        self.assertIn("token_oracle", reloaded.SAMPLING_BACKEND_CHOICES)  # 断言包含

        parsed = reloaded.prepare_server_args(
            [
                "--model-path",
                DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
                "--sampling-backend",
                "token_oracle",
                # Explicit device so ServerArgs.__post_init__ does not call
                # get_device() (fails on CPU-only CI runners) and does not run
                # _handle_cpu_backends (which would override sampling_backend
                # to "pytorch", masking what we want to verify).
                "--device",
                "cuda",
            ]
        )
        self.assertEqual(parsed.sampling_backend, "token_oracle")  # 断言相等


if __name__ == "__main__":
    unittest.main()
