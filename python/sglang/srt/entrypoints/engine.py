# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
The entry point of inference server. (SRT = SGLang Runtime)

This file implements python APIs for the inference engine.
"""

from __future__ import annotations

import asyncio
import atexit
import dataclasses
import logging
import multiprocessing as mp
import os
import random
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import torch
import uvloop
import zmq

from sglang.srt.elastic_ep.expert_backup_manager import run_expert_backup_manager
from sglang.srt.entrypoints.engine_info_bootstrap_server import (
    EngineInfoBootstrapServer,
)
from sglang.srt.entrypoints.engine_score_mixin import EngineScoreMixin
from sglang.srt.entrypoints.EngineBase import EngineBase
from sglang.srt.managers.data_parallel_controller import (
    SCHEDULER_PIDS_ARG,
    run_data_parallel_controller_process,
)
from sglang.srt.managers.detokenizer_manager import run_detokenizer_process
from sglang.srt.managers.io_struct import (
    CloseSessionReqInput,
    DestroyWeightsUpdateGroupReqInput,
    EmbeddingReqInput,
    GenerateReqInput,
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterReqInput,
    MultimodalDataInputFormat,
    OpenSessionReqInput,
    ProfileReq,
    ProfileReqType,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    RpcReqInput,
    RpcReqOutput,
    UnloadLoRAAdapterReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
    sock_recv,
    sock_send,
)
from sglang.srt.managers.multi_tokenizer_mixin import (
    MultiTokenizerRouter,
    run_multi_detokenizer_router_process,
)
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.observability.trace import process_tracing_init, trace_set_thread_info
from sglang.srt.parser.template_detection import resolve_auto_parsers
from sglang.srt.parser.template_manager import TemplateManager
from sglang.srt.plugins import load_plugins
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    MultiprocessingSerializer,
    SerializedTensorPayload,
    assert_pkg_version,
    configure_logger,
    get_bool_env_var,
    is_cuda,
    kill_process_tree,
    launch_dummy_health_check_server,
    maybe_reindex_device_id,
    normalize_serialized_named_tensor_payloads,
    numa_utils,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from sglang.srt.utils.msgspec_utils import msgspec_to_builtins
from sglang.srt.utils.network import (
    NetworkAddress,
    get_free_port,
    get_zmq_socket,
    is_port_available,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils.watchdog import SubprocessWatchdog
from sglang.version import __version__

logger = logging.getLogger(__name__)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

_is_cuda = is_cuda()


@dataclasses.dataclass
class SchedulerInitResult:
    """Result from launching schedulers."""

    scheduler_infos: List[Dict[str, Any]]
    all_child_pids: List[int] = dataclasses.field(default_factory=list)
    wait_for_ready: Callable[[], None] = lambda: None
    wait_for_completion: Callable[[], None] = lambda: None
    engine_info_bootstrap_server: Optional[Any] = None


def init_tokenizer_manager(
    server_args: ServerArgs,
    port_args: PortArgs,
    TokenizerManagerClass: Optional[TokenizerManager] = None,
) -> Tuple[TokenizerManager, TemplateManager]:
    # Launch tokenizer process
    TokenizerManagerClass = TokenizerManagerClass or TokenizerManager
    tokenizer_manager = TokenizerManagerClass(server_args, port_args)

    # Initialize templates
    template_manager = TemplateManager()
    template_manager.initialize_templates(
        tokenizer_manager=tokenizer_manager,
        model_path=server_args.model_path,
        chat_template=server_args.chat_template,
        completion_template=server_args.completion_template,
    )

    # Resolve any remaining auto parsers using template manager's detection results
    for attr, suggested, label in (
        (
            "reasoning_parser",
            template_manager.suggested_reasoning_parser,
            "reasoning parser",
        ),
        (
            "tool_call_parser",
            template_manager.suggested_tool_call_parser,
            "tool-call parser",
        ),
    ):
        if getattr(server_args, attr) != "auto":
            continue
        if suggested is not None:
            server_args.override(source="template-detection", **{attr: suggested})
            logger.info(
                f"Auto-detected --{attr.replace('_', '-')} as '{suggested}' from chat template"
            )
        else:
            logger.warning(
                f"--{attr.replace('_', '-')}=auto specified but could not detect "
                f"{label} from chat template. Disabling {label}."
            )
            server_args.override(source="template-detection", **{attr: None})

    return tokenizer_manager, template_manager


class Engine(EngineScoreMixin, EngineBase):
    """
    The entry point to the inference engine.

    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager all run in the main process.
    2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
    """

    # Some fields to allow people to override the server args
    # and launch processes for their private forks.
    server_args_class: ServerArgs = ServerArgs
    init_tokenizer_manager_func: Callable = staticmethod(init_tokenizer_manager)
    run_scheduler_process_func: Callable = staticmethod(run_scheduler_process)
    run_detokenizer_process_func: Callable = staticmethod(run_detokenizer_process)

    def __init__(self, **kwargs):
        """
        The arguments of this function is the same as `sglang/srt/server_args.py::ServerArgs`.
        Please refer to `ServerArgs` for the documentation.
        """

        # Ensure plugins are loaded before ServerArgs construction,
        # so hooks on ServerArgs.__post_init__ fire correctly.
        load_plugins()

        # Parse server_args
        if "server_args" in kwargs:
            # Directly load server_args
            server_args = kwargs["server_args"]
        else:
            # Construct server_args from kwargs
            if "log_level" not in kwargs:
                # Do not print logs by default
                kwargs["log_level"] = "error"
            server_args = self.server_args_class(**kwargs)
        self.server_args = server_args
        logger.info(f"{server_args=}")

        # Pre-initialize tokenizer_manager so the atexit handler in
        # shutdown() won't hit AttributeError.
        self.tokenizer_manager = None

        # Shutdown the subprocesses automatically when the program exits
        atexit.register(self.shutdown)

        # Launch subprocesses
        (
            tokenizer_manager,
            template_manager,
            port_args,
            scheduler_init_result,
            subprocess_watchdog,
            weight_cache_daemon_procs,
        ) = self._launch_subprocesses(
            server_args=server_args,
            init_tokenizer_manager_func=self.init_tokenizer_manager_func,
            run_scheduler_process_func=self.run_scheduler_process_func,
            run_detokenizer_process_func=self.run_detokenizer_process_func,
        )
        self.tokenizer_manager = tokenizer_manager
        self.template_manager = template_manager
        self._scheduler_init_result = scheduler_init_result
        # Engine-spawned weight cache daemons owned by *this* instance (empty
        # unless --weight-cache-mode daemon). Kept per-instance so two Engines
        # in one process each reap only their own daemons in shutdown().
        self._weight_cache_daemon_procs = weight_cache_daemon_procs
        if tokenizer_manager is not None:
            tokenizer_manager._subprocess_watchdog = subprocess_watchdog
        self.port_args = port_args

        # Initialize ZMQ sockets
        context = zmq.Context(2)
        if self.server_args.node_rank == 0:
            self.send_to_rpc = get_zmq_socket(
                context, zmq.DEALER, self.port_args.rpc_ipc_name, True
            )
        else:
            self.send_to_rpc = None

        # Enable tracing
        if server_args.enable_trace:
            process_tracing_init(
                server_args.otlp_traces_endpoint,
                "sglang",
                trace_modules=server_args.trace_modules,
            )
            thread_label = "Tokenizer"
            if server_args.disaggregation_mode == "prefill":
                thread_label = "Prefill Tokenizer"
            elif server_args.disaggregation_mode == "decode":
                thread_label = "Decode Tokenizer"
            trace_set_thread_info(thread_label)

        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    def get_all_child_pids(self) -> List[int]:
        """Returns a list of all child process PIDs."""
        return self._scheduler_init_result.all_child_pids

    def _resolve_routed_dp_rank(
        self,
        routed_dp_rank: Optional[int],
        data_parallel_rank: Optional[int],
    ) -> Optional[int]:
        if data_parallel_rank is not None:
            import warnings

            warnings.warn(
                "'data_parallel_rank' is deprecated, use 'routed_dp_rank' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            if routed_dp_rank is None:
                routed_dp_rank = data_parallel_rank

        if routed_dp_rank is not None:
            dp_size = self.server_args.dp_size
            if dp_size <= 1 and routed_dp_rank == 0:
                logger.debug(
                    f"routed_dp_rank={routed_dp_rank} is ignored because dp_size={dp_size}"
                )
                return None
            if routed_dp_rank < 0 or routed_dp_rank >= dp_size:
                raise ValueError(
                    f"routed_dp_rank={routed_dp_rank} out of range [0, {dp_size})"
                )

        logger.debug(f"routed_dp_rank: {routed_dp_rank}")
        return routed_dp_rank

    def generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        # The image input. It can be an image instance, file name, URL, or base64 encoded string.
        # Can be formatted as:
        # - Single image for a single request
        # - List of images (one per request in a batch)
        # - List of lists of images (multiple images per request)
        # - List of preprocessed outputs from a Huggingface processor, each as a dict containing `format`: 'processor_output' and other data
        # - List of precomputed image embeddings, each as a dict containing field `format`: 'precomputed_embedding' and `feature`: the precomputed embedding
        # See also python/sglang/srt/utils.py:load_image for more details.
        image_data: Optional[MultimodalDataInputFormat] = None,
        audio_data: Optional[MultimodalDataInputFormat] = None,
        video_data: Optional[MultimodalDataInputFormat] = None,
        # See GenerateReqInput.mm_hashes / async_generate for the contract.
        mm_hashes: Optional[Union[List[str], List[List[str]]]] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
        require_reasoning: bool = False,
        return_hidden_states: bool = False,
        return_routed_experts: bool = False,
        routed_experts_start_len: int = 0,
        stream: bool = False,
        bootstrap_host: Optional[Union[List[str], str]] = None,
        bootstrap_port: Optional[Union[List[int], int]] = None,
        bootstrap_room: Optional[Union[List[int], int]] = None,
        routed_dp_rank: Optional[int] = None,
        disagg_prefill_dp_rank: Optional[int] = None,
        # Deprecated: use routed_dp_rank instead
        data_parallel_rank: Optional[int] = None,
        external_trace_header: Optional[Dict] = None,
        rid: Optional[Union[List[str], str]] = None,
        session_params: Optional[Dict] = None,
        priority: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> Union[Dict, Iterator[Dict]]:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
        Please refer to `GenerateReqInput` for the documentation.
        """
        routed_dp_rank = self._resolve_routed_dp_rank(
            routed_dp_rank, data_parallel_rank
        )

        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
            mm_hashes=mm_hashes,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            lora_path=lora_path,
            custom_logit_processor=custom_logit_processor,
            require_reasoning=require_reasoning,
            return_hidden_states=return_hidden_states,
            return_routed_experts=return_routed_experts,
            routed_experts_start_len=routed_experts_start_len,
            stream=stream,
            bootstrap_host=bootstrap_host,
            bootstrap_port=bootstrap_port,
            bootstrap_room=bootstrap_room,
            routed_dp_rank=routed_dp_rank,
            disagg_prefill_dp_rank=disagg_prefill_dp_rank,
            external_trace_header=external_trace_header,
            rid=rid,
            session_id=session_id,
            session_params=session_params,
            priority=priority,
        )
        generator = self.tokenizer_manager.generate_request(obj, None)

        if stream:

            def generator_wrapper():
                while True:
                    try:
                        chunk = self.loop.run_until_complete(generator.__anext__())
                        yield chunk
                    except StopAsyncIteration:
                        break

            return generator_wrapper()
        else:
            ret = self.loop.run_until_complete(generator.__anext__())
            return ret

    async def async_generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        # The image input. It can be an image instance, file name, URL, or base64 encoded string.
        # Can be formatted as:
        # - Single image for a single request
        # - List of images (one per request in a batch)
        # - List of lists of images (multiple images per request)
        # - List of preprocessed outputs from a Huggingface processor, each as a dict containing `format`: 'processor_output' and other data
        # - List of precomputed image embeddings, each as a dict containing field `format`: 'precomputed_embedding' and `feature`: the precomputed embedding
        # See also python/sglang/srt/utils.py:load_image for more details.
        image_data: Optional[MultimodalDataInputFormat] = None,
        audio_data: Optional[MultimodalDataInputFormat] = None,
        video_data: Optional[MultimodalDataInputFormat] = None,
        # Optional per-image hashes the caller has already computed (hex strings,
        # one per image in `image_data`). When supplied, each MultimodalDataItem's
        # `hash` is initialised from this list and `set_pad_value` skips the
        # internal `hash_feature()` recompute. Intended for external KV routers
        # that compute their own per-image hash for routing decisions and need
        # sglang's prefix-cache key to align. See GenerateReqInput.mm_hashes.
        mm_hashes: Optional[Union[List[str], List[List[str]]]] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
        require_reasoning: bool = False,
        return_hidden_states: bool = False,
        return_routed_experts: bool = False,
        routed_experts_start_len: int = 0,
        stream: bool = False,
        bootstrap_host: Optional[Union[List[str], str]] = None,
        bootstrap_port: Optional[Union[List[int], int]] = None,
        bootstrap_room: Optional[Union[List[int], int]] = None,
        routed_dp_rank: Optional[int] = None,
        disagg_prefill_dp_rank: Optional[int] = None,
        # Deprecated: use routed_dp_rank instead
        data_parallel_rank: Optional[int] = None,
        external_trace_header: Optional[Dict] = None,
        rid: Optional[Union[List[str], str]] = None,
        session_params: Optional[Dict] = None,
        priority: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> Union[Dict, AsyncIterator[Dict]]:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
        Please refer to `GenerateReqInput` for the documentation.
        """
        routed_dp_rank = self._resolve_routed_dp_rank(
            routed_dp_rank, data_parallel_rank
        )

        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
            mm_hashes=mm_hashes,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            lora_path=lora_path,
            require_reasoning=require_reasoning,
            return_hidden_states=return_hidden_states,
            return_routed_experts=return_routed_experts,
            routed_experts_start_len=routed_experts_start_len,
            stream=stream,
            custom_logit_processor=custom_logit_processor,
            bootstrap_host=bootstrap_host,
            bootstrap_port=bootstrap_port,
            bootstrap_room=bootstrap_room,
            routed_dp_rank=routed_dp_rank,
            disagg_prefill_dp_rank=disagg_prefill_dp_rank,
            external_trace_header=external_trace_header,
            rid=rid,
            session_id=session_id,
            session_params=session_params,
            priority=priority,
        )
        generator = self.tokenizer_manager.generate_request(obj, None)

        if stream is True:
            return generator
        else:
            return await generator.__anext__()

    def encode(
        self,
        prompt: Union[str, List[str], List[Dict], List[List[Dict]]],
        image_data: Optional[MultimodalDataInputFormat] = None,
        audio_data: Optional[MultimodalDataInputFormat] = None,
        video_data: Optional[MultimodalDataInputFormat] = None,
        dimensions: Optional[int] = None,
        lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None,
        embed_override_token_id: Optional[int] = None,
        embed_overrides: Optional[List[List[torch.Tensor]]] = None,
        external_trace_header: Optional[Dict] = None,
        rid: Optional[Union[List[str], str]] = None,
    ) -> Dict:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::EmbeddingReqInput`.
        Please refer to `EmbeddingReqInput` for the documentation.
        """
        obj = EmbeddingReqInput(
            text=prompt,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
            dimensions=dimensions,
            lora_path=lora_path,
            embed_override_token_id=embed_override_token_id,
            embed_overrides=embed_overrides,
            external_trace_header=external_trace_header,
            rid=rid,
        )
        generator = self.tokenizer_manager.generate_request(obj, None)
        ret = self.loop.run_until_complete(generator.__anext__())
        return ret

    async def async_encode(
        self,
        prompt: Union[str, List[str], List[Dict], List[List[Dict]]],
        image_data: Optional[MultimodalDataInputFormat] = None,
        audio_data: Optional[MultimodalDataInputFormat] = None,
        video_data: Optional[MultimodalDataInputFormat] = None,
        dimensions: Optional[int] = None,
        lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None,
        embed_override_token_id: Optional[int] = None,
        embed_overrides: Optional[List[List[torch.Tensor]]] = None,
        external_trace_header: Optional[Dict] = None,
        rid: Optional[Union[List[str], str]] = None,
    ) -> Dict:
        """
        Asynchronous version of encode method.

        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::EmbeddingReqInput`.
        Please refer to `EmbeddingReqInput` for the documentation.
        """
        obj = EmbeddingReqInput(
            text=prompt,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
            dimensions=dimensions,
            lora_path=lora_path,
            embed_override_token_id=embed_override_token_id,
            embed_overrides=embed_overrides,
            external_trace_header=external_trace_header,
            rid=rid,
        )
        generator = self.tokenizer_manager.generate_request(obj, None)
        return await generator.__anext__()

    def rerank(
        self,
        prompt: Union[List[List[str]]],
    ) -> Dict:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::EmbeddingReqInput`.
        Please refer to `EmbeddingReqInput` for the documentation.
        """
        obj = EmbeddingReqInput(text=prompt, is_cross_encoder_request=True)
        generator = self.tokenizer_manager.generate_request(obj, None)
        ret = self.loop.run_until_complete(generator.__anext__())
        return ret

    @classmethod
    def _launch_weight_cache_daemons(cls, server_args: ServerArgs):
        """Launch weight cache daemon processes for this node's PP×TP ranks.

        All daemon processes join the same NCCL distributed group so that
        TP-sharded model loading works correctly. Each daemon holds its
        rank's weight shard in GPU memory and serves IPC handles.

        Lifecycle: these daemons are *co-terminal* with the engine. They are
        children of this process (kill_itself_when_parent_died installs
        PR_SET_PDEATHSIG) and are gracefully reaped in ``shutdown()``. They do
        NOT persist across engine restarts, so ``--weight-cache-mode daemon``
        on its own does not deliver a faster restart -- the first start is in
        fact slower (disk-load into the daemon plus the IPC handshake). The
        fast-recovery story is the standalone launcher
        (``python -m sglang.srt.weight_cache.daemon``) plus
        ``--weight-cache-mode client``, where the daemon outlives the engine.
        """
        if server_args.dp_size > 1:
            raise ValueError(
                "Weight cache daemon mode does not support dp_size > 1. "
                "Please set --dp-size 1 when using --weight-cache-mode daemon."
            )

        # Multi-node needs an explicit rendezvous address; otherwise each node
        # picks its own local 127.0.0.1 port (below) and the per-node daemons
        # can never form the joint process group.
        if server_args.nnodes > 1 and not server_args.dist_init_addr:
            raise ValueError(
                "Multi-node weight cache daemons (nnodes > 1) require "
                "--dist-init-addr so all nodes rendezvous at the same endpoint."
            )

        tp_size = server_args.tp_size

        pp_rank_range, tp_rank_range, pp_size_per_node, tp_size_per_node = (
            _calculate_rank_ranges(
                server_args.nnodes,
                server_args.pp_size,
                tp_size,
                server_args.node_rank,
            )
        )

        # Build the distributed init method (multi-node uses the user-provided
        # dist_init_addr so all nodes reach the same endpoint).
        if server_args.dist_init_addr:
            host, port = server_args.dist_init_addr.rsplit(":", 1)
            dist_init_method = f"tcp://{host}:{port}"
        else:
            # Fresh free port for the daemons' own rendezvous, not the engine's
            # nccl_port: a pinned --nccl-port would otherwise collide with the
            # engine's own NCCL TCPStore.
            dist_init_method = NetworkAddress("127.0.0.1", get_free_port()).to_tcp()

        num_daemons = len(pp_rank_range) * len(tp_rank_range)
        daemon_procs = []
        logger.info(
            f"Launching {num_daemons} weight cache daemon(s) on node "
            f"{server_args.node_rank} for model={server_args.model_path}, "
            f"pp_ranks={pp_rank_range.start}..{pp_rank_range.stop - 1}, "
            f"tp_ranks={tp_rank_range.start}..{tp_rank_range.stop - 1}, "
            f"dist_init_method={dist_init_method}"
        )

        # Validate and clean up stale .ready/.sock files from prior runs.
        # If a daemon is still alive at this rank, raise instead of clobbering.
        from sglang.srt.weight_cache.protocol import (
            cleanup_stale_daemon_files,
            compute_global_rank,
            compute_local_gpu_id,
            get_ready_path,
        )

        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                global_rank = compute_global_rank(tp_size, pp_rank, tp_rank)
                cleanup_stale_daemon_files(global_rank)

        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                gpu_id = compute_local_gpu_id(
                    pp_rank,
                    tp_rank,
                    pp_size_per_node,
                    tp_size_per_node,
                    base_gpu_id=server_args.base_gpu_id,
                    gpu_id_step=server_args.gpu_id_step,
                )
                cmd = [
                    sys.executable,
                    "-m",
                    "sglang.srt.weight_cache.daemon",
                    "--model-path",
                    server_args.model_path,
                    "--gpu-id",
                    str(gpu_id),
                    "--tp-size",
                    str(tp_size),
                    "--tp-rank",
                    str(tp_rank),
                    "--pp-size",
                    str(server_args.pp_size),
                    "--pp-rank",
                    str(pp_rank),
                    "--dp-size",
                    "1",
                    "--ep-size",
                    str(server_args.ep_size),
                    "--load-format",
                    server_args.load_format,
                    "--dtype",
                    server_args.dtype,
                    "--dist-init-method",
                    dist_init_method,
                ]
                if server_args.quantization:
                    cmd += ["--quantization", server_args.quantization]
                if (
                    server_args.model_loader_extra_config
                    and server_args.model_loader_extra_config != "{}"
                ):
                    cmd += [
                        "--model-loader-extra-config",
                        server_args.model_loader_extra_config,
                    ]
                if server_args.trust_remote_code:
                    cmd += ["--trust-remote-code"]
                if server_args.revision:
                    cmd += ["--revision", server_args.revision]

                proc = subprocess.Popen(cmd)
                daemon_procs.append(proc)

        # Wait for all daemons to be ready (ready file exists). On any failure
        # (readiness timeout or a daemon exiting early) terminate the siblings
        # we already spawned before propagating, so a partial launch does not
        # leak GPU-resident daemons.
        timeout = server_args.weight_cache_timeout
        check_interval = 2
        start_time = time.time()
        try:
            for pp_rank in pp_rank_range:
                for tp_rank in tp_rank_range:
                    global_rank = compute_global_rank(tp_size, pp_rank, tp_rank)
                    ready_path = get_ready_path(global_rank)
                    while not os.path.exists(ready_path):
                        time.sleep(check_interval)
                        if time.time() - start_time > timeout:
                            raise TimeoutError(
                                f"Weight cache daemon for pp_rank={pp_rank} "
                                f"tp_rank={tp_rank} did not become ready "
                                f"within {timeout}s"
                            )
                        # Check if daemon process is still alive
                        for p in daemon_procs:
                            if p.poll() is not None:
                                raise RuntimeError(
                                    f"Weight cache daemon (pid={p.pid}) exited prematurely "
                                    f"with code {p.returncode}"
                                )
                    logger.info(
                        f"Weight cache daemon for pp_rank={pp_rank} "
                        f"tp_rank={tp_rank} is ready"
                    )
        except BaseException:
            cls._terminate_weight_cache_daemons(daemon_procs)
            raise

        logger.info(
            f"All {num_daemons} weight cache daemons on node "
            f"{server_args.node_rank} are ready"
        )
        return daemon_procs

    @staticmethod
    def _terminate_weight_cache_daemons(procs, timeout: float = 10.0):
        """Gracefully stop engine-spawned weight cache daemons.

        Send SIGTERM first so each daemon's signal handler can unlink its
        ``.sock``/``.ready`` files, then SIGKILL any straggler. This matters
        because ``shutdown()`` otherwise reaps children via
        ``kill_process_tree`` (SIGKILL), which would skip that cleanup and
        leave stale files that make the next client-mode boot fail with a
        confusing "socket exists but connection refused" instead of a clean
        "no daemon" path.
        """
        if not procs:
            return
        for p in procs:
            if p.poll() is None:
                p.terminate()  # SIGTERM -> daemon cleanup handler runs
        for p in procs:
            try:
                p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning(
                    f"Weight cache daemon (pid={p.pid}) did not exit within "
                    f"{timeout}s of SIGTERM; sending SIGKILL."
                )
                p.kill()

    @classmethod
    def _launch_scheduler_processes(
        cls,
        server_args: ServerArgs,
        port_args: PortArgs,
        run_scheduler_process_func: Callable,
    ) -> Tuple[SchedulerInitResult, Optional[List]]:
        """使用 multiprocessing 启动 Scheduler 进程。

        子类可以重写该方法以接入不同的进程后端，例如 RayEngine 使用 Ray Actor。

        返回：
            (SchedulerInitResult, scheduler_procs)；RayEngine 不使用 mp.Process，
            因此它返回的 scheduler_procs 可以为 None。
        """
        scheduler_procs = []

        # DP 大于 1 或弹性 EP 处于扩缩容模式时，由 DataParallelController 统一创建
        # 和管理各 DP Scheduler；其他情况由当前主进程直接创建本节点的 Scheduler。
        use_dp_controller = (
            server_args.dp_size > 1 or server_args.ep_join_mode == "scale"
        )

        if not use_dp_controller:
            # 非 DP Controller 模式：按照本节点负责的 PP/TP rank 启动 Scheduler。
            # MemorySaverAdapter 会为子进程配置内存节省机制（未启用时为空操作）。
            memory_saver_adapter = TorchMemorySaverAdapter.create(
                enable=server_args.enable_memory_saver
            )

            scheduler_pipe_readers = []
            # 计算当前 node_rank 实际负责的 PP/TP rank 范围，以及每个节点上的
            # PP/TP 规模；多节点场景下每个节点只创建属于自己的 Scheduler。
            pp_rank_range, tp_rank_range, pp_size_per_node, tp_size_per_node = (
                _calculate_rank_ranges(
                    server_args.nnodes,
                    server_args.pp_size,
                    server_args.tp_size,
                    server_args.node_rank,
                )
            )

            # 遍历当前节点负责的所有 (pp_rank, tp_rank) 组合，并为每个组合启动一个
            # 独立的 Scheduler 子进程。每个 Scheduler 会绑定到计算得到的 GPU，使用
            # 对应的 PP/TP 及 Attention/MoE 并行 rank 初始化模型和通信组，并通过专属
            # 单向 Pipe 向父进程报告启动状态。创建完成后，父进程统一保存 Process 和
            # Pipe reader，分别用于进程生命周期管理以及等待 Scheduler 初始化完成。
            for pp_rank in pp_rank_range:
                for tp_rank in tp_rank_range:
                    # 使用单向 Pipe 接收子进程的就绪信号和初始化信息：
                    # 父进程持有 reader，Scheduler 子进程持有 writer。
                    reader, writer = mp.Pipe(duplex=False)

                    # 根据本节点内的 PP/TP 位置计算物理 GPU 编号。gpu_id_step
                    # 支持按固定步长选择设备，例如只使用编号为 0、2、4、6 的 GPU。
                    gpu_id = (
                        server_args.base_gpu_id
                        + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                        + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                    )

                    # 由全局 TP rank 推导 Attention CP、MoE DP 和 MoE EP rank，(这三个rank都是局部rank。)
                    # 并作为独立参数传给 Scheduler 初始化并行通信组。
                    attn_cp_rank, moe_dp_rank, moe_ep_rank = _compute_parallelism_ranks(
                        server_args, tp_rank
                    )

                    # maybe_reindex_device_id 处理 CUDA_VISIBLE_DEVICES 等设备重映射；
                    # 创建进程时同时配置 MemorySaver 和 NUMA 亲和性。
                    with maybe_reindex_device_id(gpu_id) as gpu_id:
                        proc = mp.Process(
                            target=run_scheduler_process_func,
                            args=(
                                server_args,
                                port_args,
                                gpu_id,
                                tp_rank,
                                attn_cp_rank,
                                moe_dp_rank,
                                moe_ep_rank,
                                pp_rank,
                                None,
                                writer,
                            ),
                        )

                        # 同时进入 MemorySaver 和 NUMA 两个子进程配置上下文；它们按
                        # 从上到下的顺序生效，并在离开 with 块时按相反顺序恢复父进程配置。
                        with (
                            # 为即将启动的 Scheduler 配置 torch-memory-saver；未启用
                            # enable_memory_saver 时，该上下文管理器不执行任何操作。
                            memory_saver_adapter.configure_subprocess(),
                            # 根据 server_args 和目标 gpu_id 配置子进程的 NUMA CPU/内存
                            # 亲和性；未开启 NUMA 绑定或无法绑定时按配置降级为空操作。
                            numa_utils.configure_subprocess(server_args, gpu_id),
                        ):
                            # 在两个临时配置都生效期间真正启动进程，使 Scheduler 子进程
                            # 继承 MemorySaver 配置，并在可用时通过 numactl 完成 NUMA 绑定。
                            proc.start()

                    scheduler_procs.append(proc)
                    scheduler_pipe_readers.append(reader)
        else:
            # DP Controller 模式：这里只启动一个控制器进程。控制器随后负责创建和
            # 管理各 DP Scheduler，并通过 Pipe 将所有 Scheduler 的初始化信息传回。
            reader, writer = mp.Pipe(duplex=False)
            scheduler_pipe_readers = [reader]
            proc = mp.Process(
                target=run_data_parallel_controller_process,
                kwargs=dict(
                    server_args=server_args,
                    port_args=port_args,
                    pipe_writer=writer,
                    run_scheduler_process_func=run_scheduler_process_func,
                ),
            )
            proc.start()
            scheduler_procs.append(proc)

        # 直接启动模式记录所有 Scheduler PID；DP Controller 模式此时只能记录
        # Controller PID，实际 Scheduler PID 会在 wait_for_ready 中补充。
        all_child_pids = [proc.pid for proc in scheduler_procs]

        # 该列表会在 wait_for_ready 被调用时原地填充。SchedulerInitResult 持有同一个
        # 列表对象，因此调用方等待完成后即可从中读取模型限制等初始化信息。
        scheduler_infos = []

        def wait_for_ready():
            # 等待每个 Pipe 返回初始化结果；若子进程提前退出，辅助函数会报告启动失败。
            infos = _wait_for_scheduler_ready(scheduler_pipe_readers, scheduler_procs)
            scheduler_infos.extend(infos)
            if use_dp_controller:
                # Controller 创建的 Scheduler 不是当前进程的直接子进程，需要从
                # Controller 上报的信息中取出 PID，供 Engine 退出时统一清理进程树。
                for info in infos:
                    if SCHEDULER_PIDS_ARG in info:
                        all_child_pids.extend(info[SCHEDULER_PIDS_ARG])

        def wait_for_completion():
            # 服务型启动方式使用该回调持续等待 Scheduler/Controller 退出，并记录
            # 退出码；join 返回通常意味着推理后端已经停止工作。
            for proc in scheduler_procs:
                proc.join()
                logger.error(
                    f"Scheduler or DataParallelController {proc.pid} "
                    f"terminated with {proc.exitcode}"
                )

        # 将可延迟执行的等待操作封装进结果对象：调用方可以先完成其他组件初始化，
        # 再通过 wait_for_ready 与 Scheduler 的模型加载阶段同步。
        return (
            SchedulerInitResult(
                scheduler_infos=scheduler_infos,
                all_child_pids=all_child_pids,
                wait_for_ready=wait_for_ready,
                wait_for_completion=wait_for_completion,
            ),
            scheduler_procs,
        )

    @classmethod
    def _launch_detokenizer_subprocesses(
        cls,
        server_args: ServerArgs,
        port_args: PortArgs,
        run_detokenizer_process_func: Callable,
    ) -> Tuple[List[mp.Process], List[str]]:
        """Launch detokenizer worker(s).

        - When ``detokenizer_worker_num == 1``: a single detokenizer process listens on
          ``port_args.detokenizer_ipc_name`` (the original behavior).
        - When ``detokenizer_worker_num > 1``: each detokenizer worker gets its own
          private IPC socket, and a ``MultiDetokenizerRouter`` process owns the
          original ``port_args.detokenizer_ipc_name`` and fans out to them.

        Returns (processes, names) for SubprocessWatchdog.
        """
        processes: List[mp.Process] = []
        names: List[str] = []

        if server_args.detokenizer_worker_num <= 1:
            proc = mp.Process(
                target=run_detokenizer_process_func,
                args=(server_args, port_args),
            )
            proc.start()
            processes.append(proc)
            names.append("detokenizer")
            return processes, names

        router_ipc_name = port_args.detokenizer_ipc_name
        worker_ipc_names: List[str] = []
        try:
            for i in range(server_args.detokenizer_worker_num):
                worker_ipc = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
                port_args.detokenizer_ipc_name = worker_ipc
                proc = mp.Process(
                    target=run_detokenizer_process_func,
                    args=(server_args, port_args),
                )
                proc.start()
                processes.append(proc)
                names.append(f"detokenizer_{i}")
                worker_ipc_names.append(worker_ipc)
        finally:
            port_args.detokenizer_ipc_name = router_ipc_name

        router_proc = mp.Process(
            target=run_multi_detokenizer_router_process,
            args=(worker_ipc_names, server_args, port_args),
        )
        router_proc.start()
        processes.append(router_proc)
        names.append("detokenizer_router")

        return processes, names

    @classmethod
    def _launch_subprocesses(
        cls,
        server_args: ServerArgs,
        init_tokenizer_manager_func: Callable,
        run_scheduler_process_func: Callable,
        run_detokenizer_process_func: Callable,
        port_args: Optional[PortArgs] = None,
    ) -> Tuple[
        TokenizerManager,
        TemplateManager,
        PortArgs,
        SchedulerInitResult,
        Optional[SubprocessWatchdog],
    ]:
        """Launch the TokenizerManager in the main process, the Scheduler in a subprocess, and the DetokenizerManager in another subprocess.

        Returns:
            Tuple of (tokenizer_manager, template_manager, port_args, scheduler_init_result, subprocess_watchdog, weight_cache_daemon_procs).
        """
        # 配置日志、环境变量和全局运行参数；这些设置必须在创建子进程前完成，
        # 以便使用 spawn 启动的进程能够继承一致的运行环境。
        configure_logger(server_args)
        _set_envs_and_config(server_args)

        # 防御性地再次加载插件。正常情况下 Engine.__init__ 或 CLI 入口已经加载，
        # 此处确保直接调用该方法时也不会遗漏插件注册。
        load_plugins()

        # 在分配端口和拉起进程之前完成参数校验及 GC 配置，尽早暴露无效配置。
        server_args.check_server_args()
        _set_gc(server_args)

        # 为 Scheduler、TokenizerManager、DetokenizerManager 等组件分配进程间通信端口。
        # 调用方传入 port_args 时复用已有端口，常用于多节点或外部统一分配端口的场景。
        if port_args is None:
            port_args = PortArgs.init_new(server_args)
        logger.info(f"{server_args=}")

        # 某些远程权重加载模式需要在 rank 0 上启动引擎信息引导服务，
        # 供其他 rank 获取启动种子等初始化信息。
        engine_info_bootstrap_server = None
        if (
            server_args.remote_instance_weight_loader_start_seed_via_transfer_engine
            and server_args.node_rank == 0
        ):
            bootstrap_port = server_args.engine_info_bootstrap_port
            if not is_port_available(bootstrap_port):
                raise RuntimeError(
                    f"engine_info_bootstrap_port {bootstrap_port} is already in use. "
                    f"When running multiple instances on the same node, each instance must use a "
                    f"different --engine-info-bootstrap-port."
                )
            engine_info_bootstrap_server = EngineInfoBootstrapServer(
                host=server_args.host, port=bootstrap_port
            )

        if (
            server_args.reasoning_parser == "auto"
            or server_args.tool_call_parser == "auto"
        ):
            # 根据模型配置自动选择推理内容和工具调用的解析器。
            resolve_auto_parsers(server_args)

        # daemon 模式下先启动权重缓存守护进程。进程句柄最终返回给当前 Engine 实例，
        # 而不是保存在类属性中，避免同一主进程内多个 Engine 相互覆盖守护进程列表。
        weight_cache_daemon_procs: List = []
        if server_args.weight_cache_mode == "daemon":
            weight_cache_daemon_procs = cls._launch_weight_cache_daemons(server_args)

        # 启动 Scheduler（或 DataParallelController）子进程。
        # wait_for_ready 会在稍后等待模型加载完成，并收集各 Scheduler 的初始化信息。
        scheduler_init_result, scheduler_procs = cls._launch_scheduler_processes(
            server_args, port_args, run_scheduler_process_func
        )
        scheduler_init_result.engine_info_bootstrap_server = (
            engine_info_bootstrap_server
        )

        if (
            server_args.enable_elastic_expert_backup
            and server_args.elastic_ep_backend is not None
        ):
            # 启用弹性专家备份时，在 Scheduler 启动后初始化专家备份管理器。
            run_expert_backup_manager(server_args, port_args)

        if server_args.node_rank >= 1:
            # 非 rank 0 节点只承载 Scheduler，不启动 Tokenizer/Detokenizer。
            # 必须先等待 Scheduler 就绪，确保模型已成功加载。
            scheduler_init_result.wait_for_ready()

            if os.getenv("SGLANG_BLOCK_NONZERO_RANK_CHILDREN") == "0":
                # 通过 Python API 使用 Engine 时允许直接返回，避免当前调用被子进程阻塞。
                return (
                    None,
                    None,
                    port_args,
                    scheduler_init_result,
                    None,
                    weight_cache_daemon_procs,
                )

            # CLI/服务模式下提供最小健康检查服务，并持续等待 Scheduler 退出。
            launch_dummy_health_check_server(
                server_args.host, server_args.port, server_args.enable_metrics
            )

            scheduler_init_result.wait_for_completion()
            return (
                None,
                None,
                port_args,
                scheduler_init_result,
                None,
                weight_cache_daemon_procs,
            )

        # 在 rank 0 上启动 Detokenizer 子进程；存在多个 worker 时，额外启动 Router，
        # 由 Router 接收统一入口上的消息并分发给各 Detokenizer worker。
        detoken_procs, detoken_names = cls._launch_detokenizer_subprocesses(
            server_args=server_args,
            port_args=port_args,
            run_detokenizer_process_func=run_detokenizer_process_func,
        )
        # 将 Detokenizer PID 纳入统一的子进程清理范围。
        for p in detoken_procs:
            scheduler_init_result.all_child_pids.append(p.pid)

        # 在主进程中初始化 TokenizerManager。单 worker 直接创建管理器；多 worker
        # 则创建 MultiTokenizerRouter，由它负责把请求路由到不同 Tokenizer worker。
        # 引导服务也会在这一初始化阶段完成，因此要先于等待 Scheduler 就绪执行。
        if server_args.tokenizer_worker_num == 1:
            tokenizer_manager, template_manager = init_tokenizer_manager_func(
                server_args, port_args
            )
        else:
            tokenizer_manager = MultiTokenizerRouter(server_args, port_args)
            template_manager = None

        # 阻塞到所有 Scheduler 完成模型加载；启动失败会在这里被感知并向上抛出。
        scheduler_init_result.wait_for_ready()

        # 将 Scheduler 根据模型和运行配置计算出的最大输入长度同步给 TokenizerManager，
        # 供主进程在接收请求时进行长度校验。
        tokenizer_manager.max_req_input_len = scheduler_init_result.scheduler_infos[0][
            "max_req_input_len"
        ]

        # 启动存活监控线程，统一检测 Scheduler 和 Detokenizer 是否异常退出。
        # RayEngine 使用 Ray Actor 而非 mp.Process，因此其 scheduler_procs 可能为 None。
        processes = list(scheduler_procs or [])
        names = [f"scheduler_{i}" for i in range(len(processes))]
        processes.extend(detoken_procs)
        names.extend(detoken_names)
        subprocess_watchdog = SubprocessWatchdog(
            processes=processes, process_names=names
        )
        subprocess_watchdog.start()

        # 返回主进程组件、通信配置、Scheduler 初始化结果以及所有需要随 Engine
        # 生命周期管理的后台进程句柄。
        return (
            tokenizer_manager,
            template_manager,
            port_args,
            scheduler_init_result,
            subprocess_watchdog,
            weight_cache_daemon_procs,
        )

    def shutdown(self):
        """Shutdown the engine; block until the scheduler subprocess releases
        its GPU context so the caller can immediately reallocate on the same
        device."""
        if (
            self.tokenizer_manager is not None
            and self.tokenizer_manager._subprocess_watchdog is not None
        ):
            self.tokenizer_manager._subprocess_watchdog.stop()

        send_to_rpc = getattr(self, "send_to_rpc", None)
        if send_to_rpc is not None:
            send_to_rpc.close(linger=0)
            self.send_to_rpc = None

        # Gracefully stop weight cache daemons *before* the blanket
        # kill_process_tree below, so their SIGTERM handlers can unlink the
        # .sock/.ready files instead of being SIGKILLed and leaving stale state.
        daemon_procs = getattr(self, "_weight_cache_daemon_procs", None)
        if daemon_procs:
            self._terminate_weight_cache_daemons(daemon_procs)
            self._weight_cache_daemon_procs = []

        kill_process_tree(os.getpid(), include_parent=False, wait_timeout=60)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return False

    def flush_cache(self):
        return self.loop.run_until_complete(self.tokenizer_manager.flush_cache())

    def open_session(
        self,
        capacity_of_str_len: int,
        session_id: Optional[str] = None,
        streaming: bool = False,
        timeout: Optional[float] = None,
    ) -> str:
        """Open a session for multi-turn conversation with shared context.

        Args:
            capacity_of_str_len: Maximum string length capacity for the session.
            session_id: Optional session ID. If not provided, a UUID will be generated.
            streaming: Use low-overhead path for realtime streaming (append-only mode).
            timeout: If set, the session is automatically closed after being inactive
                for this many seconds. Inactivity is measured from session open or the
                most recent request submission.

        Returns:
            The session ID (either the provided one or a newly generated UUID).
        """
        obj = OpenSessionReqInput(
            capacity_of_str_len=capacity_of_str_len,
            session_id=session_id,
            streaming=streaming,
            timeout=timeout,
        )
        return self.loop.run_until_complete(
            self.tokenizer_manager.open_session(obj, None)
        )

    def close_session(self, session_id: str) -> None:
        """Close a session and release its resources.

        Args:
            session_id: The session ID to close.
        """
        obj = CloseSessionReqInput(session_id=session_id)
        self.loop.run_until_complete(self.tokenizer_manager.close_session(obj, None))

    def start_profile(self, **kwargs):
        req = ProfileReq(req_type=ProfileReqType.START_PROFILE, **kwargs)
        self.loop.run_until_complete(self.tokenizer_manager.start_profile(req))

    def stop_profile(self):
        self.loop.run_until_complete(self.tokenizer_manager.stop_profile())

    def start_expert_distribution_record(self):
        self.loop.run_until_complete(
            self.tokenizer_manager.start_expert_distribution_record()
        )

    def stop_expert_distribution_record(self):
        self.loop.run_until_complete(
            self.tokenizer_manager.stop_expert_distribution_record()
        )

    def dump_expert_distribution_record(self):
        self.loop.run_until_complete(
            self.tokenizer_manager.dump_expert_distribution_record()
        )

    def get_server_info(self):
        internal_states = self.loop.run_until_complete(
            self.tokenizer_manager.get_internal_state()
        )
        return msgspec_to_builtins(
            {
                **dataclasses.asdict(self.tokenizer_manager.server_args),
                **self._scheduler_init_result.scheduler_infos[0],
                "internal_states": internal_states,
                "version": __version__,
            }
        )

    def init_weights_update_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
    ):
        """Initialize parameter update group."""
        obj = InitWeightsUpdateGroupReqInput(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )
        return self.loop.run_until_complete(
            self.tokenizer_manager.init_weights_update_group(obj, None)
        )

    def destroy_weights_update_group(
        self,
        group_name: str,
    ):
        """Destroy parameter update group."""
        obj = DestroyWeightsUpdateGroupReqInput(
            group_name=group_name,
        )
        return self.loop.run_until_complete(
            self.tokenizer_manager.destroy_weights_update_group(obj, None)
        )

    def update_weights_from_distributed(
        self,
        names: list[str],
        dtypes: list[str],
        shapes: list[list[int]],
        group_name: str = "weight_update_group",
        flush_cache: bool = True,
        load_format: Optional[str] = None,
    ):
        """Update weights from distributed source."""
        obj = UpdateWeightsFromDistributedReqInput(
            names=names,
            dtypes=dtypes,
            shapes=shapes,
            group_name=group_name,
            flush_cache=flush_cache,
            load_format=load_format,
        )
        return self.loop.run_until_complete(
            self.tokenizer_manager.update_weights_from_distributed(obj, None)
        )

    def update_weights_from_tensor(
        self,
        named_tensors: Union[
            List[Tuple[str, torch.Tensor]],
            List[SerializedTensorPayload],
        ],
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ):
        """Update weights from distributed source. If there are going to be more updates, set `flush_cache` to be false
        to avoid duplicated cache cleaning operation."""
        serialized_named_tensors = self._serialize_tensors_per_rank(
            named_tensors, load_format
        )
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=serialized_named_tensors,
            load_format=load_format,
            flush_cache=flush_cache,
        )
        return self.loop.run_until_complete(
            self.tokenizer_manager.update_weights_from_tensor(obj, None)
        )

    def update_weights_from_disk(
        self,
        model_path: str,
        load_format: Optional[str] = None,
    ):
        """Update the weights from disk inplace without re-launching the engine.

        This method allows updating the model weights from disk without restarting
        the engine. It can be used to load a different model or update weights with
        new training.
        """
        obj = UpdateWeightFromDiskReqInput(
            model_path=model_path,
            load_format=load_format,
        )

        return self.loop.run_until_complete(
            self.tokenizer_manager.update_weights_from_disk(obj, None)
        )

    def update_weights_from_ipc(
        self,
        zmq_handles: Dict[str, str],
        flush_cache: bool = True,
    ):
        """Update weights from IPC for checkpoint-engine integration."""
        obj = UpdateWeightsFromIPCReqInput(
            zmq_handles=zmq_handles,
            flush_cache=flush_cache,
        )
        return self.loop.run_until_complete(
            self.tokenizer_manager.update_weights_from_ipc(obj, None)
        )

    def get_weights_by_name(self, name: str, truncate_size: int = 100):
        """Get weights by parameter name."""
        obj = GetWeightsByNameReqInput(name=name, truncate_size=truncate_size)
        return self.loop.run_until_complete(
            self.tokenizer_manager.get_weights_by_name(obj, None)
        )

    def _serialize_tensors_per_rank(
        self,
        tensors,
        load_format: Optional[str],
    ) -> List[bytes]:
        """One serialized payload per TP rank: each rank deserializes only its
        own copy, so producer-side CUDA-IPC refcounts drop cleanly after every
        load. flattened_bucket callers pass pre-serialized per-rank payloads."""
        if load_format == "flattened_bucket":
            return normalize_serialized_named_tensor_payloads(
                cast(List[SerializedTensorPayload], tensors)
            )
        else:
            return [
                MultiprocessingSerializer.serialize(tensors)
                for _ in range(self.server_args.tp_size)
            ]

    def load_lora_adapter_from_tensors(
        self,
        lora_name: str,
        tensors: Union[Dict[str, torch.Tensor], List[SerializedTensorPayload]],
        config_dict: Dict,
        load_format: Optional[str] = None,
    ):
        serialized_named_tensors = self._serialize_tensors_per_rank(
            tensors, load_format
        )
        lora_req = LoadLoRAAdapterFromTensorsReqInput(
            lora_name=lora_name,
            config_dict=config_dict,
            serialized_named_tensors=serialized_named_tensors,
            load_format=load_format,
        )
        return self.loop.run_until_complete(
            self.tokenizer_manager.load_lora_adapter_from_tensors(lora_req, None)
        )

    def load_lora_adapter(self, lora_name: str, lora_path: str, pinned: bool = False):
        """Load a new LoRA adapter without re-launching the engine."""

        obj = LoadLoRAAdapterReqInput(
            lora_name=lora_name,
            lora_path=lora_path,
            pinned=pinned,
        )

        return self.loop.run_until_complete(
            self.tokenizer_manager.load_lora_adapter(obj, None)
        )

    def unload_lora_adapter(self, lora_name: str):
        """Unload a LoRA adapter without re-launching the engine."""

        obj = UnloadLoRAAdapterReqInput(lora_name=lora_name)

        return self.loop.run_until_complete(
            self.tokenizer_manager.unload_lora_adapter(obj, None)
        )

    async def async_load_lora_adapter(
        self, lora_name: str, lora_path: str, pinned: bool = False
    ):
        """
        Asynchronous version of load_lora_adapter.

        See load_lora_adapter() for detailed documentation.
        """

        obj = LoadLoRAAdapterReqInput(
            lora_name=lora_name,
            lora_path=lora_path,
            pinned=pinned,
        )

        return await self.tokenizer_manager.load_lora_adapter(obj, None)

    async def async_unload_lora_adapter(self, lora_name: str):
        """
        Asynchronous version of unload_lora_adapter.

        See unload_lora_adapter() for detailed documentation.
        """

        obj = UnloadLoRAAdapterReqInput(lora_name=lora_name)

        return await self.tokenizer_manager.unload_lora_adapter(obj, None)

    def release_memory_occupation(self, tags: Optional[List[str]] = None):
        obj = ReleaseMemoryOccupationReqInput(tags=tags)
        return self.loop.run_until_complete(
            self.tokenizer_manager.release_memory_occupation(obj, None)
        )

    def resume_memory_occupation(self, tags: Optional[List[str]] = None):
        obj = ResumeMemoryOccupationReqInput(tags=tags)
        return self.loop.run_until_complete(
            self.tokenizer_manager.resume_memory_occupation(obj, None)
        )

    def freeze_gc(self):
        """
        To maintain a high performance server with low latency, we want to reduce the
        stalls caused by the garbage collector scanning through a large number of objects.

        It is usually helpful to start the server and warm it up with real requests to
        initialize many of the long-lived objects that do not need to be garbage collected.

        After sufficient warmup, we can call this function to freeze the garbage collector
        so that all objects created before this point are considered out of scope for garbage
        collection.
        """

        self.loop.run_until_complete(self.tokenizer_manager.freeze_gc())

    """
    Execute an RPC call on all scheduler processes.
    """

    def collective_rpc(self, method: str, **kwargs):
        obj = RpcReqInput(method=method, parameters=kwargs)
        sock_send(self.send_to_rpc, obj)
        recv_req = sock_recv(self.send_to_rpc, flags=zmq.BLOCKY)
        assert isinstance(recv_req, RpcReqOutput)
        assert recv_req.success, recv_req.message

    def save_remote_model(self, **kwargs):
        self.collective_rpc("save_remote_model", **kwargs)

    def save_sharded_model(self, **kwargs):
        self.collective_rpc("save_sharded_model", **kwargs)

    # score() and async_score() are provided by EngineScoreMixin


def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    if "NCCL_CUMEM_ENABLE" not in os.environ or server_args.enable_symm_mem:
        os.environ["NCCL_CUMEM_ENABLE"] = str(int(server_args.enable_symm_mem))
    if (
        "NCCL_NVLS_ENABLE" not in os.environ
        or server_args.enable_nccl_nvls
        or server_args.enable_symm_mem
    ):
        os.environ["NCCL_NVLS_ENABLE"] = str(
            int(server_args.enable_nccl_nvls or server_args.enable_symm_mem)
        )
    if "NCCL_GRAPH_MIXING_SUPPORT" not in os.environ or server_args.enable_symm_mem:
        # Note(wh): NCCL_GRAPH_MIXING_SUPPORT=0 can help improve performance for symmetric kernels.
        # details in https://github.com/NVIDIA/nccl-tests/issues/333#issuecomment-3103636985
        if server_args.dcp_size > 1:
            os.environ["NCCL_GRAPH_MIXING_SUPPORT"] = "0"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "8"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"

    if os.environ.get("TRTLLM_ENABLE_PDL", "1") != "0":
        # flashinfer uses this environment variable for various kernels from MoE to quant kernels
        os.environ["TRTLLM_ENABLE_PDL"] = "1"

    if os.environ.get("CUTE_DSL_LOG_LEVEL") is None:
        # Default to warning level, to avoid too many logs
        os.environ["CUTE_DSL_LOG_LEVEL"] = "30"

    if os.environ.get("CUTE_DSL_LOG_TO_CONSOLE") is None:
        # Need to set log to console, otherwise the log level won't take effect
        os.environ["CUTE_DSL_LOG_TO_CONSOLE"] = "1"

    # Can also be passed as argument
    os.environ["SGLANG_RUN_ID"] = (
        f"sglang-run-{time.time()}-{random.randint(0, 100000000)}"
    )

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Check flashinfer version
    if not get_bool_env_var("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"):
        if server_args.attention_backend == "flashinfer":
            assert_pkg_version(
                "flashinfer_python",
                "0.6.15.post1",
                "Please uninstall the old version and "
                "reinstall the latest version by following the instructions "
                "at https://docs.flashinfer.ai/installation.html.",
            )
        if _is_cuda:
            assert_pkg_version(
                "sglang-kernel",
                "0.4.5",
                "Please reinstall the latest version with `pip install sglang-kernel --force-reinstall`",
            )

    # Signal handlers can only be registered from the main thread.
    if threading.current_thread() is threading.main_thread():
        if server_args.custom_sigquit_handler is None:
            # Register the signal handler.
            # The child processes will send SIGQUIT to this process when any error happens
            # This process then clean up the whole process tree
            # Note: This sigquit handler is used in the launch phase, and may be replaced by
            # the running_phase_sigquit_handler in the tokenizer manager after the grpc server is launched.
            def launch_phase_sigquit_handler(signum, frame):
                logger.error(
                    "Received sigquit from a child process. It usually means the child failed."
                )
                kill_process_tree(os.getpid())

            signal.signal(signal.SIGQUIT, launch_phase_sigquit_handler)
        else:
            # Allow users to register a custom SIGQUIT handler for things like crash dump
            logger.error(
                f"Using custom SIGQUIT handler: {server_args.custom_sigquit_handler}"
            )
            signal.signal(signal.SIGQUIT, server_args.custom_sigquit_handler)
    else:
        logger.warning(
            "Signal handler is not added because the engine is not in the "
            "main thread. This disables the SIGQUIT handler for cleaning up "
            "the process tree when a child process fails."
        )

    # Set mp start method
    mp.set_start_method("spawn", force=True)


def _set_gc(server_args: ServerArgs):
    if gc_threshold := server_args.gc_threshold:
        import gc

        gc.set_threshold(*gc_threshold)


def _scheduler_died_error(rank: int, proc) -> RuntimeError:
    """Build a descriptive error for a scheduler process that died during init."""
    proc.join(timeout=10)
    return RuntimeError(
        f"Rank {rank} scheduler died during initialization "
        f"(exit code: {proc.exitcode}). "
        f"If exit code is -9 (SIGKILL), a common cause is the OS OOM killer. "
        f"Run `dmesg -T | grep -i oom` to check."
    )


def _wait_for_scheduler_ready(
    scheduler_pipe_readers: List,
    scheduler_procs: List,
) -> List[Dict]:
    """Wait for the model to finish loading and return scheduler infos.

    Uses poll() with timeout instead of blocking recv(), so that child process
    death (e.g. OOM SIGKILL) is detected promptly instead of hanging forever.
    """
    scheduler_infos = []
    for i in range(len(scheduler_pipe_readers)):
        while True:
            if scheduler_pipe_readers[i].poll(timeout=5.0):
                try:
                    data = scheduler_pipe_readers[i].recv()
                except EOFError:
                    raise _scheduler_died_error(i, scheduler_procs[i])
                if data["status"] != "ready":
                    raise RuntimeError(
                        "Initialization failed. Please see the error messages above."
                    )
                scheduler_infos.append(data)
                break

            # Poll timed out — check all processes for early death
            for j in range(len(scheduler_procs)):
                if not scheduler_procs[j].is_alive():
                    raise _scheduler_died_error(j, scheduler_procs[j])

    return scheduler_infos


def _calculate_rank_ranges(
    nnodes: int, pp_size: int, tp_size: int, node_rank: int
) -> Tuple[range, range, int, int]:
    """计算指定节点负责的流水线并行（PP）和张量并行（TP）rank 范围。

    该函数把全局 PP/TP 拓扑映射到当前 ``node_rank``，供调用方只在本节点上
    启动对应的 Scheduler 进程。计算逻辑如下：

    1. 使用 ``max(pp_size // nnodes, 1)`` 计算每个节点承载的 PP rank 数量。
       当 PP 数量不少于节点数时，每个节点获得一段连续的 PP rank；当节点数
       多于 PP 数量时，每个节点至少承载一个 PP rank。
    2. 使用 ``max(nnodes // pp_size, 1)`` 计算同一个 PP rank 横跨的节点数。
       节点数多于 PP 数量时，相邻的若干节点组成一个 PP 组，并共享同一个
       PP rank；否则一个 PP rank 只位于一个节点上。
    3. ``node_rank // nnodes_per_pp_rank`` 确定当前节点属于哪个 PP 组，进而
       得到该节点负责的半开区间 ``pp_rank_range``。
    4. 同一个 PP 组内的节点共同划分 TP ranks。先计算每个节点负责的 TP rank
       数量，再通过 ``node_rank % nnodes_per_tp_group`` 得到节点在组内的位置，
       最终生成该节点对应的连续半开区间 ``tp_rank_range``。

    这里使用整数除法，依赖调用方已经完成并行规模的整除性和合法性校验。

    参数：
        nnodes: 集群的节点总数。
        pp_size: 全局流水线并行规模。
        tp_size: 全局张量并行规模。
        node_rank: 当前节点的编号，取值范围为 ``[0, nnodes)``。

    返回：
        ``(pp_rank_range, tp_rank_range, pp_size_per_node, tp_size_per_node)``：

        - ``pp_rank_range``：当前节点负责的 PP rank 半开区间。
        - ``tp_rank_range``：当前节点负责的 TP rank 半开区间。
        - ``pp_size_per_node``：每个节点承载的 PP rank 数量。
        - ``tp_size_per_node``：每个节点承载的 TP rank 数量。
    """
    pp_size_per_node = max(pp_size // nnodes, 1)
    nnodes_per_pp_rank = max(nnodes // pp_size, 1)
    pp_rank_range = range(
        pp_size_per_node * (node_rank // nnodes_per_pp_rank),
        pp_size_per_node * (node_rank // nnodes_per_pp_rank + 1),
    )

    nnodes_per_tp_group = nnodes_per_pp_rank
    tp_size_per_node = tp_size // nnodes_per_tp_group
    tp_rank_range = range(
        tp_size_per_node * (node_rank % nnodes_per_tp_group),
        tp_size_per_node * (node_rank % nnodes_per_tp_group + 1),
    )

    return pp_rank_range, tp_rank_range, pp_size_per_node, tp_size_per_node


def _compute_parallelism_ranks(
    server_args: ServerArgs, tp_rank: int
) -> Tuple[int, int, int]:
    """根据全局 TP rank 计算 Attention CP、MoE DP 和 MoE EP 的局部 rank。

    同一个全局 TP rank 在 Attention 和 MoE 模块中会按照不同的并行层级重新解释：

    - Attention：``Global TP -> Attention DP -> Attention CP -> Attention TP``。
    - MoE：``Global TP -> MoE DP -> EP -> MoE TP``。

    该函数通过整数除法定位外层并行分组，通过取模定位当前 TP rank 在分组内的
    位置。计算依赖 ``tp_size`` 能被相关 DP、CP 和 EP 规模整除；这些合法性条件
    应在 ServerArgs 校验阶段保证。

    参数：
        server_args: 服务器及并行配置，提供 TP、DP、Attention CP、MoE DP 和
            EP 的规模。
        tp_rank: 当前 Scheduler 对应的全局 TP rank。

    返回：
        ``(attn_cp_rank, moe_dp_rank, moe_ep_rank)``：

        - ``attn_cp_rank``：当前 TP rank 在 Attention CP 维度上的编号。
        - ``moe_dp_rank``：当前 TP rank 所属的 MoE DP 组编号。
        - ``moe_ep_rank``：当前 TP rank 在所属 MoE DP 组内的 EP 编号。
    """

    # 只有启用 DP Attention 时才按实际 dp_size 划分 Attention DP 维度；
    # 未启用时将该维度视为 1，即所有 TP ranks 属于同一个 Attention DP 组。
    attn_dp_size = server_args.dp_size if server_args.enable_dp_attention else 1

    # 从全局 tp_size 中依次除去 Attention DP 和 CP 维度，得到最内层每个
    # Attention TP 组包含的 rank 数量。
    attn_tp_size = server_args.tp_size // attn_dp_size // server_args.attn_cp_size

    # 先整除 attn_tp_size 跳过最内层 Attention TP 维度，再对 attn_cp_size
    # 取模，得到当前 TP rank 在 Attention CP 维度上的局部编号。
    attn_cp_rank = (tp_rank // attn_tp_size) % server_args.attn_cp_size

    # 每个 MoE DP 组包含 tp_size // moe_dp_size 个全局 TP ranks；整除该组大小
    # 即可得到当前 TP rank 位于第几个 MoE DP 组。
    moe_dp_rank = tp_rank // (server_args.tp_size // server_args.moe_dp_size)

    # 计算当前 TP rank 在所属 MoE DP 组中的 EP 编号。
    moe_ep_rank = (
        # 先对每个 MoE DP 组的大小取模，去掉外层 MoE DP 维度，得到组内 rank。
        tp_rank
        % (server_args.tp_size // server_args.moe_dp_size)
        # 每个 EP 分片包含 tp_size // moe_dp_size // ep_size 个 MoE TP ranks；
        # 用组内 rank 整除该分片大小，即得到当前 rank 所属的 EP 分片编号。
        // (server_args.tp_size // server_args.moe_dp_size // server_args.ep_size)
    )

    # 将三个并行维度上的局部 rank 返回给 Scheduler，用于初始化对应通信组。
    return attn_cp_rank, moe_dp_rank, moe_ep_rank
