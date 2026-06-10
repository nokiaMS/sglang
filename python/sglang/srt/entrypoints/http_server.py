# 本文件是SGLang推理服务器的HTTP入口点，通过FastAPI实现各种HTTP API接口，
# 包括原生API、OpenAI兼容API、Ollama兼容API、Anthropic兼容API等
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

This file implements HTTP APIs for the inference engine via fastapi.
"""

import asyncio
import dataclasses
import logging
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

# 修复Python线程的一个bug
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)


import numpy as np
import requests
import uvicorn
import uvloop
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response, StreamingResponse

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST, DisaggregationMode
from sglang.srt.entrypoints.anthropic.protocol import (
    AnthropicCountTokensRequest,
    AnthropicMessagesRequest,
)
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
from sglang.srt.entrypoints.engine import (
    Engine,
    init_tokenizer_manager,
    run_detokenizer_process,
    run_scheduler_process,
)
from sglang.srt.entrypoints.ollama.protocol import (
    OllamaChatRequest,
    OllamaGenerateRequest,
    OllamaShowRequest,
)
from sglang.srt.entrypoints.ollama.serving import OllamaServing
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ClassifyRequest,
    CompletionRequest,
    DetokenizeRequest,
    EmbeddingRequest,
    ErrorResponse,
    ModelCard,
    ModelList,
    ResponsesRequest,
    ScoringRequest,
    TokenizeRequest,
    V1RerankReqInput,
)
from sglang.srt.entrypoints.openai.serving_classify import OpenAIServingClassify
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from sglang.srt.entrypoints.openai.serving_rerank import OpenAIServingRerank
from sglang.srt.entrypoints.openai.serving_score import OpenAIServingScore
from sglang.srt.entrypoints.openai.serving_tokenize import (
    OpenAIServingDetokenize,
    OpenAIServingTokenize,
)
from sglang.srt.entrypoints.openai.serving_transcription import (
    OpenAIServingTranscription,
)
from sglang.srt.entrypoints.warmup import execute_warmups
from sglang.srt.environ import envs
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.managers.io_struct import (
    AbortReq,
    AttachHiCacheStorageReqInput,
    CheckWeightsReqInput,
    CloseSessionReqInput,
    ConfigureLoggingReq,
    ContinueGenerationReqInput,
    DestroyWeightsUpdateGroupReqInput,
    DumperControlReqInput,
    EmbeddingReqInput,
    GenerateReqInput,
    GetWeightsByNameReqInput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterReqInput,
    OpenSessionReqInput,
    ParseFunctionCallReq,
    PauseGenerationReqInput,
    ProfileReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    SendWeightsToRemoteInstanceReqInput,
    SeparateReasoningReqInput,
    SetInternalStateReq,
    SlowDownReqInput,
    UnloadLoRAAdapterReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightVersionReqInput,
    VertexGenerateReqInput,
)
from sglang.srt.managers.multi_tokenizer_mixin import (
    MultiTokenizerRouter,
    TokenizerWorker,
    get_main_process_id,
    read_from_shared_memory,
    write_data_for_multi_tokenizer,
)
from sglang.srt.managers.template_manager import TemplateManager
from sglang.srt.managers.tokenizer_manager import ServerStatus, TokenizerManager
from sglang.srt.observability.func_timer import enable_func_timer
from sglang.srt.observability.trace import (
    process_tracing_init,
    set_global_trace_level,
    trace_set_thread_info,
)
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    add_prometheus_middleware,
    add_prometheus_track_response_middleware,
    delete_directory,
    get_bool_env_var,
    is_mps,
    kill_process_tree,
    set_uvicorn_logging_configs,
)
from sglang.srt.utils.auth import AuthLevel, app_has_admin_force_endpoints, auth_level
from sglang.srt.utils.json_response import (
    SGLangORJSONResponse,
    dumps_json,
    orjson_response,
)
from sglang.srt.utils.watchdog import SubprocessWatchdog
from sglang.utils import get_exception_traceback
from sglang.version import __version__

logger = logging.getLogger(__name__)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# 全局常量
HEALTH_CHECK_TIMEOUT = int(os.getenv("SGLANG_HEALTH_CHECK_TIMEOUT", 20))
WAIT_WEIGHTS_READY_TIMEOUT = int(os.getenv("SGLANG_WAIT_WEIGHTS_READY_TIMEOUT", 120))


# 存储全局状态的数据类
@dataclasses.dataclass
class _GlobalState:
    tokenizer_manager: Union[TokenizerManager, MultiTokenizerRouter, TokenizerWorker]
    template_manager: TemplateManager
    scheduler_info: Dict


# 全局状态实例
_global_state: Optional[_GlobalState] = None


# 设置全局状态
def set_global_state(global_state: _GlobalState):
    global _global_state
    _global_state = global_state


# 获取全局状态
def get_global_state() -> _GlobalState:
    return _global_state


# 初始化Granian工作进程
async def _init_granian_worker() -> ServerArgs:
    main_pid = get_main_process_id()
    port_args, server_args, scheduler_info = read_from_shared_memory(
        f"multi_tokenizer_args_{main_pid}"
    )

    tokenizer_manager = TokenizerManager(server_args, port_args)
    template_manager = TemplateManager()
    template_manager.initialize_templates(
        tokenizer_manager=tokenizer_manager,
        model_path=server_args.model_path,
        chat_template=server_args.chat_template,
        completion_template=server_args.completion_template,
    )
    tokenizer_manager.max_req_input_len = scheduler_info["max_req_input_len"]

    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            template_manager=template_manager,
            scheduler_info=scheduler_info,
        )
    )
    return server_args


# 多分词器模式的初始化函数，从共享内存读取参数并初始化分词器管理器
async def init_multi_tokenizer() -> ServerArgs:
    """
    Initialization function for multi-process tokenizer mode.
    It read args information from shm and inits tokenizer manager for current process.
    """

    # 从共享内存读取配置
    main_pid = get_main_process_id()
    port_args, server_args, scheduler_info = read_from_shared_memory(
        f"multi_tokenizer_args_{main_pid}"
    )
    server_args: ServerArgs
    port_args: PortArgs

    # 多分词器模式不支持API密钥认证
    assert (
        server_args.api_key is None
    ), "API key is not supported in multi-tokenizer mode"

    # 为当前进程创建新的IPC名称
    port_args.tokenizer_ipc_name = (
        f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
    )
    logger.info(
        f"Start multi-tokenizer worker process {os.getpid()}, "
        f"ipc_name={port_args.tokenizer_ipc_name}"
    )

    # 启动多分词器管理器进程
    tokenizer_manager = TokenizerWorker(server_args, port_args)
    template_manager = TemplateManager()
    template_manager.initialize_templates(
        tokenizer_manager=tokenizer_manager,
        model_path=server_args.model_path,
        chat_template=server_args.chat_template,
        completion_template=server_args.completion_template,
    )

    tokenizer_manager.max_req_input_len = scheduler_info["max_req_input_len"]

    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            template_manager=template_manager,
            scheduler_info=scheduler_info,
        )
    )

    return server_args


# FastAPI应用的生命周期管理上下文
@asynccontextmanager
async def lifespan(fast_api_app: FastAPI):
    # 根据运行模式选择初始化方式
    if getattr(fast_api_app, "is_single_tokenizer_mode", False):
        server_args = fast_api_app.server_args
        warmup_thread_kwargs = fast_api_app.warmup_thread_kwargs
        thread_label = "Tokenizer"
    elif envs.SGLANG_GRANIAN_PARENT_PID.get() is not None:
        server_args = await _init_granian_worker()
        warmup_thread_kwargs = dict(server_args=server_args)
        thread_label = "Tokenizer"
    else:
        # 为工作进程初始化多分词器支持
        server_args = await init_multi_tokenizer()
        warmup_thread_kwargs = dict(server_args=server_args)
        thread_label = f"MultiTokenizer-{_global_state.tokenizer_manager.worker_id}"

    # 添加Prometheus中间件
    if server_args.enable_metrics:
        add_prometheus_middleware(app)
        enable_func_timer()

    # 初始化链路追踪
    if server_args.enable_trace:
        process_tracing_init(server_args.otlp_traces_endpoint, "sglang")
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill" + thread_label
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode" + thread_label
        trace_set_thread_info(thread_label)

    # 初始化OpenAI服务处理器
    fast_api_app.state.openai_serving_completion = OpenAIServingCompletion(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_chat = (
        _global_state.tokenizer_manager.serving_chat_class(
            _global_state.tokenizer_manager, _global_state.template_manager
        )
    )
    fast_api_app.state.openai_serving_embedding = OpenAIServingEmbedding(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_classify = OpenAIServingClassify(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_score = OpenAIServingScore(
        _global_state.tokenizer_manager
    )
    fast_api_app.state.openai_serving_rerank = OpenAIServingRerank(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_tokenize = OpenAIServingTokenize(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_detokenize = OpenAIServingDetokenize(
        _global_state.tokenizer_manager
    )
    fast_api_app.state.openai_serving_transcription = OpenAIServingTranscription(
        _global_state.tokenizer_manager
    )

    # 初始化Ollama兼容服务处理器
    fast_api_app.state.ollama_serving = OllamaServing(_global_state.tokenizer_manager)

    # 初始化Anthropic兼容服务处理器
    fast_api_app.state.anthropic_serving = AnthropicServing(
        fast_api_app.state.openai_serving_chat
    )

    # 启动工具服务器
    tool_server = None
    if server_args.tool_server == "demo":
        from sglang.srt.entrypoints.openai.tool_server import DemoToolServer

        tool_server = DemoToolServer()
    elif server_args.tool_server:
        from sglang.srt.entrypoints.openai.tool_server import MCPToolServer

        tool_server = MCPToolServer()
        await tool_server.add_tool_server(server_args.tool_server)

    # 初始化OpenAI Responses API服务处理器
    try:
        from sglang.srt.entrypoints.openai.serving_responses import (
            OpenAIServingResponses,
        )

        fast_api_app.state.openai_serving_responses = OpenAIServingResponses(
            _global_state.tokenizer_manager,
            _global_state.template_manager,
            enable_prompt_tokens_details=True,
            tool_server=tool_server,
        )
    except Exception:
        traceback = get_exception_traceback()
        logger.warning(f"Can not initialize OpenAIServingResponses, error: {traceback}")

    # 执行自定义预热
    if server_args.warmups is not None:
        await execute_warmups(
            server_args.disaggregation_mode,
            server_args.warmups.split(","),
            _global_state.tokenizer_manager,
        )
        logger.info("Warmup ended")

    # 执行通用预热
    warmup_thread = threading.Thread(
        target=_wait_and_warmup,
        kwargs=warmup_thread_kwargs,
    )
    warmup_thread.start()

    # 启动HTTP服务器
    try:
        yield
    finally:
        warmup_thread.join()


# FastAPI应用实例
app = FastAPI(
    lifespan=lifespan,
    openapi_url=None if get_bool_env_var("DISABLE_OPENAPI_DOC") else "/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 包含路由
from sglang.srt.entrypoints.v1_loads import router as v1_loads_router

app.include_router(v1_loads_router)


# HTTP异常处理器，丰富错误响应
@app.exception_handler(HTTPException)
async def validation_exception_handler(request: Request, exc: HTTPException):
    """Enrich HTTP exception with status code and other details.

    For /v1/responses, emit OpenAI-style nested error envelope:
    {"error": {"message": "...", "type": "...", "param": null, "code": <status>}}
    """
    # 为responses API调整格式
    if request.url.path.startswith("/v1/responses"):
        nested_error = {
            "message": exc.detail,
            "type": HTTPStatus(exc.status_code).phrase,
            "param": None,
            "code": exc.status_code,
        }
        return ORJSONResponse(
            content={"error": nested_error}, status_code=exc.status_code
        )

    error = ErrorResponse(
        object="error",
        message=exc.detail,
        type=str(exc.status_code),
        code=exc.status_code,
    )
    return ORJSONResponse(content=error.model_dump(), status_code=exc.status_code)


# 自定义异常处理器，将验证错误状态码从422改为400
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Override FastAPI's default 422 validation error with 400.

    For /v1/responses, emit OpenAI-style nested error envelope; for other endpoints keep legacy format.
    """
    exc_str = str(exc)
    errors_str = str(exc.errors())

    if errors_str and errors_str != exc_str:
        message = f"{exc_str} {errors_str}"
    else:
        message = exc_str

    # 为v1/responses API特别适配（注意error键不同）
    if request.url.path.startswith("/v1/responses"):
        nested_error = {
            "message": message,
            "type": HTTPStatus.BAD_REQUEST.phrase,
            "param": None,
            "code": HTTPStatus.BAD_REQUEST.value,
        }
        return ORJSONResponse(status_code=400, content={"error": nested_error})

    err = ErrorResponse(
        message=message,
        type=HTTPStatus.BAD_REQUEST.phrase,
        code=HTTPStatus.BAD_REQUEST.value,
    )

    return ORJSONResponse(
        status_code=400,
        content=err.model_dump(),
    )


# 验证请求内容类型为application/json
async def validate_json_request(raw_request: Request):
    """Validate that the request content-type is application/json."""
    content_type = raw_request.headers.get("content-type", "").lower()
    media_type = content_type.split(";", maxsplit=1)[0]
    if media_type != "application/json":
        raise RequestValidationError(
            errors=[
                {
                    "loc": ["header", "content-type"],
                    "msg": "Unsupported Media Type: Only 'application/json' is allowed",
                    "type": "value_error",
                }
            ]
        )


##### 原生API端点 #####


# 健康检查端点，通过生成一个token来检查服务器是否健康
@app.get("/health")
@app.get("/health_generate")
async def health_generate(request: Request) -> Response:
    """
    Check the health of the inference server by sending a special request to generate one token.

    If the server is running something, this request will be ignored, so it creates zero overhead.
    If the server is not running anything, this request will be run, so we know whether the server is healthy.
    """

    # 关闭期间收到健康检查请求，返回503
    if _global_state.tokenizer_manager.gracefully_exit:
        logger.info("Health check request received during shutdown. Returning 503.")
        return Response(status_code=503)

    # 服务器正在启动中，返回503
    if _global_state.tokenizer_manager.server_status == ServerStatus.Starting:
        return Response(status_code=503)

    # 如果未启用健康端点生成，直接返回200
    if (
        not envs.SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION.get()
        and request.url.path == "/health"
    ):
        return Response(status_code=200)

    sampling_params = {"max_new_tokens": 1, "temperature": 0.0}
    rid = f"{HEALTH_CHECK_RID_PREFIX}_{time.time()}"

    # 根据引擎类型构建不同的请求
    if _global_state.tokenizer_manager.is_generation:
        gri = GenerateReqInput(
            rid=rid,
            input_ids=[0],
            sampling_params=sampling_params,
            log_metrics=False,
        )
        if (
            _global_state.tokenizer_manager.server_args.disaggregation_mode
            != DisaggregationMode.NULL.value
        ):
            gri.bootstrap_host = FAKE_BOOTSTRAP_HOST
            gri.bootstrap_room = 0
    else:
        gri = EmbeddingReqInput(
            rid=rid, input_ids=[0], sampling_params=sampling_params, log_metrics=False
        )

    async def gen():
        async for _ in _global_state.tokenizer_manager.generate_request(gri, request):
            break

    task = asyncio.create_task(gen())

    # 只要从detokenizer/scheduler收到任何响应，就认为服务器是健康的
    tic = time.time()
    while time.time() < tic + HEALTH_CHECK_TIMEOUT:
        await asyncio.sleep(1)
        if _global_state.tokenizer_manager.last_receive_tstamp > tic:
            task.cancel()
            _global_state.tokenizer_manager.rid_to_state.pop(rid, None)
            _global_state.tokenizer_manager.server_status = ServerStatus.Up
            return Response(status_code=200)

    # 健康检查超时
    task.cancel()
    tic_time = time.strftime("%H:%M:%S", time.localtime(tic))
    last_receive_time = time.strftime(
        "%H:%M:%S", time.localtime(_global_state.tokenizer_manager.last_receive_tstamp)
    )
    logger.error(
        f"Health check failed. Server couldn't get a response from detokenizer for last "
        f"{HEALTH_CHECK_TIMEOUT} seconds. tic start time: {tic_time}. "
        f"last_heartbeat time: {last_receive_time}"
    )
    _global_state.tokenizer_manager.rid_to_state.pop(rid, None)
    _global_state.tokenizer_manager.server_status = ServerStatus.UnHealthy
    return Response(status_code=503)


# 获取模型信息（已弃用，使用/model_info）
@app.get("/get_model_info")
async def get_model_info():
    """Get the model information (deprecated - use /model_info instead)."""
    logger.warning(
        "Endpoint '/get_model_info' is deprecated and will be removed in a future version. "
        "Please use '/model_info' instead."
    )
    return await model_info()


# 获取模型信息
@app.get("/model_info")
async def model_info():
    """Get the model information."""
    model_config = _global_state.tokenizer_manager.model_config
    result = {
        "model_path": _global_state.tokenizer_manager.model_path,
        "tokenizer_path": _global_state.tokenizer_manager.server_args.tokenizer_path,
        "is_generation": _global_state.tokenizer_manager.is_generation,
        "preferred_sampling_params": _global_state.tokenizer_manager.server_args.preferred_sampling_params,
        "weight_version": _global_state.tokenizer_manager.server_args.weight_version,
        "has_image_understanding": model_config.is_image_understandable_model,
        "has_audio_understanding": model_config.is_audio_understandable_model,
        "model_type": getattr(model_config.hf_config, "model_type", None),
        "architectures": getattr(model_config.hf_config, "architectures", None),
        "weight_version": _global_state.tokenizer_manager.server_args.weight_version,
        # "hf_config": model_config.hf_config.to_dict(),
    }
    return result


# 获取权重版本（已弃用，使用/model_info）
@app.get("/get_weight_version")
@app.get("/weight_version")
async def weight_version():
    """Get the current weight version."""
    raise HTTPException(
        status_code=404,
        detail="Endpoint '/get_weight_version' or '/weight_version' is deprecated. Please use '/model_info' instead.",
    )


# 获取服务器信息（已弃用，使用/server_info）
@app.get("/get_server_info")
async def get_server_info():
    """Get the server information (deprecated - use /server_info instead)."""
    logger.warning(
        "Endpoint '/get_server_info' is deprecated and will be removed in a future version. "
        "Please use '/server_info' instead."
    )
    return await server_info()


# 获取服务器信息，包括内部状态和配置
@app.get("/server_info")
async def server_info():
    """Get the server information."""
    # 返回每个DP的内部状态
    internal_states: List[Dict[Any, Any]] = (
        await _global_state.tokenizer_manager.get_internal_state()
    )

    server_args = _global_state.tokenizer_manager.server_args

    # server_args.model_config不可序列化，但应被asdict排除
    return {
        **dataclasses.asdict(server_args),
        **_global_state.scheduler_info,
        "internal_states": internal_states,
        "version": __version__,
        # 用于KV感知路由器的结构化KV事件发布器描述符。
        # 当发布被禁用或配置错误时为None；参见
        # `ServerArgs.describe_kv_events_publisher`获取精确约定。
        "kv_events": server_args.describe_kv_events_publisher(),
    }


# 获取负载指标（已弃用，使用/v1/loads）
@app.get("/get_load")
async def get_load():
    """Get load metrics (deprecated - use /v1/loads instead).

    Legacy shim backed by /v1/loads. Projects GetLoadsReqOutput down to the
    historical field shape (dp_rank, num_reqs, num_waiting_reqs, num_tokens,
    num_pending_tokens, ts_tic) so existing clients keep working.
    """
    logger.warning(
        "Endpoint '/get_load' is deprecated and will be removed in a future version. "
        "Please use '/v1/loads' instead."
    )
    load_results = await _global_state.tokenizer_manager.get_loads(include=["core"])
    ts = time.perf_counter()
    return [
        {
            "dp_rank": r.dp_rank,
            "num_reqs": r.num_running_reqs + r.num_waiting_reqs,
            "num_waiting_reqs": r.num_waiting_reqs,
            "num_tokens": r.num_total_tokens,
            "num_pending_tokens": r.num_total_tokens - r.num_used_tokens,
            "ts_tic": ts,
        }
        for r in load_results
    ]


# 示例用法：
# curl -s -X POST http://localhost:30000/set_internal_state -H "Content-Type: application/json" -d '{"server_args": {"pp_max_micro_batch_size": 8}}'
# 设置内部状态的端点
@app.api_route("/set_internal_state", methods=["POST", "PUT"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def set_internal_state(obj: SetInternalStateReq, request: Request):
    res = await _global_state.tokenizer_manager.set_internal_state(obj)
    return res


# Dumper控制端点（避免导入dumper.py以减少依赖）
if os.environ.get("DUMPER_SERVER_PORT") == "reuse":

    @app.api_route("/dumper/{method}", methods=["POST"])
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def _dumper_control_handler(method: str, request: Request):
        body_bytes = await request.body()
        body = await request.json() if body_bytes else {}
        obj = DumperControlReqInput(method=method, body=body)
        results = await _global_state.tokenizer_manager.dumper_control(obj)
        if any(not r.success for r in results):
            errors = [r.error for r in results if not r.success]
            return ORJSONResponse(status_code=400, content={"error": errors})
        return [x for result in results for x in result.response]


# 生成请求端点，支持流式和非流式响应
@app.api_route(
    "/generate",
    methods=["POST", "PUT"],
    response_class=SGLangORJSONResponse,
)
async def generate_request(obj: GenerateReqInput, request: Request):
    """Handle a generate request."""
    if obj.stream:

        async def stream_results() -> AsyncIterator[bytes]:
            try:
                async for out in _global_state.tokenizer_manager.generate_request(
                    obj, request
                ):
                    yield b"data: " + dumps_json(out) + b"\n\n"
            except ValueError as e:
                out = {"error": {"message": str(e)}}
                logger.error(f"[http_server] Error: {e}")
                yield b"data: " + dumps_json(out) + b"\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
            background=_global_state.tokenizer_manager.create_abort_task(obj),
        )
    else:
        try:
            ret = await _global_state.tokenizer_manager.generate_request(
                obj, request
            ).__anext__()
            return orjson_response(ret)
        except ValueError as e:
            logger.error(f"[http_server] Error: {e}")
            return _create_error_response(e)


# 嵌入请求端点
@app.api_route("/encode", methods=["POST", "PUT"])
async def encode_request(obj: EmbeddingReqInput, request: Request):
    """Handle an embedding request."""
    try:
        ret = await _global_state.tokenizer_manager.generate_request(
            obj, request
        ).__anext__()
        return ret
    except ValueError as e:
        return _create_error_response(e)


# 分类请求端点（奖励模型）
@app.api_route("/classify", methods=["POST", "PUT"])
async def classify_request(obj: EmbeddingReqInput, request: Request):
    """Handle a reward model request. Now the arguments and return values are the same as embedding models."""
    try:
        ret = await _global_state.tokenizer_manager.generate_request(
            obj, request
        ).__anext__()
        return ret
    except ValueError as e:
        return _create_error_response(e)


# 清空缓存端点
@app.api_route("/flush_cache", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def flush_cache(timeout: float = Query(0.0, ge=0.0)):
    """Flush the radix cache."""
    ret = await _global_state.tokenizer_manager.flush_cache(timeout_s=timeout)
    if ret.success:
        content = (
            "Cache flushed.\nPlease check backend logs for more details. "
            "(When there are running or waiting requests, the operation will not be performed.)\n"
        )
    else:
        content = ret.message or "Flush cache failed.\n"
    return Response(
        content=content,
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


# 添加外部语料库（用于ngram投机解码）
@app.post("/add_external_corpus")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def add_external_corpus(request: Request):
    """Add an external corpus for ngram speculative decoding."""
    from sglang.srt.managers.io_struct import AddExternalCorpusReqInput

    try:
        obj = AddExternalCorpusReqInput(**(await request.json()))
    except TypeError as e:
        return ORJSONResponse(
            {"success": False, "message": str(e)},
            status_code=HTTPStatus.BAD_REQUEST,
        )
    result = await _global_state.tokenizer_manager.add_external_corpus(obj)
    return ORJSONResponse(
        {
            "success": result.success,
            "corpus_id": result.corpus_id,
            "message": result.message,
            "loaded_token_count": result.loaded_token_count,
        },
        status_code=200 if result.success else HTTPStatus.BAD_REQUEST,
    )


# 移除外部语料库
@app.post("/remove_external_corpus")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def remove_external_corpus(request: Request):
    """Remove an external corpus by ID."""
    body = await request.json()
    corpus_id = body.get("corpus_id")
    if not corpus_id:
        return ORJSONResponse(
            {"success": False, "message": "corpus_id is required."},
            status_code=HTTPStatus.BAD_REQUEST,
        )
    result = await _global_state.tokenizer_manager.remove_external_corpus(corpus_id)
    return ORJSONResponse(
        {"success": result.success, "message": result.message},
        status_code=200 if result.success else HTTPStatus.BAD_REQUEST,
    )


# 列出所有活跃的外部语料库
@app.get("/list_external_corpora")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def list_external_corpora():
    """List all active external corpora."""
    result = await _global_state.tokenizer_manager.list_external_corpora()
    return ORJSONResponse(
        {
            "success": result.success,
            "corpus_token_counts": result.corpus_token_counts,
            "message": result.message,
        },
        status_code=200 if result.success else HTTPStatus.BAD_REQUEST,
    )


# 清理分层缓存存储后端（已弃用）
@app.api_route("/clear_hicache_storage_backend", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def clear_hicache_storage_backend_deprecated():
    """Deprecated: use POST /hicache/storage-backend/clear."""
    ret = await _global_state.tokenizer_manager.clear_hicache_storage()
    return Response(
        content=(
            "Deprecated endpoint. Use POST /hicache/storage-backend/clear.\n"
            "Hierarchical cache storage backend cleared.\n"
        ),
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


# 清理分层缓存存储后端
@app.api_route("/hicache/storage-backend/clear", methods=["POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def clear_hicache_storage_backend():
    """Clear the hierarchical cache storage backend."""
    ret = await _global_state.tokenizer_manager.clear_hicache_storage()
    return Response(
        content="Hierarchical cache storage backend cleared.\n",
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


# 示例用法：
# curl -s -X PUT http://127.0.0.1:30000/hicache/storage-backend \
#  -H 'Content-Type: application/json' \
#   -d '{
#     "hicache_storage_backend": "file",
#     "hicache_storage_backend_extra_config_json": "{}",
#     "hicache_storage_prefetch_policy": "timeout",
#     "hicache_write_policy": "write_through"
#   }'
# 附加HiCache存储后端
@app.api_route("/hicache/storage-backend", methods=["PUT"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def attach_hicache_storage_backend(obj: AttachHiCacheStorageReqInput):
    """Attach (enable) HiCache storage backend at runtime.

    Only allowed when there are NO running / queued requests.
    """
    if not _global_state.tokenizer_manager.server_args.admin_api_key:
        return _admin_api_key_missing_response()

    ret = await _global_state.tokenizer_manager.attach_hicache_storage(
        hicache_storage_backend=obj.hicache_storage_backend,
        hicache_storage_backend_extra_config_json=obj.hicache_storage_backend_extra_config_json,
        hicache_storage_prefetch_policy=obj.hicache_storage_prefetch_policy,
        hicache_write_policy=obj.hicache_write_policy,
    )
    msg = getattr(ret, "message", "")
    return Response(
        content=(
            (
                "HiCache storage backend attached.\n"
                if ret.success
                else "Failed to attach HiCache storage backend.\n"
            )
            + (msg + "\n" if msg else "")
        ),
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


# 示例用法：
# curl -s -X DELETE http://127.0.0.1:30000/hicache/storage-backend
# 分离HiCache存储后端
@app.api_route("/hicache/storage-backend", methods=["DELETE"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def detach_hicache_storage_backend():
    """Detach (disable) HiCache storage backend at runtime.

    Only allowed when there are NO running / queued requests.
    """
    if not _global_state.tokenizer_manager.server_args.admin_api_key:
        return _admin_api_key_missing_response()

    ret = await _global_state.tokenizer_manager.detach_hicache_storage()
    msg = getattr(ret, "message", "")
    return Response(
        content=(
            (
                "HiCache storage backend detached.\n"
                if ret.success
                else "Failed to detach HiCache storage backend.\n"
            )
            + (msg + "\n" if msg else "")
        ),
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


# 示例用法：
# curl -s http://127.0.0.1:30000/hicache/storage-backend
# 获取HiCache存储后端状态
@app.get("/hicache/storage-backend")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def hicache_storage_backend_status():
    """Get current HiCache storage backend status (tokenizer-side view)."""
    if not _global_state.tokenizer_manager.server_args.admin_api_key:
        return _admin_api_key_missing_response()

    return {
        "hicache_storage_backend": _global_state.tokenizer_manager.server_args.hicache_storage_backend,
        "hicache_storage_backend_extra_config": _global_state.tokenizer_manager.server_args.hicache_storage_backend_extra_config,
        "hicache_storage_prefetch_policy": _global_state.tokenizer_manager.server_args.hicache_storage_prefetch_policy,
        "hicache_write_policy": _global_state.tokenizer_manager.server_args.hicache_write_policy,
    }


# 开始性能分析
@app.api_route("/start_profile", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def start_profile_async(obj: Optional[ProfileReqInput] = None):
    """Start profiling."""
    if obj is None:
        obj = ProfileReqInput()

    await _global_state.tokenizer_manager.start_profile(
        output_dir=obj.output_dir,
        start_step=obj.start_step,
        num_steps=obj.num_steps,
        activities=obj.activities,
        with_stack=obj.with_stack,
        record_shapes=obj.record_shapes,
        profile_by_stage=obj.profile_by_stage,
        merge_profiles=obj.merge_profiles,
        profile_prefix=obj.profile_prefix,
        profile_stages=obj.profile_stages,
    )
    return Response(
        content="Start profiling.\n",
        status_code=200,
    )


# 停止性能分析
@app.api_route("/stop_profile", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def stop_profile_async():
    """Stop profiling."""
    await _global_state.tokenizer_manager.stop_profile()
    return Response(
        content="Stop profiling. This will take some time.\n",
        status_code=200,
    )


# 设置链路追踪级别
@app.api_route("/set_trace_level", methods=["GET", "POST"])
def set_trace_level(level: int = Query(..., ge=0)):
    set_global_trace_level(level)

    return Response(
        content="success",
        status_code=200,
    )


# 冻结垃圾回收
@app.api_route("/freeze_gc", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def freeze_gc_async():
    """
    See engine.freeze_gc for more details.
    """
    await _global_state.tokenizer_manager.freeze_gc()
    return Response(
        content="Garbage collection frozen.\n",
        status_code=200,
    )


# 开始记录专家分布
@app.api_route("/start_expert_distribution_record", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def start_expert_distribution_record_async():
    """Start recording the expert distribution. Clear the previous record if any."""
    await _global_state.tokenizer_manager.start_expert_distribution_record()
    return Response(
        content="Start recording the expert distribution.\n",
        status_code=200,
    )


# 停止记录专家分布
@app.api_route("/stop_expert_distribution_record", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def stop_expert_distribution_record_async():
    """Stop recording the expert distribution."""
    await _global_state.tokenizer_manager.stop_expert_distribution_record()
    return Response(
        content="Stop recording the expert distribution.\n",
        status_code=200,
    )


# 导出专家分布记录
@app.api_route("/dump_expert_distribution_record", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def dump_expert_distribution_record_async():
    """Dump expert distribution record."""
    await _global_state.tokenizer_manager.dump_expert_distribution_record()
    return Response(
        content="Dump expert distribution record.\n",
        status_code=200,
    )


# 从磁盘更新权重，无需重启服务器
@app.post("/update_weights_from_disk")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def update_weights_from_disk(obj: UpdateWeightFromDiskReqInput, request: Request):
    """Update the weights from disk inplace without re-launching the server."""
    success, message, num_paused_requests = (
        await _global_state.tokenizer_manager.update_weights_from_disk(obj, request)
    )

    content = {
        "success": success,
        "message": message,
        "num_paused_requests": num_paused_requests,
    }
    if success:
        return ORJSONResponse(
            content,
            status_code=HTTPStatus.OK,
        )
    else:
        return ORJSONResponse(
            content,
            status_code=HTTPStatus.BAD_REQUEST,
        )


# 为远程实例初始化权重发送组
@app.post("/init_weights_send_group_for_remote_instance")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def init_weights_send_group_for_remote_instance(
    obj: InitWeightsSendGroupForRemoteInstanceReqInput, request: Request
):
    success, message = (
        await _global_state.tokenizer_manager.init_weights_send_group_for_remote_instance(
            obj, request
        )
    )
    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


# 发送权重到远程实例
@app.post("/send_weights_to_remote_instance")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def send_weights_to_remote_instance(
    obj: SendWeightsToRemoteInstanceReqInput, request: Request
):
    success, message = (
        await _global_state.tokenizer_manager.send_weights_to_remote_instance(
            obj, request
        )
    )
    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


# 获取远程实例传输引擎信息（已弃用）
@app.get("/get_remote_instance_transfer_engine_info")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def get_remote_instance_transfer_engine_info(rank: int = None):
    """Get the server information (deprecated - use /remote_instance_transfer_engine_info instead)."""
    logger.warning(
        "Endpoint '/get_remote_instance_transfer_engine_info' is deprecated and will be removed in a future version. "
        "Please use '/remote_instance_transfer_engine_info' instead."
    )
    return await remote_instance_transfer_engine_info(rank=rank)


# 获取远程实例传输引擎信息
@app.get("/remote_instance_transfer_engine_info")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def remote_instance_transfer_engine_info(rank: int = None):
    if rank is None or rank < 0:
        return ORJSONResponse(
            {"error": {"message": "Missing or invalid rank parameter"}},
            status_code=HTTPStatus.BAD_REQUEST,
        )

    server_args = _global_state.tokenizer_manager.server_args
    try:
        resp = requests.get(
            f"{server_args.engine_info_bootstrap_url}/get_transfer_engine_info",
            params={"rank": rank},
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        logger.warning(f"Failed to get transfer engine info for rank {rank}: {e}")

    return ORJSONResponse(
        {"error": {"message": f"Failed to get transfer engine info for rank {rank}"}},
        status_code=HTTPStatus.BAD_REQUEST,
    )


# 初始化参数更新组
@app.post("/init_weights_update_group")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def init_weights_update_group(
    obj: InitWeightsUpdateGroupReqInput, request: Request
):
    """Initialize the parameter update group."""
    success, message = await _global_state.tokenizer_manager.init_weights_update_group(
        obj, request
    )
    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


# 销毁参数更新组
@app.post("/destroy_weights_update_group")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def destroy_weights_update_group(
    obj: DestroyWeightsUpdateGroupReqInput, request: Request
):
    """Destroy the parameter update group."""
    success, message = (
        await _global_state.tokenizer_manager.destroy_weights_update_group(obj, request)
    )
    content = {"success": success, "message": message}
    return ORJSONResponse(
        content, status_code=200 if success else HTTPStatus.BAD_REQUEST
    )


# 通过张量数据更新权重，无需重启服务器
@app.post("/update_weights_from_tensor")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def update_weights_from_tensor(
    obj: UpdateWeightsFromTensorReqInput, request: Request
):
    """Update the weights from tensor inplace without re-launching the server.
    Notes:
    1. Ensure that the model is on the correct device (e.g., GPU) before calling this endpoint. If the model is moved to the CPU unexpectedly, it may cause performance issues or runtime errors.
    2. HTTP will transmit only the metadata of the tensor, while the tensor itself will be directly copied to the model.
    3. Any binary data in the named tensors should be base64 encoded.
    """

    success, message = await _global_state.tokenizer_manager.update_weights_from_tensor(
        obj, request
    )

    content = {"success": success, "message": message}
    return ORJSONResponse(
        content, status_code=200 if success else HTTPStatus.BAD_REQUEST
    )


# 从分布式在线更新模型参数
@app.post("/update_weights_from_distributed")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def update_weights_from_distributed(
    obj: UpdateWeightsFromDistributedReqInput, request: Request
):
    """Update model parameter from distributed online."""
    success, message = (
        await _global_state.tokenizer_manager.update_weights_from_distributed(
            obj, request
        )
    )

    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


# 通过IPC更新权重（用于检查点引擎集成）
@app.post("/update_weights_from_ipc")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def update_weights_from_ipc(obj: UpdateWeightsFromIPCReqInput, request: Request):
    """Update the weights from IPC (Inter-Process Communication) for checkpoint-engine integration."""
    success, message = await _global_state.tokenizer_manager.update_weights_from_ipc(
        obj, request
    )

    content = {"success": success, "message": message}
    if success:
        if _global_state.tokenizer_manager.initial_weights_loaded is False:
            _global_state.tokenizer_manager.initial_weights_loaded = True
        return ORJSONResponse(content)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


# 更新权重版本
@app.post("/update_weight_version")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def update_weight_version(obj: UpdateWeightVersionReqInput, request: Request):
    """Update the weight version. This operation requires no active requests."""
    if obj.abort_all_requests:
        _global_state.tokenizer_manager.abort_request(abort_all=True)

    # 使用简单方法，无需复杂的锁机制
    # 因为weight_version更新是简单操作，不影响模型权重
    try:
        # 在server args中更新权重版本（唯一真实来源）
        _global_state.tokenizer_manager.server_args.weight_version = obj.new_version

        return ORJSONResponse(
            {
                "success": True,
                "message": f"Weight version updated to {obj.new_version}",
                "new_version": obj.new_version,
            },
            status_code=HTTPStatus.OK,
        )
    except Exception as e:
        return ORJSONResponse(
            {
                "success": False,
                "message": f"Failed to update weight version: {str(e)}",
            },
            status_code=HTTPStatus.BAD_REQUEST,
        )


# 按名称获取模型参数
@app.api_route("/get_weights_by_name", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def get_weights_by_name(obj: GetWeightsByNameReqInput, request: Request):
    """Get model parameter by name."""
    try:
        ret = await _global_state.tokenizer_manager.get_weights_by_name(obj, request)
        if ret is None:
            return _create_error_response("Get parameter by name failed")
        else:
            return ORJSONResponse(ret, status_code=200)
    except Exception as e:
        return _create_error_response(e)


# 临时释放GPU显存占用
@app.api_route("/release_memory_occupation", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def release_memory_occupation(
    obj: ReleaseMemoryOccupationReqInput, request: Request
):
    """Release GPU memory occupation temporarily."""
    try:
        await _global_state.tokenizer_manager.release_memory_occupation(obj, request)
    except Exception as e:
        return _create_error_response(e)


# 恢复GPU显存占用
@app.api_route("/resume_memory_occupation", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def resume_memory_occupation(
    obj: ResumeMemoryOccupationReqInput, request: Request
):
    """Resume GPU memory occupation."""
    try:
        await _global_state.tokenizer_manager.resume_memory_occupation(obj, request)
    except Exception as e:
        return _create_error_response(e)


# 权重校验端点
@app.api_route("/weights_checker", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def check_weights(
    obj: Optional[CheckWeightsReqInput] = None, request: Request = None
):
    if obj is None:
        obj = CheckWeightsReqInput()
    success, message, ranks, per_engine_checksum = (
        await _global_state.tokenizer_manager.check_weights(obj, request)
    )
    body = {"success": success, "message": message}
    if ranks is not None:
        body["ranks"] = ranks
    if per_engine_checksum is not None:
        body["per_engine_checksum"] = per_engine_checksum
    return ORJSONResponse(body, status_code=200 if success else HTTPStatus.BAD_REQUEST)


# 人为减速系统（仅用于测试）
@app.api_route("/slow_down", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def slow_down(obj: SlowDownReqInput, request: Request):
    """Slow down the system deliberately. Only for testing. Example scenario:
    when we want to test performance of D in large-scale PD disaggregation and have no enough nodes for P,
    we can use this to slow down D to let it have enough running sequences, and then disable slowdown
    to let it run in full batch size.
    """
    try:
        await _global_state.tokenizer_manager.slow_down(obj, request)
    except Exception as e:
        return _create_error_response(e)


# 加载LoRA适配器，无需重启服务器
@app.api_route("/load_lora_adapter", methods=["POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def load_lora_adapter(obj: LoadLoRAAdapterReqInput, request: Request):
    """Load a new LoRA adapter without re-launching the server."""
    result = await _global_state.tokenizer_manager.load_lora_adapter(obj, request)

    if result.success:
        return ORJSONResponse(
            result,
            status_code=HTTPStatus.OK,
        )
    else:
        return ORJSONResponse(
            result,
            status_code=HTTPStatus.BAD_REQUEST,
        )


# 从张量加载LoRA适配器，无需重启服务器
@app.api_route("/load_lora_adapter_from_tensors", methods=["POST"])
async def load_lora_adapter_from_tensors(
    obj: LoadLoRAAdapterFromTensorsReqInput, request: Request
):
    """Load a new LoRA adapter from tensors without re-launching the server."""
    result = await _global_state.tokenizer_manager.load_lora_adapter_from_tensors(
        obj, request
    )

    if result.success:
        return ORJSONResponse(result, status_code=HTTPStatus.OK)
    else:
        return ORJSONResponse(result, status_code=HTTPStatus.BAD_REQUEST)


# 卸载LoRA适配器，无需重启服务器
@app.api_route("/unload_lora_adapter", methods=["POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def unload_lora_adapter(obj: UnloadLoRAAdapterReqInput, request: Request):
    """Load a new LoRA adapter without re-launching the server."""
    result = await _global_state.tokenizer_manager.unload_lora_adapter(obj, request)

    if result.success:
        return ORJSONResponse(
            result,
            status_code=HTTPStatus.OK,
        )
    else:
        return ORJSONResponse(
            result,
            status_code=HTTPStatus.BAD_REQUEST,
        )


# 打开会话，返回唯一的会话ID
@app.api_route("/open_session", methods=["GET", "POST"])
async def open_session(obj: OpenSessionReqInput, request: Request):
    """Open a session, and return its unique session id."""
    try:
        session_id = await _global_state.tokenizer_manager.open_session(obj, request)
        if session_id is None:
            raise Exception(
                "Failed to open the session. Check if a session with the same id is still open."
            )
        return session_id
    except Exception as e:
        return _create_error_response(e)


# 关闭会话
@app.api_route("/close_session", methods=["GET", "POST"])
async def close_session(obj: CloseSessionReqInput, request: Request):
    """Close the session."""
    try:
        await _global_state.tokenizer_manager.close_session(obj, request)
        return Response(status_code=200)
    except Exception as e:
        return _create_error_response(e)


# 配置请求日志选项
@app.api_route("/configure_logging", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def configure_logging(obj: ConfigureLoggingReq, request: Request):
    """Configure the request logging options."""
    _global_state.tokenizer_manager.configure_logging(obj)
    return Response(status_code=200)


# 中止请求
@app.post("/abort_request")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def abort_request(obj: AbortReq, request: Request):
    """Abort a request."""
    try:
        _global_state.tokenizer_manager.abort_request(
            rid=obj.rid, abort_all=obj.abort_all
        )
        return Response(status_code=200)
    except Exception as e:
        return _create_error_response(e)


# 解析函数调用的原生API端点
@app.post("/parse_function_call")
async def parse_function_call_request(obj: ParseFunctionCallReq, request: Request):
    """
    A native API endpoint to parse function calls from a text.
    """
    # 1) 根据请求体初始化解析器
    parser = FunctionCallParser(tools=obj.tools, tool_call_parser=obj.tool_call_parser)

    # 2) 调用非流式解析方法
    normal_text, calls = parser.parse_non_stream(obj.text)

    # 3) 组织响应内容
    response_data = {
        "normal_text": normal_text,
        "calls": [
            call.model_dump() for call in calls
        ],  # 将pydantic对象转换为字典
    }

    return ORJSONResponse(content=response_data, status_code=200)


# 分离推理文本的原生API端点
@app.post("/separate_reasoning")
async def separate_reasoning_request(obj: SeparateReasoningReqInput, request: Request):
    """
    A native API endpoint to separate reasoning from a text.
    """
    # 1) 根据请求体初始化解析器
    parser = ReasoningParser(model_type=obj.reasoning_parser, request=request)

    # 2) 调用非流式解析方法
    reasoning_text, normal_text = parser.parse_non_stream(obj.text)

    # 3) 组织响应内容
    response_data = {
        "reasoning_text": reasoning_text,
        "text": normal_text,
    }

    return ORJSONResponse(content=response_data, status_code=200)


# 暂停生成
@app.post("/pause_generation")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def pause_generation(obj: PauseGenerationReqInput, request: Request):
    """Pause generation."""
    await _global_state.tokenizer_manager.pause_generation(obj)
    return ORJSONResponse(
        content={"message": "Generation paused successfully.", "status": "ok"},
        status_code=200,
    )


# 继续生成
@app.post("/continue_generation")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def continue_generation(obj: ContinueGenerationReqInput, request: Request):
    """Continue generation."""
    await _global_state.tokenizer_manager.continue_generation(obj)
    return ORJSONResponse(
        content={"message": "Generation continued successfully.", "status": "ok"},
        status_code=200,
    )


##### OpenAI兼容API端点 #####


# OpenAI兼容的文本补全端点
@app.post("/v1/completions", dependencies=[Depends(validate_json_request)])
async def openai_v1_completions(request: CompletionRequest, raw_request: Request):
    """OpenAI-compatible text completion endpoint."""
    return await raw_request.app.state.openai_serving_completion.handle_request(
        request, raw_request
    )


# OpenAI兼容的聊天补全端点
@app.post("/v1/chat/completions", dependencies=[Depends(validate_json_request)])
async def openai_v1_chat_completions(
    request: ChatCompletionRequest, raw_request: Request
):
    """OpenAI-compatible chat completion endpoint."""
    return await raw_request.app.state.openai_serving_chat.handle_request(
        request, raw_request
    )


# OpenAI兼容的嵌入端点
@app.post(
    "/v1/embeddings",
    response_class=ORJSONResponse,
    dependencies=[Depends(validate_json_request)],
)
async def openai_v1_embeddings(request: EmbeddingRequest, raw_request: Request):
    """OpenAI-compatible embeddings endpoint."""
    return await raw_request.app.state.openai_serving_embedding.handle_request(
        request, raw_request
    )


# OpenAI兼容的分类端点
@app.post(
    "/v1/classify",
    response_class=ORJSONResponse,
    dependencies=[Depends(validate_json_request)],
)
async def openai_v1_classify(request: ClassifyRequest, raw_request: Request):
    """OpenAI-compatible classification endpoint."""
    return await raw_request.app.state.openai_serving_classify.handle_request(
        request, raw_request
    )


# OpenAI兼容的分词端点
@app.post(
    "/v1/tokenize",
    response_class=ORJSONResponse,
    dependencies=[Depends(validate_json_request)],
)
@app.post(
    "/tokenize",
    response_class=ORJSONResponse,
    dependencies=[Depends(validate_json_request)],
    include_in_schema=False,
)
async def openai_v1_tokenize(request: TokenizeRequest, raw_request: Request):
    """OpenAI-compatible tokenization endpoint."""
    return await raw_request.app.state.openai_serving_tokenize.handle_request(
        request, raw_request
    )


# OpenAI兼容的反分词端点
@app.post(
    "/v1/detokenize",
    response_class=ORJSONResponse,
    dependencies=[Depends(validate_json_request)],
)
@app.post(
    "/detokenize",
    response_class=ORJSONResponse,
    dependencies=[Depends(validate_json_request)],
    include_in_schema=False,
)
async def openai_v1_detokenize(request: DetokenizeRequest, raw_request: Request):
    """OpenAI-compatible detokenization endpoint."""
    return await raw_request.app.state.openai_serving_detokenize.handle_request(
        request, raw_request
    )


# OpenAI兼容的音频转录端点
@app.post("/v1/audio/transcriptions")
async def openai_v1_audio_transcriptions(
    raw_request: Request,
    file: UploadFile = File(...),
    model: str = Form(default="default"),
    language: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    stream: bool = Form(default=False),
    timestamp_granularities: Optional[List[str]] = Form(
        default=None, alias="timestamp_granularities[]"
    ),
):
    """OpenAI-compatible audio transcription endpoint."""
    if response_format not in ["json", "text", "verbose_json"]:
        return ORJSONResponse(
            content={
                "error": {
                    "message": "Only 'json', 'text', and 'verbose_json' formats supported"
                }
            },
            status_code=400,
        )

    audio_data = await file.read()

    return (
        await raw_request.app.state.openai_serving_transcription.create_transcription(
            audio_data=audio_data,
            model=model,
            language=language,
            response_format=response_format,
            temperature=temperature,
            stream=stream,
            timestamp_granularities=timestamp_granularities,
            raw_request=raw_request,
        )
    )


# OpenAI实时转录WebSocket端点
@app.websocket("/v1/realtime")
async def openai_v1_realtime_transcription(ws: WebSocket):
    """OpenAI Realtime transcription WebSocket endpoint."""
    # /v1/realtime是OpenAI的统一实时URL，涵盖转录+
    # 聊天模式。此处理器仅实现转录子集；
    # 聊天模式的session.update载荷会被
    # TranscriptionSessionConfig.type上的`Literal["transcription"]`约束拒绝
    # （参见realtime/protocol.py）。
    await ws.app.state.openai_serving_transcription.handle_websocket(ws)


# 列出可用模型（OpenAI兼容）
@app.get("/v1/models", response_class=ORJSONResponse)
async def available_models():
    """Show available models. OpenAI-compatible endpoint."""
    served_model_names = [_global_state.tokenizer_manager.served_model_name]
    model_cards = []

    # 添加基础模型
    for served_model_name in served_model_names:
        model_cards.append(
            ModelCard(
                id=served_model_name,
                root=served_model_name,
                max_model_len=_global_state.tokenizer_manager.model_config.context_len,
            )
        )

    # 添加已加载的LoRA适配器
    if _global_state.tokenizer_manager.server_args.enable_lora:
        lora_registry = _global_state.tokenizer_manager.lora_registry
        for _, lora_ref in lora_registry.get_all_adapters().items():
            model_cards.append(
                ModelCard(
                    id=lora_ref.lora_name,
                    root=lora_ref.lora_path,
                    parent=served_model_names[0],
                    max_model_len=None,
                )
            )

    return ModelList(data=model_cards)


# 检索特定模型信息（OpenAI兼容）
@app.get("/v1/models/{model:path}", response_class=ORJSONResponse)
async def retrieve_model(model: str):
    """Retrieves a model instance, providing basic information about the model."""
    served_model_names = [_global_state.tokenizer_manager.served_model_name]

    if model not in served_model_names:
        return ORJSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"The model '{model}' does not exist",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )

    return ModelCard(
        id=model,
        root=model,
        max_model_len=_global_state.tokenizer_manager.model_config.context_len,
    )


# 评分API端点
@app.post("/v1/score", dependencies=[Depends(validate_json_request)])
async def v1_score_request(request: ScoringRequest, raw_request: Request):
    """Endpoint for the scoring API. Supports CausalLM (logprob-based) and SequenceClassification (class logit-based) models. See Engine.score() for documentation."""
    return await raw_request.app.state.openai_serving_score.handle_request(
        request, raw_request
    )


# 带推理支持的Responses API端点
@app.post("/v1/responses", dependencies=[Depends(validate_json_request)])
async def v1_responses_request(request: dict, raw_request: Request):
    """Endpoint for the responses API with reasoning support."""

    request_obj = ResponsesRequest(**request)
    result = await raw_request.app.state.openai_serving_responses.create_responses(
        request_obj, raw_request
    )

    # 处理流式响应
    if isinstance(result, AsyncGenerator):
        return StreamingResponse(
            result,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    return result


# 通过ID检索响应
@app.get("/v1/responses/{response_id}")
async def v1_retrieve_responses(response_id: str, raw_request: Request):
    """Retrieve a response by ID."""
    return await raw_request.app.state.openai_serving_responses.retrieve_responses(
        response_id
    )


# 取消后台响应
@app.post("/v1/responses/{response_id}/cancel")
async def v1_cancel_responses(response_id: str, raw_request: Request):
    """Cancel a background response."""
    return await raw_request.app.state.openai_serving_responses.cancel_responses(
        response_id
    )


# 重排序API端点
@app.api_route(
    "/v1/rerank", methods=["POST", "PUT"], dependencies=[Depends(validate_json_request)]
)
async def v1_rerank_request(request: V1RerankReqInput, raw_request: Request):
    """Endpoint for reranking documents based on query relevance."""
    return await raw_request.app.state.openai_serving_rerank.handle_request(
        request, raw_request
    )


##### Ollama兼容API端点 #####

_ollama_root_route = os.environ.get("SGLANG_OLLAMA_ROOT_ROUTE")
if _ollama_root_route is not None:

    # Ollama兼容的根端点
    @app.get(_ollama_root_route)
    @app.head(_ollama_root_route)
    async def ollama_root():
        """Ollama-compatible root endpoint."""
        return "Ollama is running"

else:

    # 默认根端点
    @app.get("/")
    @app.head("/")
    async def sglang_root():
        """Default root endpoint."""
        return "SGLang is running"


# Ollama兼容的聊天端点
@app.post(os.environ.get("SGLANG_OLLAMA_CHAT_ROUTE", "/api/chat"))
async def ollama_chat(request: OllamaChatRequest, raw_request: Request):
    """Ollama-compatible chat endpoint."""
    return await raw_request.app.state.ollama_serving.handle_chat(request, raw_request)


# Ollama兼容的生成端点
@app.post(os.environ.get("SGLANG_OLLAMA_GENERATE_ROUTE", "/api/generate"))
async def ollama_generate(request: OllamaGenerateRequest, raw_request: Request):
    """Ollama-compatible generate endpoint."""
    return await raw_request.app.state.ollama_serving.handle_generate(
        request, raw_request
    )


# Ollama兼容的列出模型端点
@app.get(os.environ.get("SGLANG_OLLAMA_TAGS_ROUTE", "/api/tags"))
async def ollama_tags(raw_request: Request):
    """Ollama-compatible list models endpoint."""
    return raw_request.app.state.ollama_serving.get_tags()


# Ollama兼容的显示模型信息端点
@app.post(os.environ.get("SGLANG_OLLAMA_SHOW_ROUTE", "/api/show"))
async def ollama_show(request: OllamaShowRequest, raw_request: Request):
    """Ollama-compatible show model info endpoint."""
    return raw_request.app.state.ollama_serving.get_show(request.model)


##### Anthropic兼容API端点 #####


# Anthropic兼容的Messages API端点
@app.post("/v1/messages", dependencies=[Depends(validate_json_request)])
async def anthropic_v1_messages(
    request: AnthropicMessagesRequest, raw_request: Request
):
    """Anthropic-compatible Messages API endpoint."""
    return await raw_request.app.state.anthropic_serving.handle_messages(
        request, raw_request
    )


# Anthropic兼容的token计数端点
@app.post("/v1/messages/count_tokens", dependencies=[Depends(validate_json_request)])
async def anthropic_v1_count_tokens(
    request: AnthropicCountTokensRequest, raw_request: Request
):
    """Anthropic-compatible token counting endpoint."""
    return await raw_request.app.state.anthropic_serving.handle_count_tokens(
        request, raw_request
    )


## SageMaker API
# SageMaker健康检查端点
@app.get("/ping")
async def sagemaker_health() -> Response:
    """Check the health of the http server."""
    return Response(status_code=200)


# SageMaker推理端点
@app.post("/invocations")
async def sagemaker_chat_completions(
    request: ChatCompletionRequest, raw_request: Request
):
    """OpenAI-compatible chat completion endpoint."""
    return await raw_request.app.state.openai_serving_chat.handle_request(
        request, raw_request
    )


## Vertex AI API
# Vertex AI预测端点
@app.post(os.environ.get("AIP_PREDICT_ROUTE", "/vertex_generate"))
async def vertex_generate(vertex_req: VertexGenerateReqInput, raw_request: Request):
    if not vertex_req.instances:
        return []
    inputs = {}
    for input_key in ("text", "input_ids", "input_embeds"):
        if vertex_req.instances[0].get(input_key):
            inputs[input_key] = [
                instance.get(input_key) for instance in vertex_req.instances
            ]
            break
    image_data = [
        instance.get("image_data")
        for instance in vertex_req.instances
        if instance.get("image_data") is not None
    ] or None
    req = GenerateReqInput(
        **inputs,
        image_data=image_data,
        **(vertex_req.parameters or {}),
    )
    ret = await generate_request(req, raw_request)
    if isinstance(ret, Response):
        return ret
    return ORJSONResponse({"predictions": ret})


# 创建错误响应
def _create_error_response(e):
    return ORJSONResponse(
        {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
    )


# FIXME: 理论上我们应该为某些端点配置ADMIN_FORCE，但这样做
# 目前会导致所有端点都通过add_api_key_middleware
# （即使没有配置api-key或admin-api-key）。
#
# 目前，我们通过显式检查admin API key参数来模拟ADMIN_FORCE。
# 一旦认证布线重构，使ADMIN_FORCE仅影响预期的
# 管理端点，我们应该将此逻辑切换为直接使用ADMIN_FORCE。
# 管理API密钥缺失时的响应
def _admin_api_key_missing_response(
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
) -> ORJSONResponse:
    return ORJSONResponse(
        content={
            "error": (
                "This endpoint requires admin API key, but this server was started "
                "without one (admin-api-key). Restart with --admin-api-key to enable."
            )
        },
        status_code=status_code,
    )


# 最小32x32黑色PNG图片（base64编码，GLM4v要求至少32x32大小的图片）
MINIMUM_PNG_PICTURE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGBcua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="


# 执行服务器预热
def _execute_server_warmup(server_args: ServerArgs):
    headers = {}
    url = server_args.url()
    if server_args.api_key:
        headers["Authorization"] = f"Bearer {server_args.api_key}"

    ssl_verify = server_args.ssl_verify()

    # 等待服务器启动
    success = False
    for _ in range(120):
        time.sleep(1)
        try:
            res = requests.get(
                url + "/model_info", timeout=5, headers=headers, verify=ssl_verify
            )
            assert res.status_code == 200, f"{res=}, {res.text=}"
            success = True
            break
        except (AssertionError, requests.exceptions.RequestException):
            last_traceback = get_exception_traceback()
            pass

    if not success:
        logger.error(f"Initialization failed. warmup error: {last_traceback}")
        kill_process_tree(os.getpid())
        return success

    model_info = res.json()

    # 构建预热请求（MLX：对VLM-advertising模型使用文本预热；TODO：启用图像预热）
    is_vlm = bool(model_info.get("has_image_understanding", False)) and not is_mps()
    if model_info["is_generation"]:
        if is_vlm and not server_args.skip_tokenizer_init:
            request_name = "/v1/chat/completions"
        else:
            request_name = "/generate"
    else:
        request_name = "/encode"
    max_new_tokens = 8 if model_info["is_generation"] else 1
    json_data = {
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
    }
    if server_args.skip_tokenizer_init:
        json_data["input_ids"] = [[10, 11, 12] for _ in range(server_args.dp_size)]
        # TODO 临时解决大小为1的列表导致嵌入错误的bug
        if server_args.dp_size == 1:
            json_data["input_ids"] = json_data["input_ids"][0]
    elif (
        is_vlm
        and server_args.disaggregation_mode == "null"
        and model_info["is_generation"]
    ):
        # TODO：ChatCompletionRequest没有disaggregation模式所需的bootstrap信息，暂时禁用图像预热
        # 仅对生成模型使用聊天补全格式，不对嵌入模型使用
        json_data = {
            "model": _global_state.tokenizer_manager.served_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{MINIMUM_PNG_PICTURE_BASE64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": "Describe the image.",
                        },
                    ],
                }
            ],
            "max_tokens": max_new_tokens,
            "stream": False,
            "temperature": 0.0,
        }
    else:
        json_data["text"] = ["The capital city of France is"] * server_args.dp_size
        # TODO 临时解决大小为1的列表导致嵌入错误的bug
        if server_args.dp_size == 1:
            json_data["text"] = json_data["text"][0]

    # 配置调试转储
    if server_args.debug_tensor_dump_input_file:
        json_data.pop("text", None)
        json_data["input_ids"] = np.load(
            server_args.debug_tensor_dump_input_file
        ).tolist()
        json_data["sampling_params"]["max_new_tokens"] = 0

    # 发送预热请求
    warmup_timeout = envs.SGLANG_WARMUP_TIMEOUT.get()
    try:
        if server_args.disaggregation_mode == "null":
            res = requests.post(
                url + request_name,
                json=json_data,
                headers=headers,
                timeout=warmup_timeout if warmup_timeout > 0 else 600,
                verify=ssl_verify,
            )
            assert res.status_code == 200, f"{res.text}"
            _global_state.tokenizer_manager.server_status = ServerStatus.Up

        else:
            logger.info(f"Start of pd disaggregation warmup ...")
            request_name = "/generate"
            json_data = {
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 8,
                    "ignore_eos": True,
                },
                "bootstrap_host": [FAKE_BOOTSTRAP_HOST] * server_args.dp_size,
                # 这是一个hack，确保在预填充预热期间启用fake transfer
                # 确保每个dp rank在预填充预热期间有唯一的bootstrap_room
                "bootstrap_room": [
                    i * (2**63 // server_args.dp_size) + (i % server_args.tp_size)
                    for i in range(server_args.dp_size)
                ],
                "input_ids": [[10, 11, 12, 13]] * server_args.dp_size,
            }
            res = requests.post(
                url + request_name,
                json=json_data,
                headers=headers,
                timeout=(
                    warmup_timeout if warmup_timeout > 0 else 1800
                ),  # 因为deep gemm预缓存如果未预缓存会非常长
                verify=ssl_verify,
            )
            if res.status_code == 200:
                logger.info(
                    f"Disaggregation warmup request completed with status {res.status_code}, resp: {res.json()}"
                )
                logger.info("End of disaggregation warmup")
                _global_state.tokenizer_manager.server_status = ServerStatus.Up
            else:
                logger.info(
                    "Disaggregation warmup failed (mode=%s), status code: %s",
                    server_args.disaggregation_mode,
                    res.status_code,
                )
                _global_state.tokenizer_manager.server_status = ServerStatus.UnHealthy

    except Exception:
        last_traceback = get_exception_traceback()
        logger.error(f"Initialization failed. warmup error: {last_traceback}")
        kill_process_tree(os.getpid())
        return False

    # 调试打印
    # logger.info(f"warmup request returns: {res.json()=}")
    return success


# 等待服务器启动并执行预热
def _wait_and_warmup(
    server_args: ServerArgs,
    launch_callback: Optional[Callable[[], None]] = None,
    execute_warmup_func: Callable = _execute_server_warmup,
):
    # 等待权重就绪
    if server_args.checkpoint_engine_wait_weights_before_ready:
        _wait_weights_ready()

    # 发送预热请求
    if not server_args.skip_server_warmup:
        if not execute_warmup_func(server_args):
            return
    else:
        _global_state.tokenizer_manager.server_status = ServerStatus.Up

    # 服务器已准备好接收请求
    logger.info("The server is fired up and ready to roll!")

    # 加载后删除检查点目录
    if server_args.delete_ckpt_after_loading:
        delete_directory(server_args.model_path)

    # 调试转储后终止进程
    if server_args.debug_tensor_dump_input_file:
        kill_process_tree(os.getpid())

    if launch_callback is not None:
        launch_callback()


# 等待权重在指定超时时间内就绪
def _wait_weights_ready():
    """Wait for weights to be ready within the specified timeout."""
    timeout = WAIT_WEIGHTS_READY_TIMEOUT
    start_time = time.time()

    for _ in range(timeout):
        if _global_state.tokenizer_manager.initial_weights_loaded:
            logger.info(
                f"Weights are ready after {time.time() - start_time:.2f} seconds"
            )
            return
        time.sleep(1)

    # 超时未就绪
    logger.error(
        f"Weights are not ready after waiting {timeout} seconds. "
        f"Consider increasing SGLANG_WAIT_WEIGHTS_READY_TIMEOUT environment variable. "
        f"Current status: initial_weights_loaded={_global_state.tokenizer_manager.initial_weights_loaded}"
    )


# 关闭主进程的ZMQ套接字，为生成Granian工作进程做准备
def _close_main_process_sockets():
    """Close the main process's ZMQ sockets before spawning Granian workers.

    Granian workers create their own TokenizerManager with fresh ZMQ sockets.
    The main process must release its sockets first to avoid binding conflicts
    on the same IPC addresses.
    """
    if _global_state is None or _global_state.tokenizer_manager is None:
        return
    tm = _global_state.tokenizer_manager
    for attr in ("recv_from_detokenizer", "send_to_scheduler"):
        sock = getattr(tm, attr, None)
        if sock is None:
            continue
        inner = getattr(sock, "socket", None)
        if inner is not None:
            inner.close()
        elif hasattr(sock, "close"):
            sock.close()
        setattr(tm, attr, None)


# 启动Granian HTTP/2服务器
def _run_granian_server(server_args: ServerArgs):
    """Launch Granian with HTTP/2 support"""
    from granian import Granian
    from granian.constants import HTTPModes, Interfaces, Loops

    granian_kwargs = dict(
        target="sglang.srt.entrypoints.http_server:app",
        address=server_args.host,
        port=server_args.port,
        interface=Interfaces.ASGI,
        http=HTTPModes.auto,
        loop=Loops.uvloop,
        log_level=server_args.log_level_http or server_args.log_level or "info",
        workers=1,
    )

    ssl_enabled = server_args.ssl_certfile and server_args.ssl_keyfile
    if ssl_enabled:
        granian_kwargs["ssl_cert"] = server_args.ssl_certfile
        granian_kwargs["ssl_key"] = server_args.ssl_keyfile

    server = Granian(**granian_kwargs)
    server.serve()


# 设置全局状态、配置中间件并运行HTTP服务器
def _setup_and_run_http_server(
    server_args: ServerArgs,
    tokenizer_manager,
    template_manager,
    port_args: PortArgs,
    scheduler_infos: List[Dict],
    subprocess_watchdog: Optional[SubprocessWatchdog],
    execute_warmup_func: Callable = _execute_server_warmup,
    launch_callback: Optional[Callable[[], None]] = None,
):
    """Set up global state, configure middleware, and run uvicorn.

    Called by launch_server after subprocesses have been launched.
    """
    # 设置全局状态
    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            template_manager=template_manager,
            scheduler_info=scheduler_infos[0],
        )
    )

    # 在tokenizer_manager上存储watchdog（SIGQUIT处理器的唯一真实来源）
    if tokenizer_manager is not None:
        tokenizer_manager._subprocess_watchdog = subprocess_watchdog

    if server_args.enable_metrics:
        add_prometheus_track_response_middleware(app)

    # 使用Granian作为HTTP/2服务器
    if server_args.enable_http2:
        # 复用多分词器共享内存机制来传递
        # 初始化参数（port_args, server_args, scheduler_info）给
        # Granian工作进程，它们是独立进程。
        multi_tokenizer_args_shm = write_data_for_multi_tokenizer(
            port_args, server_args, scheduler_infos[0]
        )
        try:
            if server_args.ssl_certfile:
                logger.info(
                    f"SSL enabled: certfile={server_args.ssl_certfile}, "
                    f"keyfile={server_args.ssl_keyfile}"
                )
            logger.info(
                f"Starting Granian HTTP/2 server on "
                f"{server_args.host}:{server_args.port}"
            )
            # 通过os.environ传播主进程PID，以便Granian
            # 工作进程（forked或spawned）能找到上面创建的共享内存段
            envs.SGLANG_GRANIAN_PARENT_PID.set(os.getpid())
            _close_main_process_sockets()
            _run_granian_server(server_args)
        finally:
            if multi_tokenizer_args_shm is not None:
                multi_tokenizer_args_shm.unlink()
        return

    # 将额外参数传递给生命周期函数
    # 它们将用于额外的初始化设置
    if server_args.tokenizer_worker_num == 1:
        # 单分词器模式下，可以通过app对象的属性传递参数
        app.is_single_tokenizer_mode = True
        app.server_args = server_args
        app.warmup_thread_kwargs = dict(
            server_args=server_args,
            launch_callback=launch_callback,
            execute_warmup_func=execute_warmup_func,
        )

        # 添加API密钥授权
        # 仅在单分词器模式下支持
        #
        # 向后兼容：
        # - 仅api_key：行为与旧版一致（所有端点需要api_key）
        # - 无密钥：旧版无限制；未配置admin_api_key时ADMIN_FORCE端点仍须拒绝
        if (
            server_args.api_key
            or server_args.admin_api_key
            or app_has_admin_force_endpoints(app)
        ):
            from sglang.srt.utils.auth import add_api_key_middleware

            add_api_key_middleware(
                app,
                api_key=server_args.api_key,
                admin_api_key=server_args.admin_api_key,
            )
    else:
        # 多分词器模式下，需要将参数写入共享内存
        # 以供其他工作进程读取
        app.is_single_tokenizer_mode = False
        multi_tokenizer_args_shm = write_data_for_multi_tokenizer(
            port_args, server_args, scheduler_infos[0]
        )

    try:
        # 更新日志配置
        set_uvicorn_logging_configs(server_args)

        if server_args.ssl_certfile:
            logger.info(
                f"SSL enabled: certfile={server_args.ssl_certfile}, "
                f"keyfile={server_args.ssl_keyfile}"
            )

        # 监听HTTP请求
        if server_args.tokenizer_worker_num == 1:
            if server_args.enable_ssl_refresh:
                # 使用Config/Server API以访问SSLContext
                config = uvicorn.Config(
                    app,
                    host=server_args.host,
                    port=server_args.port,
                    root_path=server_args.fastapi_root_path,
                    log_level=server_args.log_level_http or server_args.log_level,
                    timeout_keep_alive=envs.SGLANG_TIMEOUT_KEEP_ALIVE.get(),
                    loop="uvloop",
                    ssl_keyfile=server_args.ssl_keyfile,
                    ssl_certfile=server_args.ssl_certfile,
                    ssl_ca_certs=server_args.ssl_ca_certs,
                    ssl_keyfile_password=server_args.ssl_keyfile_password,
                )
                config.load()  # 创建SSLContext

                from sglang.srt.entrypoints.ssl_utils import SSLCertRefresher

                server = uvicorn.Server(config)

                # SSL证书自动刷新
                async def _run_with_ssl_refresh():
                    refresher = SSLCertRefresher(
                        config.ssl,
                        server_args.ssl_keyfile,
                        server_args.ssl_certfile,
                        server_args.ssl_ca_certs,
                    )
                    logger.info("SSL certificate auto-refresh enabled.")
                    try:
                        await server.serve()
                    finally:
                        refresher.stop()

                import asyncio

                asyncio.run(_run_with_ssl_refresh())
            else:
                # 默认情况，单个分词器进程
                uvicorn.run(
                    app,
                    host=server_args.host,
                    port=server_args.port,
                    root_path=server_args.fastapi_root_path,
                    log_level=server_args.log_level_http or server_args.log_level,
                    timeout_keep_alive=envs.SGLANG_TIMEOUT_KEEP_ALIVE.get(),
                    loop="uvloop",
                    ssl_keyfile=server_args.ssl_keyfile,
                    ssl_certfile=server_args.ssl_certfile,
                    ssl_ca_certs=server_args.ssl_ca_certs,
                    ssl_keyfile_password=server_args.ssl_keyfile_password,
                )
        else:
            # 多个分词器和HTTP进程
            from uvicorn.config import LOGGING_CONFIG

            LOGGING_CONFIG["loggers"]["sglang.srt.entrypoints.http_server"] = {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
            }

            if server_args.enable_ssl_refresh:
                logger.warning(
                    "--enable-ssl-refresh is not supported with multiple "
                    "tokenizer workers (--tokenizer-worker-num > 1). "
                    "SSL refresh will be disabled."
                )

            uvicorn.run(
                "sglang.srt.entrypoints.http_server:app",
                host=server_args.host,
                port=server_args.port,
                root_path=server_args.fastapi_root_path,
                log_level=server_args.log_level_http or server_args.log_level,
                timeout_keep_alive=envs.SGLANG_TIMEOUT_KEEP_ALIVE.get(),
                timeout_worker_healthcheck=envs.SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT.get(),
                loop="uvloop",
                workers=server_args.tokenizer_worker_num,
                ssl_keyfile=server_args.ssl_keyfile,
                ssl_certfile=server_args.ssl_certfile,
                ssl_ca_certs=server_args.ssl_ca_certs,
                ssl_keyfile_password=server_args.ssl_keyfile_password,
            )
    finally:
        if server_args.tokenizer_worker_num > 1:
            if multi_tokenizer_args_shm is not None:
                multi_tokenizer_args_shm.unlink()
            if _global_state is not None:
                _global_state.tokenizer_manager.socket_mapping.clear_all_sockets()


# 启动SRT（SGLang运行时）服务器的主入口函数
def launch_server(
    server_args: ServerArgs,
    init_tokenizer_manager_func: Callable = init_tokenizer_manager,
    run_scheduler_process_func: Callable = run_scheduler_process,
    run_detokenizer_process_func: Callable = run_detokenizer_process,
    execute_warmup_func: Callable = _execute_server_warmup,
    launch_callback: Optional[Callable[[], None]] = None,
):
    """
    Launch SRT (SGLang Runtime) Server.

    The SRT server consists of an HTTP server and an SRT engine.

    - HTTP server: A FastAPI server that routes requests to the engine.
    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager all run in the main process.
    2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
    """
    # 启动子进程
    (
        tokenizer_manager,
        template_manager,
        port_args,
        scheduler_init_result,
        subprocess_watchdog,
    ) = Engine._launch_subprocesses(
        server_args=server_args,
        init_tokenizer_manager_func=init_tokenizer_manager_func,
        run_scheduler_process_func=run_scheduler_process_func,
        run_detokenizer_process_func=run_detokenizer_process_func,
    )

    _setup_and_run_http_server(
        server_args,
        tokenizer_manager,
        template_manager,
        port_args,
        scheduler_init_result.scheduler_infos,
        subprocess_watchdog,
        execute_warmup_func=execute_warmup_func,
        launch_callback=launch_callback,
    )
