# SGLang 启动模块交互关系

本文基于当前仓库代码，梳理默认启动路径：

- `sglang serve`
- `python -m sglang.launch_server`
- 非 `grpc_mode`
- 非 `use_ray`
- 非 `encoder_only`

也就是普通 HTTP 服务启动路径。

## 核心入口与模块

- 启动入口：`python/sglang/launch_server.py`
- HTTP 服务：`python/sglang/srt/entrypoints/http_server.py`
- Engine 与子进程编排：`python/sglang/srt/entrypoints/engine.py`
- TokenizerManager：`python/sglang/srt/managers/tokenizer_manager.py`
- Scheduler：`python/sglang/srt/managers/scheduler.py`
- DetokenizerManager：`python/sglang/srt/managers/detokenizer_manager.py`
- IPC 地址分配：`python/sglang/srt/server_args.py` 中的 `PortArgs`

## 启动阶段时序图

```mermaid
sequenceDiagram
    participant CLI as CLI / sglang.launch_server
    participant Args as ServerArgs
    participant HTTP as http_server.launch_server
    participant Engine as Engine._launch_subprocesses
    participant Ports as PortArgs
    participant SchedProc as Scheduler subprocess(es)
    participant DetokProc as Detokenizer subprocess(es)
    participant TM as TokenizerManager
    participant Template as TemplateManager
    participant FastAPI as FastAPI/Uvicorn
    participant Warmup as Warmup Thread

    CLI->>Args: prepare_server_args(argv)
    CLI->>CLI: run_server(server_args)

    alt encoder_only
        CLI->>CLI: launch encoder server / grpc encoder
    else grpc_mode
        CLI->>CLI: launch legacy grpc_server
    else use_ray
        CLI->>CLI: launch ray.http_server
    else default HTTP
        CLI->>HTTP: launch_server(server_args)
    end

    HTTP->>Engine: _launch_subprocesses(server_args)
    Engine->>Engine: configure_logger / set env / check args
    Engine->>Ports: PortArgs.init_new(server_args)
    Ports-->>Engine: tokenizer_ipc, scheduler_input_ipc, detokenizer_ipc, rpc_ipc, metrics_ipc

    Engine->>SchedProc: mp.Process(run_scheduler_process, ranks, port_args, pipe_writer)
    SchedProc->>SchedProc: configure_scheduler_process()
    SchedProc->>SchedProc: Scheduler(...)
    SchedProc->>SchedProc: init model config / tokenizer / TP worker / KV cache
    SchedProc-->>Engine: pipe_writer.send(get_init_info(): ready, max_req_input_len)

    Engine->>DetokProc: mp.Process(run_detokenizer_process, server_args, port_args)
    DetokProc->>DetokProc: DetokenizerManager(...)
    DetokProc->>DetokProc: bind detokenizer_ipc, connect tokenizer_ipc, init tokenizer

    alt tokenizer_worker_num == 1
        Engine->>TM: TokenizerManager(server_args, port_args)
        TM->>TM: init model config
        TM->>TM: init tokenizer / multimodal processor
        TM->>TM: bind tokenizer_ipc, connect scheduler_input_ipc
        Engine->>Template: initialize_templates(TM, model_path, chat_template)
    else tokenizer_worker_num > 1
        Engine->>TM: MultiTokenizerRouter(server_args, port_args)
        Note over TM,FastAPI: worker 进程在 FastAPI lifespan 中从 shared memory 初始化 TokenizerWorker
    end

    Engine->>Engine: wait_for_scheduler_ready()
    Engine->>TM: set max_req_input_len from scheduler_info
    Engine->>Engine: start SubprocessWatchdog
    Engine-->>HTTP: tokenizer_manager, template_manager, port_args, scheduler_infos

    HTTP->>FastAPI: set_global_state(TM, Template, scheduler_info)
    HTTP->>FastAPI: configure middleware / auth / metrics
    HTTP->>FastAPI: uvicorn.run(app)
    FastAPI->>FastAPI: lifespan()
    FastAPI->>FastAPI: init OpenAI/Ollama/Anthropic serving handlers
    FastAPI->>Warmup: start _wait_and_warmup thread
    Warmup->>FastAPI: GET /model_info
    Warmup->>FastAPI: POST /generate or /v1/chat/completions or /encode
    FastAPI->>TM: warmup request
    TM->>SchedProc: tokenized request via scheduler_input_ipc
    SchedProc->>DetokProc: generated token ids via detokenizer_ipc
    DetokProc->>TM: decoded output via tokenizer_ipc
    TM-->>FastAPI: warmup response
    Warmup->>TM: server_status = Up
```

## 请求阶段时序图

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant Serving as OpenAI/Native Serving Handler
    participant TM as TokenizerManager
    participant Scheduler
    participant Model as TpModelWorker / ModelRunner
    participant Detok as DetokenizerManager

    Client->>FastAPI: HTTP request (/generate, /v1/chat/completions, /encode...)
    FastAPI->>Serving: validate / convert protocol
    Serving->>TM: generate_request / embedding_request

    TM->>TM: 分配 rid，记录 ReqState
    TM->>TM: tokenize text / process multimodal input
    TM->>Scheduler: PUSH tokenized req to scheduler_input_ipc

    Scheduler->>Scheduler: recv_requests()
    Scheduler->>Scheduler: schedule batch / prefix cache / memory pool
    Scheduler->>Model: run_batch()
    Model-->>Scheduler: logits / next tokens / embeddings

    alt skip_tokenizer_init
        Scheduler->>TM: send token-id output directly to tokenizer_ipc
    else generation text output
        Scheduler->>Detok: PUSH BatchTokenIDOutput to detokenizer_ipc
        Detok->>Detok: decode token ids to text
        Detok->>TM: PUSH BatchStrOutput to tokenizer_ipc
    else embedding / control output
        Scheduler->>TM: send output directly to tokenizer_ipc
    end

    TM->>TM: merge output into ReqState
    TM-->>Serving: yield stream chunk or final response
    Serving-->>FastAPI: protocol response
    FastAPI-->>Client: HTTP response / SSE stream
```

## 模块间关系要点

- 主进程包含 HTTP server、FastAPI app、TokenizerManager 和 TemplateManager。
- 子进程包含 Scheduler 进程组和 Detokenizer 进程组。
- Scheduler 初始化完成后，会通过 `mp.Pipe` 将 `status=ready`、`max_total_num_tokens`、`max_req_input_len` 回传给父进程。
- 普通请求链路主要通过 ZMQ IPC 连接：
  - `TokenizerManager -> Scheduler`：`scheduler_input_ipc_name`
  - `Scheduler -> DetokenizerManager`：`detokenizer_ipc_name`
  - `DetokenizerManager -> TokenizerManager`：`tokenizer_ipc_name`
  - `Engine/Python API -> Scheduler RPC`：`rpc_ipc_name`
- `dp_size > 1` 时，Engine 先启动 `DataParallelController`，再由它管理 DP scheduler。
- `tokenizer_worker_num > 1` 时，主进程使用 `MultiTokenizerRouter`，FastAPI worker 通过 shared memory 初始化自己的 `TokenizerWorker`。
