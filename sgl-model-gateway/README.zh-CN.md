# SGLang Model Gateway

面向大规模 LLM 部署的高性能模型路由控制面和数据面。Gateway 负责编排 worker 集群，在 HTTP 与 gRPC 后端之间做流量均衡，并暴露 OpenAI 兼容 API；同时支持可插拔的历史存储和工具集成，并针对 SGLang serving runtime 做了深度优化。

## 概览

- 统一控制面：注册、监控并编排异构模型集群中的 prefill、decode 和 regular worker。
- 数据面：在 HTTP、PD（prefill/decode）、gRPC 和 OpenAI 兼容后端之间路由请求，并共享可靠性能力。
- 业界领先的 gRPC pipeline：使用 Rust 原生 tokenizer、reasoning parser 和 tool-call 执行，为 OpenAI 兼容 serving 提供高吞吐能力。
- 多模型 inference gateway 模式（`--enable-igw`）：同时运行多个 router，并应用模型级路由策略。
- conversation、response 和 chat-history connector：在 router 层集中管理状态，支持在模型/MCP loop 之间合规共享；可选内存、禁用、Oracle ATP 等存储。
- 内置可靠性原语：指数退避重试、circuit breaker、token-bucket 限流和队列。
- 一等观测能力：结构化日志、OpenTelemetry trace 和 Prometheus metrics。

### 架构速览

**控制面**

- Worker Manager 校验 worker、发现能力，并保持 registry 同步。
- Job Queue 串行化后台操作（add/remove），并通过 `/workers/{worker_id}` 暴露状态。
- 后台健康检查器和 load monitor 持续为 circuit breaker 与 policy 提供信息。
- 可选 Kubernetes service discovery，用于让 registry 与 pod 状态保持一致。

**数据面**

- SGLang HTTP router 支持 regular 与 PD（prefill/decode）流量，并根据策略选择 worker。
- SGLang gRPC router 和 pipeline 将 tokenized request 以流式方式发送到 SRT gRPC worker；tokenizer、reasoning parser、tool parser 都以 Rust 实现，用于最大化 OpenAI API 性能，并同时支持单阶段与 PD serving 拓扑。
- OpenAI router 代理 OpenAI 风格的 requests、responses 和 conversations 到远端服务商（OpenAI、xAI、Gemini 和其他 OpenAI 兼容 provider），同时保持 streaming/SSE 语义。
- 启用 IGW 时，Router Manager 协调多个 router 实现。
- Resilience layer 提供 token-bucket 限流、请求排队、retry executor 和 worker 级 circuit breaker，保证故障期间流量仍可继续转发。
- 高级负载均衡：cache-aware 请求复用、load-aware（power-of-two）选择和模型级 policy override。

## 功能亮点

- 多种负载均衡策略（`random`、`round_robin`、`cache_aware`、`power_of_two`、`bucket`），并支持 DP-aware 调度。
- 多模型 HTTP serving 与 inference gateway 路由，支持模型级策略。
- Prefill/decode 分离，包括 bootstrap port 处理和 cache-aware 合并。
- gRPC 路由：完整 Rust tokenizer 加载、reasoning parser 选择和 tool parser 集成，支持 OpenAI 兼容 endpoint；在 DeepSeek、Llama、Kimi K2、Qwen、GPT-OSS、Mistral、Step-3、GLM4、GLM4.7 等推理模型上支持 streaming 与 non-streaming。
- OpenAI 兼容 endpoint：`/v1/chat/completions`、`/v1/responses`、`/v1/conversations`、`/v1/embeddings`、`/v1/rerank`、`/v1/classify`。
- **Tokenization APIs**：提供 tokenize（`/v1/tokenize`）和 detokenize（`/v1/detokenize`）HTTP endpoint，支持 batch；同时提供 tokenizer 动态注册管理 API。
- **Parser endpoints**：reasoning parser（`/parse/reasoning`）和 function call parser（`/parse/function_call`），用于分离 reasoning 内容和提取 tool calls。
- 原生 MCP client 集成，支持 STDIO、HTTP、SSE 和 Streamable 等 MCP transport，用于工具执行 loop。
- 可插拔 history connector：内存、禁用、Oracle ATP 或 PostgreSQL（支持连接池与凭据）。
- 可靠性控制：带 jitter 的 retry、worker 级 circuit breaker、可选队列的 token bucket limiter，以及 cache flush API。
- regular 和 PD workload 的 service discovery，支持独立 selector。
- **完整观测能力**：40+ Prometheus metrics，覆盖 HTTP、router、worker、circuit breaker、retry、discovery、MCP 和数据库层；支持 OpenTelemetry tracing、OTLP export 和 request ID 传播。

## 文档

- **用户指南**：[docs.sglang.io/advanced_features/sgl_model_gateway.html](https://docs.sglang.io/advanced_features/sgl_model_gateway.html)
- 更多指南、API reference 和部署模式会随 SGLang release 持续更新。

## 安装

### Docker

Docker Hub 提供预构建镜像，支持多架构（x86_64 和 ARM64）：

```bash
docker pull lmsysorg/sgl-model-gateway:latest
```

### 前置条件

- **Rust 和 Cargo**
  ```bash
  # Install rustup (Rust installer and version manager)
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

  # Reload shell environment
  source "$HOME/.cargo/env"

  # Verify installation
  rustc --version
  cargo --version
  ```
- **Python**，并可使用 `pip` 和 virtualenv 工具。

### Rust 二进制

```bash
# Build release binary
cargo build --release
```

### Python 包

```bash
pip install maturin

# Fast development mode (debug build, no wheel, instant)
# Uses system OpenSSL (requires libssl-dev/openssl-devel)
cd bindings/python
maturin develop

# Production build (optimized, creates wheel)
# Uses vendored OpenSSL (cross-platform compatibility)
cd bindings/python
maturin build --release --out dist --features vendored-openssl
pip install --force-reinstall dist/*.whl

# Development build with system OpenSSL (faster)
# Requires: apt install libssl-dev pkg-config (Ubuntu/Debian)
#       or: yum install openssl-devel (RHEL/CentOS)
cd bindings/python
maturin build --release --out dist
pip install --force-reinstall dist/*.whl
```

> **注意：** Python binding 位于 `bindings/python/`，并有自己的 `Cargo.toml`。开发时使用 `maturin develop` 可快速迭代（debug 构建并直接安装）。生产 wheel 使用 `maturin build --release --features vendored-openssl`，会启用完整优化（`opt-level="z"`、`lto="fat"`）和跨平台兼容性。该包使用 abi3，兼容 Python 3.8+。

## 检查版本

安装后，可以用以下命令验证安装并查看版本信息：

```bash
# Simple version (Rust binary)
./target/release/sgl-model-gateway --version
# or use aliases
./target/release/smg --version
./target/release/amg --version

# Full version info with build details
./target/release/sgl-model-gateway --version-verbose

# Python CLI
amg --version
amg --version-verbose
python3 -m sglang_router --version
```

`--version`（或 `-V`）显示版本字符串。使用 `--version-verbose` 可查看完整构建信息，包括 Git commit、构建时间、编译器版本和平台细节。

## 快速开始

### Regular HTTP 路由

- **Rust 二进制**
  ```bash
  ./target/release/sgl-model-gateway \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy cache_aware
  ```
  开发时可使用 `cargo run --release -- ...` 获得相同行为。
- **Python launcher**
  ```bash
  python3 -m sglang_router.launch_router \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy cache_aware
  ```

### Prefill/Decode 分离（PD）

- **Rust 二进制**
  ```bash
  ./target/release/sgl-model-gateway \
    --pd-disaggregation \
    --prefill http://prefill1:30001 9001 \
    --prefill http://prefill2:30002 \
    --decode http://decode1:30011 \
    --decode http://decode2:30012 \
    --policy cache_aware \
    --prefill-policy cache_aware \
    --decode-policy power_of_two
  ```
- **Python launcher**
  ```bash
  python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://prefill1:30001 9001 \
    --prefill http://prefill2:30002 \
    --decode http://decode1:30011 \
    --decode http://decode2:30012 \
    --policy cache_aware
  ```

Prefill entry 可接受可选 bootstrap port。PD 模式会将 prefill metadata 与 decode output 合并，并以流式方式返回给客户端。

### 多模型 Inference Gateway

启用 IGW 模式后，可通过单个 router 路由多个模型，并应用模型级 policy：

```bash
./target/release/sgl-model-gateway \
  --enable-igw \
  --policy cache_aware \
  --max-concurrent-requests 512

# Register workers dynamically
curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{
        "url": "http://worker-a:8000",
        "model_id": "mistral",
        "priority": 10,
        "labels": {"tier": "gold"}
      }'

# Add another worker with a different model/policy hint
curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{
        "url": "http://worker-b:8000",
        "model_id": "llama3",
        "priority": 20,
        "labels": {"policy": "power_of_two", "tier": "silver"}
      }'

# Inspect registered workers
curl http://localhost:30000/workers
```

示例响应（HTTP worker）：

```json
{
  "workers": [
    {"id":"2f3a0c3e-3a7b-4c3f-8c70-1b7d4c3a6e1f","url":"http://0.0.0.0:31378","model_id":"mistral","priority":50,"cost":1.0,"worker_type":"regular","is_healthy":true,"load":0,"connection_mode":"Http"},
    {"id":"9b0f6c2a-1c4f-4c2a-9f4a-1f2a6c0b9d3e","url":"http://0.0.0.0:34881","model_id":"llama3","priority":50,"cost":1.0,"worker_type":"regular","is_healthy":true,"load":0,"connection_mode":"Http"}
  ],
  "total": 2,
  "stats": {
    "prefill_count": 0,
    "decode_count": 0,
    "regular_count": 2
  }
}
```

可使用同一 API 添加更多 worker；根据需要包含可选 `labels`（模型级 policy）、`tokenizer_path`、`reasoning_parser`、`tool_parser` 字段。后台 job 完成注册期间，`/workers/{worker_id}` 可查看排队 job 状态。

### gRPC 路由

- **Rust 二进制**
  ```bash
  ./target/release/sgl-model-gateway \
    --worker-urls grpc://worker-grpc-0:31001 grpc://worker-grpc-1:31002 \
    --tokenizer-path /path/to/tokenizer.json \
    --reasoning-parser deepseek-r1 \
    --tool-call-parser json
  ```
- **Python router**
  ```bash
  python3 -m sglang_router.launch_router \
    --worker-urls grpc://127.0.0.1:20000 \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8080
  ```

gRPC router 会在本地 tokenize input，支持 tool-call parsing，并以流式方式返回响应。当 worker registry 中包含 PD worker 时，它同时支持 regular HTTP 等价 serving 和 PD（prefill/decode）serving。只要连接模式解析为 gRPC，就应提供 `--model-path` 或 `--tokenizer-path`（HuggingFace ID 或本地目录）。

使用 `--reasoning-parser` 选择内置 reasoning pipeline（DeepSeek-R1、Qwen3、Step-3、GLM4、GLM4.7 等），使用 `--tool-call-parser` 支持 JSON/Pythonic/XML tool contract，可用于 streaming 或 non-streaming。

### OpenAI Backend 模式

将请求路由到 OpenAI 或 OpenAI 兼容 endpoint：

```bash
# Route to OpenAI API
python3 -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \

# Route to custom OpenAI-compatible endpoint (Gemini, xAI, etc.)
python3 -m sglang_router.launch_router \
  --backend openai \
  --worker-urls http://my-openai-compatible-service:8000 \
```

**说明**

- OpenAI backend 模式是对单个远端 endpoint 的代理，不执行负载均衡。
- 每个 router 实例只应提供一个 `--worker-urls` 条目。
- Rust 二进制支持相同参数（`./target/release/sgl-model-gateway --backend openai ...`）。

### MCP 集成

SGL Model Gateway 提供原生 Model Context Protocol（MCP）client 集成，支持通过 STDIO、SSE 和 Streamable transport 执行 tool calling。MCP server 通过 YAML 配置文件配置，并在启动时通过 workflow engine 注册。

#### 基本用法

```bash
# Rust binary
./target/release/sgl-model-gateway \
  --mcp-config-path /path/to/mcp-config.yaml \
  --worker-urls http://worker1:8000

# Python launcher
python3 -m sglang_router.launch_router \
  --mcp-config-path /path/to/mcp-config.yaml \
  --worker-urls http://worker1:8000
```

#### MCP 配置文件

创建 MCP 配置文件以定义 server、transport 和连接设置：

```yaml
servers:
  - name: "filesystem"
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    protocol: "stdio"
    required: false

  - name: "github"
    url: "https://api.github.com/mcp"
    token: "ghp_xxxxx"
    protocol: "sse"
    required: false

  - name: "custom-tools"
    url: "https://tools.example.com/mcp"
    protocol: "streamable"
    required: true

pool:
  max_connections: 100
  idle_timeout: 300  # seconds

proxy:
  http: "http://proxy.internal:8080"
  https: "https://proxy.internal:8443"
  no_proxy: "localhost,127.0.0.1,*.internal"

inventory:
  enable_refresh: true
  tool_ttl: 300  # seconds - how long tools are considered fresh
  refresh_interval: 300  # seconds - background refresh interval
```

#### 配置选项

**Server 配置**（`servers` 数组）：

- `name`：MCP server 的唯一标识。
- `command` + `args`：用于 STDIO transport（本地进程执行）。
- `url`：用于 SSE 或 Streamable transport（HTTP/HTTPS endpoint）。
- `token`：HTTP transport 的可选认证 token。
- `protocol`：协议类型（`"sse"`、`"streamable"` 或 `"stdio"`）。
- `required`：为 `true` 时，如果 server 不可达，router 启动失败（默认 `false`）。
- `envs`：STDIO 进程的环境变量（可选）。
- `proxy`：单个 server 的代理覆盖配置（设为 `null` 可绕过全局代理）。

**连接池**（`pool`）：

- `max_connections`：动态 server 的最大池化连接数（默认 100）。
- `idle_timeout`：空闲连接清理超时时间，单位秒（默认 300）。

**代理配置**（`proxy`）：

- `http`/`https`：MCP server 连接使用的代理 URL（不是 LLM 流量）。
- `no_proxy`：不走代理的 host，逗号分隔，支持通配符。
- **注意**：当前 `streamable` transport 会忽略代理设置。如需代理支持，请使用 STDIO 或 SSE transport。

**Inventory 设置**（`inventory`）：

- `enable_refresh`：启用工具 inventory 后台自动刷新（默认 true）。
- `tool_ttl`：tool cache TTL，单位秒，表示工具信息被视为新鲜的时长（默认 300）。
- `refresh_interval`：后台刷新间隔，单位秒，用于主动刷新 inventory（默认 300）。

#### Transport 类型

**STDIO**（本地进程）：

```yaml
name: "local-tools"
command: "python"
args: ["-m", "my_mcp_server"]
envs:
  API_KEY: "secret"
  DEBUG: "true"
```

**SSE**（Server-Sent Events）：

```yaml
name: "remote-sse"
url: "https://mcp.example.com/events"
token: "bearer-token"
protocol: "sse"
```

**Streamable**（双向流式）：

```yaml
name: "streaming-tools"
url: "https://mcp.example.com/stream"
protocol: "streamable"
required: true
```

#### Server 生命周期

- MCP server 通过 workflow engine 注册，并带有 retry 逻辑（STDIO server 为 100 次尝试、2 小时超时）。
- discovery 阶段识别 tools、prompts 和 resources。
- tool inventory 使用可配置 TTL 缓存，并周期刷新。
- 可选 server 失败时记录 warning；required server 失败会阻止启动。
- 静态 server（来自配置）是永久的；动态 server（per-request）使用连接池。

可通过 Prometheus metrics 查看 MCP 活动（`mcp_*` 指标），并通过 admin API 查看 workflow job 状态。

### Python Launcher（Router + Workers）

同时启动 router 和 SGLang worker 进程；`launch_server` 会一次性拉起 worker（HTTP 或 gRPC）与 router。

```bash
python3 -m sglang_router.launch_server --host 0.0.0.0
```

生产部署可根据需要添加参数：

```bash
python3 -m sglang_router.launch_server \
  --host 0.0.0.0 \
  --port 8080 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tp-size 1 \
  --dp-size 8 \
  --grpc-mode
```

省略 `--grpc-mode` 会启动 HTTP worker；router 会根据给定 DP size 自动配置 worker URL 并调度。

### Mini Load Balancer（调试）

```bash
python3 -m sglang_router.launch_router \
  --mini-lb \
  --pd-disaggregation \
  --prefill http://localhost:30001 \
  --decode http://localhost:30011
```

MiniLB 使用简单随机路由转发 PD 请求，仅用于本地调试。

### 运行 Worker Server

使用上游 SGLang 二进制启动专用 worker 进程。

- **Prefill worker server（gRPC 模式）**：
  ```bash
  python3 -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 20000 \
    --tp-size 1 \
    --grpc-mode
  ```
  移除 `--grpc-mode` 可启动 HTTP worker。结合前面的 router 命令，通过 CLI 参数或控制面 API 注册 worker。

## 控制面

### Worker 生命周期与 Job Queue

- `JobQueue` 处理异步 add/remove 操作，避免阻塞客户端。
- `WorkerManager` 检查 worker metadata（`/server_info`、`/get_model_info`），跟踪 load，并暴露 `flush_cache` 和 `get_loads`。
- worker 级 circuit breaker 和健康探针保持 registry 健康；load monitor 向 cache-aware 和 power-of-two policy 提供 metrics。

### 管理与 Worker API

| Method | Path | Description |
|---|---|---|
| `POST` | `/workers` | 排队注册 worker（prefill/decode/regular）。Body 匹配 `WorkerConfigRequest`。Job queue 处理期间返回 `202 Accepted`。 |
| `GET` | `/workers` | 列出 worker，包括 health、load、policy metadata 和排队 job 状态。 |
| `GET` | `/workers/{worker_id}` | 查看指定 worker 或 job queue entry（UUID）。 |
| `PUT` | `/workers/{worker_id}` | 按 UUID 排队更新 worker。 |
| `DELETE` | `/workers/{worker_id}` | 按 UUID 排队移除 worker。 |
| `POST` | `/flush_cache` | 触发 HTTP worker cache flush，并返回成功/失败拆分。 |
| `GET` | `/get_loads` | 采样每个 worker 当前上报的 load。 |

提供 `--api-key` 时，所有管理路由继承 router API-key 保护。Job 状态包括带 timestamp 的 `pending`、`processing` 和 `failed` 阶段。

### Service Discovery

启用 Kubernetes discovery 可自动 reconcile worker：

```bash
./target/release/sgl-model-gateway \
  --service-discovery \
  --selector app=sglang-worker role=inference \
  --service-discovery-namespace sglang-system \
  --service-discovery-port 8000
```

PD 模式支持专用 selector：

```bash
--pd-disaggregation \
--prefill-selector app=sglang component=prefill \
--decode-selector app=sglang component=decode \
--service-discovery
```

Prefill pod 可通过 `sglang.ai/bootstrap-port` annotation 暴露 bootstrap port。RBAC 必须允许对 pod 执行 `get`、`list` 和 `watch`。

## 数据面

### Router 能力（HTTP 与 gRPC）

两个 router stack 都：

- 共享负载均衡 policy（random、round-robin、cache-aware、power-of-two），支持 DP-aware 调度、retry、circuit breaker 和 rate limiting。
- 按请求记录 metrics，跟踪 running load，并与 router-wide policy registry 集成。

HTTP router 暴露完整 OpenAI 兼容 API 面（`/generate`、`/v1/chat/completions`、`/v1/completions`、`/v1/embeddings`、`/v1/responses`、`/v1/rerank` 等）。gRPC router 当前提供高速 `/generate` 和 `/v1/chat/completions`；其余 endpoint 在 pipeline 完成前返回 `501 Not Implemented`。

#### HTTP Router 细节

- **Regular router** 处理经典单阶段 worker，并支持模型级 policy override。
- **Prefill/Decode router** 协调分离的 prefill 和 decode worker，合并 metadata，并管理 streaming fan-in。

#### gRPC Router 细节

- 业界领先的全 Rust OpenAI 兼容 gRPC inference gateway，在进程内执行 tokenizer、reasoning parser 和 tool parser，以获得最大吞吐。
- 同时支持单阶段和 PD（prefill/decode）worker 拓扑；router 会按模型自动选择合适 pipeline。
- 提供与 HTTP router 相同的 `/v1/*` API，并将 tokenized request/response 直接流式传输给 SRT gRPC worker。
- 内置 reasoning parser 支持 DeepSeek、Qwen、Llama、Mistral、GPT-OSS、Step-3、GLM4、GLM4.7、Kimi K2 以及其他 structured-thought 模型。
- Tool-call parser 支持 JSON、Pythonic、XML 和 custom schema，并支持 streaming/non-streaming 执行 loop。
- Tokenizer factory 支持 HuggingFace 模型、本地 `tokenizer.json` 文件和 chat template override（见 `src/tokenizer`）。
- 可查看 `src/reasoning_parser`、`src/tool_parser` 和 `src/tokenizer` 中的代码路径，理解 gRPC 模式的端到端 Rust 实现。

### OpenAI Router

- 代理 OpenAI 兼容 chat completions 和 responses API，端到端保持 headers 和 SSE stream。
- 支持 `/v1/responses` background job，包括 cancel、delete 和 list input items，允许在不将数据持久化到远端 vendor 的前提下实现 agentic、多轮编排。
- Conversation API（`/v1/conversations` 和 `/v1/conversations/{id}/items`）与配置的 conversation storage backend 交互，用于合规管理 chat history。Conversation state 存在 router 层，因此相同历史可驱动不同模型或 MCP loop，而不会泄露给上游 vendor。
- Chat history、agentic multi-turn `/v1/responses` 和原生 MCP client（STDIO/HTTP/SSE/Streamable transport）旨在满足企业数据隐私要求，把敏感状态保留在 router 内。

### 请求 Endpoint

| Endpoint | Notes |
|---|---|
| `POST /generate` | SGLang generate API。 |
| `POST /v1/chat/completions` | OpenAI 兼容 chat。支持 streaming 和 tool calls。 |
| `POST /v1/completions` | OpenAI 兼容 text completions。 |
| `POST /v1/responses` | 创建 background responses，返回 response ID。 |
| `GET /v1/responses/{id}` | 获取已存储 response。 |
| Conversation endpoints（`/v1/conversations`、`/v1/conversations/{id}`、`/v1/conversations/{id}/items`） | 管理 chat history。 |
| `POST /v1/embeddings` | 转发 embedding request（HTTP 和 gRPC）。 |
| `POST /v1/rerank`、`POST /rerank` | Ranking API。 |
| `POST /v1/classify` | 文本分类 endpoint。 |

### Classification API

`/v1/classify` endpoint 使用 sequence classification 模型进行文本分类，例如 `Qwen2ForSequenceClassification`、`BertForSequenceClassification`。

**请求：**

```bash
curl http://localhost:30000/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jason9693/Qwen2.5-1.5B-apeach",
    "input": "I love this product!"
  }'
```

**响应：**

```json
{
  "id": "classify-a1b2c3d4-5678-90ab-cdef-1234567890ab",
  "object": "list",
  "created": 1767034308,
  "model": "jason9693/Qwen2.5-1.5B-apeach",
  "data": [
    {
      "index": 0,
      "label": "positive",
      "probs": [0.12, 0.88],
      "num_classes": 2
    }
  ],
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 0,
    "total_tokens": 6
  }
}
```

**字段：**

- `label`：预测类别标签，来自模型 `id2label` 配置；没有配置时 fallback 到 `LABEL_N`。
- `probs`：所有类别的概率分布（logits 经 softmax）。
- `num_classes`：分类类别数量。

**说明：**

- Classification 复用 embedding backend；scheduler 返回 logits，再通过 softmax 转换为概率。
- 标签来自模型 HuggingFace config 的 `id2label` 字段；没有该映射的模型使用通用标签（`LABEL_0`、`LABEL_1` 等）。
- HTTP 和 gRPC router 都支持 classification。

公共健康 endpoint（`/liveness`、`/readiness`、`/health`、`/health_generate`）反映 registry 状态；readiness 会确保 PD worker 已配对，且 IGW 至少有一个健康 route。

### Tokenization Endpoints

Gateway 提供文本 tokenization HTTP endpoint，设计目标是对齐 SGLang Python tokenization API，并支持 batch 操作。

| Endpoint | Method | Description |
|---|---|---|
| `POST /v1/tokenize` | `POST` | 将文本 tokenize 为 token IDs（单条或 batch）。 |
| `POST /v1/detokenize` | `POST` | 将 token IDs 转回文本（单条或 batch）。 |
| `POST /v1/tokenizers` | `POST` | 注册新 tokenizer（异步，返回 job 状态）。 |
| `GET /v1/tokenizers` | `GET` | 列出所有已注册 tokenizer。 |
| `GET /v1/tokenizers/{id}` | `GET` | 按 UUID 获取 tokenizer 信息。 |
| `GET /v1/tokenizers/{id}/status` | `GET` | 检查异步 tokenizer 加载状态。 |
| `DELETE /v1/tokenizers/{id}` | `DELETE` | 从 registry 移除 tokenizer。 |

**Tokenize Request：**

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "prompt": "Hello, world!"
}
```

**Batch Tokenize Request：**

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "prompt": ["Hello", "World", "How are you?"]
}
```

**Tokenize Response：**

```json
{
  "tokens": [15339, 11, 1917, 0],
  "count": 4,
  "char_count": 13
}
```

**Detokenize Request：**

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "tokens": [15339, 11, 1917, 0],
  "skip_special_tokens": true
}
```

**添加 Tokenizer（异步注册）：**

```bash
# Register from HuggingFace
curl -X POST http://localhost:30000/v1/tokenizers \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3", "source": "meta-llama/Llama-3.1-8B-Instruct"}'

# Check status
curl http://localhost:30000/v1/tokenizers/{tokenizer_id}/status
```

### Parser Endpoints

Gateway 提供 admin endpoint，用于从 LLM output 中解析 reasoning content 和 function call。

| Endpoint | Method | Description |
|---|---|---|
| `POST /parse/reasoning` | `POST` | 从普通文本中分离 reasoning（`<think>`）。 |
| `POST /parse/function_call` | `POST` | 从文本中解析 function/tool calls。 |

**Separate Reasoning Request：**

```json
{
  "text": "<think>Let me analyze this step by step...</think>The answer is 42.",
  "parser": "deepseek-r1"
}
```

**Response：**

```json
{
  "normal_text": "The answer is 42.",
  "reasoning_text": "Let me analyze this step by step..."
}
```

**支持的 Reasoning Parser：**

- `deepseek-r1` - DeepSeek-R1（初始 reasoning 模式）
- `qwen3` - Qwen-3 模型
- `qwen3-thinking` / `qwen-thinking` - Qwen thinking 变体
- `kimi` - 使用 Unicode token 的 Kimi K2
- `glm45` / `glm47` - GLM-4.5/4.6/4.7 模型
- `step3` - Step-3 模型
- `minimax` - MiniMax 模型

**Function Call Parsing：**

```json
{
  "text": "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"NYC\"}}",
  "parser": "json"
}
```

支持的 tool parser：`json`、`python`、`xml`。

## Conversations、Responses 和 Data Connectors

- `--history-backend memory`（默认）在进程内存储 responses 和 conversations。
- `--history-backend none` 禁用持久化，但保留 API。
- `--history-backend oracle` 使用 Oracle Autonomous Database；通过 flag 或环境变量提供凭据。
- `--history-backend postgres` 使用 PostgreSQL Database。
- `--history-backend redis` 使用 Redis。
- Conversation item storage 与 history backend 对齐（Oracle 或 memory）。同一存储支撑 OpenAI `/responses` 和 conversation API。

### History Backend（OpenAI Router 模式）

存储 conversation 和 response 数据，用于跟踪、调试或分析。

> **注意：** History backend 当前仅支持 `--backend openai` 模式。gRPC 模式对 `/v1/responses` API 的支持仍在规划中。

#### 可用存储选项

- **Memory**（默认）：内存存储，速度快但非持久。
- **None**：不存储，开销最小。
- **Oracle**：由 Oracle Autonomous Database 支撑的持久存储。
- **Postgres**：由 PostgreSQL Database 支撑的持久存储。
- **Redis**：由 Redis 支撑的持久存储。

```bash
# Memory backend (default)
python3 -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend memory

# No storage for maximum performance
python3 -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend none

# Oracle ATP backend (see configuration below)
python3 -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend oracle

# PostgreSQL backend
python3 -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend postgres

# Redis backend
python3 -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend redis
```

#### Oracle 配置

安装 Oracle Instant Client，并相应设置 `LD_LIBRARY_PATH`。选择 **一种** 连接方式：

```bash
# Option 1: Full connection descriptor
export ATP_DSN="(description=(address=(protocol=tcps)(port=1522)(host=adb.region.oraclecloud.com))(connect_data=(service_name=service_name)))"

# Option 2: TNS alias (requires wallet)
export ATP_TNS_ALIAS="sglroutertestatp_high"
export ATP_WALLET_PATH="/path/to/wallet"
```

提供数据库凭据和可选连接池大小：

```bash
export ATP_USER="admin"
export ATP_PASSWORD="YourPassword123"
export ATP_POOL_MIN=4
export ATP_POOL_MAX=32
```

Router flag 映射关系：

- `--oracle-dsn`（env: `ATP_DSN`），或同时使用 `--oracle-tns-alias` 与 `--oracle-wallet-path`。
- `--oracle-user` / `--oracle-password`（`ATP_USER` / `ATP_PASSWORD`）。
- 使用 TNS alias 时设置 `--oracle-wallet-path`（`ATP_WALLET_PATH`）。
- `--oracle-pool-min`、`--oracle-pool-max`、`--oracle-pool-timeout-secs`。

`--oracle-dsn` 和 `--oracle-tns-alias` 只能提供一个。

#### Redis 配置

提供 Redis 连接 URL 和可选连接池大小：

```bash
export REDIS_URL="redis://localhost:6379"
export REDIS_POOL_MAX=16
export REDIS_RETENTION_DAYS=30
```

Router flag 映射关系：

- `--redis-url`（env: `REDIS_URL`）
- `--redis-pool-max`（env: `REDIS_POOL_MAX`）
- `--redis-retention-days`（env: `REDIS_RETENTION_DAYS`）。设为 `-1` 表示持久存储（默认：30 天）。

## 可靠性与流控

- **HTTP Client**：上游 HTTP client 连接设置默认值为 pool idle timeout 50s、connect timeout 10s、每个 host 最大空闲连接 500、TCP keepalive 30s。可通过 `--pool-idle-timeout-secs`、`--connect-timeout-secs`、`--pool-max-idle-per-host`、`--tcp-keepalive-secs` 或对应 `SMG_*` 环境变量配置。
- **Retries**：默认最大重试次数为 5，使用指数退避（`--retry-max-retries`、`--retry-initial-backoff-ms`、`--retry-max-backoff-ms`、`--retry-backoff-multiplier`、`--retry-jitter-factor`）。408/429/500/502/503/504 会触发重试。
- **Circuit Breakers**：worker 级阈值（`--cb-failure-threshold`、`--cb-success-threshold`、`--cb-timeout-duration-secs`、`--cb-window-duration-secs`）。可通过 `--disable-circuit-breaker` 禁用。
- **Rate Limiting**：由 `--max-concurrent-requests` 驱动 token bucket。可设置 `--rate-limit-tokens-per-second` 覆盖 refill rate。通过 `--queue-size` 和 `--queue-timeout-secs` 配置请求队列；排队请求遵循 FIFO，并尊重 cancellation。
- **Health Checks**：通过 `--health-check-interval-secs`、`--health-check-timeout-secs`、失败/成功阈值和 `--health-check-endpoint` 配置 runtime probe。使用 `--disable-health-check` 可完全跳过健康检查。
- **Cache Management**：重新部署 PD worker 时，`/flush_cache` 可确保 LRU eviction。

## 负载均衡策略

- `random`：均匀随机选择 worker。
- `round_robin`：使用原子计数器顺序轮转。
- `cache_aware`：维护 prompt prefix tree，用于重复流量路由，并通过可配置阈值均衡负载（`--cache-threshold`、`--balance-abs-threshold`、`--balance-rel-threshold`、`--eviction-interval`、`--max-tree-size`）。
- `power_of_two`：在两个随机候选 worker 中选择更轻的一个；与 `LoadMonitor` 集成。
  PD 模式可通过 `--prefill-policy`、`--decode-policy` 做模型级 override；IGW 模式可通过 worker registry 实现。

## 观测

### 日志

通过 `tracing` 输出结构化 tracing，可选文件 sink（`--log-dir`）和 `--log-level`（`debug`、`info`、`warn`、`error`）。

### Prometheus Metrics

使用 `--prometheus-host`/`--prometheus-port` 启用（默认 `0.0.0.0:29000`）。

**指标类别（40+ metrics）：**

| Layer | Metric Prefix | Description |
|---|---|---|
| HTTP | `smg_http_*` | 请求数、耗时、活跃连接、限流 |
| Router | `smg_router_*` | 按 model/endpoint 统计的请求、延迟、错误、上游响应 |
| Inference | `smg_router_ttft/tpot/tokens_*` | 首 token 时间、每输出 token 时间、token 计数（gRPC） |
| Worker | `smg_worker_*` | 池大小、活跃连接、健康检查、选择事件 |
| Circuit Breaker | `smg_worker_cb_*` | 状态（closed/open/half-open）、转换、结果 |
| Retry | `smg_worker_retries_*` | 重试次数、耗尽重试、退避时长 |
| Discovery | `smg_discovery_*` | K8s 注册、同步耗时、发现的 worker |
| MCP | `smg_mcp_*` | Tool call、耗时、活跃 server、迭代 |
| Database | `smg_db_*` | 操作、耗时、连接、已存储 item |

**关键指标：**

- `smg_router_ttft_seconds` - 首 token 时间 histogram（gRPC 模式）
- `smg_router_tpot_seconds` - 每输出 token 时间 histogram（gRPC 模式）
- `smg_router_tokens_total` - 按模型统计的 input/output token 总数
- `smg_router_generation_duration_seconds` - 端到端 generation 时间
- `smg_worker_cb_state` - Circuit breaker 状态 gauge（0=closed，1=open，2=half-open）

**Duration Buckets：**

1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s, 15s, 30s, 45s, 60s, 90s, 120s, 180s, 240s

### OpenTelemetry Tracing

使用 OTLP export 启用分布式 tracing：

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --enable-trace \
  --otlp-traces-endpoint localhost:4317
```

**功能：**

- OTLP/gRPC exporter（默认端口 4317）
- HTTP 和 gRPC 的 W3C Trace Context 传播
- batch span processing（500ms delay、64 span batch size）
- 自定义过滤以减少噪声（只导出相关 span）
- 向上游 worker request 注入 trace context

**配置：**

- `--enable-trace` - 启用 OpenTelemetry tracing
- `--otlp-traces-endpoint <host:port>` - OTLP collector endpoint

### Request ID 传播

配置用于提取 request ID 的 header：

```bash
--request-id-headers x-request-id x-trace-id x-correlation-id
```

响应会包含 `x-request-id` header 以便关联。

### CORS

通过 `--cors-allowed-origins` 设置浏览器访问来源。

## 安全

### Router 和 Worker API Keys

- **Router API key（`--api-key`）** 保护客户端访问 router endpoint；所有受保护路由都期望 `Authorization: Bearer <key>`。
- `--worker-urls` 中列出的 worker 会自动继承 router API key。
- 动态添加 worker 时，需要通过 payload 或 query string 显式提供 API key；不会自动继承。

```bash
# Router and initial workers share the same key
python3 -m sglang_router.launch_router \
  --api-key "shared-api-key" \
  --worker-urls http://worker1:8000 http://worker2:8000

# Adding a worker without key while router has one triggers a warning and leaves the worker unprotected
curl -X POST http://localhost:8080/add_worker?url=http://worker3:8000

# Add worker with explicit key
curl -X POST "http://localhost:8080/add_worker?url=http://worker3:8000&api_key=worker3-specific-key"
```

### 安全配置

1. **无认证**（默认）：router 和 worker 接受无 key 请求；仅应在可信环境使用。
2. **仅 Router 认证**：提供 `--api-key`；客户端必须提供 key，router 访问 worker 时不带凭据。
3. **仅 Worker 认证**：router 对客户端开放；每个 worker 需要自己的 key。调用 `/workers` 或 `/add_worker` 时提供 key。
4. **完整认证**：设置 router API key，并提供每个 worker 的 key。例如：
   ```bash
   python3 -m sglang_router.launch_router --api-key "router-key"
   curl -H "Authorization: Bearer router-key" \
     -X POST http://localhost:8080/add_worker?url=http://worker:8000&api_key=worker-key
   ```

### 重要说明

- 通过 CLI 声明的初始 worker 会继承 router key；动态 worker 必须显式提供 key。
- 当 router 期望认证但 worker 注册时未带 key，router 会记录 warning。
- 当 router 和 worker 共享同一 key 时，调用动态注册 API 仍应包含该 key。

### Gateway Server 的 TLS（HTTPS）

启用 TLS，通过 HTTPS 提供 gateway：

```bash
python3 -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --tls-cert-path /path/to/server.crt \
  --tls-key-path /path/to/server.key
```

| Parameter | Description |
|---|---|
| `--tls-cert-path` | server certificate 路径（PEM 格式） |
| `--tls-key-path` | server private key 路径（PEM 格式） |

两个参数必须同时提供。Gateway 使用 rustls 和 ring crypto provider 做 TLS termination。未配置 TLS 时，gateway 回退到普通 HTTP。

### Worker 通信的 mTLS

HTTP 模式下，启用 mutual TLS（mTLS）以安全访问 worker：

```bash
python3 -m sglang_router.launch_router \
  --worker-urls https://worker1:8443 https://worker2:8443 \
  --client-cert-path /path/to/client.crt \
  --client-key-path /path/to/client.key \
  --ca-cert-path /path/to/ca.crt
```

| Parameter | Description |
|---|---|
| `--client-cert-path` | mTLS client certificate 路径（PEM 格式） |
| `--client-key-path` | mTLS client private key 路径（PEM 格式） |
| `--ca-cert-path` | 用于验证 worker TLS 的 CA certificate 路径（PEM 格式） |

**关键点：**

- client certificate 和 key 必须同时提供。
- 可通过多个 `--ca-cert-path` flag 添加多个 CA certificate。
- 配置 TLS 时使用 rustls backend。
- 为所有 worker 创建单个 HTTP client（假设处于同一安全域）。
- 长连接启用 TCP keepalive（30 秒）。

**完整 TLS 示例（Gateway HTTPS + Worker mTLS）：**

```bash
python3 -m sglang_router.launch_router \
  --worker-urls https://worker1:8443 https://worker2:8443 \
  --tls-cert-path /etc/certs/server.crt \
  --tls-key-path /etc/certs/server.key \
  --client-cert-path /etc/certs/client.crt \
  --client-key-path /etc/certs/client.key \
  --ca-cert-path /etc/certs/ca.crt \
  --api-key "secure-api-key"
```

### 控制面认证

Gateway 支持控制面 API（worker 管理、tokenizer 注册、cache 操作）的基于角色访问控制（RBAC）。可用两种认证方式：

#### 认证方式

| Method | Use Case | Configuration |
|---|---|---|
| **API Keys** | service account、内部服务 | `--control-plane-api-keys` |
| **JWT/OIDC** | 通过 Identity Provider 进行用户认证 | `--jwt-issuer`、`--jwt-audience` |

两种方式可同时使用。请求认证顺序为：API key，然后 JWT token。

#### 角色

| Role | Access |
|---|---|
| `admin` | 可访问所有控制面 API（workers、tokenizers、cache 等） |
| `user` | 仅可访问 inference/data plane API（chat completions、embeddings 等） |

#### API Key 认证

为 service account 和自动化任务配置静态 API key：

```bash
python3 -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --control-plane-api-keys 'svc1:CI Pipeline:admin:secret-key-123' \
                           'svc2:Monitoring:user:readonly-key-456' \
  --control-plane-audit-enabled
```

**格式：** `id:name:role:key`

- `id` - key 的唯一标识。
- `name` - 人类可读描述。
- `role` - `admin` 或 `user`。
- `key` - secret key（内部以 SHA-256 hash 存储）。

**用法：**

```bash
curl -H "Authorization: Bearer secret-key-123" \
  http://localhost:30000/workers
```

#### JWT/OIDC 认证

通过外部 Identity Provider（Azure AD、Okta、Auth0、Keycloak 等）认证用户：

```bash
python3 -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --jwt-issuer "https://login.microsoftonline.com/{tenant-id}/v2.0" \
  --jwt-audience "api://my-gateway-client-id" \
  --jwt-jwks-uri "https://login.microsoftonline.com/{tenant-id}/discovery/v2.0/keys" \
  --jwt-role-mapping 'Gateway.Admins=admin' 'Gateway.Users=user' \
  --control-plane-audit-enabled
```

| Parameter | Description |
|---|---|
| `--jwt-issuer` | OIDC issuer URL。用于验证 `iss` claim，并通过 `.well-known/openid-configuration` 发现 JWKS endpoint。 |
| `--jwt-audience` | 期望 audience（`aud` claim）。通常是应用 client ID 或 API identifier，例如 `api://client-id`。 |
| `--jwt-jwks-uri` | （可选）显式 JWKS URI。省略时会从 issuer 的 OIDC 配置自动发现。 |
| `--jwt-role-mapping` | 将 IDP group/role 名映射为 gateway role。格式：`idp_role=gateway_role`。 |

**工作方式：**

1. 用户通过 Identity Provider 认证（OAuth2/OIDC flow）。
2. IDP 签发 JWT token。
3. 用户向 gateway 发送 token：`Authorization: Bearer <jwt-token>`。
4. Gateway 校验 JWT：
   - 使用 JWKS 校验签名。
   - 检查 `iss` 是否匹配 `--jwt-issuer`。
   - 检查 `aud` 是否匹配 `--jwt-audience`。
   - 校验过期时间和其他标准 claim。
   - 从 `roles` claim 提取角色（或 fallback 到 `groups`）。
   - 通过 `--jwt-role-mapping` 将 IDP role 映射为 gateway role。

**Azure AD 配置示例：**

```bash
# Azure AD issues tokens with:
#   iss: https://login.microsoftonline.com/{tenant}/v2.0
#   aud: api://your-client-id (or the client ID itself)
#   roles: ["Gateway.Admins"] or groups: ["group-id"]

python3 -m sglang_router.launch_router \
  --jwt-issuer "https://login.microsoftonline.com/your-tenant-id/v2.0" \
  --jwt-audience "api://your-client-id" \
  --jwt-role-mapping 'Gateway.Admins=admin' 'Gateway.Users=user'
```

#### Audit Logging

启用 `--control-plane-audit-enabled` 可记录所有控制面操作，包含：

- Timestamp
- Principal（API key ID 或 JWT subject）
- Role
- 执行动作
- 成功/失败状态

#### 组合认证示例

针对不同用例同时使用 API key 和 JWT：

```bash
python3 -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  # API keys for service accounts
  --control-plane-api-keys 'ci:CI/CD Pipeline:admin:ci-secret' \
  # JWT for human users via Azure AD
  --jwt-issuer "https://login.microsoftonline.com/{tenant}/v2.0" \
  --jwt-audience "api://gateway" \
  --jwt-role-mapping 'Platform.Admins=admin' 'Platform.Users=user' \
  # Enable audit logging
  --control-plane-audit-enabled
```

## 开发与测试

```bash
# Build Rust components (debug mode, fast)
cargo build

# Run Rust tests
cargo test

# Fast Python development (rebuilds and installs in debug mode)
cd bindings/python && maturin develop

# Run Python tests
cd ../..  # Back to sgl-model-gateway root
pytest e2e_test/
```

生产构建时，在 `bindings/python/` 目录下使用 `maturin build --release --out dist` 创建优化 wheel。开发时，`maturin develop` 会即时重建并安装，不生成 wheel 文件。可使用 `python -m sglang_router.launch_server` 在本地小集群中同时启动 router 和 SGLang worker 进行验证。

### 构建缓存

**本地开发** 默认使用增量编译（在 `.cargo/config.toml` 中配置），适合编辑-编译-测试循环。

**release 构建或 CI** 可选使用 [sccache](https://github.com/mozilla/sccache) 缓存编译产物：

```bash
# Install sccache
cargo install sccache

# Option 1: Set environment variable (per-session)
export RUSTC_WRAPPER=sccache
cargo build --release

# Option 2: Add to your global cargo config (~/.cargo/config.toml)
# [build]
# rustc-wrapper = "sccache"
```

> **注意：** sccache 和增量编译互斥，sccache 无法缓存增量编译 crate。项目默认对本地迭代使用增量编译以提升速度。对 clean/release build，如果跨构建缓存更重要，可使用 sccache。CI workflow 使用 sccache 和 GitHub Actions cache backend 实现跨 job 编译缓存。

---

## Release 管理

### 创建 Gateway Release

为 Gateway/Router 组件创建 release，并按路径过滤 commit：

```bash
# Using make
make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0

# Save to file
make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0 OUTPUT=RELEASE_NOTES.md

# Create draft release (requires gh CLI, DEFAULT behavior)
make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0 CREATE_RELEASE=1

# Publish release immediately (requires gh CLI)
make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0 CREATE_RELEASE=1 DRAFT=0
```

**Tag 命名**：使用 `gateway-*` 或 `router-*` 前缀，避免触发无关 CI workflow。

### Release Workflow

1. **创建并 push tag**：
   ```bash
   git tag -a gateway-v1.0.0 <commit-hash> -m "Gateway release v1.0.0"
   git push origin gateway-v1.0.0
   ```

2. **生成 release notes**（自动过滤 gateway 相关 commit）：
   ```bash
   make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0
   ```

3. **创建 GitHub release**：
   ```bash
   # Create draft (DEFAULT - review before publishing)
   make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0 CREATE_RELEASE=1

   # Or publish immediately (skip draft)
   make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0 CREATE_RELEASE=1 DRAFT=0
   ```

### 过滤路径

Release notes 只包含触及以下路径的 commit：

- `sgl-model-gateway/` - Router codebase
- `python/sglang/srt/grpc/` - gRPC protocol
- `python/sglang/srt/entrypoints/grpc_server.py` - gRPC server

脚本会自动提取作者归属、PR 链接，并识别新 contributor。

---

SGLang Model Gateway 会随核心 SGLang runtime 持续演进。贡献时应保持 CLI flag、文档和 Python binding 与 Rust 实现同步。
