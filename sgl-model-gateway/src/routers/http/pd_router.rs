// CODEx 中文注释：本文件已按用户要求追加中文文件注释、函数注释和逐行注释；原始源码行保持不变。
// 中文文件注释：本文件实现 SGLang model gateway 的 HTTP PD Router，负责把请求拆分/分发到 prefill 与 decode worker，并处理响应、流式输出、日志概率合并、worker 负载与熔断状态。
// 中文文件注释：以下新增中文注释仅用于阅读说明，不改变 Rust 代码逻辑；原有英文注释全部保留。
// 中文逐行注释：下一行是原始源码第 1 行，保持原始代码不变。
use std::{sync::Arc, time::Instant};

// 中文逐行注释：下一行是原始源码第 3 行，保持原始代码不变。
use async_trait::async_trait;
// 中文逐行注释：下一行是原始源码第 4 行，保持原始代码不变。
use axum::{
    // 中文逐行注释：下一行是原始源码第 5 行，保持原始代码不变。
    body::Body,
    // 中文逐行注释：下一行是原始源码第 6 行，保持原始代码不变。
    extract::Request,
    // 中文逐行注释：下一行是原始源码第 7 行，保持原始代码不变。
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    // 中文逐行注释：下一行是原始源码第 8 行，保持原始代码不变。
    response::{IntoResponse, Response},
// 中文逐行注释：下一行是原始源码第 9 行，保持原始代码不变。
};
// 中文逐行注释：下一行是原始源码第 10 行，保持原始代码不变。
use futures_util::StreamExt;
// 中文逐行注释：下一行是原始源码第 11 行，保持原始代码不变。
use memchr::memmem;
// 中文逐行注释：下一行是原始源码第 12 行，保持原始代码不变。
use reqwest::Client;
// 中文逐行注释：下一行是原始源码第 13 行，保持原始代码不变。
use serde::Serialize;
// 中文逐行注释：下一行是原始源码第 14 行，保持原始代码不变。
use serde_json::{json, Value};
// 中文逐行注释：下一行是原始源码第 15 行，保持原始代码不变。
use tokio_stream::wrappers::UnboundedReceiverStream;
// 中文逐行注释：下一行是原始源码第 16 行，保持原始代码不变。
use tracing::{debug, error, warn};

// 中文逐行注释：下一行是原始源码第 18 行，保持原始代码不变。
use super::pd_types::api_path;
// 中文逐行注释：下一行是原始源码第 19 行，保持原始代码不变。
use crate::{
    // 中文逐行注释：下一行是原始源码第 20 行，保持原始代码不变。
    config::types::RetryConfig,
    // 中文逐行注释：下一行是原始源码第 21 行，保持原始代码不变。
    core::{
        // 中文逐行注释：下一行是原始源码第 22 行，保持原始代码不变。
        is_retryable_status, HashRing, RetryExecutor, Worker, WorkerLoadGuard, WorkerRegistry,
        // 中文逐行注释：下一行是原始源码第 23 行，保持原始代码不变。
        WorkerType, UNKNOWN_MODEL_ID,
    // 中文逐行注释：下一行是原始源码第 24 行，保持原始代码不变。
    },
    // 中文逐行注释：下一行是原始源码第 25 行，保持原始代码不变。
    observability::{
        // 中文逐行注释：下一行是原始源码第 26 行，保持原始代码不变。
        events::{self, Event},
        // 中文逐行注释：下一行是原始源码第 27 行，保持原始代码不变。
        metrics::{bool_to_static_str, metrics_labels, Metrics},
        // 中文逐行注释：下一行是原始源码第 28 行，保持原始代码不变。
        otel_trace::inject_trace_context_http,
    // 中文逐行注释：下一行是原始源码第 29 行，保持原始代码不变。
    },
    // 中文逐行注释：下一行是原始源码第 30 行，保持原始代码不变。
    policies::{LoadBalancingPolicy, PolicyRegistry, SelectWorkerInfo},
    // 中文逐行注释：下一行是原始源码第 31 行，保持原始代码不变。
    protocols::{
        // 中文逐行注释：下一行是原始源码第 32 行，保持原始代码不变。
        chat::{ChatCompletionRequest, ChatMessage, MessageContent},
        // 中文逐行注释：下一行是原始源码第 33 行，保持原始代码不变。
        classify::ClassifyRequest,
        // 中文逐行注释：下一行是原始源码第 34 行，保持原始代码不变。
        common::{InputIds, StringOrArray},
        // 中文逐行注释：下一行是原始源码第 35 行，保持原始代码不变。
        completion::CompletionRequest,
        // 中文逐行注释：下一行是原始源码第 36 行，保持原始代码不变。
        embedding::EmbeddingRequest,
        // 中文逐行注释：下一行是原始源码第 37 行，保持原始代码不变。
        generate::GenerateRequest,
        // 中文逐行注释：下一行是原始源码第 38 行，保持原始代码不变。
        rerank::RerankRequest,
    // 中文逐行注释：下一行是原始源码第 39 行，保持原始代码不变。
    },
    // 中文逐行注释：下一行是原始源码第 40 行，保持原始代码不变。
    routers::{
        // 中文逐行注释：下一行是原始源码第 41 行，保持原始代码不变。
        error,
        // 中文逐行注释：下一行是原始源码第 42 行，保持原始代码不变。
        grpc::utils::{error_type_from_status, route_to_endpoint},
        // 中文逐行注释：下一行是原始源码第 43 行，保持原始代码不变。
        header_utils,
        // 中文逐行注释：下一行是原始源码第 44 行，保持原始代码不变。
        streaming_utils::BreakerTrackedStream,
        // 中文逐行注释：下一行是原始源码第 45 行，保持原始代码不变。
        RouterTrait,
    // 中文逐行注释：下一行是原始源码第 46 行，保持原始代码不变。
    },
// 中文逐行注释：下一行是原始源码第 47 行，保持原始代码不变。
};

// 中文逐行注释：下一行是原始源码第 49 行，保持原始代码不变。
#[derive(Debug)]
// 中文逐行注释：下一行是原始源码第 50 行，保持原始代码不变。
pub struct PDRouter {
    // 中文逐行注释：下一行是原始源码第 51 行，保持原始代码不变。
    pub worker_registry: Arc<WorkerRegistry>,
    // 中文逐行注释：下一行是原始源码第 52 行，保持原始代码不变。
    pub policy_registry: Arc<PolicyRegistry>,
    // 中文逐行注释：下一行是原始源码第 53 行，保持原始代码不变。
    pub client: Client,
    // 中文逐行注释：下一行是原始源码第 54 行，保持原始代码不变。
    pub retry_config: RetryConfig,
    // 中文逐行注释：下一行是原始源码第 55 行，保持原始代码不变。
    pub api_key: Option<String>,
    // 中文逐行注释：下一行是原始源码第 56 行，保持原始代码不变。
    pub enable_igw: bool,
// 中文逐行注释：下一行是原始源码第 57 行，保持原始代码不变。
}

// 中文逐行注释：下一行是原始源码第 59 行，保持原始代码不变。
#[derive(Clone)]
// 中文逐行注释：下一行是原始源码第 60 行，保持原始代码不变。
struct PDRequestContext<'a> {
    // 中文逐行注释：下一行是原始源码第 61 行，保持原始代码不变。
    route: &'static str,
    // 中文逐行注释：下一行是原始源码第 62 行，保持原始代码不变。
    batch_size: Option<usize>,
    // 中文逐行注释：下一行是原始源码第 63 行，保持原始代码不变。
    is_stream: bool,
    // 中文逐行注释：下一行是原始源码第 64 行，保持原始代码不变。
    return_logprob: bool,
    // 中文逐行注释：下一行是原始源码第 65 行，保持原始代码不变。
    request_text: Option<String>,
    // 中文逐行注释：下一行是原始源码第 66 行，保持原始代码不变。
    model_id: Option<&'a str>,
    // 中文逐行注释：下一行是原始源码第 67 行，保持原始代码不变。
    headers: Option<HeaderMap>,
// 中文逐行注释：下一行是原始源码第 68 行，保持原始代码不变。
}

// 中文逐行注释：下一行是原始源码第 70 行，保持原始代码不变。
/// Marker placed on a `Response` by paths inside
// 中文逐行注释：下一行是原始源码第 71 行，保持原始代码不变。
/// `execute_dual_dispatch_internal` that have already recorded prefill and
// 中文逐行注释：下一行是原始源码第 72 行，保持原始代码不变。
/// decode breaker outcomes against the workers' actual per-side results
// 中文逐行注释：下一行是原始源码第 73 行，保持原始代码不变。
/// (rather than the final response status). The outer dispatcher reads this
// 中文逐行注释：下一行是原始源码第 74 行，保持原始代码不变。
/// and skips its own status-based `record_outcome` calls so a decode-only
// 中文逐行注释：下一行是原始源码第 75 行，保持原始代码不变。
/// transport failure can't be misattributed to a healthy prefill.
// 中文逐行注释：下一行是原始源码第 76 行，保持原始代码不变。
#[derive(Clone, Copy)]
// 中文逐行注释：下一行是原始源码第 77 行，保持原始代码不变。
struct BreakerOutcomesRecorded;

// 中文逐行注释：下一行是原始源码第 79 行，保持原始代码不变。
impl PDRouter {
    // 中文函数注释：代理 GET 请求到第一个可用的 prefill worker，用于简单透传类接口。
    // 中文逐行注释：下一行是原始源码第 80 行，保持原始代码不变。
    async fn proxy_to_first_prefill_worker(
        // 中文逐行注释：下一行是原始源码第 81 行，保持原始代码不变。
        &self,
        // 中文逐行注释：下一行是原始源码第 82 行，保持原始代码不变。
        endpoint: &str,
        // 中文逐行注释：下一行是原始源码第 83 行，保持原始代码不变。
        headers: Option<Vec<(String, String)>>,
    // 中文逐行注释：下一行是原始源码第 84 行，保持原始代码不变。
    ) -> Response {
        // 中文逐行注释：下一行是原始源码第 85 行，保持原始代码不变。
        let workers = self.worker_registry.get_prefill_workers();
        // 中文逐行注释：下一行是原始源码第 86 行，保持原始代码不变。
        let first_worker_url = workers.first().map(|w| w.url().to_string());

        // 中文逐行注释：下一行是原始源码第 88 行，保持原始代码不变。
        if let Some(worker_url) = first_worker_url {
            // 中文逐行注释：下一行是原始源码第 89 行，保持原始代码不变。
            self.proxy_to_worker(worker_url, endpoint, headers).await
        // 中文逐行注释：下一行是原始源码第 90 行，保持原始代码不变。
        } else {
            // 中文逐行注释：下一行是原始源码第 91 行，保持原始代码不变。
            error::service_unavailable("no_prefill_servers", "No prefill servers available")
        // 中文逐行注释：下一行是原始源码第 92 行，保持原始代码不变。
        }
    // 中文逐行注释：下一行是原始源码第 93 行，保持原始代码不变。
    }

    // 中文函数注释：向指定 worker 发起 GET 代理请求，并把后端响应转换为 router 响应。
    // 中文逐行注释：下一行是原始源码第 95 行，保持原始代码不变。
    async fn proxy_to_worker(
        // 中文逐行注释：下一行是原始源码第 96 行，保持原始代码不变。
        &self,
        // 中文逐行注释：下一行是原始源码第 97 行，保持原始代码不变。
        worker_url: String,
        // 中文逐行注释：下一行是原始源码第 98 行，保持原始代码不变。
        endpoint: &str,
        // 中文逐行注释：下一行是原始源码第 99 行，保持原始代码不变。
        headers: Option<Vec<(String, String)>>,
    // 中文逐行注释：下一行是原始源码第 100 行，保持原始代码不变。
    ) -> Response {
        // 中文逐行注释：下一行是原始源码第 101 行，保持原始代码不变。
        let url = format!("{}/{}", worker_url, endpoint);
        // 中文逐行注释：下一行是原始源码第 102 行，保持原始代码不变。
        let mut request_builder = self.client.get(&url);

        // 中文逐行注释：下一行是原始源码第 104 行，保持原始代码不变。
        if let Some(headers) = headers {
            // 中文逐行注释：下一行是原始源码第 105 行，保持原始代码不变。
            for (name, value) in headers {
                // 中文逐行注释：下一行是原始源码第 106 行，保持原始代码不变。
                request_builder = request_builder.header(name, value);
            // 中文逐行注释：下一行是原始源码第 107 行，保持原始代码不变。
            }
        // 中文逐行注释：下一行是原始源码第 108 行，保持原始代码不变。
        }

        // 中文逐行注释：下一行是原始源码第 110 行，保持原始代码不变。
        match request_builder.send().await {
            // 中文逐行注释：下一行是原始源码第 111 行，保持原始代码不变。
            Ok(res) if res.status().is_success() => {
                // 中文逐行注释：下一行是原始源码第 112 行，保持原始代码不变。
                let response_headers = header_utils::preserve_response_headers(res.headers());

                // 中文逐行注释：下一行是原始源码第 114 行，保持原始代码不变。
                match res.bytes().await {
                    // 中文逐行注释：下一行是原始源码第 115 行，保持原始代码不变。
                    Ok(body) => {
                        // 中文逐行注释：下一行是原始源码第 116 行，保持原始代码不变。
                        let mut response = Response::new(Body::from(body));
                        // 中文逐行注释：下一行是原始源码第 117 行，保持原始代码不变。
                        *response.status_mut() = StatusCode::OK;
                        // 中文逐行注释：下一行是原始源码第 118 行，保持原始代码不变。
                        *response.headers_mut() = response_headers;
                        // 中文逐行注释：下一行是原始源码第 119 行，保持原始代码不变。
                        response
                    // 中文逐行注释：下一行是原始源码第 120 行，保持原始代码不变。
                    }
                    // 中文逐行注释：下一行是原始源码第 121 行，保持原始代码不变。
                    Err(e) => {
                        // 中文逐行注释：下一行是原始源码第 122 行，保持原始代码不变。
                        error!("Failed to read response body: {}", e);
                        // 中文逐行注释：下一行是原始源码第 123 行，保持原始代码不变。
                        error::internal_error(
                            // 中文逐行注释：下一行是原始源码第 124 行，保持原始代码不变。
                            "read_response_body_failed",
                            // 中文逐行注释：下一行是原始源码第 125 行，保持原始代码不变。
                            format!("Failed to read response body: {}", e),
                        // 中文逐行注释：下一行是原始源码第 126 行，保持原始代码不变。
                        )
                    // 中文逐行注释：下一行是原始源码第 127 行，保持原始代码不变。
                    }
                // 中文逐行注释：下一行是原始源码第 128 行，保持原始代码不变。
                }
            // 中文逐行注释：下一行是原始源码第 129 行，保持原始代码不变。
            }
            // 中文逐行注释：下一行是原始源码第 130 行，保持原始代码不变。
            Ok(res) => {
                // 中文逐行注释：下一行是原始源码第 131 行，保持原始代码不变。
                let status = StatusCode::from_u16(res.status().as_u16())
                    // 中文逐行注释：下一行是原始源码第 132 行，保持原始代码不变。
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                // 中文逐行注释：下一行是原始源码第 133 行，保持原始代码不变。
                // Use the status code to determine which error function to use
                // 中文逐行注释：下一行是原始源码第 134 行，保持原始代码不变。
                match status {
                    // 中文逐行注释：下一行是原始源码第 135 行，保持原始代码不变。
                    StatusCode::BAD_REQUEST => error::bad_request(
                        // 中文逐行注释：下一行是原始源码第 136 行，保持原始代码不变。
                        "server_bad_request",
                        // 中文逐行注释：下一行是原始源码第 137 行，保持原始代码不变。
                        format!("Server returned status: {}", res.status()),
                    // 中文逐行注释：下一行是原始源码第 138 行，保持原始代码不变。
                    ),
                    // 中文逐行注释：下一行是原始源码第 139 行，保持原始代码不变。
                    StatusCode::NOT_FOUND => error::not_found(
                        // 中文逐行注释：下一行是原始源码第 140 行，保持原始代码不变。
                        "server_not_found",
                        // 中文逐行注释：下一行是原始源码第 141 行，保持原始代码不变。
                        format!("Server returned status: {}", res.status()),
                    // 中文逐行注释：下一行是原始源码第 142 行，保持原始代码不变。
                    ),
                    // 中文逐行注释：下一行是原始源码第 143 行，保持原始代码不变。
                    StatusCode::INTERNAL_SERVER_ERROR => error::internal_error(
                        // 中文逐行注释：下一行是原始源码第 144 行，保持原始代码不变。
                        "server_internal_error",
                        // 中文逐行注释：下一行是原始源码第 145 行，保持原始代码不变。
                        format!("Server returned status: {}", res.status()),
                    // 中文逐行注释：下一行是原始源码第 146 行，保持原始代码不变。
                    ),
                    // 中文逐行注释：下一行是原始源码第 147 行，保持原始代码不变。
                    StatusCode::SERVICE_UNAVAILABLE => error::service_unavailable(
                        // 中文逐行注释：下一行是原始源码第 148 行，保持原始代码不变。
                        "server_unavailable",
                        // 中文逐行注释：下一行是原始源码第 149 行，保持原始代码不变。
                        format!("Server returned status: {}", res.status()),
                    // 中文逐行注释：下一行是原始源码第 150 行，保持原始代码不变。
                    ),
                    // 中文逐行注释：下一行是原始源码第 151 行，保持原始代码不变。
                    StatusCode::BAD_GATEWAY => error::bad_gateway(
                        // 中文逐行注释：下一行是原始源码第 152 行，保持原始代码不变。
                        "server_bad_gateway",
                        // 中文逐行注释：下一行是原始源码第 153 行，保持原始代码不变。
                        format!("Server returned status: {}", res.status()),
                    // 中文逐行注释：下一行是原始源码第 154 行，保持原始代码不变。
                    ),
                    // 中文逐行注释：下一行是原始源码第 155 行，保持原始代码不变。
                    _ => error::internal_error(
                        // 中文逐行注释：下一行是原始源码第 156 行，保持原始代码不变。
                        "server_error",
                        // 中文逐行注释：下一行是原始源码第 157 行，保持原始代码不变。
                        format!("Server returned status: {}", res.status()),
                    // 中文逐行注释：下一行是原始源码第 158 行，保持原始代码不变。
                    ),
                // 中文逐行注释：下一行是原始源码第 159 行，保持原始代码不变。
                }
            // 中文逐行注释：下一行是原始源码第 160 行，保持原始代码不变。
            }
            // 中文逐行注释：下一行是原始源码第 161 行，保持原始代码不变。
            Err(e) => {
                // 中文逐行注释：下一行是原始源码第 162 行，保持原始代码不变。
                error!("Failed to proxy request server: {}", e);
                // 中文逐行注释：下一行是原始源码第 163 行，保持原始代码不变。
                error::internal_error(
                    // 中文逐行注释：下一行是原始源码第 164 行，保持原始代码不变。
                    "proxy_request_failed",
                    // 中文逐行注释：下一行是原始源码第 165 行，保持原始代码不变。
                    format!("Failed to proxy request: {}", e),
                // 中文逐行注释：下一行是原始源码第 166 行，保持原始代码不变。
                )
            // 中文逐行注释：下一行是原始源码第 167 行，保持原始代码不变。
            }
        // 中文逐行注释：下一行是原始源码第 168 行，保持原始代码不变。
        }
    // 中文逐行注释：下一行是原始源码第 169 行，保持原始代码不变。
    }

    // 中文函数注释：根据应用上下文创建 PD Router 实例并复制依赖组件。
    // 中文逐行注释：下一行是原始源码第 171 行，保持原始代码不变。
    pub async fn new(ctx: &Arc<crate::app_context::AppContext>) -> Result<Self, String> {
        // 中文逐行注释：下一行是原始源码第 172 行，保持原始代码不变。
        Ok(PDRouter {
            // 中文逐行注释：下一行是原始源码第 173 行，保持原始代码不变。
            worker_registry: Arc::clone(&ctx.worker_registry),
            // 中文逐行注释：下一行是原始源码第 174 行，保持原始代码不变。
            policy_registry: Arc::clone(&ctx.policy_registry),
            // 中文逐行注释：下一行是原始源码第 175 行，保持原始代码不变。
            client: ctx.client.clone(),
            // 中文逐行注释：下一行是原始源码第 176 行，保持原始代码不变。
            retry_config: ctx.router_config.effective_retry_config(),
            // 中文逐行注释：下一行是原始源码第 177 行，保持原始代码不变。
            api_key: ctx.router_config.api_key.clone(),
            // 中文逐行注释：下一行是原始源码第 178 行，保持原始代码不变。
            enable_igw: ctx.router_config.enable_igw,
        // 中文逐行注释：下一行是原始源码第 179 行，保持原始代码不变。
        })
    // 中文逐行注释：下一行是原始源码第 180 行，保持原始代码不变。
    }

    // 中文函数注释：把 PD worker 选择失败转换为统一的 503 响应。
    // 中文逐行注释：下一行是原始源码第 182 行，保持原始代码不变。
    fn handle_server_selection_error(error: String) -> Response {
        // 中文逐行注释：下一行是原始源码第 183 行，保持原始代码不变。
        error!("Failed to select PD pair error={}", error);
        // 中文逐行注释：下一行是原始源码第 184 行，保持原始代码不变。
        error::service_unavailable(
            // 中文逐行注释：下一行是原始源码第 185 行，保持原始代码不变。
            "server_selection_failed",
            // 中文逐行注释：下一行是原始源码第 186 行，保持原始代码不变。
            format!("No available servers: {}", error),
        // 中文逐行注释：下一行是原始源码第 187 行，保持原始代码不变。
        )
    // 中文逐行注释：下一行是原始源码第 188 行，保持原始代码不变。
    }

    // 中文函数注释：把请求序列化失败转换为统一的 500 响应。
    // 中文逐行注释：下一行是原始源码第 190 行，保持原始代码不变。
    fn handle_serialization_error(error: impl std::fmt::Display) -> Response {
        // 中文逐行注释：下一行是原始源码第 191 行，保持原始代码不变。
        error!("Failed to serialize request error={}", error);
        // 中文逐行注释：下一行是原始源码第 192 行，保持原始代码不变。
        error::internal_error("serialization_failed", "Failed to serialize request")
    // 中文逐行注释：下一行是原始源码第 193 行，保持原始代码不变。
    }

    // 中文函数注释：从 GenerateRequest 中推断 batch size。
    // 中文逐行注释：下一行是原始源码第 195 行，保持原始代码不变。
    fn get_generate_batch_size(req: &GenerateRequest) -> Option<usize> {
        // 中文逐行注释：下一行是原始源码第 196 行，保持原始代码不变。
        // GenerateRequest doesn't support batch via arrays, only via input_ids
        // 中文逐行注释：下一行是原始源码第 197 行，保持原始代码不变。
        if let Some(InputIds::Batch(batches)) = &req.input_ids {
            // 中文逐行注释：下一行是原始源码第 198 行，保持原始代码不变。
            if !batches.is_empty() {
                // 中文逐行注释：下一行是原始源码第 199 行，保持原始代码不变。
                return Some(batches.len());
            // 中文逐行注释：下一行是原始源码第 200 行，保持原始代码不变。
            }
        // 中文逐行注释：下一行是原始源码第 201 行，保持原始代码不变。
        }
        // 中文逐行注释：下一行是原始源码第 202 行，保持原始代码不变。
        None
    // 中文逐行注释：下一行是原始源码第 203 行，保持原始代码不变。
    }

    // 中文函数注释：从 ChatCompletionRequest 中推断 batch size。
    // 中文逐行注释：下一行是原始源码第 205 行，保持原始代码不变。
    fn get_chat_batch_size(req: &ChatCompletionRequest) -> Option<usize> {
        // 中文逐行注释：下一行是原始源码第 206 行，保持原始代码不变。
        if let Some(n) = req.n {
            // 中文逐行注释：下一行是原始源码第 207 行，保持原始代码不变。
            if n > 1 {
                // 中文逐行注释：下一行是原始源码第 208 行，保持原始代码不变。
                return Some(n as usize);
            // 中文逐行注释：下一行是原始源码第 209 行，保持原始代码不变。
            }
        // 中文逐行注释：下一行是原始源码第 210 行，保持原始代码不变。
        }
        // 中文逐行注释：下一行是原始源码第 211 行，保持原始代码不变。
        None
    // 中文逐行注释：下一行是原始源码第 212 行，保持原始代码不变。
    }

    // 中文函数注释：从 CompletionRequest 中推断 batch size。
    // 中文逐行注释：下一行是原始源码第 214 行，保持原始代码不变。
    fn get_completion_batch_size(req: &CompletionRequest) -> Option<usize> {
        // 中文逐行注释：下一行是原始源码第 215 行，保持原始代码不变。
        if let StringOrArray::Array(arr) = &req.prompt {
            // 中文逐行注释：下一行是原始源码第 216 行，保持原始代码不变。
            if !arr.is_empty() {
                // 中文逐行注释：下一行是原始源码第 217 行，保持原始代码不变。
                return Some(arr.len());
            // 中文逐行注释：下一行是原始源码第 218 行，保持原始代码不变。
            }
        // 中文逐行注释：下一行是原始源码第 219 行，保持原始代码不变。
        }
        // 中文逐行注释：下一行是原始源码第 220 行，保持原始代码不变。
        None
    // 中文逐行注释：下一行是原始源码第 221 行，保持原始代码不变。
    }

    // 中文逐行注释：下一行是原始源码第 223 行，保持原始代码不变。
    // Static key strings to avoid per-request allocations
    // 中文逐行注释：下一行是原始源码第 224 行，保持原始代码不变。
    const BOOTSTRAP_HOST_KEY: &'static str = "bootstrap_host";
    // 中文逐行注释：下一行是原始源码第 225 行，保持原始代码不变。
    const BOOTSTRAP_PORT_KEY: &'static str = "bootstrap_port";
    // 中文逐行注释：下一行是原始源码第 226 行，保持原始代码不变。
    const BOOTSTRAP_ROOM_KEY: &'static str = "bootstrap_room";

    // 中文函数注释：向 JSON 请求体注入 PD bootstrap host、port 和 room 信息。
    // 中文逐行注释：下一行是原始源码第 228 行，保持原始代码不变。
    fn inject_bootstrap_into_value(
        // 中文逐行注释：下一行是原始源码第 229 行，保持原始代码不变。
        mut original: Value,
        // 中文逐行注释：下一行是原始源码第 230 行，保持原始代码不变。
        prefill_worker: &dyn Worker,
        // 中文逐行注释：下一行是原始源码第 231 行，保持原始代码不变。
        batch_size: Option<usize>,
    // 中文逐行注释：下一行是原始源码第 232 行，保持原始代码不变。
    ) -> Result<Value, String> {
        // 中文逐行注释：下一行是原始源码第 233 行，保持原始代码不变。
        let obj = original
            // 中文逐行注释：下一行是原始源码第 234 行，保持原始代码不变。
            .as_object_mut()
            // 中文逐行注释：下一行是原始源码第 235 行，保持原始代码不变。
            .ok_or_else(|| "Request must be a JSON object".to_string())?;

        // 中文逐行注释：下一行是原始源码第 237 行，保持原始代码不变。
        if let Some(n) = batch_size {
            // 中文逐行注释：下一行是原始源码第 238 行，保持原始代码不变。
            let mut hosts = Vec::with_capacity(n);
            // 中文逐行注释：下一行是原始源码第 239 行，保持原始代码不变。
            let mut ports = Vec::with_capacity(n);
            // 中文逐行注释：下一行是原始源码第 240 行，保持原始代码不变。
            let mut rooms = Vec::with_capacity(n);
            // 中文逐行注释：下一行是原始源码第 241 行，保持原始代码不变。
            for _ in 0..n {
                // 中文逐行注释：下一行是原始源码第 242 行，保持原始代码不变。
                hosts.push(prefill_worker.bootstrap_host());
                // 中文逐行注释：下一行是原始源码第 243 行，保持原始代码不变。
                ports.push(prefill_worker.bootstrap_port());
                // 中文逐行注释：下一行是原始源码第 244 行，保持原始代码不变。
                rooms.push(super::pd_types::generate_room_id());
            // 中文逐行注释：下一行是原始源码第 245 行，保持原始代码不变。
            }
            // 中文逐行注释：下一行是原始源码第 246 行，保持原始代码不变。
            // Use static string keys to avoid per-request allocations
            // 中文逐行注释：下一行是原始源码第 247 行，保持原始代码不变。
            obj.insert(
                // 中文逐行注释：下一行是原始源码第 248 行，保持原始代码不变。
                Self::BOOTSTRAP_HOST_KEY.to_string(),
                // 中文逐行注释：下一行是原始源码第 249 行，保持原始代码不变。
                Value::Array(hosts.into_iter().map(Value::from).collect()),
            // 中文逐行注释：下一行是原始源码第 250 行，保持原始代码不变。
            );
            // 中文逐行注释：下一行是原始源码第 251 行，保持原始代码不变。
            obj.insert(
                // 中文逐行注释：下一行是原始源码第 252 行，保持原始代码不变。
                Self::BOOTSTRAP_PORT_KEY.to_string(),
                // 中文逐行注释：下一行是原始源码第 253 行，保持原始代码不变。
                Value::Array(
                    // 中文逐行注释：下一行是原始源码第 254 行，保持原始代码不变。
                    ports
                        // 中文逐行注释：下一行是原始源码第 255 行，保持原始代码不变。
                        .into_iter()
                        // 中文逐行注释：下一行是原始源码第 256 行，保持原始代码不变。
                        .map(|p| match p {
                            // 中文逐行注释：下一行是原始源码第 257 行，保持原始代码不变。
                            Some(v) => Value::from(v),
                            // 中文逐行注释：下一行是原始源码第 258 行，保持原始代码不变。
                            None => Value::Null,
                        // 中文逐行注释：下一行是原始源码第 259 行，保持原始代码不变。
                        })
                        // 中文逐行注释：下一行是原始源码第 260 行，保持原始代码不变。
                        .collect(),
                // 中文逐行注释：下一行是原始源码第 261 行，保持原始代码不变。
                ),
            // 中文逐行注释：下一行是原始源码第 262 行，保持原始代码不变。
            );
            // 中文逐行注释：下一行是原始源码第 263 行，保持原始代码不变。
            obj.insert(
                // 中文逐行注释：下一行是原始源码第 264 行，保持原始代码不变。
                Self::BOOTSTRAP_ROOM_KEY.to_string(),
                // 中文逐行注释：下一行是原始源码第 265 行，保持原始代码不变。
                Value::Array(rooms.into_iter().map(Value::from).collect()),
            // 中文逐行注释：下一行是原始源码第 266 行，保持原始代码不变。
            );
        // 中文逐行注释：下一行是原始源码第 267 行，保持原始代码不变。
        } else {
            // 中文逐行注释：下一行是原始源码第 268 行，保持原始代码不变。
            // Use static string keys to avoid per-request allocations
            // 中文逐行注释：下一行是原始源码第 269 行，保持原始代码不变。
            obj.insert(
                // 中文逐行注释：下一行是原始源码第 270 行，保持原始代码不变。
                Self::BOOTSTRAP_HOST_KEY.to_string(),
                // 中文逐行注释：下一行是原始源码第 271 行，保持原始代码不变。
                Value::from(prefill_worker.bootstrap_host()),
            // 中文逐行注释：下一行是原始源码第 272 行，保持原始代码不变。
            );
            // 中文逐行注释：下一行是原始源码第 273 行，保持原始代码不变。
            obj.insert(
                // 中文逐行注释：下一行是原始源码第 274 行，保持原始代码不变。
                Self::BOOTSTRAP_PORT_KEY.to_string(),
                // 中文逐行注释：下一行是原始源码第 275 行，保持原始代码不变。
                match prefill_worker.bootstrap_port() {
                    // 中文逐行注释：下一行是原始源码第 276 行，保持原始代码不变。
                    Some(v) => Value::from(v),
                    // 中文逐行注释：下一行是原始源码第 277 行，保持原始代码不变。
                    None => Value::Null,
                // 中文逐行注释：下一行是原始源码第 278 行，保持原始代码不变。
                },
            // 中文逐行注释：下一行是原始源码第 279 行，保持原始代码不变。
            );
            // 中文逐行注释：下一行是原始源码第 280 行，保持原始代码不变。
            obj.insert(
                // 中文逐行注释：下一行是原始源码第 281 行，保持原始代码不变。
                Self::BOOTSTRAP_ROOM_KEY.to_string(),
                // 中文逐行注释：下一行是原始源码第 282 行，保持原始代码不变。
                Value::from(super::pd_types::generate_room_id()),
            // 中文逐行注释：下一行是原始源码第 283 行，保持原始代码不变。
            );
        // 中文逐行注释：下一行是原始源码第 284 行，保持原始代码不变。
        }
        // 中文逐行注释：下一行是原始源码第 285 行，保持原始代码不变。
        Ok(original)
    // 中文逐行注释：下一行是原始源码第 286 行，保持原始代码不变。
    }

    // 中文函数注释：执行带重试封装的 PD prefill/decode 双发调度。
    // 中文逐行注释：下一行是原始源码第 288 行，保持原始代码不变。
    async fn execute_dual_dispatch<T: Serialize + Clone>(
        // 中文逐行注释：下一行是原始源码第 289 行，保持原始代码不变。
        &self,
        // 中文逐行注释：下一行是原始源码第 290 行，保持原始代码不变。
        headers: Option<&HeaderMap>,
        // 中文逐行注释：下一行是原始源码第 291 行，保持原始代码不变。
        original_request: &T,
        // 中文逐行注释：下一行是原始源码第 292 行，保持原始代码不变。
        context: PDRequestContext<'_>,
    // 中文逐行注释：下一行是原始源码第 293 行，保持原始代码不变。
    ) -> Response {
        // 中文逐行注释：下一行是原始源码第 294 行，保持原始代码不变。
        let start_time = Instant::now();

        // 中文逐行注释：下一行是原始源码第 296 行，保持原始代码不变。
        let route = context.route;
        // 中文逐行注释：下一行是原始源码第 297 行，保持原始代码不变。
        let model = context.model_id.unwrap_or(UNKNOWN_MODEL_ID);
        // 中文逐行注释：下一行是原始源码第 298 行，保持原始代码不变。
        let endpoint = route_to_endpoint(route);

        // 中文逐行注释：下一行是原始源码第 300 行，保持原始代码不变。
        // Record request start (Layer 2)
        // 中文逐行注释：下一行是原始源码第 301 行，保持原始代码不变。
        Metrics::record_router_request(
            // 中文逐行注释：下一行是原始源码第 302 行，保持原始代码不变。
            metrics_labels::ROUTER_HTTP,
            // 中文逐行注释：下一行是原始源码第 303 行，保持原始代码不变。
            metrics_labels::BACKEND_PD,
            // 中文逐行注释：下一行是原始源码第 304 行，保持原始代码不变。
            metrics_labels::CONNECTION_HTTP,
            // 中文逐行注释：下一行是原始源码第 305 行，保持原始代码不变。
            model,
            // 中文逐行注释：下一行是原始源码第 306 行，保持原始代码不变。
            endpoint,
            // 中文逐行注释：下一行是原始源码第 307 行，保持原始代码不变。
            bool_to_static_str(context.is_stream),
        // 中文逐行注释：下一行是原始源码第 308 行，保持原始代码不变。
        );
        // 中文逐行注释：下一行是原始源码第 309 行，保持原始代码不变。
        // Clone request once outside the retry loop, then use Arc to share across attempts
        // 中文逐行注释：下一行是原始源码第 310 行，保持原始代码不变。
        // This avoids O(retries) clones by sharing the same data
        // 中文逐行注释：下一行是原始源码第 311 行，保持原始代码不变。
        let shared_request = Arc::new(original_request.clone());
        // 中文逐行注释：下一行是原始源码第 312 行，保持原始代码不变。
        let response = RetryExecutor::execute_response_with_retry(
            // 中文逐行注释：下一行是原始源码第 313 行，保持原始代码不变。
            &self.retry_config,
            // 中文逐行注释：下一行是原始源码第 314 行，保持原始代码不变。
            {
                // 中文逐行注释：下一行是原始源码第 315 行，保持原始代码不变。
                move |attempt: u32| {
                    // 中文逐行注释：下一行是原始源码第 316 行，保持原始代码不变。
                    // Clone Arc (cheap reference count increment) instead of cloning the entire request
                    // 中文逐行注释：下一行是原始源码第 317 行，保持原始代码不变。
                    let shared_request = Arc::clone(&shared_request);
                    // 中文逐行注释：下一行是原始源码第 318 行，保持原始代码不变。
                    let context = context.clone();
                    // 中文逐行注释：下一行是原始源码第 319 行，保持原始代码不变。
                    async move {
                        // 中文逐行注释：下一行是原始源码第 320 行，保持原始代码不变。
                        let (prefill, decode) = match self
                            // 中文逐行注释：下一行是原始源码第 321 行，保持原始代码不变。
                            .select_pd_pair(
                                // 中文逐行注释：下一行是原始源码第 322 行，保持原始代码不变。
                                context.request_text.as_deref(),
                                // 中文逐行注释：下一行是原始源码第 323 行，保持原始代码不变。
                                context.model_id,
                                // 中文逐行注释：下一行是原始源码第 324 行，保持原始代码不变。
                                context.headers.as_ref(),
                            // 中文逐行注释：下一行是原始源码第 325 行，保持原始代码不变。
                            )
                            // 中文逐行注释：下一行是原始源码第 326 行，保持原始代码不变。
                            .await
                        // 中文逐行注释：下一行是原始源码第 327 行，保持原始代码不变。
                        {
                            // 中文逐行注释：下一行是原始源码第 328 行，保持原始代码不变。
                            Ok(pair) => pair,
                            // 中文逐行注释：下一行是原始源码第 329 行，保持原始代码不变。
                            Err(e) => {
                                // 中文逐行注释：下一行是原始源码第 330 行，保持原始代码不变。
                                return Self::handle_server_selection_error(e);
                            // 中文逐行注释：下一行是原始源码第 331 行，保持原始代码不变。
                            }
                        // 中文逐行注释：下一行是原始源码第 332 行，保持原始代码不变。
                        };

                        // 中文逐行注释：下一行是原始源码第 334 行，保持原始代码不变。
                        debug!(
                            // 中文逐行注释：下一行是原始源码第 335 行，保持原始代码不变。
                            "PD retry attempt {} using prefill={} decode={}",
                            // 中文逐行注释：下一行是原始源码第 336 行，保持原始代码不变。
                            attempt,
                            // 中文逐行注释：下一行是原始源码第 337 行，保持原始代码不变。
                            prefill.url(),
                            // 中文逐行注释：下一行是原始源码第 338 行，保持原始代码不变。
                            decode.url()
                        // 中文逐行注释：下一行是原始源码第 339 行，保持原始代码不变。
                        );

                        // 中文逐行注释：下一行是原始源码第 341 行，保持原始代码不变。
                        let mut json_request = match serde_json::to_value(shared_request.as_ref()) {
                            // 中文逐行注释：下一行是原始源码第 342 行，保持原始代码不变。
                            Ok(v) => v,
                            // 中文逐行注释：下一行是原始源码第 343 行，保持原始代码不变。
                            Err(e) => return Self::handle_serialization_error(e),
                        // 中文逐行注释：下一行是原始源码第 344 行，保持原始代码不变。
                        };

                        // 中文逐行注释：下一行是原始源码第 346 行，保持原始代码不变。
                        json_request = match Self::inject_bootstrap_into_value(
                            // 中文逐行注释：下一行是原始源码第 347 行，保持原始代码不变。
                            json_request,
                            // 中文逐行注释：下一行是原始源码第 348 行，保持原始代码不变。
                            prefill.as_ref(),
                            // 中文逐行注释：下一行是原始源码第 349 行，保持原始代码不变。
                            context.batch_size,
                        // 中文逐行注释：下一行是原始源码第 350 行，保持原始代码不变。
                        ) {
                            // 中文逐行注释：下一行是原始源码第 351 行，保持原始代码不变。
                            Ok(v) => v,
                            // 中文逐行注释：下一行是原始源码第 352 行，保持原始代码不变。
                            Err(e) => return Self::handle_serialization_error(e),
                        // 中文逐行注释：下一行是原始源码第 353 行，保持原始代码不变。
                        };

                        // 中文逐行注释：下一行是原始源码第 355 行，保持原始代码不变。
                        let ctx_is_stream = context.is_stream;
                        // 中文逐行注释：下一行是原始源码第 356 行，保持原始代码不变。
                        let response = self
                            // 中文逐行注释：下一行是原始源码第 357 行，保持原始代码不变。
                            .execute_dual_dispatch_internal(
                                // 中文逐行注释：下一行是原始源码第 358 行，保持原始代码不变。
                                headers,
                                // 中文逐行注释：下一行是原始源码第 359 行，保持原始代码不变。
                                json_request,
                                // 中文逐行注释：下一行是原始源码第 360 行，保持原始代码不变。
                                context,
                                // 中文逐行注释：下一行是原始源码第 361 行，保持原始代码不变。
                                Arc::clone(&prefill),
                                // 中文逐行注释：下一行是原始源码第 362 行，保持原始代码不变。
                                Arc::clone(&decode),
                                // 中文逐行注释：下一行是原始源码第 363 行，保持原始代码不变。
                                start_time,
                            // 中文逐行注释：下一行是原始源码第 364 行，保持原始代码不变。
                            )
                            // 中文逐行注释：下一行是原始源码第 365 行，保持原始代码不变。
                            .await;

                        // 中文逐行注释：下一行是原始源码第 367 行，保持原始代码不变。
                        let status = response.status();
                        // 中文逐行注释：下一行是原始源码第 368 行，保持原始代码不变。
                        let outcomes_already_recorded = response
                            // 中文逐行注释：下一行是原始源码第 369 行，保持原始代码不变。
                            .extensions()
                            // 中文逐行注释：下一行是原始源码第 370 行，保持原始代码不变。
                            .get::<BreakerOutcomesRecorded>()
                            // 中文逐行注释：下一行是原始源码第 371 行，保持原始代码不变。
                            .is_some();
                        // 中文逐行注释：下一行是原始源码第 372 行，保持原始代码不变。
                        if !outcomes_already_recorded {
                            // 中文逐行注释：下一行是原始源码第 373 行，保持原始代码不变。
                            let not_error = status.is_success() || status.is_client_error();
                            // 中文逐行注释：下一行是原始源码第 374 行，保持原始代码不变。
                            // Prefill is always non-streaming and fully read before
                            // 中文逐行注释：下一行是原始源码第 375 行，保持原始代码不变。
                            // we get here, so its outcome is final.
                            // 中文逐行注释：下一行是原始源码第 376 行，保持原始代码不变。
                            prefill.record_outcome(not_error);
                            // 中文逐行注释：下一行是原始源码第 377 行，保持原始代码不变。
                            // Decode for a streaming request is still mid-flight at
                            // 中文逐行注释：下一行是原始源码第 378 行，保持原始代码不变。
                            // this point; the `BreakerTrackedStream` wrapped around
                            // 中文逐行注释：下一行是原始源码第 379 行，保持原始代码不变。
                            // its byte stream records the outcome on drop. Skip the
                            // 中文逐行注释：下一行是原始源码第 380 行，保持原始代码不变。
                            // eager success record to avoid masking "200-then-broken"
                            // 中文逐行注释：下一行是原始源码第 381 行，保持原始代码不变。
                            // decode workers.
                            // 中文逐行注释：下一行是原始源码第 382 行，保持原始代码不变。
                            if !ctx_is_stream {
                                // 中文逐行注释：下一行是原始源码第 383 行，保持原始代码不变。
                                decode.record_outcome(not_error);
                            // 中文逐行注释：下一行是原始源码第 384 行，保持原始代码不变。
                            }
                        // 中文逐行注释：下一行是原始源码第 385 行，保持原始代码不变。
                        }

                        // 中文逐行注释：下一行是原始源码第 387 行，保持原始代码不变。
                        // Record worker errors for server errors (5xx)
                        // 中文逐行注释：下一行是原始源码第 388 行，保持原始代码不变。
                        if status.is_server_error() {
                            // 中文逐行注释：下一行是原始源码第 389 行，保持原始代码不变。
                            let error_type = error_type_from_status(status);
                            // 中文逐行注释：下一行是原始源码第 390 行，保持原始代码不变。
                            Metrics::record_worker_error(
                                // 中文逐行注释：下一行是原始源码第 391 行，保持原始代码不变。
                                metrics_labels::WORKER_PREFILL,
                                // 中文逐行注释：下一行是原始源码第 392 行，保持原始代码不变。
                                metrics_labels::CONNECTION_HTTP,
                                // 中文逐行注释：下一行是原始源码第 393 行，保持原始代码不变。
                                error_type,
                            // 中文逐行注释：下一行是原始源码第 394 行，保持原始代码不变。
                            );
                            // 中文逐行注释：下一行是原始源码第 395 行，保持原始代码不变。
                            Metrics::record_worker_error(
                                // 中文逐行注释：下一行是原始源码第 396 行，保持原始代码不变。
                                metrics_labels::WORKER_DECODE,
                                // 中文逐行注释：下一行是原始源码第 397 行，保持原始代码不变。
                                metrics_labels::CONNECTION_HTTP,
                                // 中文逐行注释：下一行是原始源码第 398 行，保持原始代码不变。
                                error_type,
                            // 中文逐行注释：下一行是原始源码第 399 行，保持原始代码不变。
                            );
                        // 中文逐行注释：下一行是原始源码第 400 行，保持原始代码不变。
                        }

                        // 中文逐行注释：下一行是原始源码第 402 行，保持原始代码不变。
                        response
                    // 中文逐行注释：下一行是原始源码第 403 行，保持原始代码不变。
                    }
                // 中文逐行注释：下一行是原始源码第 404 行，保持原始代码不变。
                }
            // 中文逐行注释：下一行是原始源码第 405 行，保持原始代码不变。
            },
            // 中文逐行注释：下一行是原始源码第 406 行，保持原始代码不变。
            |res, _attempt| is_retryable_status(res.status()),
            // 中文逐行注释：下一行是原始源码第 407 行，保持原始代码不变。
            |delay, attempt| {
                // 中文逐行注释：下一行是原始源码第 408 行，保持原始代码不变。
                // Layer 3 worker metrics (PD mode uses both prefill and decode workers)
                // 中文逐行注释：下一行是原始源码第 409 行，保持原始代码不变。
                Metrics::record_worker_retry(metrics_labels::WORKER_PREFILL, endpoint);
                // 中文逐行注释：下一行是原始源码第 410 行，保持原始代码不变。
                Metrics::record_worker_retry(metrics_labels::WORKER_DECODE, endpoint);
                // 中文逐行注释：下一行是原始源码第 411 行，保持原始代码不变。
                Metrics::record_worker_retry_backoff(attempt, delay);
            // 中文逐行注释：下一行是原始源码第 412 行，保持原始代码不变。
            },
            // 中文逐行注释：下一行是原始源码第 413 行，保持原始代码不变。
            || {
                // 中文逐行注释：下一行是原始源码第 414 行，保持原始代码不变。
                Metrics::record_worker_retries_exhausted(metrics_labels::WORKER_PREFILL, endpoint);
                // 中文逐行注释：下一行是原始源码第 415 行，保持原始代码不变。
                Metrics::record_worker_retries_exhausted(metrics_labels::WORKER_DECODE, endpoint);
            // 中文逐行注释：下一行是原始源码第 416 行，保持原始代码不变。
            },
        // 中文逐行注释：下一行是原始源码第 417 行，保持原始代码不变。
        )
        // 中文逐行注释：下一行是原始源码第 418 行，保持原始代码不变。
        .await;

        // 中文逐行注释：下一行是原始源码第 420 行，保持原始代码不变。
        // Record Layer 2 metrics
        // 中文逐行注释：下一行是原始源码第 421 行，保持原始代码不变。
        let duration = start_time.elapsed();
        // 中文逐行注释：下一行是原始源码第 422 行，保持原始代码不变。
        if response.status().is_success() {
            // 中文逐行注释：下一行是原始源码第 423 行，保持原始代码不变。
            Metrics::record_router_duration(
                // 中文逐行注释：下一行是原始源码第 424 行，保持原始代码不变。
                metrics_labels::ROUTER_HTTP,
                // 中文逐行注释：下一行是原始源码第 425 行，保持原始代码不变。
                metrics_labels::BACKEND_PD,
                // 中文逐行注释：下一行是原始源码第 426 行，保持原始代码不变。
                metrics_labels::CONNECTION_HTTP,
                // 中文逐行注释：下一行是原始源码第 427 行，保持原始代码不变。
                model,
                // 中文逐行注释：下一行是原始源码第 428 行，保持原始代码不变。
                endpoint,
                // 中文逐行注释：下一行是原始源码第 429 行，保持原始代码不变。
                duration,
            // 中文逐行注释：下一行是原始源码第 430 行，保持原始代码不变。
            );
        // 中文逐行注释：下一行是原始源码第 431 行，保持原始代码不变。
        } else if !is_retryable_status(response.status()) {
            // 中文逐行注释：下一行是原始源码第 432 行，保持原始代码不变。
            Metrics::record_router_error(
                // 中文逐行注释：下一行是原始源码第 433 行，保持原始代码不变。
                metrics_labels::ROUTER_HTTP,
                // 中文逐行注释：下一行是原始源码第 434 行，保持原始代码不变。
                metrics_labels::BACKEND_PD,
                // 中文逐行注释：下一行是原始源码第 435 行，保持原始代码不变。
                metrics_labels::CONNECTION_HTTP,
                // 中文逐行注释：下一行是原始源码第 436 行，保持原始代码不变。
                model,
                // 中文逐行注释：下一行是原始源码第 437 行，保持原始代码不变。
                endpoint,
                // 中文逐行注释：下一行是原始源码第 438 行，保持原始代码不变。
                error_type_from_status(response.status()),
            // 中文逐行注释：下一行是原始源码第 439 行，保持原始代码不变。
            );
        // 中文逐行注释：下一行是原始源码第 440 行，保持原始代码不变。
        }

        // 中文逐行注释：下一行是原始源码第 442 行，保持原始代码不变。
        response
    // 中文逐行注释：下一行是原始源码第 443 行，保持原始代码不变。
    }

    // 中文函数注释：处理 decode 侧错误响应，并结合 prefill 结果记录 breaker 状态。
    // 中文逐行注释：下一行是原始源码第 445 行，保持原始代码不变。
    async fn handle_decode_error_response(
        // 中文逐行注释：下一行是原始源码第 446 行，保持原始代码不变。
        &self,
        // 中文逐行注释：下一行是原始源码第 447 行，保持原始代码不变。
        res: reqwest::Response,
        // 中文逐行注释：下一行是原始源码第 448 行，保持原始代码不变。
        context: &PDRequestContext<'_>,
        // 中文逐行注释：下一行是原始源码第 449 行，保持原始代码不变。
        prefill: Arc<dyn Worker>,
        // 中文逐行注释：下一行是原始源码第 450 行，保持原始代码不变。
        decode: Arc<dyn Worker>,
    // 中文逐行注释：下一行是原始源码第 451 行，保持原始代码不变。
    ) -> Response {
        // 中文逐行注释：下一行是原始源码第 452 行，保持原始代码不变。
        let status = res.status();

        // 中文逐行注释：下一行是原始源码第 454 行，保持原始代码不变。
        if context.is_stream {
            // 中文逐行注释：下一行是原始源码第 455 行，保持原始代码不变。
            // Handle streaming error response
            // 中文逐行注释：下一行是原始源码第 456 行，保持原始代码不变。
            let response_headers = header_utils::preserve_response_headers(res.headers());
            // 中文逐行注释：下一行是原始源码第 457 行，保持原始代码不变。
            let error_payload = match res.bytes().await {
                // 中文逐行注释：下一行是原始源码第 458 行，保持原始代码不变。
                Ok(error_body) => match serde_json::from_slice::<Value>(&error_body) {
                    // 中文逐行注释：下一行是原始源码第 459 行，保持原始代码不变。
                    Ok(error_json) => {
                        // 中文逐行注释：下一行是原始源码第 460 行，保持原始代码不变。
                        json!({ "message": error_json, "status": status.as_u16() })
                    // 中文逐行注释：下一行是原始源码第 461 行，保持原始代码不变。
                    }
                    // 中文逐行注释：下一行是原始源码第 462 行，保持原始代码不变。
                    Err(parse_err) => {
                        // 中文逐行注释：下一行是原始源码第 463 行，保持原始代码不变。
                        let body_text = String::from_utf8_lossy(&error_body).to_string();
                        // 中文逐行注释：下一行是原始源码第 464 行，保持原始代码不变。
                        let preview: String = body_text.chars().take(256).collect();
                        // 中文逐行注释：下一行是原始源码第 465 行，保持原始代码不变。
                        tracing::warn!(
                            // 中文逐行注释：下一行是原始源码第 466 行，保持原始代码不变。
                            "Failed to parse decode error body as JSON from {}: {} \
                             // 中文逐行注释：下一行是原始源码第 467 行，保持原始代码不变。
                             (status={}, body preview: {:?})",
                            // 中文逐行注释：下一行是原始源码第 468 行，保持原始代码不变。
                            decode.url(),
                            // 中文逐行注释：下一行是原始源码第 469 行，保持原始代码不变。
                            parse_err,
                            // 中文逐行注释：下一行是原始源码第 470 行，保持原始代码不变。
                            status.as_u16(),
                            // 中文逐行注释：下一行是原始源码第 471 行，保持原始代码不变。
                            preview
                        // 中文逐行注释：下一行是原始源码第 472 行，保持原始代码不变。
                        );
                        // 中文逐行注释：下一行是原始源码第 473 行，保持原始代码不变。
                        json!({ "message": body_text, "status": status.as_u16() })
                    // 中文逐行注释：下一行是原始源码第 474 行，保持原始代码不变。
                    }
                // 中文逐行注释：下一行是原始源码第 475 行，保持原始代码不变。
                },
                // 中文逐行注释：下一行是原始源码第 476 行，保持原始代码不变。
                Err(e) => {
                    // 中文逐行注释：下一行是原始源码第 477 行，保持原始代码不变。
                    json!({ "message": format!("Decode server error: {}", e), "status": status.as_u16() })
                // 中文逐行注释：下一行是原始源码第 478 行，保持原始代码不变。
                }
            // 中文逐行注释：下一行是原始源码第 479 行，保持原始代码不变。
            };

            // 中文逐行注释：下一行是原始源码第 481 行，保持原始代码不变。
            let sse_data = format!(
                // 中文逐行注释：下一行是原始源码第 482 行，保持原始代码不变。
                "data: {{'error': {}}}",
                // 中文逐行注释：下一行是原始源码第 483 行，保持原始代码不变。
                serde_json::to_string(&error_payload).unwrap_or_default()
            // 中文逐行注释：下一行是原始源码第 484 行，保持原始代码不变。
            );
            // 中文逐行注释：下一行是原始源码第 485 行，保持原始代码不变。
            let error_stream = tokio_stream::once(Ok(axum::body::Bytes::from(sse_data)));

            // 中文逐行注释：下一行是原始源码第 487 行，保持原始代码不变。
            self.create_streaming_response(
                // 中文逐行注释：下一行是原始源码第 488 行，保持原始代码不变。
                error_stream,
                // 中文逐行注释：下一行是原始源码第 489 行，保持原始代码不变。
                status,
                // 中文逐行注释：下一行是原始源码第 490 行，保持原始代码不变。
                None,
                // 中文逐行注释：下一行是原始源码第 491 行，保持原始代码不变。
                context.return_logprob,
                // 中文逐行注释：下一行是原始源码第 492 行，保持原始代码不变。
                Some(response_headers),
                // 中文逐行注释：下一行是原始源码第 493 行，保持原始代码不变。
                prefill,
                // 中文逐行注释：下一行是原始源码第 494 行，保持原始代码不变。
                decode,
            // 中文逐行注释：下一行是原始源码第 495 行，保持原始代码不变。
            )
        // 中文逐行注释：下一行是原始源码第 496 行，保持原始代码不变。
        } else {
            // 中文逐行注释：下一行是原始源码第 497 行，保持原始代码不变。
            // Handle non-streaming error response
            // 中文逐行注释：下一行是原始源码第 498 行，保持原始代码不变。
            match res.bytes().await {
                // 中文逐行注释：下一行是原始源码第 499 行，保持原始代码不变。
                Ok(error_body) => {
                    // 中文逐行注释：下一行是原始源码第 500 行，保持原始代码不变。
                    // Try to parse error message from body, fallback to status-based error
                    // 中文逐行注释：下一行是原始源码第 501 行，保持原始代码不变。
                    let error_message = if let Ok(error_json) =
                        // 中文逐行注释：下一行是原始源码第 502 行，保持原始代码不变。
                        serde_json::from_slice::<Value>(&error_body)
                    // 中文逐行注释：下一行是原始源码第 503 行，保持原始代码不变。
                    {
                        // 中文逐行注释：下一行是原始源码第 504 行，保持原始代码不变。
                        if let Some(msg) = error_json
                            // 中文逐行注释：下一行是原始源码第 505 行，保持原始代码不变。
                            .get("error")
                            // 中文逐行注释：下一行是原始源码第 506 行，保持原始代码不变。
                            .and_then(|e| e.get("message"))
                            // 中文逐行注释：下一行是原始源码第 507 行，保持原始代码不变。
                            .and_then(|m| m.as_str())
                        // 中文逐行注释：下一行是原始源码第 508 行，保持原始代码不变。
                        {
                            // 中文逐行注释：下一行是原始源码第 509 行，保持原始代码不变。
                            msg.to_string()
                        // 中文逐行注释：下一行是原始源码第 510 行，保持原始代码不变。
                        } else if let Some(msg) = error_json.get("message").and_then(|m| m.as_str())
                        // 中文逐行注释：下一行是原始源码第 511 行，保持原始代码不变。
                        {
                            // 中文逐行注释：下一行是原始源码第 512 行，保持原始代码不变。
                            msg.to_string()
                        // 中文逐行注释：下一行是原始源码第 513 行，保持原始代码不变。
                        } else {
                            // 中文逐行注释：下一行是原始源码第 514 行，保持原始代码不变。
                            String::from_utf8_lossy(&error_body).to_string()
                        // 中文逐行注释：下一行是原始源码第 515 行，保持原始代码不变。
                        }
                    // 中文逐行注释：下一行是原始源码第 516 行，保持原始代码不变。
                    } else {
                        // 中文逐行注释：下一行是原始源码第 517 行，保持原始代码不变。
                        String::from_utf8_lossy(&error_body).to_string()
                    // 中文逐行注释：下一行是原始源码第 518 行，保持原始代码不变。
                    };

                    // 中文逐行注释：下一行是原始源码第 520 行，保持原始代码不变。
                    let status_code = StatusCode::from_u16(status.as_u16())
                        // 中文逐行注释：下一行是原始源码第 521 行，保持原始代码不变。
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    // 中文逐行注释：下一行是原始源码第 522 行，保持原始代码不变。
                    match status_code {
                        // 中文逐行注释：下一行是原始源码第 523 行，保持原始代码不变。
                        StatusCode::BAD_REQUEST => {
                            // 中文逐行注释：下一行是原始源码第 524 行，保持原始代码不变。
                            error::bad_request("decode_bad_request", error_message)
                        // 中文逐行注释：下一行是原始源码第 525 行，保持原始代码不变。
                        }
                        // 中文逐行注释：下一行是原始源码第 526 行，保持原始代码不变。
                        StatusCode::NOT_FOUND => {
                            // 中文逐行注释：下一行是原始源码第 527 行，保持原始代码不变。
                            error::not_found("decode_not_found", error_message)
                        // 中文逐行注释：下一行是原始源码第 528 行，保持原始代码不变。
                        }
                        // 中文逐行注释：下一行是原始源码第 529 行，保持原始代码不变。
                        StatusCode::INTERNAL_SERVER_ERROR => {
                            // 中文逐行注释：下一行是原始源码第 530 行，保持原始代码不变。
                            error::internal_error("decode_internal_error", error_message)
                        // 中文逐行注释：下一行是原始源码第 531 行，保持原始代码不变。
                        }
                        // 中文逐行注释：下一行是原始源码第 532 行，保持原始代码不变。
                        StatusCode::SERVICE_UNAVAILABLE => {
                            // 中文逐行注释：下一行是原始源码第 533 行，保持原始代码不变。
                            error::service_unavailable("decode_unavailable", error_message)
                        // 中文逐行注释：下一行是原始源码第 534 行，保持原始代码不变。
                        }
                        // 中文逐行注释：下一行是原始源码第 535 行，保持原始代码不变。
                        StatusCode::BAD_GATEWAY => {
                            // 中文逐行注释：下一行是原始源码第 536 行，保持原始代码不变。
                            error::bad_gateway("decode_bad_gateway", error_message)
                        // 中文逐行注释：下一行是原始源码第 537 行，保持原始代码不变。
                        }
                        // 中文逐行注释：下一行是原始源码第 538 行，保持原始代码不变。
                        _ => error::internal_error("decode_error", error_message),
                    // 中文逐行注释：下一行是原始源码第 539 行，保持原始代码不变。
                    }
                // 中文逐行注释：下一行是原始源码第 540 行，保持原始代码不变。
                }
                // 中文逐行注释：下一行是原始源码第 541 行，保持原始代码不变。
                Err(e) => {
                    // 中文逐行注释：下一行是原始源码第 542 行，保持原始代码不变。
                    let error_message = format!("Decode server error: {}", e);
                    // 中文逐行注释：下一行是原始源码第 543 行，保持原始代码不变。
                    let status_code = StatusCode::from_u16(status.as_u16())
                        // 中文逐行注释：下一行是原始源码第 544 行，保持原始代码不变。
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    // 中文逐行注释：下一行是原始源码第 545 行，保持原始代码不变。
                    match status_code {
                        // 中文逐行注释：下一行是原始源码第 546 行，保持原始代码不变。
                        StatusCode::BAD_REQUEST => {
                            // 中文逐行注释：下一行是原始源码第 547 行，保持原始代码不变。
                            error::bad_request("decode_read_failed", error_message)
                        // 中文逐行注释：下一行是原始源码第 548 行，保持原始代码不变。
                        }
                        // 中文逐行注释：下一行是原始源码第 549 行，保持原始代码不变。
                        StatusCode::NOT_FOUND => {
                            // 中文逐行注释：下一行是原始源码第 550 行，保持原始代码不变。
                            error::not_found("decode_read_failed", error_message)
                        // 中文逐行注释：下一行是原始源码第 551 行，保持原始代码不变。
                        }
                        // 中文逐行注释：下一行是原始源码第 552 行，保持原始代码不变。
                        StatusCode::INTERNAL_SERVER_ERROR => {
                            // 中文逐行注释：下一行是原始源码第 553 行，保持原始代码不变。
                            error::internal_error("decode_read_failed", error_message)
                        // 中文逐行注释：下一行是原始源码第 554 行，保持原始代码不变。
                        }
                        // 中文逐行注释：下一行是原始源码第 555 行，保持原始代码不变。
                        StatusCode::SERVICE_UNAVAILABLE => {
                            // 中文逐行注释：下一行是原始源码第 556 行，保持原始代码不变。
                            error::service_unavailable("decode_read_failed", error_message)
                        // 中文逐行注释：下一行是原始源码第 557 行，保持原始代码不变。
                        }
                        // 中文逐行注释：下一行是原始源码第 558 行，保持原始代码不变。
                        StatusCode::BAD_GATEWAY => {
                            // 中文逐行注释：下一行是原始源码第 559 行，保持原始代码不变。
                            error::bad_gateway("decode_read_failed", error_message)
                        // 中文逐行注释：下一行是原始源码第 560 行，保持原始代码不变。
                        }
                        // 中文逐行注释：下一行是原始源码第 561 行，保持原始代码不变。
                        _ => error::internal_error("decode_read_failed", error_message),
                    // 中文逐行注释：下一行是原始源码第 562 行，保持原始代码不变。
                    }
                // 中文逐行注释：下一行是原始源码第 563 行，保持原始代码不变。
                }
            // 中文逐行注释：下一行是原始源码第 564 行，保持原始代码不变。
            }
        // 中文逐行注释：下一行是原始源码第 565 行，保持原始代码不变。
        }
    // 中文逐行注释：下一行是原始源码第 566 行，保持原始代码不变。
    }

    // 中文逐行注释：下一行是原始源码第 568 行，保持原始代码不变。
    // Internal method that performs the actual dual dispatch (without retry logic)
    // 中文函数注释：执行单次 PD 双发，请求 prefill 与 decode worker 并合并处理结果。
    // 中文逐行注释：下一行是原始源码第 569 行，保持原始代码不变。
    async fn execute_dual_dispatch_internal(
        // 中文逐行注释：下一行是原始源码第 570 行，保持原始代码不变。
        &self,
        // 中文逐行注释：下一行是原始源码第 571 行，保持原始代码不变。
        headers: Option<&HeaderMap>,
        // 中文逐行注释：下一行是原始源码第 572 行，保持原始代码不变。
        json_request: Value,
        // 中文逐行注释：下一行是原始源码第 573 行，保持原始代码不变。
        context: PDRequestContext<'_>,
        // 中文逐行注释：下一行是原始源码第 574 行，保持原始代码不变。
        prefill: Arc<dyn Worker>,
        // 中文逐行注释：下一行是原始源码第 575 行，保持原始代码不变。
        decode: Arc<dyn Worker>,
        // 中文逐行注释：下一行是原始源码第 576 行，保持原始代码不变。
        _start_time: Instant,
    // 中文逐行注释：下一行是原始源码第 577 行，保持原始代码不变。
    ) -> Response {
        // 中文逐行注释：下一行是原始源码第 578 行，保持原始代码不变。
        // For non-streaming: use guard for automatic load management
        // 中文逐行注释：下一行是原始源码第 579 行，保持原始代码不变。
        // For streaming: load will be managed in create_streaming_response
        // 中文逐行注释：下一行是原始源码第 580 行，保持原始代码不变。
        let _prefill_guard =
            // 中文逐行注释：下一行是原始源码第 581 行，保持原始代码不变。
            (!context.is_stream).then(|| WorkerLoadGuard::new(prefill.clone(), headers));
        // 中文逐行注释：下一行是原始源码第 582 行，保持原始代码不变。
        let _decode_guard =
            // 中文逐行注释：下一行是原始源码第 583 行，保持原始代码不变。
            (!context.is_stream).then(|| WorkerLoadGuard::new(decode.clone(), headers));

        // 中文逐行注释：下一行是原始源码第 585 行，保持原始代码不变。
        let mut headers_with_trace = headers.cloned().unwrap_or_default();
        // 中文逐行注释：下一行是原始源码第 586 行，保持原始代码不变。
        inject_trace_context_http(&mut headers_with_trace);
        // 中文逐行注释：下一行是原始源码第 587 行，保持原始代码不变。
        let headers = Some(&headers_with_trace);

        // 中文逐行注释：下一行是原始源码第 589 行，保持原始代码不变。
        // Build both requests
        // 中文逐行注释：下一行是原始源码第 590 行，保持原始代码不变。
        let prefill_request = self.build_post_with_headers(
            // 中文逐行注释：下一行是原始源码第 591 行，保持原始代码不变。
            &self.client,
            // 中文逐行注释：下一行是原始源码第 592 行，保持原始代码不变。
            prefill.url(),
            // 中文逐行注释：下一行是原始源码第 593 行，保持原始代码不变。
            context.route,
            // 中文逐行注释：下一行是原始源码第 594 行，保持原始代码不变。
            &json_request,
            // 中文逐行注释：下一行是原始源码第 595 行，保持原始代码不变。
            headers,
            // 中文逐行注释：下一行是原始源码第 596 行，保持原始代码不变。
            false,
        // 中文逐行注释：下一行是原始源码第 597 行，保持原始代码不变。
        );
        // 中文逐行注释：下一行是原始源码第 598 行，保持原始代码不变。
        let decode_request = self.build_post_with_headers(
            // 中文逐行注释：下一行是原始源码第 599 行，保持原始代码不变。
            &self.client,
            // 中文逐行注释：下一行是原始源码第 600 行，保持原始代码不变。
            decode.url(),
            // 中文逐行注释：下一行是原始源码第 601 行，保持原始代码不变。
            context.route,
            // 中文逐行注释：下一行是原始源码第 602 行，保持原始代码不变。
            &json_request,
            // 中文逐行注释：下一行是原始源码第 603 行，保持原始代码不变。
            headers,
            // 中文逐行注释：下一行是原始源码第 604 行，保持原始代码不变。
            false,
        // 中文逐行注释：下一行是原始源码第 605 行，保持原始代码不变。
        );

        // 中文逐行注释：下一行是原始源码第 607 行，保持原始代码不变。
        // Send both requests concurrently and wait for both
        // 中文逐行注释：下一行是原始源码第 608 行，保持原始代码不变。
        // Note: Using borrowed references avoids heap allocation
        // 中文逐行注释：下一行是原始源码第 609 行，保持原始代码不变。
        events::RequestPDSentEvent {
            // 中文逐行注释：下一行是原始源码第 610 行，保持原始代码不变。
            prefill_url: prefill.url(),
            // 中文逐行注释：下一行是原始源码第 611 行，保持原始代码不变。
            decode_url: decode.url(),
        // 中文逐行注释：下一行是原始源码第 612 行，保持原始代码不变。
        }
        // 中文逐行注释：下一行是原始源码第 613 行，保持原始代码不变。
        .emit();

        // 中文逐行注释：下一行是原始源码第 615 行，保持原始代码不变。
        let (prefill_result, decode_result) =
            // 中文逐行注释：下一行是原始源码第 616 行，保持原始代码不变。
            tokio::join!(prefill_request.send(), decode_request.send());

        // 中文逐行注释：下一行是原始源码第 618 行，保持原始代码不变。
        events::RequestReceivedEvent {}.emit();

        // 中文逐行注释：下一行是原始源码第 620 行，保持原始代码不变。
        // Process decode response
        // 中文逐行注释：下一行是原始源码第 621 行，保持原始代码不变。
        match decode_result {
            // 中文逐行注释：下一行是原始源码第 622 行，保持原始代码不变。
            Ok(res) => {
                // 中文逐行注释：下一行是原始源码第 623 行，保持原始代码不变。
                let status = StatusCode::from_u16(res.status().as_u16())
                    // 中文逐行注释：下一行是原始源码第 624 行，保持原始代码不变。
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                // 中文逐行注释：下一行是原始源码第 625 行，保持原始代码不变。
                debug!("Decode response status: {}", status);

                // 中文逐行注释：下一行是原始源码第 627 行，保持原始代码不变。
                if !status.is_success() {
                    // 中文逐行注释：下一行是原始源码第 628 行，保持原始代码不变。
                    error!(
                        // 中文逐行注释：下一行是原始源码第 629 行，保持原始代码不变。
                        "Decode server returned error status decode_url={} status={}",
                        // 中文逐行注释：下一行是原始源码第 630 行，保持原始代码不变。
                        decode.url(),
                        // 中文逐行注释：下一行是原始源码第 631 行，保持原始代码不变。
                        status
                    // 中文逐行注释：下一行是原始源码第 632 行，保持原始代码不变。
                    );

                    // 中文逐行注释：下一行是原始源码第 634 行，保持原始代码不变。
                    // Per-worker breaker attribution before the synthetic 5xx
                    // 中文逐行注释：下一行是原始源码第 635 行，保持原始代码不变。
                    // response takes over. Prefill ran concurrently in the
                    // 中文逐行注释：下一行是原始源码第 636 行，保持原始代码不变。
                    // `tokio::join!`: tick it based on its actual response
                    // 中文逐行注释：下一行是原始源码第 637 行，保持原始代码不变。
                    // status, not on the decode-driven failure. For
                    // 中文逐行注释：下一行是原始源码第 638 行，保持原始代码不变。
                    // non-streaming the response carries no tracked stream
                    // 中文逐行注释：下一行是原始源码第 639 行，保持原始代码不变。
                    // so record decode's outcome here too — but treat 4xx
                    // 中文逐行注释：下一行是原始源码第 640 行，保持原始代码不变。
                    // as a client fault rather than a worker fault, matching
                    // 中文逐行注释：下一行是原始源码第 641 行，保持原始代码不变。
                    // the legacy outer-dispatcher rule and the streaming
                    // 中文逐行注释：下一行是原始源码第 642 行，保持原始代码不变。
                    // `BreakerTrackedStream` pre-mark in
                    // 中文逐行注释：下一行是原始源码第 643 行，保持原始代码不变。
                    // `create_streaming_response`. For streaming
                    // 中文逐行注释：下一行是原始源码第 644 行，保持原始代码不变。
                    // `handle_decode_error_response` wraps the synthetic
                    // 中文逐行注释：下一行是原始源码第 645 行，保持原始代码不变。
                    // error SSE in a `BreakerTrackedStream` that ticks
                    // 中文逐行注释：下一行是原始源码第 646 行，保持原始代码不变。
                    // decode on drop, so skip to avoid double-counting.
                    // 中文逐行注释：下一行是原始源码第 647 行，保持原始代码不变。
                    // Mark the response so the outer dispatcher skips its
                    // 中文逐行注释：下一行是原始源码第 648 行，保持原始代码不变。
                    // status-derived `record_outcome`.
                    // 中文逐行注释：下一行是原始源码第 649 行，保持原始代码不变。
                    let prefill_ok = match &prefill_result {
                        // 中文逐行注释：下一行是原始源码第 650 行，保持原始代码不变。
                        Ok(r) => {
                            // 中文逐行注释：下一行是原始源码第 651 行，保持原始代码不变。
                            let s = r.status();
                            // 中文逐行注释：下一行是原始源码第 652 行，保持原始代码不变。
                            s.is_success() || s.is_client_error()
                        // 中文逐行注释：下一行是原始源码第 653 行，保持原始代码不变。
                        }
                        // 中文逐行注释：下一行是原始源码第 654 行，保持原始代码不变。
                        Err(_) => false,
                    // 中文逐行注释：下一行是原始源码第 655 行，保持原始代码不变。
                    };
                    // 中文逐行注释：下一行是原始源码第 656 行，保持原始代码不变。
                    prefill.record_outcome(prefill_ok);
                    // 中文逐行注释：下一行是原始源码第 657 行，保持原始代码不变。
                    if !context.is_stream {
                        // 中文逐行注释：下一行是原始源码第 658 行，保持原始代码不变。
                        let decode_ok = status.is_success() || status.is_client_error();
                        // 中文逐行注释：下一行是原始源码第 659 行，保持原始代码不变。
                        decode.record_outcome(decode_ok);
                    // 中文逐行注释：下一行是原始源码第 660 行，保持原始代码不变。
                    }

                    // 中文逐行注释：下一行是原始源码第 662 行，保持原始代码不变。
                    let mut response = self
                        // 中文逐行注释：下一行是原始源码第 663 行，保持原始代码不变。
                        .handle_decode_error_response(res, &context, prefill, decode)
                        // 中文逐行注释：下一行是原始源码第 664 行，保持原始代码不变。
                        .await;
                    // 中文逐行注释：下一行是原始源码第 665 行，保持原始代码不变。
                    response.extensions_mut().insert(BreakerOutcomesRecorded);
                    // 中文逐行注释：下一行是原始源码第 666 行，保持原始代码不变。
                    return response;
                // 中文逐行注释：下一行是原始源码第 667 行，保持原始代码不变。
                }

                // 中文逐行注释：下一行是原始源码第 669 行，保持原始代码不变。
                // Process prefill response
                // 中文逐行注释：下一行是原始源码第 670 行，保持原始代码不变。
                let prefill_body = if context.return_logprob {
                    // 中文逐行注释：下一行是原始源码第 671 行，保持原始代码不变。
                    match self
                        // 中文逐行注释：下一行是原始源码第 672 行，保持原始代码不变。
                        .process_prefill_response(
                            // 中文逐行注释：下一行是原始源码第 673 行，保持原始代码不变。
                            prefill_result,
                            // 中文逐行注释：下一行是原始源码第 674 行，保持原始代码不变。
                            prefill.url(),
                            // 中文逐行注释：下一行是原始源码第 675 行，保持原始代码不变。
                            context.return_logprob,
                        // 中文逐行注释：下一行是原始源码第 676 行，保持原始代码不变。
                        )
                        // 中文逐行注释：下一行是原始源码第 677 行，保持原始代码不变。
                        .await
                    // 中文逐行注释：下一行是原始源码第 678 行，保持原始代码不变。
                    {
                        // 中文逐行注释：下一行是原始源码第 679 行，保持原始代码不变。
                        Ok((_, body)) => body,
                        // 中文逐行注释：下一行是原始源码第 680 行，保持原始代码不变。
                        Err(error_response) => return error_response,
                    // 中文逐行注释：下一行是原始源码第 681 行，保持原始代码不变。
                    }
                // 中文逐行注释：下一行是原始源码第 682 行，保持原始代码不变。
                } else {
                    // 中文逐行注释：下一行是原始源码第 683 行，保持原始代码不变。
                    // Even if we don't need logprobs, we should check prefill status
                    // 中文逐行注释：下一行是原始源码第 684 行，保持原始代码不变。
                    match self
                        // 中文逐行注释：下一行是原始源码第 685 行，保持原始代码不变。
                        .process_prefill_response(prefill_result, prefill.url(), false)
                        // 中文逐行注释：下一行是原始源码第 686 行，保持原始代码不变。
                        .await
                    // 中文逐行注释：下一行是原始源码第 687 行，保持原始代码不变。
                    {
                        // 中文逐行注释：下一行是原始源码第 688 行，保持原始代码不变。
                        Ok((_, body)) => body,
                        // 中文逐行注释：下一行是原始源码第 689 行，保持原始代码不变。
                        Err(error_response) => return error_response,
                    // 中文逐行注释：下一行是原始源码第 690 行，保持原始代码不变。
                    }
                // 中文逐行注释：下一行是原始源码第 691 行，保持原始代码不变。
                };

                // 中文逐行注释：下一行是原始源码第 693 行，保持原始代码不变。
                if context.is_stream {
                    // 中文逐行注释：下一行是原始源码第 694 行，保持原始代码不变。
                    // Streaming response
                    // 中文逐行注释：下一行是原始源码第 695 行，保持原始代码不变。
                    let prefill_logprobs = if context.return_logprob {
                        // 中文逐行注释：下一行是原始源码第 696 行，保持原始代码不变。
                        prefill_body
                            // 中文逐行注释：下一行是原始源码第 697 行，保持原始代码不变。
                            .as_ref()
                            // 中文逐行注释：下一行是原始源码第 698 行，保持原始代码不变。
                            .and_then(|body| serde_json::from_slice::<Value>(body).ok())
                            // 中文逐行注释：下一行是原始源码第 699 行，保持原始代码不变。
                            .and_then(|json| {
                                // 中文逐行注释：下一行是原始源码第 700 行，保持原始代码不变。
                                json.pointer("/meta_info/input_token_logprobs").cloned()
                            // 中文逐行注释：下一行是原始源码第 701 行，保持原始代码不变。
                            })
                    // 中文逐行注释：下一行是原始源码第 702 行，保持原始代码不变。
                    } else {
                        // 中文逐行注释：下一行是原始源码第 703 行，保持原始代码不变。
                        None
                    // 中文逐行注释：下一行是原始源码第 704 行，保持原始代码不变。
                    };

                    // 中文逐行注释：下一行是原始源码第 706 行，保持原始代码不变。
                    let response_headers = header_utils::preserve_response_headers(res.headers());

                    // 中文逐行注释：下一行是原始源码第 708 行，保持原始代码不变。
                    self.create_streaming_response(
                        // 中文逐行注释：下一行是原始源码第 709 行，保持原始代码不变。
                        res.bytes_stream(),
                        // 中文逐行注释：下一行是原始源码第 710 行，保持原始代码不变。
                        status,
                        // 中文逐行注释：下一行是原始源码第 711 行，保持原始代码不变。
                        prefill_logprobs,
                        // 中文逐行注释：下一行是原始源码第 712 行，保持原始代码不变。
                        context.return_logprob,
                        // 中文逐行注释：下一行是原始源码第 713 行，保持原始代码不变。
                        Some(response_headers),
                        // 中文逐行注释：下一行是原始源码第 714 行，保持原始代码不变。
                        prefill,
                        // 中文逐行注释：下一行是原始源码第 715 行，保持原始代码不变。
                        decode,
                    // 中文逐行注释：下一行是原始源码第 716 行，保持原始代码不变。
                    )
                // 中文逐行注释：下一行是原始源码第 717 行，保持原始代码不变。
                } else {
                    // 中文逐行注释：下一行是原始源码第 718 行，保持原始代码不变。
                    // Non-streaming response
                    // 中文逐行注释：下一行是原始源码第 719 行，保持原始代码不变。
                    if context.return_logprob {
                        // 中文逐行注释：下一行是原始源码第 720 行，保持原始代码不变。
                        self.process_non_streaming_response(
                            // 中文逐行注释：下一行是原始源码第 721 行，保持原始代码不变。
                            res,
                            // 中文逐行注释：下一行是原始源码第 722 行，保持原始代码不变。
                            status,
                            // 中文逐行注释：下一行是原始源码第 723 行，保持原始代码不变。
                            context.return_logprob,
                            // 中文逐行注释：下一行是原始源码第 724 行，保持原始代码不变。
                            prefill_body,
                        // 中文逐行注释：下一行是原始源码第 725 行，保持原始代码不变。
                        )
                        // 中文逐行注释：下一行是原始源码第 726 行，保持原始代码不变。
                        .await
                    // 中文逐行注释：下一行是原始源码第 727 行，保持原始代码不变。
                    } else {
                        // 中文逐行注释：下一行是原始源码第 728 行，保持原始代码不变。
                        // Direct passthrough when no logprobs needed
                        // 中文逐行注释：下一行是原始源码第 729 行，保持原始代码不变。
                        let response_headers =
                            // 中文逐行注释：下一行是原始源码第 730 行，保持原始代码不变。
                            header_utils::preserve_response_headers(res.headers());

                        // 中文逐行注释：下一行是原始源码第 732 行，保持原始代码不变。
                        match res.bytes().await {
                            // 中文逐行注释：下一行是原始源码第 733 行，保持原始代码不变。
                            Ok(decode_body) => {
                                // 中文逐行注释：下一行是原始源码第 734 行，保持原始代码不变。
                                let mut response = Response::new(Body::from(decode_body));
                                // 中文逐行注释：下一行是原始源码第 735 行，保持原始代码不变。
                                *response.status_mut() = status;
                                // 中文逐行注释：下一行是原始源码第 736 行，保持原始代码不变。
                                *response.headers_mut() = response_headers;
                                // 中文逐行注释：下一行是原始源码第 737 行，保持原始代码不变。
                                response
                            // 中文逐行注释：下一行是原始源码第 738 行，保持原始代码不变。
                            }
                            // 中文逐行注释：下一行是原始源码第 739 行，保持原始代码不变。
                            Err(e) => {
                                // 中文逐行注释：下一行是原始源码第 740 行，保持原始代码不变。
                                error!("Failed to read decode response: {}", e);
                                // 中文逐行注释：下一行是原始源码第 741 行，保持原始代码不变。
                                error::internal_error(
                                    // 中文逐行注释：下一行是原始源码第 742 行，保持原始代码不变。
                                    "read_response_failed",
                                    // 中文逐行注释：下一行是原始源码第 743 行，保持原始代码不变。
                                    "Failed to read response",
                                // 中文逐行注释：下一行是原始源码第 744 行，保持原始代码不变。
                                )
                            // 中文逐行注释：下一行是原始源码第 745 行，保持原始代码不变。
                            }
                        // 中文逐行注释：下一行是原始源码第 746 行，保持原始代码不变。
                        }
                    // 中文逐行注释：下一行是原始源码第 747 行，保持原始代码不变。
                    }
                // 中文逐行注释：下一行是原始源码第 748 行，保持原始代码不变。
                }
            // 中文逐行注释：下一行是原始源码第 749 行，保持原始代码不变。
            }
            // 中文逐行注释：下一行是原始源码第 750 行，保持原始代码不变。
            Err(e) => {
                // 中文逐行注释：下一行是原始源码第 751 行，保持原始代码不变。
                error!(
                    // 中文逐行注释：下一行是原始源码第 752 行，保持原始代码不变。
                    decode_url = %decode.url(),
                    // 中文逐行注释：下一行是原始源码第 753 行，保持原始代码不变。
                    error = %e,
                    // 中文逐行注释：下一行是原始源码第 754 行，保持原始代码不变。
                    "Decode request failed"
                // 中文逐行注释：下一行是原始源码第 755 行，保持原始代码不变。
                );
                // 中文逐行注释：下一行是原始源码第 756 行，保持原始代码不变。
                // Decode failed at TCP/transport level. No tracked
                // 中文逐行注释：下一行是原始源码第 757 行，保持原始代码不变。
                // stream will ever wrap a response (streaming path) and
                // 中文逐行注释：下一行是原始源码第 758 行，保持原始代码不变。
                // we shortcut past the outer non-streaming
                // 中文逐行注释：下一行是原始源码第 759 行，保持原始代码不变。
                // `record_outcome` too — so record decode failure
                // 中文逐行注释：下一行是原始源码第 760 行，保持原始代码不变。
                // directly. Prefill ran concurrently in the
                // 中文逐行注释：下一行是原始源码第 761 行，保持原始代码不变。
                // `tokio::join!`: record its real per-worker outcome
                // 中文逐行注释：下一行是原始源码第 762 行，保持原始代码不变。
                // (success on a 2xx/4xx send, failure on transport
                // 中文逐行注释：下一行是原始源码第 763 行，保持原始代码不变。
                // error) so the decode-driven 502 doesn't penalise a
                // 中文逐行注释：下一行是原始源码第 764 行，保持原始代码不变。
                // healthy prefill. Mark the response so the outer
                // 中文逐行注释：下一行是原始源码第 765 行，保持原始代码不变。
                // dispatcher skips its status-derived `record_outcome`
                // 中文逐行注释：下一行是原始源码第 766 行，保持原始代码不变。
                // and we don't double-count.
                // 中文逐行注释：下一行是原始源码第 767 行，保持原始代码不变。
                decode.record_outcome(false);
                // 中文逐行注释：下一行是原始源码第 768 行，保持原始代码不变。
                let prefill_ok = match &prefill_result {
                    // 中文逐行注释：下一行是原始源码第 769 行，保持原始代码不变。
                    Ok(res) => {
                        // 中文逐行注释：下一行是原始源码第 770 行，保持原始代码不变。
                        let s = res.status();
                        // 中文逐行注释：下一行是原始源码第 771 行，保持原始代码不变。
                        s.is_success() || s.is_client_error()
                    // 中文逐行注释：下一行是原始源码第 772 行，保持原始代码不变。
                    }
                    // 中文逐行注释：下一行是原始源码第 773 行，保持原始代码不变。
                    Err(_) => false,
                // 中文逐行注释：下一行是原始源码第 774 行，保持原始代码不变。
                };
                // 中文逐行注释：下一行是原始源码第 775 行，保持原始代码不变。
                prefill.record_outcome(prefill_ok);

                // 中文逐行注释：下一行是原始源码第 777 行，保持原始代码不变。
                let mut response = error::bad_gateway(
                    // 中文逐行注释：下一行是原始源码第 778 行，保持原始代码不变。
                    "decode_server_error",
                    // 中文逐行注释：下一行是原始源码第 779 行，保持原始代码不变。
                    format!("Decode server error: {}", e),
                // 中文逐行注释：下一行是原始源码第 780 行，保持原始代码不变。
                );
                // 中文逐行注释：下一行是原始源码第 781 行，保持原始代码不变。
                response.extensions_mut().insert(BreakerOutcomesRecorded);
                // 中文逐行注释：下一行是原始源码第 782 行，保持原始代码不变。
                response
            // 中文逐行注释：下一行是原始源码第 783 行，保持原始代码不变。
            }
        // 中文逐行注释：下一行是原始源码第 784 行，保持原始代码不变。
        }
    // 中文逐行注释：下一行是原始源码第 785 行，保持原始代码不变。
    }

    // 中文函数注释：判断当前策略是否需要提取请求文本用于路由。
    // 中文逐行注释：下一行是原始源码第 787 行，保持原始代码不变。
    fn policies_need_request_text(&self) -> bool {
        // 中文逐行注释：下一行是原始源码第 788 行，保持原始代码不变。
        let prefill_policy = self.policy_registry.get_prefill_policy();
        // 中文逐行注释：下一行是原始源码第 789 行，保持原始代码不变。
        let decode_policy = self.policy_registry.get_decode_policy();
        // 中文逐行注释：下一行是原始源码第 790 行，保持原始代码不变。
        prefill_policy.needs_request_text() || decode_policy.needs_request_text()
    // 中文逐行注释：下一行是原始源码第 791 行，保持原始代码不变。
    }

    // 中文函数注释：选择一组 prefill 和 decode worker。
    // 中文逐行注释：下一行是原始源码第 793 行，保持原始代码不变。
    async fn select_pd_pair(
        // 中文逐行注释：下一行是原始源码第 794 行，保持原始代码不变。
        &self,
        // 中文逐行注释：下一行是原始源码第 795 行，保持原始代码不变。
        request_text: Option<&str>,
        // 中文逐行注释：下一行是原始源码第 796 行，保持原始代码不变。
        model_id: Option<&str>,
        // 中文逐行注释：下一行是原始源码第 797 行，保持原始代码不变。
        headers: Option<&HeaderMap>,
    // 中文逐行注释：下一行是原始源码第 798 行，保持原始代码不变。
    ) -> Result<(Arc<dyn Worker>, Arc<dyn Worker>), String> {
        // 中文逐行注释：下一行是原始源码第 799 行，保持原始代码不变。
        let effective_model_id = if !self.enable_igw { None } else { model_id };

        // 中文逐行注释：下一行是原始源码第 801 行，保持原始代码不变。
        debug!(
            // 中文逐行注释：下一行是原始源码第 802 行，保持原始代码不变。
            "Selecting PD pair: enable_igw={}, model_id={:?}, effective_model_id={:?}",
            // 中文逐行注释：下一行是原始源码第 803 行，保持原始代码不变。
            self.enable_igw, model_id, effective_model_id
        // 中文逐行注释：下一行是原始源码第 804 行，保持原始代码不变。
        );

        // 中文逐行注释：下一行是原始源码第 806 行，保持原始代码不变。
        let prefill_workers = if let Some(model) = effective_model_id {
            // 中文逐行注释：下一行是原始源码第 807 行，保持原始代码不变。
            self.worker_registry
                // 中文逐行注释：下一行是原始源码第 808 行，保持原始代码不变。
                .get_by_model(model)
                // 中文逐行注释：下一行是原始源码第 809 行，保持原始代码不变。
                .iter()
                // 中文逐行注释：下一行是原始源码第 810 行，保持原始代码不变。
                .filter(|w| matches!(w.worker_type(), WorkerType::Prefill { .. }))
                // 中文逐行注释：下一行是原始源码第 811 行，保持原始代码不变。
                .cloned()
                // 中文逐行注释：下一行是原始源码第 812 行，保持原始代码不变。
                .collect()
        // 中文逐行注释：下一行是原始源码第 813 行，保持原始代码不变。
        } else {
            // 中文逐行注释：下一行是原始源码第 814 行，保持原始代码不变。
            self.worker_registry.get_prefill_workers()
        // 中文逐行注释：下一行是原始源码第 815 行，保持原始代码不变。
        };

        // 中文逐行注释：下一行是原始源码第 817 行，保持原始代码不变。
        let decode_workers = if let Some(model) = effective_model_id {
            // 中文逐行注释：下一行是原始源码第 818 行，保持原始代码不变。
            self.worker_registry
                // 中文逐行注释：下一行是原始源码第 819 行，保持原始代码不变。
                .get_by_model(model)
                // 中文逐行注释：下一行是原始源码第 820 行，保持原始代码不变。
                .iter()
                // 中文逐行注释：下一行是原始源码第 821 行，保持原始代码不变。
                .filter(|w| matches!(w.worker_type(), WorkerType::Decode))
                // 中文逐行注释：下一行是原始源码第 822 行，保持原始代码不变。
                .cloned()
                // 中文逐行注释：下一行是原始源码第 823 行，保持原始代码不变。
                .collect()
        // 中文逐行注释：下一行是原始源码第 824 行，保持原始代码不变。
        } else {
            // 中文逐行注释：下一行是原始源码第 825 行，保持原始代码不变。
            self.worker_registry.get_decode_workers()
        // 中文逐行注释：下一行是原始源码第 826 行，保持原始代码不变。
        };

        // 中文逐行注释：下一行是原始源码第 828 行，保持原始代码不变。
        let prefill_policy = self.policy_registry.get_prefill_policy();
        // 中文逐行注释：下一行是原始源码第 829 行，保持原始代码不变。
        let decode_policy = self.policy_registry.get_decode_policy();

        // 中文逐行注释：下一行是原始源码第 831 行，保持原始代码不变。
        // Get cached hash ring for consistent hashing
        // 中文逐行注释：下一行是原始源码第 832 行，保持原始代码不变。
        let hash_ring = self
            // 中文逐行注释：下一行是原始源码第 833 行，保持原始代码不变。
            .worker_registry
            // 中文逐行注释：下一行是原始源码第 834 行，保持原始代码不变。
            .get_hash_ring(effective_model_id.unwrap_or(UNKNOWN_MODEL_ID));

        // 中文逐行注释：下一行是原始源码第 836 行，保持原始代码不变。
        let prefill = Self::pick_worker_by_policy_arc(
            // 中文逐行注释：下一行是原始源码第 837 行，保持原始代码不变。
            &prefill_workers,
            // 中文逐行注释：下一行是原始源码第 838 行，保持原始代码不变。
            &*prefill_policy,
            // 中文逐行注释：下一行是原始源码第 839 行，保持原始代码不变。
            request_text,
            // 中文逐行注释：下一行是原始源码第 840 行，保持原始代码不变。
            headers,
            // 中文逐行注释：下一行是原始源码第 841 行，保持原始代码不变。
            hash_ring.clone(),
            // 中文逐行注释：下一行是原始源码第 842 行，保持原始代码不变。
            "prefill",
        // 中文逐行注释：下一行是原始源码第 843 行，保持原始代码不变。
        )
        // 中文逐行注释：下一行是原始源码第 844 行，保持原始代码不变。
        .await?;

        // 中文逐行注释：下一行是原始源码第 846 行，保持原始代码不变。
        let decode = Self::pick_worker_by_policy_arc(
            // 中文逐行注释：下一行是原始源码第 847 行，保持原始代码不变。
            &decode_workers,
            // 中文逐行注释：下一行是原始源码第 848 行，保持原始代码不变。
            &*decode_policy,
            // 中文逐行注释：下一行是原始源码第 849 行，保持原始代码不变。
            request_text,
            // 中文逐行注释：下一行是原始源码第 850 行，保持原始代码不变。
            headers,
            // 中文逐行注释：下一行是原始源码第 851 行，保持原始代码不变。
            hash_ring,
            // 中文逐行注释：下一行是原始源码第 852 行，保持原始代码不变。
            "decode",
        // 中文逐行注释：下一行是原始源码第 853 行，保持原始代码不变。
        )
        // 中文逐行注释：下一行是原始源码第 854 行，保持原始代码不变。
        .await?;

        // 中文逐行注释：下一行是原始源码第 856 行，保持原始代码不变。
        // Record worker selection metrics (Layer 3)
        // 中文逐行注释：下一行是原始源码第 857 行，保持原始代码不变。
        let model = model_id.unwrap_or(UNKNOWN_MODEL_ID);
        // 中文逐行注释：下一行是原始源码第 858 行，保持原始代码不变。
        Metrics::record_worker_selection(
            // 中文逐行注释：下一行是原始源码第 859 行，保持原始代码不变。
            metrics_labels::WORKER_PREFILL,
            // 中文逐行注释：下一行是原始源码第 860 行，保持原始代码不变。
            metrics_labels::CONNECTION_HTTP,
            // 中文逐行注释：下一行是原始源码第 861 行，保持原始代码不变。
            model,
            // 中文逐行注释：下一行是原始源码第 862 行，保持原始代码不变。
            prefill_policy.name(),
        // 中文逐行注释：下一行是原始源码第 863 行，保持原始代码不变。
        );
        // 中文逐行注释：下一行是原始源码第 864 行，保持原始代码不变。
        Metrics::record_worker_selection(
            // 中文逐行注释：下一行是原始源码第 865 行，保持原始代码不变。
            metrics_labels::WORKER_DECODE,
            // 中文逐行注释：下一行是原始源码第 866 行，保持原始代码不变。
            metrics_labels::CONNECTION_HTTP,
            // 中文逐行注释：下一行是原始源码第 867 行，保持原始代码不变。
            model,
            // 中文逐行注释：下一行是原始源码第 868 行，保持原始代码不变。
            decode_policy.name(),
        // 中文逐行注释：下一行是原始源码第 869 行，保持原始代码不变。
        );

        // 中文逐行注释：下一行是原始源码第 871 行，保持原始代码不变。
        Ok((prefill, decode))
    // 中文逐行注释：下一行是原始源码第 872 行，保持原始代码不变。
    }

    // 中文函数注释：按负载均衡策略从候选 worker 中选择目标 worker。
    // 中文逐行注释：下一行是原始源码第 874 行，保持原始代码不变。
    async fn pick_worker_by_policy_arc(
        // 中文逐行注释：下一行是原始源码第 875 行，保持原始代码不变。
        workers: &[Arc<dyn Worker>],
        // 中文逐行注释：下一行是原始源码第 876 行，保持原始代码不变。
        policy: &dyn LoadBalancingPolicy,
        // 中文逐行注释：下一行是原始源码第 877 行，保持原始代码不变。
        request_text: Option<&str>,
        // 中文逐行注释：下一行是原始源码第 878 行，保持原始代码不变。
        headers: Option<&HeaderMap>,
        // 中文逐行注释：下一行是原始源码第 879 行，保持原始代码不变。
        hash_ring: Option<Arc<HashRing>>,
        // 中文逐行注释：下一行是原始源码第 880 行，保持原始代码不变。
        worker_type: &str,
    // 中文逐行注释：下一行是原始源码第 881 行，保持原始代码不变。
    ) -> Result<Arc<dyn Worker>, String> {
        // 中文逐行注释：下一行是原始源码第 882 行，保持原始代码不变。
        if workers.is_empty() {
            // 中文逐行注释：下一行是原始源码第 883 行，保持原始代码不变。
            return Err(format!(
                // 中文逐行注释：下一行是原始源码第 884 行，保持原始代码不变。
                "No {} workers available. Please check if {} servers are configured and healthy.",
                // 中文逐行注释：下一行是原始源码第 885 行，保持原始代码不变。
                worker_type, worker_type
            // 中文逐行注释：下一行是原始源码第 886 行，保持原始代码不变。
            ));
        // 中文逐行注释：下一行是原始源码第 887 行，保持原始代码不变。
        }

        // 中文逐行注释：下一行是原始源码第 889 行，保持原始代码不变。
        let available_workers: Vec<Arc<dyn Worker>> = workers
            // 中文逐行注释：下一行是原始源码第 890 行，保持原始代码不变。
            .iter()
            // 中文逐行注释：下一行是原始源码第 891 行，保持原始代码不变。
            .filter(|w| w.is_available())
            // 中文逐行注释：下一行是原始源码第 892 行，保持原始代码不变。
            .cloned()
            // 中文逐行注释：下一行是原始源码第 893 行，保持原始代码不变。
            .collect();

        // 中文逐行注释：下一行是原始源码第 895 行，保持原始代码不变。
        if available_workers.is_empty() {
            // 中文逐行注释：下一行是原始源码第 896 行，保持原始代码不变。
            return Err(format!(
                // 中文逐行注释：下一行是原始源码第 897 行，保持原始代码不变。
                "No available {} workers (all circuits open or unhealthy)",
                // 中文逐行注释：下一行是原始源码第 898 行，保持原始代码不变。
                worker_type
            // 中文逐行注释：下一行是原始源码第 899 行，保持原始代码不变。
            ));
        // 中文逐行注释：下一行是原始源码第 900 行，保持原始代码不变。
        }

        // 中文逐行注释：下一行是原始源码第 902 行，保持原始代码不变。
        let selected_idx = policy
            // 中文逐行注释：下一行是原始源码第 903 行，保持原始代码不变。
            .select_worker(
                // 中文逐行注释：下一行是原始源码第 904 行，保持原始代码不变。
                &available_workers,
                // 中文逐行注释：下一行是原始源码第 905 行，保持原始代码不变。
                &SelectWorkerInfo {
                    // 中文逐行注释：下一行是原始源码第 906 行，保持原始代码不变。
                    request_text,
                    // 中文逐行注释：下一行是原始源码第 907 行，保持原始代码不变。
                    tokens: None, // HTTP doesn't have tokens, use gRPC for PrefixHash
                    // 中文逐行注释：下一行是原始源码第 908 行，保持原始代码不变。
                    headers,
                    // 中文逐行注释：下一行是原始源码第 909 行，保持原始代码不变。
                    hash_ring,
                // 中文逐行注释：下一行是原始源码第 910 行，保持原始代码不变。
                },
            // 中文逐行注释：下一行是原始源码第 911 行，保持原始代码不变。
            )
            // 中文逐行注释：下一行是原始源码第 912 行，保持原始代码不变。
            .await
            // 中文逐行注释：下一行是原始源码第 913 行，保持原始代码不变。
            .ok_or_else(|| {
                // 中文逐行注释：下一行是原始源码第 914 行，保持原始代码不变。
                format!(
                    // 中文逐行注释：下一行是原始源码第 915 行，保持原始代码不变。
                    "Policy {} failed to select a {} worker",
                    // 中文逐行注释：下一行是原始源码第 916 行，保持原始代码不变。
                    policy.name(),
                    // 中文逐行注释：下一行是原始源码第 917 行，保持原始代码不变。
                    worker_type
                // 中文逐行注释：下一行是原始源码第 918 行，保持原始代码不变。
                )
            // 中文逐行注释：下一行是原始源码第 919 行，保持原始代码不变。
            })?;

        // 中文逐行注释：下一行是原始源码第 921 行，保持原始代码不变。
        Ok(available_workers[selected_idx].clone())
    // 中文逐行注释：下一行是原始源码第 922 行，保持原始代码不变。
    }

    // 中文逐行注释：下一行是原始源码第 924 行，保持原始代码不变。
    #[allow(clippy::too_many_arguments)]
    // 中文函数注释：根据 decode 字节流创建 streaming HTTP 响应并跟踪 worker 状态。
    // 中文逐行注释：下一行是原始源码第 925 行，保持原始代码不变。
    fn create_streaming_response(
        // 中文逐行注释：下一行是原始源码第 926 行，保持原始代码不变。
        &self,
        // 中文逐行注释：下一行是原始源码第 927 行，保持原始代码不变。
        stream: impl futures_util::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
        // 中文逐行注释：下一行是原始源码第 928 行，保持原始代码不变。
        status: StatusCode,
        // 中文逐行注释：下一行是原始源码第 929 行，保持原始代码不变。
        prefill_logprobs: Option<Value>,
        // 中文逐行注释：下一行是原始源码第 930 行，保持原始代码不变。
        return_logprob: bool,
        // 中文逐行注释：下一行是原始源码第 931 行，保持原始代码不变。
        headers: Option<HeaderMap>,
        // 中文逐行注释：下一行是原始源码第 932 行，保持原始代码不变。
        prefill: Arc<dyn Worker>,
        // 中文逐行注释：下一行是原始源码第 933 行，保持原始代码不变。
        decode: Arc<dyn Worker>,
    // 中文逐行注释：下一行是原始源码第 934 行，保持原始代码不变。
    ) -> Response {
        // 中文逐行注释：下一行是原始源码第 935 行，保持原始代码不变。
        use crate::core::AttachedBody;

        // 中文逐行注释：下一行是原始源码第 937 行，保持原始代码不变。
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        // 中文逐行注释：下一行是原始源码第 939 行，保持原始代码不变。
        // Uses select! to race stream.next() against tx.closed() so that
        // 中文逐行注释：下一行是原始源码第 940 行，保持原始代码不变。
        // when the client disconnects the upstream HTTP connection is dropped
        // 中文逐行注释：下一行是原始源码第 941 行，保持原始代码不变。
        // promptly, allowing the engine to abort the request.
        // 中文逐行注释：下一行是原始源码第 942 行，保持原始代码不变。
        // `biased;` drains a ready upstream chunk before observing client
        // 中文逐行注释：下一行是原始源码第 943 行，保持原始代码不变。
        // disconnect, so a chunk already produced by reqwest reaches the
        // 中文逐行注释：下一行是原始源码第 944 行，保持原始代码不变。
        // client (and the logprob merger) before we tear the loop down.
        // 中文逐行注释：下一行是原始源码第 945 行，保持原始代码不变。
        //
        // 中文逐行注释：下一行是原始源码第 946 行，保持原始代码不变。
        // The upstream stream is wrapped in `BreakerTrackedStream` so the
        // 中文逐行注释：下一行是原始源码第 947 行，保持原始代码不变。
        // decode worker's circuit breaker is updated once on drop: success
        // 中文逐行注释：下一行是原始源码第 948 行，保持原始代码不变。
        // on clean completion (`[DONE]` sentinel or `None`), failure on
        // 中文逐行注释：下一行是原始源码第 949 行，保持原始代码不变。
        // stream error, neither on client disconnect. PD's pre-PR semantics
        // 中文逐行注释：下一行是原始源码第 950 行，保持原始代码不变。
        // treated 4xx (client error) as not-a-worker-fault, so we only
        // 中文逐行注释：下一行是原始源码第 951 行，保持原始代码不变。
        // pre-mark the wrapper as Errored on 5xx — `handle_decode_error_response`
        // 中文逐行注释：下一行是原始源码第 952 行，保持原始代码不变。
        // synthesizes a single-chunk SSE error envelope that would otherwise
        // 中文逐行注释：下一行是原始源码第 953 行，保持原始代码不变。
        // stream cleanly to None and record a spurious success.
        // 中文逐行注释：下一行是原始源码第 954 行，保持原始代码不变。
        let mut tracked =
            // 中文逐行注释：下一行是原始源码第 955 行，保持原始代码不变。
            BreakerTrackedStream::new(stream, Arc::clone(&decode), decode.url().to_string());
        // 中文逐行注释：下一行是原始源码第 956 行，保持原始代码不变。
        if !(status.is_success() || status.is_client_error()) {
            // 中文逐行注释：下一行是原始源码第 957 行，保持原始代码不变。
            tracked.mark_errored();
        // 中文逐行注释：下一行是原始源码第 958 行，保持原始代码不变。
        }
        // 中文逐行注释：下一行是原始源码第 959 行，保持原始代码不变。
        let decode_for_log = decode.clone();
        // 中文逐行注释：下一行是原始源码第 960 行，保持原始代码不变。
        tokio::spawn(async move {
            // 中文逐行注释：下一行是原始源码第 961 行，保持原始代码不变。
            loop {
                // 中文逐行注释：下一行是原始源码第 962 行，保持原始代码不变。
                tokio::select! {
                    // 中文逐行注释：下一行是原始源码第 963 行，保持原始代码不变。
                    biased;
                    // 中文逐行注释：下一行是原始源码第 964 行，保持原始代码不变。
                    chunk_result = tracked.next() => {
                        // 中文逐行注释：下一行是原始源码第 965 行，保持原始代码不变。
                        match chunk_result {
                            // 中文逐行注释：下一行是原始源码第 966 行，保持原始代码不变。
                            Some(Ok(chunk)) => {
                                // 中文逐行注释：下一行是原始源码第 967 行，保持原始代码不变。
                                let is_done = memmem::find(&chunk, b"data: [DONE]").is_some();

                                // 中文逐行注释：下一行是原始源码第 969 行，保持原始代码不变。
                                let result = if return_logprob && prefill_logprobs.is_some() {
                                    // 中文逐行注释：下一行是原始源码第 970 行，保持原始代码不变。
                                    Self::merge_streaming_logprobs(prefill_logprobs.clone(), &chunk)
                                        // 中文逐行注释：下一行是原始源码第 971 行，保持原始代码不变。
                                        .unwrap_or(chunk)
                                // 中文逐行注释：下一行是原始源码第 972 行，保持原始代码不变。
                                } else {
                                    // 中文逐行注释：下一行是原始源码第 973 行，保持原始代码不变。
                                    chunk
                                // 中文逐行注释：下一行是原始源码第 974 行，保持原始代码不变。
                                };

                                // 中文逐行注释：下一行是原始源码第 976 行，保持原始代码不变。
                                // Mark the wrapper completed before the client
                                // 中文逐行注释：下一行是原始源码第 977 行，保持原始代码不变。
                                // send: upstream finished cleanly regardless of
                                // 中文逐行注释：下一行是原始源码第 978 行，保持原始代码不变。
                                // whether the client is still listening, and
                                // 中文逐行注释：下一行是原始源码第 979 行，保持原始代码不变。
                                // the worker deserves the success tick either
                                // 中文逐行注释：下一行是原始源码第 980 行，保持原始代码不变。
                                // way. `mark_completed` is a no-op once Errored
                                // 中文逐行注释：下一行是原始源码第 981 行，保持原始代码不变。
                                // is set, so the synthetic-error path is unaffected.
                                // 中文逐行注释：下一行是原始源码第 982 行，保持原始代码不变。
                                if is_done {
                                    // 中文逐行注释：下一行是原始源码第 983 行，保持原始代码不变。
                                    tracked.mark_completed();
                                // 中文逐行注释：下一行是原始源码第 984 行，保持原始代码不变。
                                }

                                // 中文逐行注释：下一行是原始源码第 986 行，保持原始代码不变。
                                if tx.send(Ok(result)).is_err() {
                                    // 中文逐行注释：下一行是原始源码第 987 行，保持原始代码不变。
                                    tracing::debug!(
                                        // 中文逐行注释：下一行是原始源码第 988 行，保持原始代码不变。
                                        "Receiver dropped (likely client disconnect), \
                                        // 中文逐行注释：下一行是原始源码第 989 行，保持原始代码不变。
                                        cancelling upstream PD stream"
                                    // 中文逐行注释：下一行是原始源码第 990 行，保持原始代码不变。
                                    );
                                    // 中文逐行注释：下一行是原始源码第 991 行，保持原始代码不变。
                                    break;
                                // 中文逐行注释：下一行是原始源码第 992 行，保持原始代码不变。
                                }

                                // 中文逐行注释：下一行是原始源码第 994 行，保持原始代码不变。
                                if is_done {
                                    // 中文逐行注释：下一行是原始源码第 995 行，保持原始代码不变。
                                    break;
                                // 中文逐行注释：下一行是原始源码第 996 行，保持原始代码不变。
                                }
                            // 中文逐行注释：下一行是原始源码第 997 行，保持原始代码不变。
                            }
                            // 中文逐行注释：下一行是原始源码第 998 行，保持原始代码不变。
                            Some(Err(e)) => {
                                // 中文逐行注释：下一行是原始源码第 999 行，保持原始代码不变。
                                // BreakerTrackedStream already logged the error
                                // 中文逐行注释：下一行是原始源码第 1000 行，保持原始代码不变。
                                // and marked the terminal state as Errored so
                                // 中文逐行注释：下一行是原始源码第 1001 行，保持原始代码不变。
                                // the worker's circuit breaker will tick on drop.
                                // 中文逐行注释：下一行是原始源码第 1002 行，保持原始代码不变。
                                let _ = tx.send(Err(format!("Stream error: {}", e)));
                                // 中文逐行注释：下一行是原始源码第 1003 行，保持原始代码不变。
                                break;
                            // 中文逐行注释：下一行是原始源码第 1004 行，保持原始代码不变。
                            }
                            // 中文逐行注释：下一行是原始源码第 1005 行，保持原始代码不变。
                            None => break,
                        // 中文逐行注释：下一行是原始源码第 1006 行，保持原始代码不变。
                        }
                    // 中文逐行注释：下一行是原始源码第 1007 行，保持原始代码不变。
                    }
                    // 中文逐行注释：下一行是原始源码第 1008 行，保持原始代码不变。
                    _ = tx.closed() => {
                        // 中文逐行注释：下一行是原始源码第 1009 行，保持原始代码不变。
                        tracing::info!(
                            // 中文逐行注释：下一行是原始源码第 1010 行，保持原始代码不变。
                            "Client disconnected, cancelling upstream PD stream from {}",
                            // 中文逐行注释：下一行是原始源码第 1011 行，保持原始代码不变。
                            decode_for_log.url()
                        // 中文逐行注释：下一行是原始源码第 1012 行，保持原始代码不变。
                        );
                        // 中文逐行注释：下一行是原始源码第 1013 行，保持原始代码不变。
                        break;
                    // 中文逐行注释：下一行是原始源码第 1014 行，保持原始代码不变。
                    }
                // 中文逐行注释：下一行是原始源码第 1015 行，保持原始代码不变。
                }
            // 中文逐行注释：下一行是原始源码第 1016 行，保持原始代码不变。
            }
        // 中文逐行注释：下一行是原始源码第 1017 行，保持原始代码不变。
        });

        // 中文逐行注释：下一行是原始源码第 1019 行，保持原始代码不变。
        let stream = UnboundedReceiverStream::new(rx);
        // 中文逐行注释：下一行是原始源码第 1020 行，保持原始代码不变。
        let body = Body::from_stream(stream);

        // 中文逐行注释：下一行是原始源码第 1022 行，保持原始代码不变。
        let guards = vec![
            // 中文逐行注释：下一行是原始源码第 1023 行，保持原始代码不变。
            WorkerLoadGuard::new(prefill, headers.as_ref()),
            // 中文逐行注释：下一行是原始源码第 1024 行，保持原始代码不变。
            WorkerLoadGuard::new(decode, headers.as_ref()),
        // 中文逐行注释：下一行是原始源码第 1025 行，保持原始代码不变。
        ];

        // 中文逐行注释：下一行是原始源码第 1027 行，保持原始代码不变。
        let mut response = Response::new(body);
        // 中文逐行注释：下一行是原始源码第 1028 行，保持原始代码不变。
        *response.status_mut() = status;

        // 中文逐行注释：下一行是原始源码第 1030 行，保持原始代码不变。
        let mut response_headers = headers.unwrap_or_default();
        // 中文逐行注释：下一行是原始源码第 1031 行，保持原始代码不变。
        response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        // 中文逐行注释：下一行是原始源码第 1032 行，保持原始代码不变。
        *response.headers_mut() = response_headers;

        // 中文逐行注释：下一行是原始源码第 1034 行，保持原始代码不变。
        AttachedBody::wrap_response(response, guards)
    // 中文逐行注释：下一行是原始源码第 1035 行，保持原始代码不变。
    }

    // 中文逐行注释：下一行是原始源码第 1037 行，保持原始代码不变。
    // Helper to process non-streaming decode response with logprob merging
    // 中文函数注释：处理非流式 decode 响应，并按需合并 prefill logprobs。
    // 中文逐行注释：下一行是原始源码第 1038 行，保持原始代码不变。
    async fn process_non_streaming_response(
        // 中文逐行注释：下一行是原始源码第 1039 行，保持原始代码不变。
        &self,
        // 中文逐行注释：下一行是原始源码第 1040 行，保持原始代码不变。
        res: reqwest::Response,
        // 中文逐行注释：下一行是原始源码第 1041 行，保持原始代码不变。
        status: StatusCode,
        // 中文逐行注释：下一行是原始源码第 1042 行，保持原始代码不变。
        return_logprob: bool,
        // 中文逐行注释：下一行是原始源码第 1043 行，保持原始代码不变。
        prefill_body: Option<bytes::Bytes>,
    // 中文逐行注释：下一行是原始源码第 1044 行，保持原始代码不变。
    ) -> Response {
        // 中文逐行注释：下一行是原始源码第 1045 行，保持原始代码不变。
        let response = res.bytes().await;
        // 中文逐行注释：下一行是原始源码第 1046 行，保持原始代码不变。
        let decode_body = match response {
            // 中文逐行注释：下一行是原始源码第 1047 行，保持原始代码不变。
            Ok(decode_body) => decode_body,
            // 中文逐行注释：下一行是原始源码第 1048 行，保持原始代码不变。
            Err(e) => {
                // 中文逐行注释：下一行是原始源码第 1049 行，保持原始代码不变。
                error!("Failed to read decode response: {}", e);
                // 中文逐行注释：下一行是原始源码第 1050 行，保持原始代码不变。
                return error::internal_error("read_response_failed", "Failed to read response");
            // 中文逐行注释：下一行是原始源码第 1051 行，保持原始代码不变。
            }
        // 中文逐行注释：下一行是原始源码第 1052 行，保持原始代码不变。
        };

        // 中文逐行注释：下一行是原始源码第 1054 行，保持原始代码不变。
        if !return_logprob {
            // 中文逐行注释：下一行是原始源码第 1055 行，保持原始代码不变。
            return (status, decode_body).into_response();
        // 中文逐行注释：下一行是原始源码第 1056 行，保持原始代码不变。
        }

        // 中文逐行注释：下一行是原始源码第 1058 行，保持原始代码不变。
        let Some(prefill_body) = prefill_body else {
            // 中文逐行注释：下一行是原始源码第 1059 行，保持原始代码不变。
            return (status, decode_body).into_response();
        // 中文逐行注释：下一行是原始源码第 1060 行，保持原始代码不变。
        };

        // 中文逐行注释：下一行是原始源码第 1062 行，保持原始代码不变。
        // Merge logprobs from prefill and decode
        // 中文逐行注释：下一行是原始源码第 1063 行，保持原始代码不变。
        let (Ok(prefill_json), Ok(mut decode_json)) = (
            // 中文逐行注释：下一行是原始源码第 1064 行，保持原始代码不变。
            serde_json::from_slice::<Value>(&prefill_body),
            // 中文逐行注释：下一行是原始源码第 1065 行，保持原始代码不变。
            serde_json::from_slice::<Value>(&decode_body),
        // 中文逐行注释：下一行是原始源码第 1066 行，保持原始代码不变。
        ) else {
            // 中文逐行注释：下一行是原始源码第 1067 行，保持原始代码不变。
            warn!("Failed to parse responses for logprob merging");
            // 中文逐行注释：下一行是原始源码第 1068 行，保持原始代码不变。
            return (status, decode_body).into_response();
        // 中文逐行注释：下一行是原始源码第 1069 行，保持原始代码不变。
        };

        // 中文逐行注释：下一行是原始源码第 1071 行，保持原始代码不变。
        Self::merge_logprobs_in_json(&prefill_json, &mut decode_json);

        // 中文逐行注释：下一行是原始源码第 1073 行，保持原始代码不变。
        // Return merged response
        // 中文逐行注释：下一行是原始源码第 1074 行，保持原始代码不变。
        match serde_json::to_vec(&decode_json) {
            // 中文逐行注释：下一行是原始源码第 1075 行，保持原始代码不变。
            Ok(body) => (status, body).into_response(),
            // 中文逐行注释：下一行是原始源码第 1076 行，保持原始代码不变。
            Err(e) => {
                // 中文逐行注释：下一行是原始源码第 1077 行，保持原始代码不变。
                error!("Failed to serialize merged response: {}", e);
                // 中文逐行注释：下一行是原始源码第 1078 行，保持原始代码不变。
                (status, decode_body).into_response()
            // 中文逐行注释：下一行是原始源码第 1079 行，保持原始代码不变。
            }
        // 中文逐行注释：下一行是原始源码第 1080 行，保持原始代码不变。
        }
    // 中文逐行注释：下一行是原始源码第 1081 行，保持原始代码不变。
    }

    // 中文逐行注释：下一行是原始源码第 1083 行，保持原始代码不变。
    // Helper to process prefill response and extract body if needed for logprobs
    // 中文函数注释：处理 prefill 响应，提取状态和内容。
    // 中文逐行注释：下一行是原始源码第 1084 行，保持原始代码不变。
    async fn process_prefill_response(
        // 中文逐行注释：下一行是原始源码第 1085 行，保持原始代码不变。
        &self,
        // 中文逐行注释：下一行是原始源码第 1086 行，保持原始代码不变。
        prefill_result: Result<reqwest::Response, reqwest::Error>,
        // 中文逐行注释：下一行是原始源码第 1087 行，保持原始代码不变。
        prefill_url: &str,
        // 中文逐行注释：下一行是原始源码第 1088 行，保持原始代码不变。
        return_logprob: bool,
    // 中文逐行注释：下一行是原始源码第 1089 行，保持原始代码不变。
    ) -> Result<(StatusCode, Option<bytes::Bytes>), Response> {
        // 中文逐行注释：下一行是原始源码第 1090 行，保持原始代码不变。
        // Check prefill result first - it's critical for disaggregated mode
        // 中文逐行注释：下一行是原始源码第 1091 行，保持原始代码不变。
        let prefill_response = match prefill_result {
            // 中文逐行注释：下一行是原始源码第 1092 行，保持原始代码不变。
            Ok(response) => response,
            // 中文逐行注释：下一行是原始源码第 1093 行，保持原始代码不变。
            Err(e) => {
                // 中文逐行注释：下一行是原始源码第 1094 行，保持原始代码不变。
                error!(
                    // 中文逐行注释：下一行是原始源码第 1095 行，保持原始代码不变。
                    "Prefill server failed (CRITICAL) prefill_url={} error={}. Decode will timeout without prefill KV cache.",
                    // 中文逐行注释：下一行是原始源码第 1096 行，保持原始代码不变。
                    prefill_url,
                    // 中文逐行注释：下一行是原始源码第 1097 行，保持原始代码不变。
                    e
                // 中文逐行注释：下一行是原始源码第 1098 行，保持原始代码不变。
                );

                // 中文逐行注释：下一行是原始源码第 1100 行，保持原始代码不变。
                // Return error immediately - don't wait for decode to timeout
                // 中文逐行注释：下一行是原始源码第 1101 行，保持原始代码不变。
                return Err(error::bad_gateway(
                    // 中文逐行注释：下一行是原始源码第 1102 行，保持原始代码不变。
                    "prefill_server_error",
                    // 中文逐行注释：下一行是原始源码第 1103 行，保持原始代码不变。
                    format!(
                        // 中文逐行注释：下一行是原始源码第 1104 行，保持原始代码不变。
                        "Prefill server error: {}. This will cause decode timeout.",
                        // 中文逐行注释：下一行是原始源码第 1105 行，保持原始代码不变。
                        e
                    // 中文逐行注释：下一行是原始源码第 1106 行，保持原始代码不变。
                    ),
                // 中文逐行注释：下一行是原始源码第 1107 行，保持原始代码不变。
                ));
            // 中文逐行注释：下一行是原始源码第 1108 行，保持原始代码不变。
            }
        // 中文逐行注释：下一行是原始源码第 1109 行，保持原始代码不变。
        };

        // 中文逐行注释：下一行是原始源码第 1111 行，保持原始代码不变。
        let prefill_status = StatusCode::from_u16(prefill_response.status().as_u16())
            // 中文逐行注释：下一行是原始源码第 1112 行，保持原始代码不变。
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        // 中文逐行注释：下一行是原始源码第 1114 行，保持原始代码不变。
        // Check if prefill succeeded
        // 中文逐行注释：下一行是原始源码第 1115 行，保持原始代码不变。
        if !prefill_status.is_success() {
            // 中文逐行注释：下一行是原始源码第 1116 行，保持原始代码不变。
            // Get error body from prefill
            // 中文逐行注释：下一行是原始源码第 1117 行，保持原始代码不变。
            let error_msg = prefill_response
                // 中文逐行注释：下一行是原始源码第 1118 行，保持原始代码不变。
                .text()
                // 中文逐行注释：下一行是原始源码第 1119 行，保持原始代码不变。
                .await
                // 中文逐行注释：下一行是原始源码第 1120 行，保持原始代码不变。
                .unwrap_or_else(|_| "Unknown prefill error".to_string());

            // 中文逐行注释：下一行是原始源码第 1122 行，保持原始代码不变。
            error!(
                // 中文逐行注释：下一行是原始源码第 1123 行，保持原始代码不变。
                "Prefill server returned error status prefill_url={} status={} body={}",
                // 中文逐行注释：下一行是原始源码第 1124 行，保持原始代码不变。
                prefill_url, prefill_status, error_msg
            // 中文逐行注释：下一行是原始源码第 1125 行，保持原始代码不变。
            );

            // 中文逐行注释：下一行是原始源码第 1127 行，保持原始代码不变。
            // Map prefill_status to appropriate error function
            // 中文逐行注释：下一行是原始源码第 1128 行，保持原始代码不变。
            let error_response = match prefill_status {
                // 中文逐行注释：下一行是原始源码第 1129 行，保持原始代码不变。
                StatusCode::BAD_REQUEST => error::bad_request(
                    // 中文逐行注释：下一行是原始源码第 1130 行，保持原始代码不变。
                    "prefill_bad_request",
                    // 中文逐行注释：下一行是原始源码第 1131 行，保持原始代码不变。
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                // 中文逐行注释：下一行是原始源码第 1132 行，保持原始代码不变。
                ),
                // 中文逐行注释：下一行是原始源码第 1133 行，保持原始代码不变。
                StatusCode::NOT_FOUND => error::not_found(
                    // 中文逐行注释：下一行是原始源码第 1134 行，保持原始代码不变。
                    "prefill_not_found",
                    // 中文逐行注释：下一行是原始源码第 1135 行，保持原始代码不变。
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                // 中文逐行注释：下一行是原始源码第 1136 行，保持原始代码不变。
                ),
                // 中文逐行注释：下一行是原始源码第 1137 行，保持原始代码不变。
                StatusCode::INTERNAL_SERVER_ERROR => error::internal_error(
                    // 中文逐行注释：下一行是原始源码第 1138 行，保持原始代码不变。
                    "prefill_internal_error",
                    // 中文逐行注释：下一行是原始源码第 1139 行，保持原始代码不变。
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                // 中文逐行注释：下一行是原始源码第 1140 行，保持原始代码不变。
                ),
                // 中文逐行注释：下一行是原始源码第 1141 行，保持原始代码不变。
                StatusCode::SERVICE_UNAVAILABLE => error::service_unavailable(
                    // 中文逐行注释：下一行是原始源码第 1142 行，保持原始代码不变。
                    "prefill_unavailable",
                    // 中文逐行注释：下一行是原始源码第 1143 行，保持原始代码不变。
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                // 中文逐行注释：下一行是原始源码第 1144 行，保持原始代码不变。
                ),
                // 中文逐行注释：下一行是原始源码第 1145 行，保持原始代码不变。
                StatusCode::BAD_GATEWAY => error::bad_gateway(
                    // 中文逐行注释：下一行是原始源码第 1146 行，保持原始代码不变。
                    "prefill_bad_gateway",
                    // 中文逐行注释：下一行是原始源码第 1147 行，保持原始代码不变。
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                // 中文逐行注释：下一行是原始源码第 1148 行，保持原始代码不变。
                ),
                // 中文逐行注释：下一行是原始源码第 1149 行，保持原始代码不变。
                _ => error::internal_error(
                    // 中文逐行注释：下一行是原始源码第 1150 行，保持原始代码不变。
                    "prefill_error",
                    // 中文逐行注释：下一行是原始源码第 1151 行，保持原始代码不变。
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                // 中文逐行注释：下一行是原始源码第 1152 行，保持原始代码不变。
                ),
            // 中文逐行注释：下一行是原始源码第 1153 行，保持原始代码不变。
            };
            // 中文逐行注释：下一行是原始源码第 1154 行，保持原始代码不变。
            return Err(error_response);
        // 中文逐行注释：下一行是原始源码第 1155 行，保持原始代码不变。
        }

        // 中文逐行注释：下一行是原始源码第 1157 行，保持原始代码不变。
        // Read prefill body if needed for logprob merging
        // 中文逐行注释：下一行是原始源码第 1158 行，保持原始代码不变。
        let prefill_body = if return_logprob {
            // 中文逐行注释：下一行是原始源码第 1159 行，保持原始代码不变。
            match prefill_response.bytes().await {
                // 中文逐行注释：下一行是原始源码第 1160 行，保持原始代码不变。
                Ok(body) => Some(body),
                // 中文逐行注释：下一行是原始源码第 1161 行，保持原始代码不变。
                Err(e) => {
                    // 中文逐行注释：下一行是原始源码第 1162 行，保持原始代码不变。
                    warn!("Failed to read prefill response body for logprobs: {}", e);
                    // 中文逐行注释：下一行是原始源码第 1163 行，保持原始代码不变。
                    None
                // 中文逐行注释：下一行是原始源码第 1164 行，保持原始代码不变。
                }
            // 中文逐行注释：下一行是原始源码第 1165 行，保持原始代码不变。
            }
        // 中文逐行注释：下一行是原始源码第 1166 行，保持原始代码不变。
        } else {
            // 中文逐行注释：下一行是原始源码第 1167 行，保持原始代码不变。
            // For non-logprob requests, just consume the response without storing
            // 中文逐行注释：下一行是原始源码第 1168 行，保持原始代码不变。
            debug!("Consuming prefill response body (non-logprob request)");
            // 中文逐行注释：下一行是原始源码第 1169 行，保持原始代码不变。
            match prefill_response.bytes().await {
                // 中文逐行注释：下一行是原始源码第 1170 行，保持原始代码不变。
                Ok(_) => debug!("Prefill response consumed successfully"),
                // 中文逐行注释：下一行是原始源码第 1171 行，保持原始代码不变。
                Err(e) => warn!("Error consuming prefill response: {}", e),
            // 中文逐行注释：下一行是原始源码第 1172 行，保持原始代码不变。
            }
            // 中文逐行注释：下一行是原始源码第 1173 行，保持原始代码不变。
            None
        // 中文逐行注释：下一行是原始源码第 1174 行，保持原始代码不变。
        };

        // 中文逐行注释：下一行是原始源码第 1176 行，保持原始代码不变。
        Ok((prefill_status, prefill_body))
    // 中文逐行注释：下一行是原始源码第 1177 行，保持原始代码不变。
    }

    // 中文函数注释：构造带过滤请求头、鉴权头和 JSON body 的 POST 请求。
    // 中文逐行注释：下一行是原始源码第 1179 行，保持原始代码不变。
    fn build_post_with_headers(
        // 中文逐行注释：下一行是原始源码第 1180 行，保持原始代码不变。
        &self,
        // 中文逐行注释：下一行是原始源码第 1181 行，保持原始代码不变。
        client: &Client,
        // 中文逐行注释：下一行是原始源码第 1182 行，保持原始代码不变。
        url: &str,
        // 中文逐行注释：下一行是原始源码第 1183 行，保持原始代码不变。
        route: &'static str,
        // 中文逐行注释：下一行是原始源码第 1184 行，保持原始代码不变。
        json_request: &Value,
        // 中文逐行注释：下一行是原始源码第 1185 行，保持原始代码不变。
        headers: Option<&HeaderMap>,
        // 中文逐行注释：下一行是原始源码第 1186 行，保持原始代码不变。
        connection_close: bool,
    // 中文逐行注释：下一行是原始源码第 1187 行，保持原始代码不变。
    ) -> reqwest::RequestBuilder {
        // 中文逐行注释：下一行是原始源码第 1188 行，保持原始代码不变。
        let mut request = client.post(api_path(url, route)).json(json_request);
        // 中文逐行注释：下一行是原始源码第 1189 行，保持原始代码不变。
        if connection_close {
            // 中文逐行注释：下一行是原始源码第 1190 行，保持原始代码不变。
            request = request.header("Connection", "close");
        // 中文逐行注释：下一行是原始源码第 1191 行，保持原始代码不变。
        }
        // 中文逐行注释：下一行是原始源码第 1192 行，保持原始代码不变。
        if let Some(headers) = headers {
            // 中文逐行注释：下一行是原始源码第 1193 行，保持原始代码不变。
            for (name, value) in headers.iter() {
                // 中文逐行注释：下一行是原始源码第 1194 行，保持原始代码不变。
                if header_utils::should_forward_request_header(name.as_str()) {
                    // 中文逐行注释：下一行是原始源码第 1195 行，保持原始代码不变。
                    if let Ok(val) = value.to_str() {
                        // 中文逐行注释：下一行是原始源码第 1196 行，保持原始代码不变。
                        request = request.header(name, val);
                    // 中文逐行注释：下一行是原始源码第 1197 行，保持原始代码不变。
                    }
                // 中文逐行注释：下一行是原始源码第 1198 行，保持原始代码不变。
                }
            // 中文逐行注释：下一行是原始源码第 1199 行，保持原始代码不变。
            }
        // 中文逐行注释：下一行是原始源码第 1200 行，保持原始代码不变。
        }
        // 中文逐行注释：下一行是原始源码第 1201 行，保持原始代码不变。
        request
    // 中文逐行注释：下一行是原始源码第 1202 行，保持原始代码不变。
    }

    // 中文逐行注释：下一行是原始源码第 1204 行，保持原始代码不变。
    // Helper to merge logprobs from prefill and decode responses
    // 中文逐行注释：下一行是原始源码第 1205 行，保持原始代码不变。
    // Optimized to avoid double cloning by taking ownership of decode array
    // 中文函数注释：把 prefill 返回的输入 logprobs 合并到 decode JSON 响应中。
    // 中文逐行注释：下一行是原始源码第 1206 行，保持原始代码不变。
    fn merge_logprobs_in_json(prefill_json: &Value, decode_json: &mut Value) -> bool {
        // 中文逐行注释：下一行是原始源码第 1207 行，保持原始代码不变。
        if let (Some(prefill_meta), Some(decode_meta)) = (
            // 中文逐行注释：下一行是原始源码第 1208 行，保持原始代码不变。
            prefill_json.get("meta_info"),
            // 中文逐行注释：下一行是原始源码第 1209 行，保持原始代码不变。
            decode_json.get_mut("meta_info"),
        // 中文逐行注释：下一行是原始源码第 1210 行，保持原始代码不变。
        ) {
            // 中文逐行注释：下一行是原始源码第 1211 行，保持原始代码不变。
            if let (Some(prefill_logprobs), Some(decode_logprobs)) = (
                // 中文逐行注释：下一行是原始源码第 1212 行，保持原始代码不变。
                prefill_meta.get("input_token_logprobs"),
                // 中文逐行注释：下一行是原始源码第 1213 行，保持原始代码不变。
                decode_meta.get_mut("input_token_logprobs"),
            // 中文逐行注释：下一行是原始源码第 1214 行，保持原始代码不变。
            ) {
                // 中文逐行注释：下一行是原始源码第 1215 行，保持原始代码不变。
                if let Some(prefill_arr) = prefill_logprobs.as_array() {
                    // 中文逐行注释：下一行是原始源码第 1216 行，保持原始代码不变。
                    // Take ownership of decode array to avoid cloning it
                    // 中文逐行注释：下一行是原始源码第 1217 行，保持原始代码不变。
                    let decode_arr = std::mem::take(decode_logprobs);
                    // 中文逐行注释：下一行是原始源码第 1218 行，保持原始代码不变。
                    if let Value::Array(decode_vec) = decode_arr {
                        // 中文逐行注释：下一行是原始源码第 1219 行，保持原始代码不变。
                        // Pre-allocate merged array with exact capacity
                        // 中文逐行注释：下一行是原始源码第 1220 行，保持原始代码不变。
                        let mut merged = Vec::with_capacity(prefill_arr.len() + decode_vec.len());
                        // 中文逐行注释：下一行是原始源码第 1221 行，保持原始代码不变。
                        merged.extend(prefill_arr.iter().cloned());
                        // 中文逐行注释：下一行是原始源码第 1222 行，保持原始代码不变。
                        merged.extend(decode_vec);
                        // 中文逐行注释：下一行是原始源码第 1223 行，保持原始代码不变。
                        decode_meta["input_token_logprobs"] = Value::Array(merged);
                        // 中文逐行注释：下一行是原始源码第 1224 行，保持原始代码不变。
                        return true;
                    // 中文逐行注释：下一行是原始源码第 1225 行，保持原始代码不变。
                    }
                // 中文逐行注释：下一行是原始源码第 1226 行，保持原始代码不变。
                }
            // 中文逐行注释：下一行是原始源码第 1227 行，保持原始代码不变。
            }
        // 中文逐行注释：下一行是原始源码第 1228 行，保持原始代码不变。
        }
        // 中文逐行注释：下一行是原始源码第 1229 行，保持原始代码不变。
        false
    // 中文逐行注释：下一行是原始源码第 1230 行，保持原始代码不变。
    }

    // 中文逐行注释：下一行是原始源码第 1232 行，保持原始代码不变。
    // Simple helper to merge logprobs in streaming responses
    // 中文逐行注释：下一行是原始源码第 1233 行，保持原始代码不变。
    // Optimized to reduce allocations in the merge path
    // 中文函数注释：把 prefill logprobs 合并到流式响应首个 chunk 中。
    // 中文逐行注释：下一行是原始源码第 1234 行，保持原始代码不变。
    fn merge_streaming_logprobs(
        // 中文逐行注释：下一行是原始源码第 1235 行，保持原始代码不变。
        prefill_logprobs: Option<Value>,
        // 中文逐行注释：下一行是原始源码第 1236 行，保持原始代码不变。
        decode_chunk: &[u8],
    // 中文逐行注释：下一行是原始源码第 1237 行，保持原始代码不变。
    ) -> Result<bytes::Bytes, ()> {
        // 中文逐行注释：下一行是原始源码第 1238 行，保持原始代码不变。
        // Skip non-data chunks
        // 中文逐行注释：下一行是原始源码第 1239 行，保持原始代码不变。
        let chunk_str = std::str::from_utf8(decode_chunk).map_err(|_| ())?;
        // 中文逐行注释：下一行是原始源码第 1240 行，保持原始代码不变。
        if !chunk_str.starts_with("data: ") || chunk_str.contains("[DONE]") {
            // 中文逐行注释：下一行是原始源码第 1241 行，保持原始代码不变。
            return Err(());
        // 中文逐行注释：下一行是原始源码第 1242 行，保持原始代码不变。
        }

        // 中文逐行注释：下一行是原始源码第 1244 行，保持原始代码不变。
        // Parse JSON from chunk
        // 中文逐行注释：下一行是原始源码第 1245 行，保持原始代码不变。
        let json_str = chunk_str.trim_start_matches("data: ").trim();
        // 中文逐行注释：下一行是原始源码第 1246 行，保持原始代码不变。
        let mut decode_json: Value = serde_json::from_str(json_str).map_err(|_| ())?;

        // 中文逐行注释：下一行是原始源码第 1248 行，保持原始代码不变。
        // Merge prefill logprobs if available
        // 中文逐行注释：下一行是原始源码第 1249 行，保持原始代码不变。
        if let Some(ref p_logprobs) = prefill_logprobs {
            // 中文逐行注释：下一行是原始源码第 1250 行，保持原始代码不变。
            if let Some(meta) = decode_json.get_mut("meta_info") {
                // 中文逐行注释：下一行是原始源码第 1251 行，保持原始代码不变。
                if let Some(d_logprobs) = meta.get_mut("input_token_logprobs") {
                    // 中文逐行注释：下一行是原始源码第 1252 行，保持原始代码不变。
                    if let Some(p_arr) = p_logprobs.as_array() {
                        // 中文逐行注释：下一行是原始源码第 1253 行，保持原始代码不变。
                        // Take ownership of decode array to avoid cloning it
                        // 中文逐行注释：下一行是原始源码第 1254 行，保持原始代码不变。
                        let decode_arr = std::mem::take(d_logprobs);
                        // 中文逐行注释：下一行是原始源码第 1255 行，保持原始代码不变。
                        if let Value::Array(d_vec) = decode_arr {
                            // 中文逐行注释：下一行是原始源码第 1256 行，保持原始代码不变。
                            // Pre-allocate merged array with exact capacity
                            // 中文逐行注释：下一行是原始源码第 1257 行，保持原始代码不变。
                            let mut merged = Vec::with_capacity(p_arr.len() + d_vec.len());
                            // 中文逐行注释：下一行是原始源码第 1258 行，保持原始代码不变。
                            merged.extend(p_arr.iter().cloned());
                            // 中文逐行注释：下一行是原始源码第 1259 行，保持原始代码不变。
                            merged.extend(d_vec);
                            // 中文逐行注释：下一行是原始源码第 1260 行，保持原始代码不变。
                            *d_logprobs = Value::Array(merged);
                        // 中文逐行注释：下一行是原始源码第 1261 行，保持原始代码不变。
                        }
                    // 中文逐行注释：下一行是原始源码第 1262 行，保持原始代码不变。
                    }
                // 中文逐行注释：下一行是原始源码第 1263 行，保持原始代码不变。
                }
            // 中文逐行注释：下一行是原始源码第 1264 行，保持原始代码不变。
            }
        // 中文逐行注释：下一行是原始源码第 1265 行，保持原始代码不变。
        }

        // 中文逐行注释：下一行是原始源码第 1267 行，保持原始代码不变。
        // Re-serialize
        // 中文逐行注释：下一行是原始源码第 1268 行，保持原始代码不变。
        let merged_str = format!(
            // 中文逐行注释：下一行是原始源码第 1269 行，保持原始代码不变。
            "data: {}\n\n",
            // 中文逐行注释：下一行是原始源码第 1270 行，保持原始代码不变。
            serde_json::to_string(&decode_json).unwrap_or_default()
        // 中文逐行注释：下一行是原始源码第 1271 行，保持原始代码不变。
        );
        // 中文逐行注释：下一行是原始源码第 1272 行，保持原始代码不变。
        Ok(bytes::Bytes::from(merged_str))
    // 中文逐行注释：下一行是原始源码第 1273 行，保持原始代码不变。
    }
// 中文逐行注释：下一行是原始源码第 1274 行，保持原始代码不变。
}

// 中文逐行注释：下一行是原始源码第 1276 行，保持原始代码不变。
#[async_trait]
// 中文逐行注释：下一行是原始源码第 1277 行，保持原始代码不变。
impl RouterTrait for PDRouter {
    // 中文函数注释：返回 Any 引用，支持 trait object 下转型。
    // 中文逐行注释：下一行是原始源码第 1278 行，保持原始代码不变。
    fn as_any(&self) -> &dyn std::any::Any {
        // 中文逐行注释：下一行是原始源码第 1279 行，保持原始代码不变。
        self
    // 中文逐行注释：下一行是原始源码第 1280 行，保持原始代码不变。
    }

    // 中文函数注释：实现 health_generate 路由，检查 prefill/decode worker 健康状态。
    // 中文逐行注释：下一行是原始源码第 1282 行，保持原始代码不变。
    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // 中文逐行注释：下一行是原始源码第 1283 行，保持原始代码不变。
        // Note: This endpoint actually causes the model to generate tokens, so we only test one pair

        // 中文逐行注释：下一行是原始源码第 1285 行，保持原始代码不变。
        // Select a random worker pair using the policy
        // 中文逐行注释：下一行是原始源码第 1286 行，保持原始代码不变。
        let (prefill, decode) = match self.select_pd_pair(None, None, None).await {
            // 中文逐行注释：下一行是原始源码第 1287 行，保持原始代码不变。
            Ok(pair) => pair,
            // 中文逐行注释：下一行是原始源码第 1288 行，保持原始代码不变。
            Err(e) => {
                // 中文逐行注释：下一行是原始源码第 1289 行，保持原始代码不变。
                return error::service_unavailable(
                    // 中文逐行注释：下一行是原始源码第 1290 行，保持原始代码不变。
                    "no_healthy_worker_pair",
                    // 中文逐行注释：下一行是原始源码第 1291 行，保持原始代码不变。
                    format!("No healthy worker pair available: {}", e),
                // 中文逐行注释：下一行是原始源码第 1292 行，保持原始代码不变。
                );
            // 中文逐行注释：下一行是原始源码第 1293 行，保持原始代码不变。
            }
        // 中文逐行注释：下一行是原始源码第 1294 行，保持原始代码不变。
        };

        // 中文逐行注释：下一行是原始源码第 1296 行，保持原始代码不变。
        let prefill_url = format!("{}/health_generate", prefill.url());
        // 中文逐行注释：下一行是原始源码第 1297 行，保持原始代码不变。
        let (prefill_result, decode_result) = tokio::join!(
            // 中文逐行注释：下一行是原始源码第 1298 行，保持原始代码不变。
            self.client.get(&prefill_url).send(),
            // 中文逐行注释：下一行是原始源码第 1299 行，保持原始代码不变。
            self.client
                // 中文逐行注释：下一行是原始源码第 1300 行，保持原始代码不变。
                .get(format!("{}/health_generate", decode.url()))
                // 中文逐行注释：下一行是原始源码第 1301 行，保持原始代码不变。
                .send()
        // 中文逐行注释：下一行是原始源码第 1302 行，保持原始代码不变。
        );

        // 中文逐行注释：下一行是原始源码第 1304 行，保持原始代码不变。
        // Check results
        // 中文逐行注释：下一行是原始源码第 1305 行，保持原始代码不变。
        let mut errors = Vec::new();

        // 中文逐行注释：下一行是原始源码第 1307 行，保持原始代码不变。
        match prefill_result {
            // 中文逐行注释：下一行是原始源码第 1308 行，保持原始代码不变。
            Ok(res) if res.status().is_success() => {
                // 中文逐行注释：下一行是原始源码第 1309 行，保持原始代码不变。
                debug!(
                    // 中文逐行注释：下一行是原始源码第 1310 行，保持原始代码不变。
                    "Health generate passed for prefill server: {}",
                    // 中文逐行注释：下一行是原始源码第 1311 行，保持原始代码不变。
                    prefill.url()
                // 中文逐行注释：下一行是原始源码第 1312 行，保持原始代码不变。
                );
            // 中文逐行注释：下一行是原始源码第 1313 行，保持原始代码不变。
            }
            // 中文逐行注释：下一行是原始源码第 1314 行，保持原始代码不变。
            Ok(res) => {
                // 中文逐行注释：下一行是原始源码第 1315 行，保持原始代码不变。
                errors.push(format!(
                    // 中文逐行注释：下一行是原始源码第 1316 行，保持原始代码不变。
                    "Prefill {} returned status {}",
                    // 中文逐行注释：下一行是原始源码第 1317 行，保持原始代码不变。
                    prefill.url(),
                    // 中文逐行注释：下一行是原始源码第 1318 行，保持原始代码不变。
                    res.status()
                // 中文逐行注释：下一行是原始源码第 1319 行，保持原始代码不变。
                ));
            // 中文逐行注释：下一行是原始源码第 1320 行，保持原始代码不变。
            }
            // 中文逐行注释：下一行是原始源码第 1321 行，保持原始代码不变。
            Err(e) => {
                // 中文逐行注释：下一行是原始源码第 1322 行，保持原始代码不变。
                errors.push(format!("Prefill {} error: {}", prefill.url(), e));
            // 中文逐行注释：下一行是原始源码第 1323 行，保持原始代码不变。
            }
        // 中文逐行注释：下一行是原始源码第 1324 行，保持原始代码不变。
        }

        // 中文逐行注释：下一行是原始源码第 1326 行，保持原始代码不变。
        match decode_result {
            // 中文逐行注释：下一行是原始源码第 1327 行，保持原始代码不变。
            Ok(res) if res.status().is_success() => {
                // 中文逐行注释：下一行是原始源码第 1328 行，保持原始代码不变。
                debug!("Health generate passed for decode server: {}", decode.url());
            // 中文逐行注释：下一行是原始源码第 1329 行，保持原始代码不变。
            }
            // 中文逐行注释：下一行是原始源码第 1330 行，保持原始代码不变。
            Ok(res) => {
                // 中文逐行注释：下一行是原始源码第 1331 行，保持原始代码不变。
                errors.push(format!(
                    // 中文逐行注释：下一行是原始源码第 1332 行，保持原始代码不变。
                    "Decode {} returned status {}",
                    // 中文逐行注释：下一行是原始源码第 1333 行，保持原始代码不变。
                    decode.url(),
                    // 中文逐行注释：下一行是原始源码第 1334 行，保持原始代码不变。
                    res.status()
                // 中文逐行注释：下一行是原始源码第 1335 行，保持原始代码不变。
                ));
            // 中文逐行注释：下一行是原始源码第 1336 行，保持原始代码不变。
            }
            // 中文逐行注释：下一行是原始源码第 1337 行，保持原始代码不变。
            Err(e) => {
                // 中文逐行注释：下一行是原始源码第 1338 行，保持原始代码不变。
                errors.push(format!("Decode {} error: {}", decode.url(), e));
            // 中文逐行注释：下一行是原始源码第 1339 行，保持原始代码不变。
            }
        // 中文逐行注释：下一行是原始源码第 1340 行，保持原始代码不变。
        }

        // 中文逐行注释：下一行是原始源码第 1342 行，保持原始代码不变。
        if errors.is_empty() {
            // 中文逐行注释：下一行是原始源码第 1343 行，保持原始代码不变。
            (
                // 中文逐行注释：下一行是原始源码第 1344 行，保持原始代码不变。
                StatusCode::OK,
                // 中文逐行注释：下一行是原始源码第 1345 行，保持原始代码不变。
                format!(
                    // 中文逐行注释：下一行是原始源码第 1346 行，保持原始代码不变。
                    "Health generate passed on selected pair: prefill={}, decode={}",
                    // 中文逐行注释：下一行是原始源码第 1347 行，保持原始代码不变。
                    prefill.url(),
                    // 中文逐行注释：下一行是原始源码第 1348 行，保持原始代码不变。
                    decode.url()
                // 中文逐行注释：下一行是原始源码第 1349 行，保持原始代码不变。
                ),
            // 中文逐行注释：下一行是原始源码第 1350 行，保持原始代码不变。
            )
                // 中文逐行注释：下一行是原始源码第 1351 行，保持原始代码不变。
                .into_response()
        // 中文逐行注释：下一行是原始源码第 1352 行，保持原始代码不变。
        } else {
            // 中文逐行注释：下一行是原始源码第 1353 行，保持原始代码不变。
            error::service_unavailable(
                // 中文逐行注释：下一行是原始源码第 1354 行，保持原始代码不变。
                "health_generate_failed",
                // 中文逐行注释：下一行是原始源码第 1355 行，保持原始代码不变。
                format!("Health generate failed: {:?}", errors),
            // 中文逐行注释：下一行是原始源码第 1356 行，保持原始代码不变。
            )
        // 中文逐行注释：下一行是原始源码第 1357 行，保持原始代码不变。
        }
    // 中文逐行注释：下一行是原始源码第 1358 行，保持原始代码不变。
    }

    // 中文函数注释：代理或返回 server info 接口响应。
    // 中文逐行注释：下一行是原始源码第 1360 行，保持原始代码不变。
    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        // 中文逐行注释：下一行是原始源码第 1361 行，保持原始代码不变。
        // Get info from the first decode server to match sglang's server info format
        // 中文逐行注释：下一行是原始源码第 1362 行，保持原始代码不变。
        // Note: We use decode workers for server info to match expected format
        // 中文逐行注释：下一行是原始源码第 1363 行，保持原始代码不变。
        self.proxy_to_first_prefill_worker("server_info", None)
            // 中文逐行注释：下一行是原始源码第 1364 行，保持原始代码不变。
            .await
    // 中文逐行注释：下一行是原始源码第 1365 行，保持原始代码不变。
    }

    // 中文函数注释：代理 models 列表请求。
    // 中文逐行注释：下一行是原始源码第 1367 行，保持原始代码不变。
    async fn get_models(&self, req: Request<Body>) -> Response {
        // 中文逐行注释：下一行是原始源码第 1368 行，保持原始代码不变。
        // Extract headers first to avoid Send issues
        // 中文逐行注释：下一行是原始源码第 1369 行，保持原始代码不变。
        let headers = header_utils::copy_request_headers(&req);

        // 中文逐行注释：下一行是原始源码第 1371 行，保持原始代码不变。
        // Proxy to first prefill worker
        // 中文逐行注释：下一行是原始源码第 1372 行，保持原始代码不变。
        self.proxy_to_first_prefill_worker("v1/models", Some(headers))
            // 中文逐行注释：下一行是原始源码第 1373 行，保持原始代码不变。
            .await
    // 中文逐行注释：下一行是原始源码第 1374 行，保持原始代码不变。
    }

    // 中文函数注释：代理 model info 请求。
    // 中文逐行注释：下一行是原始源码第 1376 行，保持原始代码不变。
    async fn get_model_info(&self, req: Request<Body>) -> Response {
        // 中文逐行注释：下一行是原始源码第 1377 行，保持原始代码不变。
        // Extract headers first to avoid Send issues
        // 中文逐行注释：下一行是原始源码第 1378 行，保持原始代码不变。
        let headers = header_utils::copy_request_headers(&req);

        // 中文逐行注释：下一行是原始源码第 1380 行，保持原始代码不变。
        // Proxy to first prefill worker
        // 中文逐行注释：下一行是原始源码第 1381 行，保持原始代码不变。
        self.proxy_to_first_prefill_worker("model_info", Some(headers))
            // 中文逐行注释：下一行是原始源码第 1382 行，保持原始代码不变。
            .await
    // 中文逐行注释：下一行是原始源码第 1383 行，保持原始代码不变。
    }

    // 中文函数注释：处理 /generate 请求并走 PD 双发链路。
    // 中文逐行注释：下一行是原始源码第 1385 行，保持原始代码不变。
    async fn route_generate(
        // 中文逐行注释：下一行是原始源码第 1386 行，保持原始代码不变。
        &self,
        // 中文逐行注释：下一行是原始源码第 1387 行，保持原始代码不变。
        headers: Option<&HeaderMap>,
        // 中文逐行注释：下一行是原始源码第 1388 行，保持原始代码不变。
        body: &GenerateRequest,
        // 中文逐行注释：下一行是原始源码第 1389 行，保持原始代码不变。
        model_id: Option<&str>,
    // 中文逐行注释：下一行是原始源码第 1390 行，保持原始代码不变。
    ) -> Response {
        // 中文逐行注释：下一行是原始源码第 1391 行，保持原始代码不变。
        let is_stream = body.stream;
        // 中文逐行注释：下一行是原始源码第 1392 行，保持原始代码不变。
        let return_logprob = body.return_logprob.unwrap_or(false);

        // 中文逐行注释：下一行是原始源码第 1394 行，保持原始代码不变。
        let request_text = if self.policies_need_request_text() {
            // 中文逐行注释：下一行是原始源码第 1395 行，保持原始代码不变。
            body.text.as_deref().map(|s| s.to_string())
        // 中文逐行注释：下一行是原始源码第 1396 行，保持原始代码不变。
        } else {
            // 中文逐行注释：下一行是原始源码第 1397 行，保持原始代码不变。
            None
        // 中文逐行注释：下一行是原始源码第 1398 行，保持原始代码不变。
        };

        // 中文逐行注释：下一行是原始源码第 1400 行，保持原始代码不变。
        let batch_size = Self::get_generate_batch_size(body);

        // 中文逐行注释：下一行是原始源码第 1402 行，保持原始代码不变。
        let context = PDRequestContext {
            // 中文逐行注释：下一行是原始源码第 1403 行，保持原始代码不变。
            route: "/generate",
            // 中文逐行注释：下一行是原始源码第 1404 行，保持原始代码不变。
            batch_size,
            // 中文逐行注释：下一行是原始源码第 1405 行，保持原始代码不变。
            is_stream,
            // 中文逐行注释：下一行是原始源码第 1406 行，保持原始代码不变。
            return_logprob,
            // 中文逐行注释：下一行是原始源码第 1407 行，保持原始代码不变。
            request_text,
            // 中文逐行注释：下一行是原始源码第 1408 行，保持原始代码不变。
            model_id,
            // 中文逐行注释：下一行是原始源码第 1409 行，保持原始代码不变。
            headers: headers.cloned(),
        // 中文逐行注释：下一行是原始源码第 1410 行，保持原始代码不变。
        };

        // 中文逐行注释：下一行是原始源码第 1412 行，保持原始代码不变。
        self.execute_dual_dispatch(headers, body, context).await
    // 中文逐行注释：下一行是原始源码第 1413 行，保持原始代码不变。
    }

    // 中文函数注释：处理 chat completions 请求并走 PD 双发链路。
    // 中文逐行注释：下一行是原始源码第 1415 行，保持原始代码不变。
    async fn route_chat(
        // 中文逐行注释：下一行是原始源码第 1416 行，保持原始代码不变。
        &self,
        // 中文逐行注释：下一行是原始源码第 1417 行，保持原始代码不变。
        headers: Option<&HeaderMap>,
        // 中文逐行注释：下一行是原始源码第 1418 行，保持原始代码不变。
        body: &ChatCompletionRequest,
        // 中文逐行注释：下一行是原始源码第 1419 行，保持原始代码不变。
        model_id: Option<&str>,
    // 中文逐行注释：下一行是原始源码第 1420 行，保持原始代码不变。
    ) -> Response {
        // 中文逐行注释：下一行是原始源码第 1421 行，保持原始代码不变。
        let is_stream = body.stream;
        // 中文逐行注释：下一行是原始源码第 1422 行，保持原始代码不变。
        let return_logprob = body.logprobs;

        // 中文逐行注释：下一行是原始源码第 1424 行，保持原始代码不变。
        let request_text = if self.policies_need_request_text() {
            // 中文逐行注释：下一行是原始源码第 1425 行，保持原始代码不变。
            body.messages.first().and_then(|msg| match msg {
                // 中文逐行注释：下一行是原始源码第 1426 行，保持原始代码不变。
                ChatMessage::User { content, .. } => match content {
                    // 中文逐行注释：下一行是原始源码第 1427 行，保持原始代码不变。
                    MessageContent::Text(text) => Some(text.clone()),
                    // 中文逐行注释：下一行是原始源码第 1428 行，保持原始代码不变。
                    MessageContent::Parts(_) => None,
                // 中文逐行注释：下一行是原始源码第 1429 行，保持原始代码不变。
                },
                // 中文逐行注释：下一行是原始源码第 1430 行，保持原始代码不变。
                ChatMessage::Developer { content, .. } => match content {
                    // 中文逐行注释：下一行是原始源码第 1431 行，保持原始代码不变。
                    MessageContent::Text(text) => Some(text.clone()),
                    // 中文逐行注释：下一行是原始源码第 1432 行，保持原始代码不变。
                    MessageContent::Parts(_) => None,
                // 中文逐行注释：下一行是原始源码第 1433 行，保持原始代码不变。
                },
                // 中文逐行注释：下一行是原始源码第 1434 行，保持原始代码不变。
                ChatMessage::System { content, .. } => Some(content.to_simple_string()),
                // 中文逐行注释：下一行是原始源码第 1435 行，保持原始代码不变。
                _ => None,
            // 中文逐行注释：下一行是原始源码第 1436 行，保持原始代码不变。
            })
        // 中文逐行注释：下一行是原始源码第 1437 行，保持原始代码不变。
        } else {
            // 中文逐行注释：下一行是原始源码第 1438 行，保持原始代码不变。
            None
        // 中文逐行注释：下一行是原始源码第 1439 行，保持原始代码不变。
        };

        // 中文逐行注释：下一行是原始源码第 1441 行，保持原始代码不变。
        // Calculate batch size
        // 中文逐行注释：下一行是原始源码第 1442 行，保持原始代码不变。
        let batch_size = Self::get_chat_batch_size(body);

        // 中文逐行注释：下一行是原始源码第 1444 行，保持原始代码不变。
        let context = PDRequestContext {
            // 中文逐行注释：下一行是原始源码第 1445 行，保持原始代码不变。
            route: "/v1/chat/completions",
            // 中文逐行注释：下一行是原始源码第 1446 行，保持原始代码不变。
            batch_size,
            // 中文逐行注释：下一行是原始源码第 1447 行，保持原始代码不变。
            is_stream,
            // 中文逐行注释：下一行是原始源码第 1448 行，保持原始代码不变。
            return_logprob,
            // 中文逐行注释：下一行是原始源码第 1449 行，保持原始代码不变。
            request_text,
            // 中文逐行注释：下一行是原始源码第 1450 行，保持原始代码不变。
            model_id,
            // 中文逐行注释：下一行是原始源码第 1451 行，保持原始代码不变。
            headers: headers.cloned(),
        // 中文逐行注释：下一行是原始源码第 1452 行，保持原始代码不变。
        };

        // 中文逐行注释：下一行是原始源码第 1454 行，保持原始代码不变。
        self.execute_dual_dispatch(headers, body, context).await
    // 中文逐行注释：下一行是原始源码第 1455 行，保持原始代码不变。
    }

    // 中文函数注释：处理 completions 请求并走 PD 双发链路。
    // 中文逐行注释：下一行是原始源码第 1457 行，保持原始代码不变。
    async fn route_completion(
        // 中文逐行注释：下一行是原始源码第 1458 行，保持原始代码不变。
        &self,
        // 中文逐行注释：下一行是原始源码第 1459 行，保持原始代码不变。
        headers: Option<&HeaderMap>,
        // 中文逐行注释：下一行是原始源码第 1460 行，保持原始代码不变。
        body: &CompletionRequest,
        // 中文逐行注释：下一行是原始源码第 1461 行，保持原始代码不变。
        model_id: Option<&str>,
    // 中文逐行注释：下一行是原始源码第 1462 行，保持原始代码不变。
    ) -> Response {
        // 中文逐行注释：下一行是原始源码第 1463 行，保持原始代码不变。
        let is_stream = body.stream;
        // 中文逐行注释：下一行是原始源码第 1464 行，保持原始代码不变。
        let return_logprob = body.logprobs.is_some();

        // 中文逐行注释：下一行是原始源码第 1466 行，保持原始代码不变。
        let request_text = if self.policies_need_request_text() {
            // 中文逐行注释：下一行是原始源码第 1467 行，保持原始代码不变。
            match &body.prompt {
                // 中文逐行注释：下一行是原始源码第 1468 行，保持原始代码不变。
                StringOrArray::String(s) => Some(s.clone()),
                // 中文逐行注释：下一行是原始源码第 1469 行，保持原始代码不变。
                StringOrArray::Array(v) => v.first().map(|s| s.to_string()),
            // 中文逐行注释：下一行是原始源码第 1470 行，保持原始代码不变。
            }
        // 中文逐行注释：下一行是原始源码第 1471 行，保持原始代码不变。
        } else {
            // 中文逐行注释：下一行是原始源码第 1472 行，保持原始代码不变。
            None
        // 中文逐行注释：下一行是原始源码第 1473 行，保持原始代码不变。
        };

        // 中文逐行注释：下一行是原始源码第 1475 行，保持原始代码不变。
        // Calculate batch size
        // 中文逐行注释：下一行是原始源码第 1476 行，保持原始代码不变。
        let batch_size = Self::get_completion_batch_size(body);

        // 中文逐行注释：下一行是原始源码第 1478 行，保持原始代码不变。
        let context = PDRequestContext {
            // 中文逐行注释：下一行是原始源码第 1479 行，保持原始代码不变。
            route: "/v1/completions",
            // 中文逐行注释：下一行是原始源码第 1480 行，保持原始代码不变。
            batch_size,
            // 中文逐行注释：下一行是原始源码第 1481 行，保持原始代码不变。
            is_stream,
            // 中文逐行注释：下一行是原始源码第 1482 行，保持原始代码不变。
            return_logprob,
            // 中文逐行注释：下一行是原始源码第 1483 行，保持原始代码不变。
            request_text,
            // 中文逐行注释：下一行是原始源码第 1484 行，保持原始代码不变。
            model_id,
            // 中文逐行注释：下一行是原始源码第 1485 行，保持原始代码不变。
            headers: headers.cloned(),
        // 中文逐行注释：下一行是原始源码第 1486 行，保持原始代码不变。
        };

        // 中文逐行注释：下一行是原始源码第 1488 行，保持原始代码不变。
        self.execute_dual_dispatch(headers, body, context).await
    // 中文逐行注释：下一行是原始源码第 1489 行，保持原始代码不变。
    }

    // 中文函数注释：处理 rerank 请求并走 PD 双发链路。
    // 中文逐行注释：下一行是原始源码第 1491 行，保持原始代码不变。
    async fn route_rerank(
        // 中文逐行注释：下一行是原始源码第 1492 行，保持原始代码不变。
        &self,
        // 中文逐行注释：下一行是原始源码第 1493 行，保持原始代码不变。
        headers: Option<&HeaderMap>,
        // 中文逐行注释：下一行是原始源码第 1494 行，保持原始代码不变。
        body: &RerankRequest,
        // 中文逐行注释：下一行是原始源码第 1495 行，保持原始代码不变。
        model_id: Option<&str>,
    // 中文逐行注释：下一行是原始源码第 1496 行，保持原始代码不变。
    ) -> Response {
        // 中文逐行注释：下一行是原始源码第 1497 行，保持原始代码不变。
        // Extract text for cache-aware routing
        // 中文逐行注释：下一行是原始源码第 1498 行，保持原始代码不变。
        let req_text = if self.policies_need_request_text() {
            // 中文逐行注释：下一行是原始源码第 1499 行，保持原始代码不变。
            Some(body.query.clone())
        // 中文逐行注释：下一行是原始源码第 1500 行，保持原始代码不变。
        } else {
            // 中文逐行注释：下一行是原始源码第 1501 行，保持原始代码不变。
            None
        // 中文逐行注释：下一行是原始源码第 1502 行，保持原始代码不变。
        };

        // 中文逐行注释：下一行是原始源码第 1504 行，保持原始代码不变。
        let context = PDRequestContext {
            // 中文逐行注释：下一行是原始源码第 1505 行，保持原始代码不变。
            route: "/v1/rerank",
            // 中文逐行注释：下一行是原始源码第 1506 行，保持原始代码不变。
            batch_size: None,
            // 中文逐行注释：下一行是原始源码第 1507 行，保持原始代码不变。
            is_stream: false,
            // 中文逐行注释：下一行是原始源码第 1508 行，保持原始代码不变。
            return_logprob: false,
            // 中文逐行注释：下一行是原始源码第 1509 行，保持原始代码不变。
            request_text: req_text,
            // 中文逐行注释：下一行是原始源码第 1510 行，保持原始代码不变。
            model_id,
            // 中文逐行注释：下一行是原始源码第 1511 行，保持原始代码不变。
            headers: headers.cloned(),
        // 中文逐行注释：下一行是原始源码第 1512 行，保持原始代码不变。
        };

        // 中文逐行注释：下一行是原始源码第 1514 行，保持原始代码不变。
        self.execute_dual_dispatch(headers, body, context).await
    // 中文逐行注释：下一行是原始源码第 1515 行，保持原始代码不变。
    }

    // 中文函数注释：处理 embeddings 请求并走 PD 双发链路。
    // 中文逐行注释：下一行是原始源码第 1517 行，保持原始代码不变。
    async fn route_embeddings(
        // 中文逐行注释：下一行是原始源码第 1518 行，保持原始代码不变。
        &self,
        // 中文逐行注释：下一行是原始源码第 1519 行，保持原始代码不变。
        headers: Option<&HeaderMap>,
        // 中文逐行注释：下一行是原始源码第 1520 行，保持原始代码不变。
        body: &EmbeddingRequest,
        // 中文逐行注释：下一行是原始源码第 1521 行，保持原始代码不变。
        model_id: Option<&str>,
    // 中文逐行注释：下一行是原始源码第 1522 行，保持原始代码不变。
    ) -> Response {
        // 中文逐行注释：下一行是原始源码第 1523 行，保持原始代码不变。
        let _ = (headers, body, model_id);
        // 中文逐行注释：下一行是原始源码第 1524 行，保持原始代码不变。
        warn!("PD mode does not support /v1/embeddings; returning bad request");
        // 中文逐行注释：下一行是原始源码第 1525 行，保持原始代码不变。
        error::bad_request(
            // 中文逐行注释：下一行是原始源码第 1526 行，保持原始代码不变。
            "pd_unsupported_embeddings",
            // 中文逐行注释：下一行是原始源码第 1527 行，保持原始代码不变。
            "PD mode does not support /v1/embeddings",
        // 中文逐行注释：下一行是原始源码第 1528 行，保持原始代码不变。
        )
    // 中文逐行注释：下一行是原始源码第 1529 行，保持原始代码不变。
    }

    // 中文函数注释：处理 classify 请求并走 PD 双发链路。
    // 中文逐行注释：下一行是原始源码第 1531 行，保持原始代码不变。
    async fn route_classify(
        // 中文逐行注释：下一行是原始源码第 1532 行，保持原始代码不变。
        &self,
        // 中文逐行注释：下一行是原始源码第 1533 行，保持原始代码不变。
        headers: Option<&HeaderMap>,
        // 中文逐行注释：下一行是原始源码第 1534 行，保持原始代码不变。
        body: &ClassifyRequest,
        // 中文逐行注释：下一行是原始源码第 1535 行，保持原始代码不变。
        model_id: Option<&str>,
    // 中文逐行注释：下一行是原始源码第 1536 行，保持原始代码不变。
    ) -> Response {
        // 中文逐行注释：下一行是原始源码第 1537 行，保持原始代码不变。
        let _ = (headers, body, model_id);
        // 中文逐行注释：下一行是原始源码第 1538 行，保持原始代码不变。
        warn!("PD mode does not support /v1/classify; returning bad request");
        // 中文逐行注释：下一行是原始源码第 1539 行，保持原始代码不变。
        error::bad_request(
            // 中文逐行注释：下一行是原始源码第 1540 行，保持原始代码不变。
            "pd_unsupported_classify",
            // 中文逐行注释：下一行是原始源码第 1541 行，保持原始代码不变。
            "PD mode does not support /v1/classify",
        // 中文逐行注释：下一行是原始源码第 1542 行，保持原始代码不变。
        )
    // 中文逐行注释：下一行是原始源码第 1543 行，保持原始代码不变。
    }

    // 中文函数注释：返回 router 类型标识。
    // 中文逐行注释：下一行是原始源码第 1545 行，保持原始代码不变。
    fn router_type(&self) -> &'static str {
        // 中文逐行注释：下一行是原始源码第 1546 行，保持原始代码不变。
        "pd"
    // 中文逐行注释：下一行是原始源码第 1547 行，保持原始代码不变。
    }
// 中文逐行注释：下一行是原始源码第 1548 行，保持原始代码不变。
}

// 中文逐行注释：下一行是原始源码第 1550 行，保持原始代码不变。
#[cfg(test)]
// 中文逐行注释：下一行是原始源码第 1551 行，保持原始代码不变。
mod tests {
    // 中文逐行注释：下一行是原始源码第 1552 行，保持原始代码不变。
    use super::*;
    // 中文逐行注释：下一行是原始源码第 1553 行，保持原始代码不变。
    use crate::core::{BasicWorkerBuilder, WorkerType};

    // 中文函数注释：构造测试用 PDRouter。
    // 中文逐行注释：下一行是原始源码第 1555 行，保持原始代码不变。
    fn create_test_pd_router() -> PDRouter {
        // 中文逐行注释：下一行是原始源码第 1556 行，保持原始代码不变。
        let worker_registry = Arc::new(WorkerRegistry::new());
        // 中文逐行注释：下一行是原始源码第 1557 行，保持原始代码不变。
        let policy_registry =
            // 中文逐行注释：下一行是原始源码第 1558 行，保持原始代码不变。
            Arc::new(PolicyRegistry::new(crate::config::PolicyConfig::RoundRobin));

        // 中文逐行注释：下一行是原始源码第 1560 行，保持原始代码不变。
        PDRouter {
            // 中文逐行注释：下一行是原始源码第 1561 行，保持原始代码不变。
            worker_registry,
            // 中文逐行注释：下一行是原始源码第 1562 行，保持原始代码不变。
            policy_registry,
            // 中文逐行注释：下一行是原始源码第 1563 行，保持原始代码不变。
            client: Client::new(),
            // 中文逐行注释：下一行是原始源码第 1564 行，保持原始代码不变。
            retry_config: RetryConfig::default(),
            // 中文逐行注释：下一行是原始源码第 1565 行，保持原始代码不变。
            api_key: Some("test_api_key".to_string()),
            // 中文逐行注释：下一行是原始源码第 1566 行，保持原始代码不变。
            enable_igw: false,
        // 中文逐行注释：下一行是原始源码第 1567 行，保持原始代码不变。
        }
    // 中文逐行注释：下一行是原始源码第 1568 行，保持原始代码不变。
    }

    // 中文函数注释：构造测试用 worker。
    // 中文逐行注释：下一行是原始源码第 1570 行，保持原始代码不变。
    fn create_test_worker(url: String, worker_type: WorkerType, healthy: bool) -> Box<dyn Worker> {
        // 中文逐行注释：下一行是原始源码第 1571 行，保持原始代码不变。
        let worker = BasicWorkerBuilder::new(url)
            // 中文逐行注释：下一行是原始源码第 1572 行，保持原始代码不变。
            .worker_type(worker_type)
            // 中文逐行注释：下一行是原始源码第 1573 行，保持原始代码不变。
            .build();
        // 中文逐行注释：下一行是原始源码第 1574 行，保持原始代码不变。
        worker.set_healthy(healthy);
        // 中文逐行注释：下一行是原始源码第 1575 行，保持原始代码不变。
        Box::new(worker)
    // 中文逐行注释：下一行是原始源码第 1576 行，保持原始代码不变。
    }

    // 中文逐行注释：下一行是原始源码第 1578 行，保持原始代码不变。
    #[tokio::test]
    // 中文函数注释：测试健康 prefill worker 选择。
    // 中文逐行注释：下一行是原始源码第 1579 行，保持原始代码不变。
    async fn test_select_healthy_prefill_worker() {
        // 中文逐行注释：下一行是原始源码第 1580 行，保持原始代码不变。
        let router = create_test_pd_router();

        // 中文逐行注释：下一行是原始源码第 1582 行，保持原始代码不变。
        let healthy_worker = create_test_worker(
            // 中文逐行注释：下一行是原始源码第 1583 行，保持原始代码不变。
            "http://healthy".to_string(),
            // 中文逐行注释：下一行是原始源码第 1584 行，保持原始代码不变。
            WorkerType::Prefill {
                // 中文逐行注释：下一行是原始源码第 1585 行，保持原始代码不变。
                bootstrap_port: None,
            // 中文逐行注释：下一行是原始源码第 1586 行，保持原始代码不变。
            },
            // 中文逐行注释：下一行是原始源码第 1587 行，保持原始代码不变。
            true,
        // 中文逐行注释：下一行是原始源码第 1588 行，保持原始代码不变。
        );
        // 中文逐行注释：下一行是原始源码第 1589 行，保持原始代码不变。
        let unhealthy_worker = create_test_worker(
            // 中文逐行注释：下一行是原始源码第 1590 行，保持原始代码不变。
            "http://unhealthy".to_string(),
            // 中文逐行注释：下一行是原始源码第 1591 行，保持原始代码不变。
            WorkerType::Prefill {
                // 中文逐行注释：下一行是原始源码第 1592 行，保持原始代码不变。
                bootstrap_port: None,
            // 中文逐行注释：下一行是原始源码第 1593 行，保持原始代码不变。
            },
            // 中文逐行注释：下一行是原始源码第 1594 行，保持原始代码不变。
            false,
        // 中文逐行注释：下一行是原始源码第 1595 行，保持原始代码不变。
        );
        // 中文逐行注释：下一行是原始源码第 1596 行，保持原始代码不变。
        let decode_worker =
            // 中文逐行注释：下一行是原始源码第 1597 行，保持原始代码不变。
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        // 中文逐行注释：下一行是原始源码第 1599 行，保持原始代码不变。
        router.worker_registry.register(Arc::from(unhealthy_worker));
        // 中文逐行注释：下一行是原始源码第 1600 行，保持原始代码不变。
        router.worker_registry.register(Arc::from(healthy_worker));
        // 中文逐行注释：下一行是原始源码第 1601 行，保持原始代码不变。
        router.worker_registry.register(Arc::from(decode_worker));

        // 中文逐行注释：下一行是原始源码第 1603 行，保持原始代码不变。
        let result = router.select_pd_pair(None, None, None).await;

        // 中文逐行注释：下一行是原始源码第 1605 行，保持原始代码不变。
        assert!(result.is_ok());
        // 中文逐行注释：下一行是原始源码第 1606 行，保持原始代码不变。
        let (prefill, _decode) = result.unwrap();

        // 中文逐行注释：下一行是原始源码第 1608 行，保持原始代码不变。
        assert_eq!(prefill.url(), "http://healthy");
        // 中文逐行注释：下一行是原始源码第 1609 行，保持原始代码不变。
        assert!(prefill.is_healthy());
    // 中文逐行注释：下一行是原始源码第 1610 行，保持原始代码不变。
    }

    // 中文逐行注释：下一行是原始源码第 1612 行，保持原始代码不变。
    #[tokio::test]
    // 中文函数注释：测试 worker 列表为空时的行为。
    // 中文逐行注释：下一行是原始源码第 1613 行，保持原始代码不变。
    async fn test_empty_worker_lists() {
        // 中文逐行注释：下一行是原始源码第 1614 行，保持原始代码不变。
        let router = create_test_pd_router();

        // 中文逐行注释：下一行是原始源码第 1616 行，保持原始代码不变。
        let result = router.select_pd_pair(None, None, None).await;

        // 中文逐行注释：下一行是原始源码第 1618 行，保持原始代码不变。
        assert!(result.is_err());
        // 中文逐行注释：下一行是原始源码第 1619 行，保持原始代码不变。
        assert!(result.unwrap_err().contains("No prefill workers available"));
    // 中文逐行注释：下一行是原始源码第 1620 行，保持原始代码不变。
    }

    // 中文逐行注释：下一行是原始源码第 1622 行，保持原始代码不变。
    #[test]
    // 中文函数注释：测试 worker load guard 指标变化。
    // 中文逐行注释：下一行是原始源码第 1623 行，保持原始代码不变。
    fn test_worker_load_metrics() {
        // 中文逐行注释：下一行是原始源码第 1624 行，保持原始代码不变。
        let prefill_worker: Arc<dyn Worker> = Arc::from(create_test_worker(
            // 中文逐行注释：下一行是原始源码第 1625 行，保持原始代码不变。
            "http://prefill".to_string(),
            // 中文逐行注释：下一行是原始源码第 1626 行，保持原始代码不变。
            WorkerType::Prefill {
                // 中文逐行注释：下一行是原始源码第 1627 行，保持原始代码不变。
                bootstrap_port: None,
            // 中文逐行注释：下一行是原始源码第 1628 行，保持原始代码不变。
            },
            // 中文逐行注释：下一行是原始源码第 1629 行，保持原始代码不变。
            true,
        // 中文逐行注释：下一行是原始源码第 1630 行，保持原始代码不变。
        ));
        // 中文逐行注释：下一行是原始源码第 1631 行，保持原始代码不变。
        let decode_worker: Arc<dyn Worker> = Arc::from(create_test_worker(
            // 中文逐行注释：下一行是原始源码第 1632 行，保持原始代码不变。
            "http://decode".to_string(),
            // 中文逐行注释：下一行是原始源码第 1633 行，保持原始代码不变。
            WorkerType::Decode,
            // 中文逐行注释：下一行是原始源码第 1634 行，保持原始代码不变。
            true,
        // 中文逐行注释：下一行是原始源码第 1635 行，保持原始代码不变。
        ));

        // 中文逐行注释：下一行是原始源码第 1637 行，保持原始代码不变。
        let _prefill_guard = WorkerLoadGuard::new(prefill_worker.clone(), None);
        // 中文逐行注释：下一行是原始源码第 1638 行，保持原始代码不变。
        let _decode_guard = WorkerLoadGuard::new(decode_worker.clone(), None);

        // 中文逐行注释：下一行是原始源码第 1640 行，保持原始代码不变。
        assert_eq!(prefill_worker.load(), 1);
        // 中文逐行注释：下一行是原始源码第 1641 行，保持原始代码不变。
        assert_eq!(decode_worker.load(), 1);

        // 中文逐行注释：下一行是原始源码第 1643 行，保持原始代码不变。
        drop(_prefill_guard);
        // 中文逐行注释：下一行是原始源码第 1644 行，保持原始代码不变。
        drop(_decode_guard);

        // 中文逐行注释：下一行是原始源码第 1646 行，保持原始代码不变。
        assert_eq!(prefill_worker.load(), 0);
        // 中文逐行注释：下一行是原始源码第 1647 行，保持原始代码不变。
        assert_eq!(decode_worker.load(), 0);
    // 中文逐行注释：下一行是原始源码第 1648 行，保持原始代码不变。
    }

    // 中文逐行注释：下一行是原始源码第 1650 行，保持原始代码不变。
    #[tokio::test]
    // 中文函数注释：测试 streaming 场景的 load tracking。
    // 中文逐行注释：下一行是原始源码第 1651 行，保持原始代码不变。
    async fn test_streaming_load_tracking() {
        // 中文逐行注释：下一行是原始源码第 1652 行，保持原始代码不变。
        use futures_util::StreamExt;
        // 中文逐行注释：下一行是原始源码第 1653 行，保持原始代码不变。
        use tokio::time::{sleep, Duration};

        // 中文逐行注释：下一行是原始源码第 1655 行，保持原始代码不变。
        let router = create_test_pd_router();

        // 中文逐行注释：下一行是原始源码第 1657 行，保持原始代码不变。
        let prefill_worker = create_test_worker(
            // 中文逐行注释：下一行是原始源码第 1658 行，保持原始代码不变。
            "http://prefill".to_string(),
            // 中文逐行注释：下一行是原始源码第 1659 行，保持原始代码不变。
            WorkerType::Prefill {
                // 中文逐行注释：下一行是原始源码第 1660 行，保持原始代码不变。
                bootstrap_port: None,
            // 中文逐行注释：下一行是原始源码第 1661 行，保持原始代码不变。
            },
            // 中文逐行注释：下一行是原始源码第 1662 行，保持原始代码不变。
            true,
        // 中文逐行注释：下一行是原始源码第 1663 行，保持原始代码不变。
        );
        // 中文逐行注释：下一行是原始源码第 1664 行，保持原始代码不变。
        let decode_worker =
            // 中文逐行注释：下一行是原始源码第 1665 行，保持原始代码不变。
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        // 中文逐行注释：下一行是原始源码第 1667 行，保持原始代码不变。
        router.worker_registry.register(Arc::from(prefill_worker));
        // 中文逐行注释：下一行是原始源码第 1668 行，保持原始代码不变。
        router.worker_registry.register(Arc::from(decode_worker));

        // 中文逐行注释：下一行是原始源码第 1670 行，保持原始代码不变。
        let prefill_workers = router.worker_registry.get_prefill_workers();
        // 中文逐行注释：下一行是原始源码第 1671 行，保持原始代码不变。
        let decode_workers = router.worker_registry.get_decode_workers();

        // 中文逐行注释：下一行是原始源码第 1673 行，保持原始代码不变。
        let prefill_ref = prefill_workers[0].clone();
        // 中文逐行注释：下一行是原始源码第 1674 行，保持原始代码不变。
        let decode_ref = decode_workers[0].clone();

        // 中文逐行注释：下一行是原始源码第 1676 行，保持原始代码不变。
        assert_eq!(prefill_ref.load(), 0);
        // 中文逐行注释：下一行是原始源码第 1677 行，保持原始代码不变。
        assert_eq!(decode_ref.load(), 0);

        // 中文逐行注释：下一行是原始源码第 1679 行，保持原始代码不变。
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        // 中文逐行注释：下一行是原始源码第 1680 行，保持原始代码不变。
        let stream = UnboundedReceiverStream::new(rx);

        // 中文逐行注释：下一行是原始源码第 1682 行，保持原始代码不变。
        {
            // 中文逐行注释：下一行是原始源码第 1683 行，保持原始代码不变。
            let response = router.create_streaming_response(
                // 中文逐行注释：下一行是原始源码第 1684 行，保持原始代码不变。
                stream.map(Ok),
                // 中文逐行注释：下一行是原始源码第 1685 行，保持原始代码不变。
                StatusCode::OK,
                // 中文逐行注释：下一行是原始源码第 1686 行，保持原始代码不变。
                None,
                // 中文逐行注释：下一行是原始源码第 1687 行，保持原始代码不变。
                false,
                // 中文逐行注释：下一行是原始源码第 1688 行，保持原始代码不变。
                None,
                // 中文逐行注释：下一行是原始源码第 1689 行，保持原始代码不变。
                prefill_ref.clone(),
                // 中文逐行注释：下一行是原始源码第 1690 行，保持原始代码不变。
                decode_ref.clone(),
            // 中文逐行注释：下一行是原始源码第 1691 行，保持原始代码不变。
            );

            // 中文逐行注释：下一行是原始源码第 1693 行，保持原始代码不变。
            // Guards are now attached to response body, so load should be 1
            // 中文逐行注释：下一行是原始源码第 1694 行，保持原始代码不变。
            assert_eq!(prefill_ref.load(), 1);
            // 中文逐行注释：下一行是原始源码第 1695 行，保持原始代码不变。
            assert_eq!(decode_ref.load(), 1);

            // 中文逐行注释：下一行是原始源码第 1697 行，保持原始代码不变。
            tx.send(bytes::Bytes::from("test data")).unwrap();

            // 中文逐行注释：下一行是原始源码第 1699 行，保持原始代码不变。
            sleep(Duration::from_millis(10)).await;

            // 中文逐行注释：下一行是原始源码第 1701 行，保持原始代码不变。
            // Load still 1 while response body exists
            // 中文逐行注释：下一行是原始源码第 1702 行，保持原始代码不变。
            assert_eq!(prefill_ref.load(), 1);
            // 中文逐行注释：下一行是原始源码第 1703 行，保持原始代码不变。
            assert_eq!(decode_ref.load(), 1);

            // 中文逐行注释：下一行是原始源码第 1705 行，保持原始代码不变。
            drop(tx);

            // 中文逐行注释：下一行是原始源码第 1707 行，保持原始代码不变。
            // Response (and its body with guards) dropped here
            // 中文逐行注释：下一行是原始源码第 1708 行，保持原始代码不变。
            drop(response);
        // 中文逐行注释：下一行是原始源码第 1709 行，保持原始代码不变。
        }

        // 中文逐行注释：下一行是原始源码第 1711 行，保持原始代码不变。
        // Guards dropped when response dropped
        // 中文逐行注释：下一行是原始源码第 1712 行，保持原始代码不变。
        assert_eq!(prefill_ref.load(), 0);
        // 中文逐行注释：下一行是原始源码第 1713 行，保持原始代码不变。
        assert_eq!(decode_ref.load(), 0);
    // 中文逐行注释：下一行是原始源码第 1714 行，保持原始代码不变。
    }
// 中文逐行注释：下一行是原始源码第 1715 行，保持原始代码不变。
}
