<!-- 本文件由 docs/ 自动生成到 docs_CN/。代码块、命令、路径、模型名和外部链接保持原样；本地 docs 链接已改写到 docs_CN。 -->

# 可观测性

## 生产环境指标
SGLang exposes the following metrics via Prometheus. 你可以 enable them by adding `--enable-metrics` when launching the server.
你可以 query them by:
```
curl http://localhost:30000/metrics
```

See [生产环境指标](../references/production_metrics.md) and [Production 请求 Tracing](../references/production_request_trace.md) for more details.

## Logging

默认情况下, SGLang does not log any request contents. 你可以 log them by using `--log-requests`.
你可以 control the verbosity by using `--log-request-level`.
See [Logging](server_arguments.md#logging) for more details.

## 请求 Dump and Replay

你可以 dump all requests and replay them later for benchmarking or other purposes.

To start dumping, use the following command to send a request to a server:
```
python3 -m sglang.srt.managers.configure_logging --url http://localhost:30000 --dump-requests-folder /tmp/sglang_request_dump --dump-requests-threshold 100
```
The server will dump the requests into a pickle file for every 100 requests.

To replay the request dump, use `scripts/playground/replay_request_dump.py`.

## Crash Dump and Replay
Sometimes the server might crash, and you may want to debug the cause of the crash.
SGLang 支持 crash dumping, which will dump all requests from the 5 minutes before the crash, allowing you to replay the requests and debug the reason later.

要启用 crash dumping, use `--crash-dump-folder /tmp/crash_dump`.
To replay the crash dump, use `scripts/playground/replay_request_dump.py`.
