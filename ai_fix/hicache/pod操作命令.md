- 删除pod
  - kubectl -n elm-test delete -f /root/guoxu/dsv4/dsv4_pro_deepep/sts-dsv4-pro-sg.yaml
- 创建pod
  - kubectl -n elm-test apply -f /root/guoxu/dsv4/dsv4_pro_deepep/sts-dsv4-pro-sg.yaml
- 注意事项
  - 每次进入pod内部启动sglang服务的时候需要重新获取pod dsv4pro-sg-gx-0的ip地址并更新sglang的启动参数dist-init-addr。
- pod dsv4pro-sg-gx-0启动命令如下
```angular2html
SGLANG_SHARED_EXPERT_TP1=1 \
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
sglang serve \
  --trust-remote-code \
  --model-path /userdata/dsv4/DeepSeek-V4-Pro \
  --served-model-name deepseek-v4-pro \
  --tp 16 \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr 172.16.78.32:20000 \
  --moe-runner-backend marlin \
  --mem-fraction-static 0.9 \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --host 0.0.0.0 \
  --port 8080 \
 --moe-dense-tp-size 1 \
--kv-cache-dtype fp8_e4m3 \
--cuda-graph-max-bs 64 \
--context-length 202752 \
--allow-auto-truncate \
--enable-hierarchical-cache --hicache-ratio 2.0 --hicache-write-policy write_through --hicache-io-backend direct
```

