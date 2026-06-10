# 文件名: _add_chinese_comments.py -  add chinese comments测试
import ast
import os
import re

FILES = [
    r"E:\code\ai_infra\sglang\test\manual\4-gpu-models\test_qwen3_next_models_mtp_archived.py",
    r"E:\code\ai_infra\sglang\test\manual\4-gpu-models\test_qwen3_next_models.py",
    r"E:\code\ai_infra\sglang\test\manual\4-gpu-models\test_qwen35_models_archived.py",
    r"E:\code\ai_infra\sglang\test\manual\4-gpu-models\test_qwen35_fp4_triton.py",
    r"E:\code\ai_infra\sglang\test\manual\8-gpu-models\test_dsa_models_basic.py",
    r"E:\code\ai_infra\sglang\test\manual\8-gpu-models\test_deepseek_v3_basic.py",
    r"E:\code\ai_infra\sglang\test\manual\ascend\test_mindspore_models.py",
    r"E:\code\ai_infra\sglang\test\manual\ascend\test_ascend_w8a8_quantization.py",
    r"E:\code\ai_infra\sglang\test\manual\ascend\test_ascend_vocab_mask.py",
    r"E:\code\ai_infra\sglang\test\manual\ascend\test_ascend_deepseek_mtp.py",
    r"E:\code\ai_infra\sglang\test\manual\ascend\disaggregation_utils.py",
    r"E:\code\ai_infra\sglang\test\manual\attention\test_trtllm_mla_backend.py",
    r"E:\code\ai_infra\sglang\test\manual\attention\test_local_attn.py",
    r"E:\code\ai_infra\sglang\test\manual\attention\test_prefix_chunk_info.py",
    r"E:\code\ai_infra\sglang\test\manual\attention\test_flashattn_backend.py",
    r"E:\code\ai_infra\sglang\test\manual\attention\test_flashattn_mla_backend.py",
    r"E:\code\ai_infra\sglang\test\manual\attention\test_fa3.py",
    r"E:\code\ai_infra\sglang\test\manual\core\test_swa_loc_translation_cache.py",
    r"E:\code\ai_infra\sglang\test\manual\core\test_gpt_oss_1gpu.py",
    r"E:\code\ai_infra\sglang\test\manual\core\test_dynamic_grad_mode.py",
    r"E:\code\ai_infra\sglang\test\manual\core\test_dsv4_stale_loc_crash.py",
    r"E:\code\ai_infra\sglang\test\manual\core\test_dsv4_hicache_swa_translation_cache.py",
    r"E:\code\ai_infra\sglang\test\manual\core\test_dsv4_cached_loc_invalidation.py",
    r"E:\code\ai_infra\sglang\test\manual\cpu\test_comm.py",
    r"E:\code\ai_infra\sglang\test\manual\debug_utils\test_log_parser.py",
    r"E:\code\ai_infra\sglang\test\manual\debug_utils\test_dump_metric.py",
    r"E:\code\ai_infra\sglang\test\manual\debug_utils\run_with_retry.py",
    r"E:\code\ai_infra\sglang\test\manual\debug_utils\get_logits_ut.py",
    r"E:\code\ai_infra\sglang\test\manual\distributed\test_dp_attention_large.py",
    r"E:\code\ai_infra\sglang\test\manual\distributed\test_dp_attention_archived.py",
    r"E:\code\ai_infra\sglang\test\manual\dsv4\__init__.py",
    r"E:\code\ai_infra\sglang\test\manual\dsv4\_common.py",
    r"E:\code\ai_infra\sglang\test\manual\dsv4\test_h200_fp8_pro.py",
    r"E:\code\ai_infra\sglang\test\manual\dsv4\test_h200_fp8_flash.py",
    r"E:\code\ai_infra\sglang\test\manual\dsv4\test_h200_fp4_pro.py",
    r"E:\code\ai_infra\sglang\test\manual\dsv4\test_h200_fp4_flash.py",
    r"E:\code\ai_infra\sglang\test\manual\dsv4\test_gb300_pro.py",
    r"E:\code\ai_infra\sglang\test\manual\dsv4\test_gb300_flash.py",
    r"E:\code\ai_infra\sglang\test\manual\dsv4\test_fused_compress_attn_hip.py",
    r"E:\code\ai_infra\sglang\test\manual\dsv4\test_dsv4_swa_radix_retract.py",
    r"E:\code\ai_infra\sglang\test\manual\dsv4\test_dsv4_pro_mtp.py",
    r"E:\code\ai_infra\sglang\test\manual\dsv4\test_dsv4_pd_disagg_nixl.py",
    r"E:\code\ai_infra\sglang\test\manual\dsv4\test_dsv4_flash_sanity_tp8.py",
    r"E:\code\ai_infra\sglang\test\manual\dsv4\test_dsv4_flash_sanity_dp4.py",
    r"E:\code\ai_infra\sglang\test\manual\dsv4\test_dsv4_flash_mtp_tp8.py",
    r"E:\code\ai_infra\sglang\test\manual\dsv4\test_dsv4_flash_mtp_dp4.py",
    r"E:\code\ai_infra\sglang\test\manual\dsv4\test_b300_pro.py",
    r"E:\code\ai_infra\sglang\test\manual\dsv4\test_b300_flash.py",
    r"E:\code\ai_infra\sglang\test\manual\dsv4\test_b200_pro.py",
    r"E:\code\ai_infra\sglang\test\manual\dsv4\test_b200_flash.py",
    r"E:\code\ai_infra\sglang\test\manual\entrypoints\http_server\test_abort_request.py",
    r"E:\code\ai_infra\sglang\test\manual\ep\test_nixl_ep.py",
    r"E:\code\ai_infra\sglang\test\manual\ep\test_mooncake_expert_backup.py",
    r"E:\code\ai_infra\sglang\test\manual\ep\test_moe_deepep_eval_accuracy_large.py",
    r"E:\code\ai_infra\sglang\test\manual\ep\test_moe_deepep.py",
    r"E:\code\ai_infra\sglang\test\manual\ep\test_flashinfer_dispatcher.py",
    r"E:\code\ai_infra\sglang\test\manual\ep\test_eplb.py",
    r"E:\code\ai_infra\sglang\test\manual\ep\test_deepep_low_latency.py",
    r"E:\code\ai_infra\sglang\test\manual\ep\test_deepep_intranode.py",
    r"E:\code\ai_infra\sglang\test\manual\ep\test_deepep_internode.py",
    r"E:\code\ai_infra\sglang\test\manual\eval\validate_longbench_v2_standalone.py",
    r"E:\code\ai_infra\sglang\test\manual\eval\validate_longbench_v2.py",
    r"E:\code\ai_infra\sglang\test\manual\eval\test_longbench_v2_eval.py",
    r"E:\code\ai_infra\sglang\test\manual\eval\test_eval_accuracy_large.py",
]

FILE_DESCS = {
    "test_qwen3_next_models_mtp_archived.py": "Qwen3-Next模型MTP(多令牌预测)归档测试",
    "test_qwen3_next_models.py": "Qwen3-Next模型精度测试(GSM8K/KL散度/前缀缓存)",
    "test_qwen35_models_archived.py": "Qwen3.5 FP4模型归档测试(GSM8K/MTP推测解码)",
    "test_qwen35_fp4_triton.py": "Qwen3.5 FP4 Triton后端精度测试",
    "test_dsa_models_basic.py": "DSA模型基本测试(DeepSeek-V3.2/GLM-5 DP和TP模式)",
    "test_deepseek_v3_basic.py": "DeepSeek-V3基本测试(GSM8K精度和推理速度)",
    "test_mindspore_models.py": "昇腾NPU上MindSpore模型测试(Qwen3-8B)",
    "test_ascend_w8a8_quantization.py": "昇腾NPU W8A8量化测试(精度和吞吐量)",
    "test_ascend_vocab_mask.py": "昇腾NPU词表掩码测试(限制解码词表)",
    "test_ascend_deepseek_mtp.py": "昇腾NPU DeepSeek MTP推测解码测试",
    "disaggregation_utils.py": "PD分离部署工具类(负载均衡/服务器就绪检测/RDMA设备配置)",
    "test_trtllm_mla_backend.py": "TRT-LLM MLA注意力后端测试(解码/预填充/元数据/CUDA图)",
    "test_local_attn.py": "FA3局部注意力测试(长上下文模型)",
    "test_prefix_chunk_info.py": "前缀分块信息测试(分块KV缓存索引计算)",
    "test_flashattn_backend.py": "FlashAttention后端测试(扩展/解码/上下文并行)",
    "test_flashattn_mla_backend.py": "FlashAttention MLA后端测试(多头潜在注意力)",
    "test_fa3.py": "FlashAttention3集成测试(MLA/推测解码变体)",
    "test_swa_loc_translation_cache.py": "SWA KV池位置翻译缓存测试(data_ptr缓存键/分配器失效)",
    "test_gpt_oss_1gpu.py": "GPT-OSS单GPU测试(MXFP4/BF16量化)",
    "test_dynamic_grad_mode.py": "动态梯度模式测试(inference_mode/no_grad嵌套优先级)",
    "test_dsv4_stale_loc_crash.py": "DSV4过期缓存位置崩溃回归测试(register_mapping未清缓存)",
    "test_dsv4_hicache_swa_translation_cache.py": "DSV4 HiCache SWA翻译缓存回归测试",
    "test_dsv4_cached_loc_invalidation.py": "DSV4缓存位置失效回归测试(register_mapping清缓存修复)",
    "test_comm.py": "CPU通信原语测试(共享内存allreduce/allgather)",
    "test_log_parser.py": "日志解析器测试(解码批次日志结构化解析)",
    "test_dump_metric.py": "指标转储函数测试(JSONL格式/环境变量/类型转换)",
    "run_with_retry.py": "带重试逻辑的测试运行器",
    "get_logits_ut.py": "logits计算优化单元测试(原始版vs优化版性能对比)",
    "test_dp_attention_large.py": "DP注意力大规模测试(DP2TP4/DeepSeekV3 MTP/VLM)",
    "test_dp_attention_archived.py": "DP注意力归档测试(DeepSeekV3 MTP DP模式)",
    "__init__.py": "DSV4测试包初始化",
    "_common.py": "DeepSeek-V4测试共享基类(AIME25/GSM8K评测框架)",
    "test_h200_fp8_pro.py": "H200 FP8 DeepSeek-V4-Pro测试(低延迟/均衡/最大吞吐)",
    "test_h200_fp8_flash.py": "H200 FP8 DeepSeek-V4-Flash测试(低延迟/均衡/最大吞吐/CP)",
    "test_h200_fp4_pro.py": "H200 FP4 Marlin DeepSeek-V4-Pro测试(低延迟/均衡/最大吞吐)",
    "test_h200_fp4_flash.py": "H200 FP4 Marlin DeepSeek-V4-Flash测试(低延迟/均衡/最大吞吐)",
    "test_gb300_pro.py": "GB300 DeepSeek-V4-Pro测试(低延迟/均衡/最大吞吐/CP)",
    "test_gb300_flash.py": "GB300 DeepSeek-V4-Flash测试(低延迟/均衡/最大吞吐/CP)",
    "test_fused_compress_attn_hip.py": "HIP平台融合压缩注意力Triton核测试",
    "test_dsv4_swa_radix_retract.py": "DSV4 SWA基数缓存+墓碑+回退交互压力测试",
    "test_dsv4_pro_mtp.py": "DSV4-Pro 1.6T MTP性能测试(模拟接受长度/红楼梦长上下文)",
    "test_dsv4_pd_disagg_nixl.py": "DSV4 Flash PD分离部署NIXL后端测试",
    "test_dsv4_flash_sanity_tp8.py": "DSV4-Flash TP8健全性测试(基本解码正确性)",
    "test_dsv4_flash_sanity_dp4.py": "DSV4-Flash TP4健全性矩阵(DP4/EP/大块预填充)",
    "test_dsv4_flash_mtp_tp8.py": "DSV4-Flash H200 TP8 MTP性能测试",
    "test_dsv4_flash_mtp_dp4.py": "DSV4-Flash DP4 MTP测试(精度/退化推测/请求中断)",
    "test_b300_pro.py": "B300 DeepSeek-V4-Pro测试(低延迟/均衡/最大吞吐/CP)",
    "test_b300_flash.py": "B300 DeepSeek-V4-Flash测试(低延迟/均衡/最大吞吐/CP)",
    "test_b200_pro.py": "B200 FP4 DeepSeek-V4-Pro测试(低延迟/均衡/最大吞吐/CP)",
    "test_b200_flash.py": "B200 FP4 DeepSeek-V4-Flash测试(低延迟/均衡/最大吞吐/CP)",
    "test_abort_request.py": "HTTP服务器请求中断功能集成测试",
    "test_nixl_ep.py": "NIXL专家并行测试(TP/DP注意力/弹性EP/Mooncake)",
    "test_mooncake_expert_backup.py": "Mooncake专家备份测试(多进程弹性专家)",
    "test_moe_deepep_eval_accuracy_large.py": "DeepEP大规模MoE精度测试(GSM8K/MMLU)",
    "test_moe_deepep.py": "DeepEP MoE测试(纯TP/DP注意力+自定义配置)",
    "test_flashinfer_dispatcher.py": "FlashInfer分发器测试(基本分发/空token/FP4量化)",
    "test_eplb.py": "专家并行负载均衡测试(动态EPLB/静态EPLB)",
    "test_deepep_low_latency.py": "DeepEP低延迟模式测试(分发/合并正确性和带宽)",
    "test_deepep_intranode.py": "DeepEP节点内通信测试(NVL链路分发/合并/调优)",
    "test_deepep_internode.py": "DeepEP节点间通信测试(RDMA+NVL分发/合并/调优)",
    "validate_longbench_v2_standalone.py": "LongBench-v2独立验证脚本(格式/答案提取/精度模拟)",
    "validate_longbench_v2.py": "LongBench-v2实现验证(格式兼容/评测流水线/类别过滤)",
    "test_longbench_v2_eval.py": "LongBench-v2评测工具测试(格式化/答案提取/难度指标)",
    "test_eval_accuracy_large.py": "大规模精度评测测试(MMLU/HumanEval/MGSM-en)",
}

FUNC_DESCS = {
    "setUpClass": "类级别初始化：启动服务器",
    "tearDownClass": "类级别清理：终止服务器进程",
    "setUp": "测试初始化：设置公共参数",
    "tearDown": "测试清理：清理环境变量",
    "test_gsm8k": "GSM8K数学推理精度测试",
    "test_mmlu": "MMLU多任务语言理解精度测试",
    "test_bs_1_speed": "批量大小为1的推理速度测试",
    "test_a_gsm8k": "GSM8K精度测试(预热服务器)",
    "test_throughput": "吞吐量测试",
    "test_vlm_generate": "视觉语言模型生成测试",
    "test_inference": "推理模式测试(inference_mode)",
    "test_no_grad": "无梯度模式测试(no_grad)",
    "test_nested_inference": "嵌套推理模式测试(inference_mode优先级高于no_grad)",
    "test_nested_no_grad": "嵌套无梯度测试(no_grad嵌套inference_mode)",
    "test_basic_functionality": "基本功能测试",
    "test_decode_output_match": "解码输出匹配测试(TRTLLM vs FlashInfer)",
    "test_page_size_consistency": "页面大小一致性测试",
    "test_shape_sanity": "输出形状健全性检查",
    "test_metadata_initialization": "元数据初始化测试",
    "test_metadata_block_calculation": "元数据块计数计算测试",
    "test_metadata_kv_indices_correctness": "KV索引正确性测试",
    "test_metadata_cuda_graph_compatibility": "CUDA图兼容性测试",
    "test_metadata_consistency_across_calls": "多次前向调用元数据一致性测试",
    "test_prefill_output_match_self_attention": "预填充输出匹配测试",
    "test_draft_extend_padding_unpadding_kernels": "草稿扩展填充/去填充核测试",
    "test_forward_extend_cp": "上下文并行扩展前向测试",
    "test_forward_extend": "标准扩展操作测试",
    "test_forward_decode": "缓存token解码操作测试",
    "test_forward_extend_with_prefix": "带前缀缓存的扩展操作测试",
    "test_draft_decode_set_expand_metadata": "草稿解码扩展元数据测试",
    "test_update_draft_decode_set_expand_metadata_multi_batch": "多批次草稿解码扩展元数据测试",
    "test_mask_blocks_disallowed_token_on_npu": "NPU上词表掩码阻止非法token测试",
    "test_npu_path_matches_reference_random": "NPU词表掩码路径与参考实现匹配测试",
    "test_same_offset_view_is_cache_hit": "相同偏移视图缓存命中测试",
    "test_different_offset_view_is_cache_miss": "不同偏移视图缓存未命中测试",
    "test_storage_base_ptr_would_collide": "存储基址指针冲突测试",
    "test_alloc_invalidates": "分配操作使缓存失效测试",
    "test_free_swa_invalidates": "释放SWA使缓存失效测试",
    "test_clear_invalidates": "清空操作使缓存失效测试",
    "test_set_full_to_swa_mapping_invalidates": "设置full-to-swa映射使缓存失效测试",
    "test_noop_does_not_raise": "空操作不抛异常测试",
    "test_base_class_noop_directly": "基类空操作直接调用测试",
    "test_fresh_translation_after_explicit_invalidation": "显式失效后新鲜翻译测试",
    "test_mxfp4_20b": "MXFP4量化20B模型测试",
    "test_bf16_20b": "BF16精度20B模型测试",
    "test_crash_without_fix": "未修复时崩溃复现测试",
    "test_fix_prevents_crash": "修复后防止崩溃测试",
    "test_stale_without_fix": "未修复时返回过期值测试",
    "test_correct_with_fix": "修复后返回正确值测试",
    "test_register_mapping_clears_cached_loc": "register_mapping清空cached_loc测试",
    "test_register_mapping_clears_none_cached_loc": "cached_loc为None时register_mapping幂等测试",
    "test_all_reduce": "AllReduce通信原语测试",
    "test_all_gather": "AllGather通信原语测试",
    "test_log_parser": "日志解析器功能测试",
    "test_writes_valid_jsonl": "写入有效JSONL测试",
    "test_no_env_no_file": "无环境变量时不创建文件测试",
    "test_labels_not_serializable_stringified": "不可序列化标签字符串化测试",
    "test_bool_to_int": "布尔值转整数测试",
    "test_pytest_current_test_parsing": "PYTEST当前测试解析测试",
    "test_abort_during_non_streaming_generation": "非流式生成期间中断请求测试",
    "test_batch_requests_with_selective_abort": "批量请求选择性中断测试",
    "test_gsm8k_fault_1": "GSM8K容错测试(杀死一个进程)",
    "test_dispatch_basic": "基本分发功能测试",
    "test_dispatch_with_empty_tokens": "空token分发测试",
    "test_dispatch_with_fp4_quantization": "FP4量化分发测试",
    "test_save_expert_distribution_and_init_expert_location": "保存专家分布并初始化专家位置测试",
    "test_smoke_gsm8k": "GSM8K冒烟测试(快速验证服务器)",
    "test_aime25": "AIME25数学竞赛精度测试",
    "test_swa_tombstone_retract_does_not_crash": "SWA墓碑回退不崩溃压力测试",
    "test_max_token_one": "退化推测步骤测试(max_token=1)",
    "test_request_abort": "CUDA图缓冲池中断恢复测试",
    "test_isl_4096": "输入序列长度4096延迟测试",
    "test_isl_900k": "输入序列长度900K延迟测试",
    "test_short_30k": "30K token短上下文基准测试",
    "test_long_full": "完整长上下文基准测试",
    "test_format_compatibility": "格式兼容性测试",
    "test_answer_extraction": "答案提取测试",
    "test_data_loading_simulation": "数据加载模拟测试",
    "run_accuracy_simulation": "精度模拟测试",
    "generate_validation_report": "生成验证报告",
    "test_evaluation_pipeline": "评测流水线测试",
    "test_category_filtering": "类别过滤测试",
    "test_difficulty_metrics": "难度指标测试",
    "test_format_longbench_v2_question": "LongBench-v2问题格式化测试",
    "test_extract_longbench_v2_answer": "LongBench-v2答案提取测试",
    "test_longbench_v2_eval_initialization": "LongBench-v2评测类初始化测试",
    "run_eval": "运行评测",
    "main": "主函数",
    "test_main": "测试主函数",
    "test_loop": "测试循环",
    "launch_lb": "启动负载均衡器",
    "wait_server_ready": "等待服务器就绪",
    "start_prefill": "启动预填充服务器",
    "start_decode": "启动解码服务器",
    "get_rdma_devices_args": "获取RDMA设备参数",
    "build_rotary_emb": "构建旋转位置编码",
    "compare_outputs": "比较输出结果(含详细分析)",
    "_build_pool": "构建SWA KV池和分配器",
    "_prime_and_check_invalidation": "预填充缓存并检查失效",
    "_make_mapping": "创建映射张量",
    "check_kv_indices": "检查KV索引正确性",
    "multinode_args": "获取多节点启动参数",
    "_run_sgl_eval": "运行sgl-eval评测工具",
    "_extract_score": "从sgl-eval JSON中提取分数",
    "_run_gsm8k": "运行GSM8K评测",
    "_run_one_batch": "运行单批次基准测试",
    "_run_custom_bench": "运行自定义基准测试",
    "_build_hongloumeng_jsonl": "构建红楼梦JSONL数据集",
    "_launch_dsv4_pro_server": "启动DSV4-Pro服务器",
    "_launch_dsv4_flash_server": "启动DSV4-Flash服务器",
    "_launch": "启动服务器辅助函数",
    "send_request": "发送请求",
    "send_requests_abort": "发送可中断请求",
    "_send_completion_request": "发送补全请求",
    "_send_abort_request": "发送中断请求",
    "_check_server_health": "检查服务器健康状态",
    "_spawn_and_check": "启动子进程并检查结果",
    "all_reduce_fn": "AllReduce测试函数",
    "all_gather_fn": "AllGather测试函数",
    "run_distributed_test": "运行分布式测试",
    "create_dispatcher": "创建分发器实例",
    "_assert_engine_generate_correct": "断言引擎生成结果正确",
    "format_longbench_v2_question": "格式化LongBench-v2问题",
    "extract_longbench_v2_answer": "从模型响应中提取答案",
    "create_official_format_samples": "创建官方格式测试样本",
    "create_alternative_format_samples": "创建替代格式测试样本",
    "create_sample_official_data": "创建官方格式样本数据",
    "create_alternative_format_data": "创建替代格式样本数据",
    "run_accuracy_benchmark": "运行精度基准测试",
    "generate_comparison_report": "生成对比报告",
    "_pack_mask": "将允许的token ID打包为位掩码",
    "_apply_ref_cpu": "CPU参考实现：应用词表掩码",
    "write_current_token_to_state": "参考写入路径：将当前token写入状态池",
    "fused_compress_attn": "参考压缩路径：融合压缩注意力",
    "_make_plan_from_params": "从参数构建测试计划",
    "_make_freqs_cis": "创建测试频率复数张量",
    "_freqs_to_real": "将复数频率转换为实数交错格式",
    "_ref_compress": "纯PyTorch参考压缩实现",
    "_run_test": "运行融合压缩注意力测试",
    "test_write_then_compress": "先写后压缩顺序验证测试",
    "_create_test_data": "创建填充核测试数据",
    "_create_test_output_data": "创建去填充核测试数据",
    "_merge_config": "合并测试用例与默认配置",
    "_create_model_components": "创建模型运行器和后端",
    "_create_qkv_tensors": "创建QKV张量",
    "_create_forward_batch": "创建前向批次",
    "_populate_kv_cache": "填充KV缓存",
    "_init_model_runner": "初始化模型运行器",
    "_create_attention_layer": "创建注意力层",
    "_run_reference_forward": "运行参考前向传播",
    "_verify_output": "验证输出张量",
    "_mock_write_to_req_to_token_pool": "模拟写入请求到token池",
    "_setup_kv_cache": "设置KV缓存",
    "_run_attention_test": "运行注意力测试",
    "_run_attention_cp_test": "运行上下文并行注意力测试",
    "_send_req": "发送请求到服务器",
    "_run_gsm8k_eval": "运行GSM8K评测辅助",
}


# get file header
def get_file_header(filepath):
    fname = os.path.basename(filepath)
    desc = FILE_DESCS.get(fname, "手动测试文件")
    return f"# 文件名: {fname} - {desc}"


# get func comment
def get_func_comment(func_name):
    return FUNC_DESCS.get(func_name, None)


# has chinese
def has_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))


# is docstring line
def is_docstring_line(lines, idx):
    if idx + 1 < len(lines):
        next_line = lines[idx + 1].strip()
        if next_line.startswith('"""') or next_line.startswith("'''"):
            return True
    return False


# process file
def process_file(filepath):
    fname = os.path.basename(filepath)
    if fname == "__init__.py":
        content = open(filepath, "r", encoding="utf-8").read()
        if not content.strip():
            header = get_file_header(filepath)
            new_content = header + "\n"
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)
            return True
        return False

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return False

    result = []
    added_header = False
    i = 0

    # Determine where to insert the file header
    # Skip leading docstrings and comments, insert header before first actual code
    header_inserted = False
    leading_docstring_end = -1
    in_docstring = False
    docstring_char = None

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                dc = stripped[:3]
                # Check if it's a single-line docstring
                if stripped.count(dc) >= 2 and len(stripped) > 3:
                    continue
                else:
                    in_docstring = True
                    docstring_char = dc
                    continue
            elif stripped.startswith('#') or stripped == '':
                continue
            else:
                leading_docstring_end = idx
                break
        else:
            if docstring_char in stripped:
                in_docstring = False
                continue

    if leading_docstring_end == -1:
        leading_docstring_end = len(lines)

    # Build new file content
    # Add header at the very top
    header = get_file_header(filepath) + "\n"

    # Find function definitions and add comments
    func_pattern = re.compile(r'^(\s*)(def\s+(\w+)\s*\()')
    class_pattern = re.compile(r'^(\s*)(class\s+(\w+)\s*[\(:])')

    new_lines = []
    # Add header first
    new_lines.append(header)

    # Add original content
    for idx, line in enumerate(lines):
        stripped = line.strip()

        # Check for function definitions
        func_match = func_pattern.match(line)
        if func_match:
            indent = func_match.group(1)
            func_name = func_match.group(3)
            func_comment = get_func_comment(func_name)
            if func_comment and not has_chinese(stripped):
                # Check if previous line is already a comment
                prev_is_comment = False
                if new_lines:
                    prev_stripped = new_lines[-1].strip()
                    if prev_stripped.startswith('#') or prev_stripped.startswith('"""') or prev_stripped.startswith("'''"):
                        prev_is_comment = True
                if not prev_is_comment:
                    new_lines.append(f"{indent}# {func_comment}\n")

        # Check for class definitions - add Chinese class description
        class_match = class_pattern.match(line)
        if class_match and not has_chinese(stripped):
            class_name = class_match.group(3)
            indent = class_match.group(1)
            # Map some common class names
            class_descs = {
                "TestQwen3NextMTP": "Qwen3-Next MTP推测解码测试类",
                "TestQwen3Next": "Qwen3-Next模型测试类",
                "TestQwen35FP4": "Qwen3.5 FP4模型测试类",
                "TestQwen35FP4MTP": "Qwen3.5 FP4 MTP推测解码测试类",
                "TestDeepseekV32DP": "DeepSeek-V3.2 DP模式测试类",
                "TestDeepseekV32TP": "DeepSeek-V3.2 TP模式测试类",
                "TestGLM5DP": "GLM-5 DP模式测试类",
                "TestGLM5TP": "GLM-5 TP模式测试类",
                "TestDeepseekV3Basic": "DeepSeek-V3基本测试类",
                "TestMindSporeQwen3": "MindSpore Qwen3模型测试类",
                "TestAscendW8A8": "昇腾W8A8量化测试类",
                "TestAscendDeepSeekMTP": "昇腾DeepSeek MTP测试类",
                "TestDisaggregationBase": "PD分离部署基类",
                "TestTRTLLMMLA": "TRT-LLM MLA后端测试类",
                "MockModelRunner": "模拟模型运行器(测试用)",
                "TestFlashAttention3LocalAttn": "FA3局部注意力测试类",
                "TestPrefixChunkInfo": "前缀分块信息测试类",
                "MockForwardBatch": "模拟前向批次(测试用)",
                "MockReqToTokenPool": "模拟请求到token池(测试用)",
                "TestFlashAttentionBackend": "FlashAttention后端测试类",
                "TestUpdateDraftDecodeSetExpandMetadata": "草稿解码扩展元数据更新测试类",
                "TestFlashAttentionMLABackend": "FlashAttention MLA后端测试类",
                "BaseFlashAttentionTest": "FlashAttention3测试基类",
                "TestFlashAttention3MLA": "FA3 MLA测试类",
                "TestFlashAttention3SpeculativeDecode": "FA3推测解码测试类",
                "TestFlashAttention3SpeculativeDecodeTopk": "FA3推测解码TopK测试类",
                "TestFlashAttention3MLASpeculativeDecode": "FA3 MLA推测解码测试类",
                "TestFlashAttention3MLASpeculativeDecodeTopk": "FA3 MLA推测解码TopK测试类",
                "TestCacheKeyDataPtr": "缓存键data_ptr测试类",
                "TestAllocatorMutationInvalidation": "分配器变更失效测试类",
                "TestBaseClassNoOp": "基类空操作测试类",
                "TestExplicitInvalidationCycle": "显式失效周期测试类",
                "TestGptOss1Gpu": "GPT-OSS单GPU测试类",
                "TestDynamicGradMode": "动态梯度模式测试类",
                "TestDSV4StaleLocCrash": "DSV4过期位置崩溃测试类",
                "TestDSV4HiCacheSWATranslationCache": "DSV4 HiCache SWA翻译缓存测试类",
                "TestDSV4CachedLocBugAndFix": "DSV4缓存位置Bug和修复测试类",
                "TestDSV4ActualPoolRegisterMapping": "DSV4实际池register_mapping测试类",
                "TestComm": "CPU通信原语测试类",
                "TestLogParser": "日志解析器测试类",
                "TestDumpMetric": "指标转储测试类",
                "TestAbortRequest": "请求中断测试类",
                "TestDPAttentionDP2TP4": "DP注意力DP2TP4测试类",
                "TestDPAttentionDP2TP2DeepseekV3MTP": "DP注意力DeepSeekV3 MTP测试类",
                "TestDPAttentionDP2TP4VLM": "DP注意力VLM测试类",
                "DSV4Aime25TestBase": "DSV4 AIME25测试基类",
                "DSV4FlashAime25TestBase": "DSV4 Flash AIME25测试基类(阈值0.93)",
                "DSV4ProAime25TestBase": "DSV4 Pro AIME25测试基类(阈值0.95)",
                "TestDSV4FlashPDDisaggNIXL": "DSV4 Flash PD分离NIXL测试类",
                "TestDSV4FlashTP8NoSpec": "DSV4 Flash TP8无推测解码测试类",
                "TestDSV4FlashTP4DP4": "DSV4 Flash TP4 DP4测试类",
                "TestDSV4FlashTP4EP": "DSV4 Flash TP4 EP测试类",
                "TestDSV4FlashTP4DP4ChunkedPrefillLarge": "DSV4 Flash TP4大块预填充测试类",
                "DSV4FlashMTPServerBase": "DSV4 Flash MTP服务器基类",
                "TestDSV4FlashMTPBasic": "DSV4 Flash MTP基本测试类",
                "TestDSV4FlashSWARadixRetract": "DSV4 Flash SWA基数回退测试类",
                "TestDSV4ProMTPSimulatedAcc": "DSV4 Pro MTP模拟接受长度测试类",
                "TestDSV4ProMTPHongloumeng": "DSV4 Pro MTP红楼梦长上下文测试类",
                "TestDSV4FlashMTPSimulatedAcc": "DSV4 Flash MTP模拟接受长度测试类",
                "_EPTestBase": "专家并行测试基类",
                "TestNixlEPTP": "NIXL EP TP模式测试类",
                "TestNixlEPDPAttn": "NIXL EP DP注意力测试类",
                "TestNixlEPElasticEP": "NIXL EP弹性EP测试类",
                "TestNixlMoeMooncakeElasticEP": "NIXL Mooncake弹性EP测试类",
                "TestBackup": "Mooncake专家备份测试类",
                "TestMoEDeepEPEvalAccuracyLarge": "DeepEP大规模MoE精度测试类",
                "TestPureTP": "纯TP模式测试类",
                "TestDPAttn": "DP注意力模式测试类",
                "TestFlashinferDispatcher": "FlashInfer分发器测试类",
                "_BaseTestDynamicEPLB": "动态EPLB测试基类",
                "TestDynamicEPLBSimple": "简单动态EPLB测试类",
                "TestDynamicEPLBMultiChunk": "多块动态EPLB测试类",
                "TestStaticEPLB": "静态EPLB测试类",
                "MockSampler": "模拟采样器(测试用)",
                "FixedSampler": "固定采样器(测试用)",
                "DummyModel": "虚拟模型(logits计算优化测试用)",
                "_SWAPoolMock": "SWA池模拟类",
                "_DSV4CacheStub": "DSV4缓存存根(测试用)",
                "TestFusedCompressAttn": "融合压缩注意力测试类",
                "TestStateOrdering": "状态顺序测试类",
                "FusedCompressPlan": "融合压缩计划数据类",
                "TestEvalAccuracyLarge": "大规模精度评测测试类",
            }
            cdesc = class_descs.get(class_name)
            if cdesc:
                # Check if previous line is already a comment
                prev_is_comment = False
                if new_lines:
                    prev_stripped = new_lines[-1].strip()
                    if prev_stripped.startswith('#') or prev_stripped.startswith('"""') or prev_stripped.startswith("'''"):
                        prev_is_comment = True
                if not prev_is_comment:
                    new_lines.append(f"{indent}# {cdesc}\n")

        # Add inline Chinese comments for key patterns
        # For lines with existing English comments, add Chinese after
        if '#' in line and not has_chinese(line):
            code_part, comment_part = line.rstrip('\n').rsplit('#', 1)
            comment_stripped = comment_part.strip()

            inline_map = {
                "Archived test classes": "归档测试类",
                "Originally registered": "最初注册于",
                "the per-commit pruning effort": "按提交裁剪工作",
                "Append an \"a\" to make this test run first": "添加\"a\"使此测试按字母序先运行以预热服务器",
                "to warm up the server": "预热服务器",
                "Patch DP-attention globals before importing backends": "在导入后端之前修补DP注意力全局变量",
                "Global configuration for all tests": "所有测试的全局配置",
                "Centralized test cases for different test scenarios": "不同测试场景的集中测试用例",
                "Minimal fake ModelRunner for testing MLA backends": "用于测试MLA后端的最小模拟ModelRunner",
                "Minimal sanity check": "最小健全性检查",
                "Medium-scale batch": "中等规模批次",
                "Single FP16 vs reference": "单样本FP16与参考对比",
                "Batch FP16 vs reference": "批量FP16与参考对比",
                "32-token pages": "32 token页面",
                "64-token pages": "64 token页面",
                "Single sequence": "单序列",
                "Different page size": "不同页面大小",
                "Batch shapes": "批量形状",
                "Single sequence metadata": "单序列元数据",
                "Mixed sequence lengths": "混合序列长度",
                "Large batch stress test": "大批量压力测试",
                "Sub-page sequences": "子页面序列",
                "Server args stub": "服务器参数存根",
                "Model-config stub with MLA attributes": "带MLA属性的模型配置存根",
                "Req-to-token pool": "请求到token池",
                "KV-token pool (MLA)": "KV token池(MLA)",
                "Basic checks": "基本检查",
                "Check for NaN/Inf": "检查NaN/Inf",
                "Element-wise differences": "逐元素差异",
                "Check numerical equivalence": "检查数值等价性",
                "Find top differences for debugging": "查找最大差异用于调试",
                "Set up global server args for testing": "设置全局服务器参数用于测试",
                "Merge test case with default configuration": "合并测试用例与默认配置",
                "Create model runners, backends, and layer for testing": "创建模型运行器、后端和层用于测试",
                "Create backends": "创建后端",
                "Create RadixAttention layer": "创建RadixAttention层",
                "Create Q, K, V random tensors": "创建QKV随机张量",
                "Create separate nope and rope components for Q": "为Q创建单独的nope和rope分量",
                "Create separate nope and rope components for K": "为K创建单独的nope和rope分量",
                "V tensor (unchanged)": "V张量(不变)",
                "Create a forward batch for the given backend": "为给定后端创建前向批次",
                "Publish backend for RadixAttention dispatch": "发布后端用于RadixAttention分发",
                "Add position information for RoPE": "添加RoPE位置信息",
                "Populate KV cache with identical data for both backends": "为两个后端填充相同的KV缓存",
                "Fixed seed for reproducible cache": "固定种子以确保缓存可复现",
                "Reset seed for each backend": "为每个后端重置种子",
                "Create random K components for MLA": "为MLA创建随机K分量",
                "Calculate cache location": "计算缓存位置",
                "Save to KV cache": "保存到KV缓存",
                "Test basic functionality with minimal setup": "使用最小设置测试基本功能",
                "Running basic functionality tests": "运行基本功能测试",
                "Create components": "创建组件",
                "Create sequence lengths": "创建序列长度",
                "For larger batch sizes, create varied sequence lengths": "对于更大批量，创建变化的序列长度",
                "Ensure at least one max length": "确保至少一个最大长度",
                "Create forward batch": "创建前向批次",
                "Populate KV cache": "填充KV缓存",
                "Create Q, K, V tensors with separate MLA components": "使用独立的MLA分量创建QKV张量",
                "Run forward decode with separate MLA components": "使用独立的MLA分量运行前向解码",
                "Test that TRTLLM and FlashInfer MLA backends produce matching outputs": "测试TRTLLM和FlashInfer MLA后端输出是否匹配",
                "Running decode output matching tests": "运行解码输出匹配测试",
                "Create identical sequence lengths for both backends": "为两个后端创建相同的序列长度",
                "Ensure at least one max length": "确保至少一个最大长度",
                "Create forward batches with identical inputs": "使用相同输入创建前向批次",
                "Initialize metadata for both backends": "为两个后端初始化元数据",
                "Populate both KV caches identically": "相同地填充两个KV缓存",
                "Create Q, K, V tensors for current decode step": "为当前解码步骤创建QKV张量",
                "TRT kernel applies RoPE + FP8 quantization internally": "TRT核内部应用RoPE+FP8量化",
                "pre-apply RoPE on the reference": "在参考路径上预应用RoPE",
                "both paths share the same rope params/cache": "两条路径共享相同的rope参数/缓存",
                "Run forward decode on both backends": "在两个后端上运行前向解码",
                "Reference backend should also take separate components": "参考后端也应使用独立分量",
                "Compare outputs": "比较输出",
                "Test output consistency across different page sizes": "测试不同页面大小的输出一致性",
                "Check decode shapes across several configurations": "检查多种配置下的解码形状",
                "Random seq lens (ensure one matches max)": "随机序列长度(确保一个匹配最大值)",
                "Test with None v": "使用None v测试",
                "Run forward decode": "运行前向解码",
                "Shape and sanity checks": "形状和健全性检查",
                "Test TRTLLM MLA metadata initialization and structure": "测试TRTLLM MLA元数据初始化和结构",
                "Create varied sequence lengths": "创建变化的序列长度",
                "Verify metadata exists": "验证元数据存在",
                "Test metadata structure": "测试元数据结构",
                "Test block KV indices properties": "测试块KV索引属性",
                "Verify block indices are valid": "验证块索引有效",
                "with -1 as padding": "使用-1作为填充",
                "Test block count calculation logic": "测试块计数计算逻辑",
                "Test internal block calculation": "测试内部块计算",
                "Should be at least the minimum required": "应至少为所需最小值",
                "Should satisfy page_size constraint": "应满足page_size约束",
                "Should satisfy TRT-LLM and Triton constraints": "应满足TRT-LLM和Triton约束",
                "Block count should be multiple of LCM of constraints": "块计数应为约束最小公倍数的倍数",
                "Test KV indices creation and correctness": "测试KV索引创建和正确性",
                "Test subset for performance": "性能测试子集",
                "Create known sequence lengths": "创建已知序列长度",
                "Populate some KV cache to have valid indices": "填充一些KV缓存以获得有效索引",
                "Verify KV indices structure": "验证KV索引结构",
                "Count valid (non -1) indices for this sequence": "计算此序列的有效(非-1)索引数",
                "Should have at least enough blocks for the sequence": "应至少有足够的块用于序列",
                "Verify indices are within valid range": "验证索引在有效范围内",
                "All block indices should be": "所有块索引应为",
                "Test metadata compatibility with CUDA graph capture/replay": "测试元数据与CUDA图捕获/重放的兼容性",
                "Initialize CUDA graph state": "初始化CUDA图状态",
                "Verify CUDA graph buffers are allocated": "验证CUDA图缓冲区已分配",
                "Test capture metadata": "测试捕获元数据",
                "Verify capture metadata": "验证捕获元数据",
                "Test replay with different sequence lengths": "使用不同序列长度测试重放",
                "Verify replay updated the metadata": "验证重放更新了元数据",
                "Test metadata consistency across multiple forward calls": "测试多次前向调用的元数据一致性",
                "First call": "第一次调用",
                "Second call with same sequence lengths": "使用相同序列长度的第二次调用",
                "Metadata structure should be consistent": "元数据结构应一致",
                "Third call with different sequence lengths": "使用不同序列长度的第三次调用",
                "Should still have valid structure": "应仍有有效结构",
                "Test prefill (forward) behavior of TRTLLM MLA backend vs reference": "测试TRTLLM MLA后端与参考的预填充行为",
                "Prefill uses full sequences": "预填充使用完整序列",
                "Create forward batches": "创建前向批次",
                "Create Q, K, V tensors for prefill": "为预填充创建QKV张量",
                "Reshape as requested": "按要求重塑",
                "Run prefill on both backends": "在两个后端上运行预填充",
                "Local attention with FA3": "FA3局部注意力",
                "requires SM 90+ / H100": "需要SM 90+/H100",
                "FlashAttention3 integration tests": "FlashAttention3集成测试",
                "Multiple test classes": "多个测试类",
                "In case of some machine lack internet connection": "如果某些机器缺少网络连接",
                "Change the path below when OFFLINE_MODE is True": "OFFLINE_MODE为True时更改以下路径",
                "Default server arguments shared across all tests": "所有测试共享的默认服务器参数",
                "disable deep gemm precompile to make launch server faster": "禁用DeepGEMM预编译以加快服务器启动",
                "Base class for testing FlashAttention3": "FlashAttention3测试基类",
                "Return the arguments for the server launch": "返回服务器启动参数",
                "Use the appropriate metric key based on the test class": "根据测试类使用适当的指标键",
                "Test FlashAttention3 with MLA": "使用MLA测试FlashAttention3",
                "Test FlashAttention3 with speculative decode enabled": "启用推测解码测试FlashAttention3",
                "Tests FlashAttention3 with enhanced speculative decoding": "增强推测解码测试FlashAttention3",
                "using top-k value > 1": "使用top-k值>1",
                "which would verify the other branches of the FA3 code": "验证FA3代码的其他分支",
                "Common test parameters": "通用测试参数",
                "only consider page=1 for unit test": "单元测试仅考虑page=1",
                "only consider layer=1 for unit test": "单元测试仅考虑layer=1",
                "Test the standard extend operation": "测试标准扩展操作",
                "input_ids and out_cache_loc are dummy tensors in this test": "此测试中input_ids和out_cache_loc是虚拟张量",
                "Pool refs are resolved via the active ForwardContext": "池引用通过活动的ForwardContext解析",
                "mock an attn_backend that carries the pools": "模拟一个携带池的attn_backend",
                "Test parameters": "测试参数",
                "Max batch size for the test": "测试的最大批量大小",
                "Total tokens(prefix + extend + decode) in the test should not exceed this length": "测试中总token数(前缀+扩展+解码)不应超过此长度",
                "Create a large enough req_to_token_pool": "创建足够大的req_to_token池",
                "Add req_to_token attribute": "添加req_to_token属性",
                "only consider layer=1 for unit test": "单元测试仅考虑layer=1",
                "Create q, k, v tensors for testing": "创建测试用qkv张量",
                "Run reference forward pass using native backend": "使用原生后端运行参考前向传播",
                "Verify output tensor shape, dtype, and values": "验证输出张量形状、数据类型和值",
                "Create a forward batch for testing based on mode and lengths": "根据模式和长度创建测试前向批次",
                "Default to self.seq_len if not specified": "未指定时默认为self.seq_len",
                "if page_size > 1, the token pool stores the index to the page": "如果page_size>1，token池存储页面索引",
                "so we need to multiply the index by page_size": "因此需要将索引乘以page_size",
                "Create constant values for the prefix cache for easy debugging": "为前缀缓存创建常量值以便调试",
                "Set the prefix KV cache": "设置前缀KV缓存",
                "Run an attention test with the specified parameters": "使用指定参数运行注意力测试",
                "KV cache for prefixed extend is prefix_len": "带前缀扩展的KV缓存长度为prefix_len",
                "KV cache for decode is same as seq_len": "解码的KV缓存与seq_len相同",
                "No KV cache for extend without prefix": "无前缀扩展无KV缓存",
                "All the test cases examples have 1 additional cache location": "所有测试用例示例有1个额外缓存位置",
                "This is to align with the current allocation logic": "这与当前分配逻辑对齐",
                "Decode span multiple pages": "解码跨多个页面",
                "duplicated kv cache": "重复的KV缓存",
                "We need 3 pages in total": "我们总共需要3个页面",
                "Ensure expand metadata works when batch size > 1": "确保批量大小>1时扩展元数据正常工作",
                "MLA with different V headdim requires Hopper architecture": "具有不同V头维度的MLA需要Hopper架构",
                "Initialize model runner and backend": "初始化模型运行器和后端",
                "Publish the backend so RadixAttention.forward resolves correctly": "发布后端以便RadixAttention.forward正确解析",
                "Set up KV cache with prefix tokens": "使用前缀token设置KV缓存",
                "For MLA, create separate nope and rope caches": "对于MLA，创建单独的nope和rope缓存",
                "latent cache has only one head in MQA": "MQA中潜在缓存只有一个头",
                "Set the prefix KV cache using MLA-specific method": "使用MLA特定方法设置前缀KV缓存",
                "Create q, kv_compressed for testing": "创建测试用q和kv_compressed",
                "For MLA, split kv_compressed into k_nope and k_rope": "对于MLA，将kv_compressed拆分为k_nope和k_rope",
                "k_nope needs to be unsqueezed for the num_heads dimension": "k_nope需要为num_heads维度增加维度",
                "k_rope also needs to be unsqueezed": "k_rope也需要增加维度",
                "v is not used for mqa": "mqa不使用v",
                "Test the standard extend operation": "测试标准扩展操作",
                "Test the decode operation with cached tokens": "使用缓存token测试解码操作",
                "Test extending from cached prefix tokens": "从缓存的前缀token测试扩展",
                "Skip the slow exhaustive DeepGEMM warmup grid": "跳过慢速的DeepGEMM全面预热网格",
                "covers the shapes DSV4 actually hits": "覆盖DSV4实际命中的形状",
                "shaves several minutes off server startup": "节省几分钟的服务器启动时间",
                "Defaults applied to every recipe's EXTRA_ENV": "应用于每个配方EXTRA_ENV的默认值",
                "Per-recipe EXTRA_ENV wins on key conflict": "配方EXTRA_ENV在键冲突时优先",
                "DeepEP \"large SMS\" config": "DeepEP大SMS配置",
                "Return CLI args for a multi-node launch, or skip the test": "返回多节点启动CLI参数或跳过测试",
                "Subclass via DSV4FlashAime25TestBase or DSV4ProAime25TestBase": "通过DSV4FlashAime25TestBase或DSV4ProAime25TestBase子类化",
                "not directly": "而非直接使用",
                "Per-recipe subclasses set MODEL / OTHER_ARGS / EXTRA_ENV": "每个配方子类设置MODEL/OTHER_ARGS/EXTRA_ENV",
                "Quick GSM8K pass to verify the server is producing math answers": "快速GSM8K测试验证服务器能产生数学答案",
                "Full AIME25 accuracy run": "完整AIME25精度运行",
                "threshold gated by Flash vs Pro base": "阈值由Flash或Pro基类决定",
                "Find metric anywhere in the sgl-eval JSON tree": "在sgl-eval JSON树中查找指标",
                "Base for DeepSeek-V4-Flash recipes": "DeepSeek-V4-Flash配方基类",
                "Base for DeepSeek-V4-Pro recipes": "DeepSeek-V4-Pro配方基类",
                "Long shared prefix forces multi-chunk prefill": "长共享前缀强制多块预填充",
                "prefix-cache hits so one req's tombstone affects later reqs": "前缀缓存命中使一个请求的墓碑影响后续请求",
                "Tight static memory so SWA pool fills up under load": "紧凑的静态内存使SWA池在负载下填满",
                "retract is forced": "强制回退",
                "Vary outputs slightly so reqs don't share decode": "略微变化输出使请求不完全共享解码路径",
                "paths perfectly": "路径",
                "we want some to finish, some to be retracted under pressure": "我们希望一些完成一些在压力下被回退",
                "Per-request success is not the gate": "单个请求成功不是门槛",
                "some requests are expected to be retracted/aborted under heavy pressure": "某些请求预期在重压下被回退/中断",
                "Stress: 64 concurrent long-prompt reqs with long generation force": "压力测试：64个并发长提示请求强制长生成",
                "retract under SWA pool pressure": "在SWA池压力下回退",
                "Reqs share a 30k+ token prefix": "请求共享30K+token前缀",
                "tombstoned leaves from retracted reqs are on the radix path of new reqs": "回退请求的墓碑叶子在新请求的基数路径上",
                "Scheduler must not crash on the swa_radix_cache assert": "调度器不能在swa_radix_cache断言上崩溃",
                "Long enough generation to push past sliding_window_size": "足够长的生成以超过滑动窗口大小",
                "fires dec_swa_lock_only": "触发dec_swa_lock_only",
                "tombstones leaves": "墓碑叶子",
                "Combined with SWA pool pressure this guarantees retract while tombstones are live": "结合SWA池压力，保证墓碑活跃时发生回退",
                "Add a small per-req suffix so reqs don't dedup at radix root": "添加小量每请求后缀使请求不在基数根去重",
                "but still share the bulk of the prefix": "但仍共享大部分前缀",
                "Stagger so requests enter prefill in waves": "交错使请求分波进入预填充",
                "some are still in decode": "一些仍在解码",
                "and have tombstoned leaves": "并有墓碑叶子",
                "when later waves of chunked-prefill reqs walk the same radix path": "当后续分波块预填充请求遍历相同基数路径",
                "The only invariant: scheduler survived": "唯一不变量：调度器存活",
                "Per-request completion is best-effort under retract pressure": "在回退压力下每个请求的完成是尽力而为",
                "MTP runs ~num_draft_tokens forward passes per step": "MTP每步运行约num_draft_tokens次前向传播",
                "so the deepep dispatch input size scales by that factor": "因此deepep分发输入大小按该因子缩放",
                "Default 256 (used by the plain server) overflows": "默认256(普通服务器使用)会溢出",
                "once cuda-graph-max-bs * num_draft_tokens": "一旦cuda-graph-max-bs * num_draft_tokens",
                "covers bs=128 * 4 draft tokens with headroom": "覆盖bs=128 * 4草稿token还有余量",
                "Accuracy + spec path full forward": "精度+推测路径完整前向",
                "Degenerate spec step": "退化推测步骤",
                "still cuda-graph captured": "仍被CUDA图捕获",
                "Cuda-graph buffer pool must survive abort+restart cycles": "CUDA图缓冲池必须在中断+重启循环中存活",
                "DSV4 Flash MTP shares the EAGLE wire path": "DSV4 Flash MTP共享EAGLE线路路径",
                "EAGLE algo + NextN head built into the target model weights": "EAGLE算法+NextN头内置于目标模型权重",
                "No separate draft model is needed": "不需要单独的草稿模型",
                "sglang auto-falls back": "sglang自动回退",
                "Test matrix mirrors test_eagle_infer_b.TestEAGLEServerBasic": "测试矩阵镜像test_eagle_infer_b.TestEAGLEServerBasic",
                "to maximize cuda-graph + buffer-pool coverage on the DSV4 path": "最大化DSV4路径上CUDA图+缓冲池的覆盖",
                "Server launch matches run_flash_dp4.sh": "服务器启动匹配run_flash_dp4.sh",
                "DSV4 FP8 (FP4 experts disabled)": "DSV4 FP8(FP4专家禁用)",
                "End-to-end PD-disagg accuracy through the LB": "通过负载均衡器的端到端PD分离精度",
                "TP8, no spec decoding": "TP8，无推测解码",
                "TP4 + DP4 + deepep + EAGLE MTP": "TP4 + DP4 + deepep + EAGLE MTP",
                "TP attn + EP MoE (no DP attn)": "TP注意力+EP MoE(无DP注意力)",
                "exercises the DeepEP + TP-attn path": "测试DeepEP+TP注意力路径",
                "No --enable-dp-attention by design": "设计上不启用--enable-dp-attention",
                "TP4 + DP4 with --chunked-prefill-size 16384": "TP4 + DP4使用--chunked-prefill-size 16384",
                "large chunked prefill": "大块预填充",
                "Launch the server": "启动服务器",
                "Clean up the server": "清理服务器",
                "Send a completion request to the server": "向服务器发送补全请求",
                "Send an abort request": "发送中断请求",
                "Check if server is healthy": "检查服务器是否健康",
                "Start all requests": "启动所有请求",
                "Abort one request": "中断一个请求",
                "Wait for completion": "等待完成",
                "Verify results": "验证结果",
                "Check aborted request": "检查被中断的请求",
                "Check other requests completed normally": "检查其他请求正常完成",
                "Helper to create dispatcher instance": "创建分发器实例的辅助函数",
                "Test basic dispatch functionality": "测试基本分发功能",
                "Single expert per token for simplicity": "为简化每个token选择单个专家",
                "One expert per rank": "每个rank一个专家",
                "Create tokens with rank number": "使用rank编号创建token",
                "Route all tokens from rank i to expert": "将rank i的所有token路由到专家",
                "Expected: we should receive tokens from rank": "预期：我们应收到来自rank的token",
                "Verify we received the right number of tokens": "验证我们收到了正确数量的token",
                "Verify tokens came from the expected source": "验证token来自预期来源",
                "Test dispatch when there are no tokens (edge case)": "无token时分发测试(边界情况)",
                "This tests the dummy token handling": "测试虚拟token处理",
                "Create tokens with rank number, rank 1 has no tokens": "使用rank编号创建token，rank 1无token",
                "Rank should receive no tokens since rank 1 was empty": "由于rank 1为空，rank不应收到token",
                "Test dispatch with FP4 quantization enabled": "启用FP4量化测试分发",
                "Create tokens with random values": "使用随机值创建token",
                "Set input global scale to enable FP4 quantization": "设置输入全局缩放以启用FP4量化",
                "Test that dump_metric writes one valid JSON line when env is set": "测试环境变量设置时dump_metric写入一条有效JSON行",
                "Check file exists with PID suffix": "检查文件存在并带有PID后缀",
                "Read and validate": "读取并验证",
                "Validate required fields": "验证必填字段",
                "Validate optional fields": "验证可选字段",
                "Test that dump_metric doesn't create file when env var not set": "测试未设置环境变量时dump_metric不创建文件",
                "Don't set env var": "不设置环境变量",
                "Verify no files created": "验证未创建文件",
                "Test that non-serializable labels are stringified": "测试不可序列化标签被字符串化",
                "Non-serializable label": "不可序列化标签",
                "Test that bool values are converted to int": "测试布尔值转换为整数",
                "Test PYTEST_CURRENT_TEST parsing for test_case": "测试PYTEST_CURRENT_TEST解析为test_case",
                "Only assert test_case parsing, not filename": "仅断言test_case解析，不断言文件名",
                "Clean up env vars before each test": "每个测试前清理环境变量",
                "Clean up env vars after each test": "每个测试后清理环境变量",
                "Check for NaN/Inf": "检查NaN/Inf",
                "Verify the padding worked correctly": "验证填充正确工作",
                "Check that valid positions are copied correctly": "验证有效位置正确复制",
                "Compare input and output for valid positions": "比较有效位置的输入和输出",
                "Check that invalid positions are zero": "检查无效位置为零",
                "Launch kernel": "启动核函数",
                "Verify the unpadding worked correctly": "验证去填充正确工作",
                "Check that valid positions are copied correctly": "验证有效位置正确复制",
                "Launch kernel": "启动核函数",
                "Intentionally omitted": "有意省略",
                "Both should return the identical tensor (cache hit)": "两者应返回相同的张量(缓存命中)",
                "Views at different offsets produce different data_ptr → cache miss": "不同偏移的视图产生不同data_ptr→缓存未命中",
                "same numel": "相同numel",
                "Different data_ptr (different storage offset)": "不同data_ptr(不同存储偏移)",
                "Prime the cache with view_lo": "用view_lo预填充缓存",
                "view_hi should be a cache miss and produce a distinct translation": "view_hi应是缓存未命中并产生不同的翻译",
                "They should NOT be the same object (different cache entries)": "它们不应是相同对象(不同缓存条目)",
                "And the content must differ (different full indices → different swa)": "且内容必须不同(不同full索引→不同swa)",
                "Demonstrate that untyped_storage().data_ptr() WOULD collide": "演示untyped_storage().data_ptr()会冲突",
                "Same storage base — old key would collide": "相同存储基址—旧键会冲突",
                "But data_ptr differs — new key is safe": "但data_ptr不同—新键安全",
                "Helper: prime cache, mutate, assert fresh translation": "辅助：预填充缓存、变更、断言新鲜翻译",
                "Prime the cache": "预填充缓存",
                "Mutate — should invalidate": "变更—应使缓存失效",
                "Cache must be cleared after mutation": "变更后缓存必须被清空",
                "Another alloc should invalidate": "另一次分配应使缓存失效",
                "HiCache load-back path: set_full_to_swa_mapping must invalidate": "HiCache回载路径：set_full_to_swa_mapping必须使缓存失效",
                "Simulate HiCache rebuild with new swa indices": "使用新swa索引模拟HiCache重建",
                "Translation after rebuild should reflect the new mapping": "重建后翻译应反映新映射",
                "BaseSWAKVPool.invalidate_loc_cache is a no-op default — must not raise": "BaseSWAKVPool.invalidate_loc_cache默认空操作—不得抛异常",
                "Calling on the concrete class uses the override — that's fine": "在具体类上调用使用覆盖—这是正确的",
                "idempotent": "幂等",
                "Call the base-class method directly to verify it's a true no-op": "直接调用基类方法验证它是真正的空操作",
                "Prime the cache first": "先预填充缓存",
                "Call the BASE class method directly — should not clear the cache": "直接调用基类方法—不应清空缓存",
                "it's a no-op; the concrete override is what clears": "它是空操作；具体覆盖才是清除的",
                "base no-op: cache untouched": "基类空操作：缓存未改变",
                "Simulates the per-forward-pass invalidation done by model_runner": "模拟model_runner执行的每次前向传播失效",
                "After invalidate_loc_cache(), a new alloc produces the right mapping": "invalidate_loc_cache()后，新分配产生正确的映射",
                "First \"forward pass\": alloc 4 tokens, translate": "第一次\"前向传播\"：分配4个token并翻译",
                "Simulate start of next forward pass: model_runner calls invalidate": "模拟下一次前向传播开始：model_runner调用失效",
                "Alloc 4 more (mapping changes), translate loc1 again": "再分配4个(映射变化)，再次翻译loc1",
                "loc1's SWA mapping hasn't changed (same full→swa assignment)": "loc1的SWA映射未变(相同full→swa分配)",
                "so result should be equal — but it must have been recomputed": "所以结果应相等—但必须重新计算",
                "cache key was None before this call": "此调用前缓存键为None",
                "loc2 should have different translation than loc1": "loc2的翻译应与loc1不同",
                "They have different indices, so translation differs": "它们有不同索引，所以翻译不同",
                "raw_loc: full-pool indices for 4 tokens": "raw_loc：4个token的全池索引",
                "cache_k: synthetic key data for 4 tokens": "cache_k：4个token的合成键数据",
                "SWA layer id > start_layer (0)": "SWA层id > start_layer(0)",
                "so cached_loc is only reset by the \"is None\" branch": "所以cached_loc仅由\"is None\"分支重置",
                "NOT by the \"layer_id == start_layer\" branch": "而非\"layer_id == start_layer\"分支",
                "mapping_v1: raw_loc [0,1,2,3] → SWA slots [4,5,6,7]": "mapping_v1：raw_loc [0,1,2,3] → SWA槽 [4,5,6,7]",
                "valid for size-8 pool": "对size-8池有效",
                "mapping_v2: same raw_loc → SWA slots [0,1,2,3]": "mapping_v2：相同raw_loc → SWA槽 [0,1,2,3]",
                "valid for size-4 pool": "对size-4池有效",
                "Without the fix, stale cached_loc [4,5,6,7] causes RuntimeError": "未修复时，过期cached_loc [4,5,6,7]导致RuntimeError",
                "Pass 1: swa_layer_id=1, cached_loc is None → compute and cache [4,5,6,7]": "第1遍：swa_layer_id=1，cached_loc为None→计算并缓存[4,5,6,7]",
                "PRE-FIX register_mapping: replace mapping WITHOUT clearing cached_loc": "修复前register_mapping：替换映射但不清空cached_loc",
                "Pass 2 on a smaller pool (size=4). cached_loc is still [4,5,6,7]": "第2遍在较小池(size=4)上。cached_loc仍为[4,5,6,7]",
                "OOB write → RuntimeError": "越界写入→RuntimeError",
                "With the fix, register_mapping() clears cached_loc": "修复后register_mapping()清空cached_loc",
                "Pass 2 recomputes [0,1,2,3] from mapping_v2": "第2遍从mapping_v2重新计算[0,1,2,3]",
                "and writes to the correct size-4 pool slots": "并写入正确的size-4池槽",
                "Pass 1: prime cache → cached_loc = [4,5,6,7]": "第1遍：预填充缓存→cached_loc = [4,5,6,7]",
                "FIXED register_mapping: clears cached_loc": "修复后register_mapping：清空cached_loc",
                "Fix: cached_loc cleared by register_mapping": "修复：cached_loc被register_mapping清空",
                "Pass 2 on the smaller pool: cached_loc is None → recompute with mapping_v2": "第2遍在较小池上：cached_loc为None→使用mapping_v2重新计算",
                "Fresh indices after fix": "修复后新鲜索引",
                "Verify data landed in the correct SWA slots [0-3], not the stale [4-7]": "验证数据落在正确的SWA槽[0-3]，非过期的[4-7]",
                "Correct SWA slots 0-3 received data": "正确的SWA槽0-3接收到数据",
                "Launch the server": "启动服务器",
                "Disable CUDA graph for abort testing": "禁用CUDA图用于中断测试",
            }

            if comment_stripped in inline_map:
                cn = inline_map[comment_stripped]
                new_line = code_part + "#" + comment_part + "  # " + cn + "\n"
                new_lines.append(new_line)
                continue

        new_lines.append(line)

    # Check if file already starts with our header pattern
    content = "".join(new_lines)
    first_line = new_lines[0] if new_lines else ""

    # If first line is already our header, don't add again
    if first_line.startswith("# 文件名:"):
        # Remove the duplicate header we added
        new_lines = new_lines[1:]
        content = "".join(new_lines)
    else:
        content = "".join(new_lines)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return True


count = 0
for fp in FILES:
    if os.path.exists(fp):
        try:
            process_file(fp)
            count += 1
            print(f"OK: {os.path.basename(fp)}")
        except Exception as e:
            print(f"ERR: {os.path.basename(fp)}: {e}")
    else:
        print(f"SKIP: {fp}")

print(f"\nDone! Processed {count} files.")
