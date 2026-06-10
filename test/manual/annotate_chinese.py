#!/usr/bin/env python3
# 文件名: annotate_chinese.py - annotate chinese测试
"""Script to add Chinese comments to .py files in test/manual/ subdirectories."""

import os
import re
import sys

FILE_DESCRIPTIONS = {
    # prefill_only/
    "test_encoder_embedding_models.py": "编码器嵌入模型测试 - 验证SRT和HuggingFace推理结果的相似度",
    "test_cross_encoder_models.py": "交叉编码器模型测试 - 验证SRT和HuggingFace重排序得分的一致性",
    # quant/
    "test_torchao.py": "TorchAO量化测试 - 验证int4wo/fp8wo量化配置下的推理吞吐量和VLM生成",
    "test_quantization.py": "量化模型夜间评估测试 - 使用GSM8K评估AWQ/GPTQ等量化模型的准确性",
    "test_nvfp4_gemm_archived.py": "NVFP4 GEMM归档测试 - 验证NVFP4量化模型的GSM8K准确性",
    "test_kvfp4_quant_dequant.py": "KV FP4量化反量化基准测试 - 比较FP8和KVFP4的量化和反量化性能与精度",
    "test_fp8_kvcache.py": "FP8 KV缓存测试 - 验证FP8 KV缓存在Llama和Qwen模型上的准确性",
    "test_eval_fp8_accuracy.py": "FP8精度评估测试 - 验证静态/动态/在线FP8量化的MMLU准确性",
    "test_deepseek_v3_fp4_4gpu_trtllm.py": "DeepSeek-V3 FP4 4卡TRTLLM测试 - 验证FP4量化下GSM8K准确性和推理速度",
    "test_deepseek_v32_fp4_4gpu.py": "DeepSeek-V3.2 FP4 4卡测试 - 验证DP和TP模式下FP4量化的GSM8K准确性和速度",
    "test_cutlass_w4a8_moe.py": "CUTLASS W4A8 MoE测试 - 验证W4A8量化MoE内核与参考实现的一致性",
    "test_cutlass_w16a16_moe.py": "CUTLASS W16A16 MoE测试 - 验证bf16 CUTLASS融合MoE API的正确性",
    "test_cutlass_moe.py": "CUTLASS MoE基准测试 - 比较CUTLASS和Triton融合MoE的FP8性能",
    "test_custom_ops.py": "自定义量化算子测试 - 验证FP8 per-tensor/per-token量化和填充的正确性",
    "test_block_fp8_deep_gemm_blackwell.py": "DeepGEMM Blackwell FP8测试 - 验证DeepGEMM Blackwell平台上块级FP8量化的GEMM正确性",
    "test_block_fp8.py": "块级FP8量化测试 - 验证per-token-group量化、静态量化、块级FP8矩阵乘法和MoE的正确性",
    "test_awq_archived.py": "AWQ归档测试 - 验证AWQ Marlin量化在float16精度下的MMLU准确性",
    "test_autoround.py": "AutoRound量化测试 - 验证AutoRound量化方法下多个模型的MMLU准确性",
    # scheduler/
    "test_no_overlap_scheduler.py": "无重叠调度器测试 - 验证禁用重叠调度时radix cache和chunked prefill的组合效果",
    "test_no_chunked_prefill.py": "无分块预填充测试 - 验证禁用chunked prefill模式下的MMLU准确性和服务吞吐量",
    # spec/
    "test_spec_utils.py": "推测解码工具函数测试 - 验证draft cache位置分配和KV缓存复制的正确性",
    "test_spec_ngram_fa3.py": "NGram推测解码FA3测试 - 使用FA3注意力后端验证NGram推测解码的GSM8K准确性",
    "eagle/test_eagle3_basic.py": "EAGLE3基础测试 - 验证EAGLE3推测解码的MMLU准确性和平均接受长度",
    # vlm/
    "test_mm_utils.py": "多模态工具函数测试 - 验证CUDA IPC代理张量的重建和from_dict流程",
    "test_anthropic_vision.py": "Anthropic视觉API测试 - 验证/v1/messages端点的图像输入、流式输出和多轮对话功能",
    # root-level
    "test_async_dynamic_batch_tokenizer.py": "异步动态批处理分词器测试 - 验证批处理效率、超时处理和错误处理",
    "test_config_integration.py": "配置文件集成测试 - 验证YAML配置文件解析、CLI参数覆盖和错误处理",
    "test_whisper_cuda_graph.py": "Whisper CUDA图测试 - 验证Whisper模型在CUDA图模式下的转录正确性和一致性",
    "test_weight_version.py": "权重版本测试 - 验证权重版本查询、更新和持久化功能",
    "test_weight_validation.py": "权重验证测试 - 验证分片缺失和损坏检测、缓存清理逻辑",
    "test_wave_attention_backend.py": "Wave注意力后端测试 - 验证wave注意力后端的延迟和MMLU准确性",
    "test_w4a8_deepseek_v3.py": "DeepSeek-V3 W4A8测试 - 验证W4AFP8量化下DeepSeek-V3的GSM8K准确性",
    "test_vlm_accuracy.py": "视觉语言模型精度测试 - 比较HF和SGLang的VLM嵌入输出一致性",
    "test_vertex_endpoint.py": "Vertex端点测试 - 验证Vertex AI兼容的generate端点",
    "test_two_batch_overlap.py": "双批次重叠测试 - 验证two-batch-overlap功能的正确性和MMLU准确性",
    "test_trtllm_fp8_kv_kernel.py": "TRTLLM FP8 KV内核测试 - 验证融合FP8 KV缓存写入内核的正确性和CUDA图兼容性",
    "test_triton_moe_wna16.py": "Triton MoE WNA16测试 - 验证W4A16/W8A16量化MoE Triton内核的正确性",
    "test_triton_attention_rocm_mla.py": "Triton MLA注意力ROCm测试 - 验证ROCm平台上MLA解码注意力的RoPE融合内核正确性",
    "test_torch_tp.py": "Torch张量并行测试 - 验证torch native后端下TP=2的离线吞吐量",
    "test_torch_flex_attention_backend.py": "Torch Flex注意力后端测试 - 验证flex_attention后端的GSM8K准确性",
    "test_tokenizer_manager.py": "分词器管理器测试 - 验证输入格式检测、分词器输入准备、结果提取和ReqState文本缓冲",
    "test_tokenizer_batch_encode.py": "分词器批处理编码测试 - 验证批处理分词的多模态输入校验和约束",
    "test_srt_engine_with_quant_args.py": "SRT引擎量化参数测试 - 验证fp8/torchao等量化配置下引擎的生成功能",
    "test_schedule_policy.py": "调度策略测试 - 验证FCFS/LPM/LOF/routing-key等调度策略的优先级排序",
    "test_sagemaker_server.py": "SageMaker服务器测试 - 验证SageMaker兼容端点的聊天补全和流式输出",
    "test_ray_engine.py": "Ray引擎测试 - 验证Ray后端的TP/PP/DP并行推理、放置组和HTTP服务器",
    "test_qwen3_235b.py": "Qwen3-235B测试 - 验证Qwen3-235B-FP8模型的准确性、性能和上下文并行",
    "test_quick_allreduce.py": "Quick AllReduce测试 - 验证ROCm Quick AllReduce在图模式和即时模式下的正确性",
    "test_mscclpp.py": "MSCCL++ AllReduce测试 - 验证MSCCL++ AllReduce在图模式和即时模式下的正确性",
    "test_mori_transfer_engine_e2e.py": "Mori传输引擎端到端测试 - 验证Mori传输引擎的端到端功能",
    "test_models_from_modelscope.py": "ModelScope模型测试 - 验证从ModelScope加载的模型的推理功能",
    "test_modelopt_fp8kvcache.py": "ModelOpt FP8 KV缓存测试 - 验证ModelOpt FP8 KV缓存的准确性",
    "test_modelopt.py": "ModelOpt测试 - 验证ModelOpt量化模型的推理功能",
    "test_mla_tp.py": "MLA张量并行测试 - 验证MLA模型在TP模式下的推理功能",
    "test_logprobs.py": "对数概率测试 - 验证生成请求的对数概率输出正确性",
    "test_kv_events.py": "KV事件测试 - 验证KV缓存事件监控功能",
    "test_health_check.py": "健康检查测试 - 验证服务器健康检查端点",
    "test_glm_46_fp8.py": "GLM-4 6B FP8测试 - 验证GLM-4 6B FP8量化模型的推理功能",
    "test_get_weights_by_name.py": "按名称获取权重测试 - 验证通过名称获取模型权重的功能",
    "test_forward_split_prefill.py": "分块预填充前向测试 - 验证分块预填充前向传播的正确性",
    "test_forward_pass_metrics.py": "前向传播指标测试 - 验证前向传播的性能指标收集功能",
    "test_fim_completion.py": "FIM补全测试 - 验证Fill-in-the-Middle补全功能",
    "test_expert_location_updater.py": "专家位置更新器测试 - 验证MoE专家位置动态更新功能",
    "test_expert_distribution.py": "专家分布测试 - 验证MoE专家路由分布的统计功能",
    "test_dsa_alias_cli_registry_env.py": "DSA别名CLI注册环境测试 - 验证DSA别名的CLI参数、注册和环境变量功能",
    "test_deepseek_v31.py": "DeepSeek-V3.1测试 - 验证DeepSeek-V3.1模型的推理功能",
    "test_deepseek_chat_templates.py": "DeepSeek聊天模板测试 - 验证DeepSeek模型的聊天模板渲染功能",
    "test_custom_allreduce.py": "自定义AllReduce测试 - 验证自定义AllReduce通信算子的正确性",
    "test_crusoe_backend.py": "Crusoe后端测试 - 验证Crusoe后端的推理功能",
    "test_cross_node_scheduler_info_sync.py": "跨节点调度器信息同步测试 - 验证分布式调度器的信息同步功能",
    "test_mori_transfer_engine_e2e.py": "Mori传输引擎端到端测试 - 验证Mori传输引擎的端到端功能",
}

FUNCTION_DESCRIPTIONS = {
    "setUpClass": "类级别初始化 - 在所有测试前执行一次的设置",
    "tearDownClass": "类级别清理 - 在所有测试后执行一次的清理",
    "setUp": "测试方法级别初始化 - 每个测试方法前执行",
    "test_": "测试方法",
    "main": "主入口函数",
}

INLINE_COMMENT_MAP = {
    "popen_launch_server": "启动SRT服务器",
    "kill_process_tree": "终止服务器进程",
    "run_eval": "运行评估",
    "run_mmlu_test": "运行MMLU测试",
    "run_bench_serving": "运行服务吞吐量基准测试",
    "run_bench_one_batch": "运行单批次基准测试",
    "run_bench_offline_throughput": "运行离线吞吐量基准测试",
    "assertGreater": "断言大于",
    "assertGreaterEqual": "断言大于等于",
    "assertEqual": "断言等于",
    "assertTrue": "断言为真",
    "assertFalse": "断言为假",
    "assertIn": "断言包含",
    "assertClose": "断言近似",
    "assert_allclose": "断言全部近似",
    "torch.testing.assert_close": "断言张量近似",
    "raise NotImplementedError": "抛出未实现异常",
    "is_in_ci": "是否在CI环境中",
    "is_in_amd_ci": "是否在AMD CI环境中",
}

GENERIC_FUNC_DESC = {
    "assert_close_prefill_logits": "验证预填充logits的相似度 - 比较HF和SRT的推理结果",
    "_truncate_prompts": "截断提示文本 - 根据模型最大长度限制截断输入",
    "preprocess_prompts": "预处理提示 - 将查询和文档配对为交叉编码器输入格式",
    "run_decode": "运行解码生成 - 发送生成请求并返回结果",
    "run_eval": "运行评估 - 执行模型评估并返回指标",
    "run_bench": "运行基准测试 - 执行性能基准测试",
    "parse_models": "解析模型列表 - 将逗号分隔的模型字符串解析为列表",
    "popen_launch_server_wrapper": "启动服务器包装器 - 根据模型配置启动SRT服务器",
    "check_model_scores": "检查模型分数 - 验证模型评估分数是否达到阈值",
    "pack_int4_values_to_int8": "将int4值打包为int8 - 将交错排列的int4值压缩存储",
    "pack_interleave": "打包交错权重 - 将量化权重和缩放因子打包为交错格式",
    "calc_diff": "计算差异 - 使用DeepGEMM方法计算两个张量之间的相似度差异",
    "get_model_config": "获取模型配置 - 从HuggingFace加载DeepSeek-R1的模型配置",
    "to_fp8": "转换为FP8 - 将张量缩放并转换为FP8 E4M3格式",
    "run_test": "运行测试 - 执行CUTLASS与Triton MoE的基准对比测试",
    "quantize_ref_per_tensor": "per-tensor量化参考实现 - 按张量级进行FP8量化的参考方法",
    "dequantize_per_tensor": "per-tensor反量化 - 按张量级进行FP8反量化",
    "quantize_ref_per_token": "per-token量化参考实现 - 按token级进行FP8量化的参考方法",
    "dequantize_per_token": "per-token反量化 - 按token级进行FP8反量化",
    "calculate_accuracy_metrics": "计算精度指标 - 计算MSE、MAE、PSNR和相对误差",
    "run_benchmark": "运行基准测试 - 比较FP8和KVFP4的量化反量化性能",
    "ceil_div": "向上整除 - 计算x除以y的向上取整结果",
    "align": "对齐 - 将x对齐到y的倍数",
    "per_token_group_quant_fp8": "per-token-group FP8量化 - 按token组进行FP8量化",
    "per_block_quant_fp8": "per-block FP8量化 - 按128x128块进行FP8量化",
    "ceil_to_ue8m0": "转换为UE8M0格式 - 将缩放因子向上取整到2的幂次",
    "per_token_group_quant_mxfp8": "per-token-group MXFP8量化 - 按token组进行MXFP8量化",
    "per_block_quant_mxfp8": "per-block MXFP8量化 - 按块进行MXFP8量化",
    "native_w8a8_block_fp8_matmul": "原生W8A8块级FP8矩阵乘法 - 使用torch实现的块级量化矩阵乘参考",
    "block_quant_dequant": "块级量化反量化 - 将块级量化的张量还原为未量化形式",
    "native_per_token_group_quant_fp8": "原生per-token-group FP8量化 - 使用原生torch实现的参考方法",
    "native_static_quant_fp8": "原生静态FP8量化 - 使用原生torch实现的静态量化参考方法",
    "torch_w8a8_block_fp8_moe": "torch W8A8块级FP8 MoE - 使用torch实现的块级量化MoE参考",
    "torch_w8a8_block_fp8_bmm": "torch W8A8块级FP8批量矩阵乘 - 使用torch实现的块级量化批量矩阵乘参考",
    "quantize_weights": "量化权重 - 将权重按指定类型和分组大小进行量化",
    "torch_moe": "torch MoE参考实现 - 使用torch实现的MoE参考",
    "torch_moe_reference": "torch MoE参考实现 - 使用torch实现的MoE参考",
    "cutlass_moe": "CUTLASS MoE包装 - 准备参数并调用CUTLASS W4A8 MoE内核",
    "ref": "参考实现 - 使用torch实现的W4A8 MoE前向传播参考",
    "get_audio_bytes": "获取音频字节 - 下载或读取本地音频文件",
    "_transcribe": "发送转录请求 - 通过OpenAI兼容音频端点发送转录请求",
    "_make_request": "发送请求 - 向/v1/messages端点发送请求",
    "_parse_sse_events": "解析SSE事件 - 从流式响应中解析服务器发送事件",
    "_verify_ironing_image_content": "验证熨衣图像内容 - 检查响应文本是否描述了熨衣图像",
    "_fetch_image_base64": "获取图像base64 - 下载图像并返回base64编码",
    "_make_proxy_with_reconstruct_result": "创建代理对象 - 构造CudaIpcTensorTransportProxy的模拟实例",
    "materialize_proxy": "物化代理 - 将CUDA IPC代理张量重建为目标设备上的实际张量",
    "compute_split_seq_index": "计算分割序列索引 - 计算双批次重叠中的序列分割点",
    "compute_split_token_index": "计算分割token索引 - 计算双批次重叠中的token分割点",
    "_detect_input_format": "检测输入格式 - 判断输入是单字符串、批量字符串还是交叉编码器对",
    "_prepare_tokenizer_input": "准备分词器输入 - 根据输入格式准备分词器的输入数据",
    "_extract_tokenizer_results": "提取分词器结果 - 从分词器输出中提取input_ids和token_type_ids",
    "_validate_batch_tokenization_constraints": "验证批处理分词约束 - 检查批处理模式下不支持的功能",
    "calc_priority": "计算优先级 - 根据调度策略计算请求的优先级排序",
    "_create_engine_on_pg": "在放置组上创建引擎 - 在Ray放置组中创建引擎Actor",
    "_cleanup": "清理 - 关闭引擎Actor并移除放置组",
    "get_open_port": "获取可用端口 - 查找系统可用的网络端口",
    "multi_process_parallel": "多进程并行测试 - 使用Ray启动多进程并行测试",
    "_test_kernel_correctness": "测试内核正确性 - 比较Triton内核和朴素实现的结果",
    "_check_index_files_exist": "检查索引文件存在 - 验证分片模型索引文件是否存在和有效",
    "_validate_sharded_model": "验证分片模型 - 检查模型分片的完整性和有效性",
    "compare_outputs": "比较输出 - 对比HF和SGLang的输出张量的统计差异",
    "get_completion_request": "获取补全请求 - 构造ChatCompletionRequest测试请求",
    "get_processor_output": "获取处理器输出 - 使用HuggingFace处理器处理多模态输入",
    "get_sglang_model": "获取SGLang模型 - 创建ModelRunner并返回模型实例",
    "_set_all_seeds": "设置所有随机种子 - 设置Python和PyTorch的随机种子以确保可重复性",
    "preprocess_kv_cache": "预处理KV缓存 - 将KV缓存分解为k_input和v_input",
    "input_helper": "输入辅助函数 - 生成MLA解码注意力测试所需的输入张量",
    "ref_compute_full_fwd": "参考前向计算 - 使用分组解码注意力进行MLA前向计算的参考实现",
    "_test_rocm_fused_mla_kernel": "测试ROCm融合MLA内核 - 比较ROCm MLA解码注意力与参考实现",
    "qr_variable_input": "QuickReduce可变输入测试 - 测试输入形状频繁变化时的QuickReduce稳定性",
    "run_chat_completion": "运行聊天补全 - 发送非流式聊天补全请求并验证响应",
    "run_chat_completion_stream": "运行流式聊天补全 - 发送流式聊天补全请求并验证响应",
    "run_generate": "运行生成 - 发送生成请求并返回结果",
    "_test_deep_gemm_blackwell": "测试DeepGEMM Blackwell - 验证Blackwell平台上DeepGEMM的FP8 GEMM正确性",
    "_per_token_group_quant_fp8": "测试per-token-group FP8量化 - 比较内核实现和原生参考实现",
    "_static_quant_fp8": "测试静态FP8量化 - 比较内核实现和原生参考实现",
    "_per_tensor_quant_mla_fp8": "测试per-tensor MLA FP8量化 - 比较内核实现和参考实现",
    "_w8a8_block_fp8_matmul": "测试W8A8块级FP8矩阵乘法 - 比较内核实现和原生参考实现",
    "_mxfp8_dense_linear": "测试MXFP8密集线性层 - 比较Triton内核和参考实现",
    "_w8a8_block_fp8_fused_moe": "测试W8A8块级FP8融合MoE - 比较内核实现和torch参考实现",
    "_w8a8_block_fp8_batched_deep_gemm": "测试W8A8块级FP8批量DeepGEMM - 比较DeepGEMM和torch参考实现",
    "_w8a8_block_fp8_batched_deep_gemm_masked_fp8": "测试W8A8块级FP8批量DeepGEMM masked - 比较DeepGEMM masked和torch参考实现",
    "_make_state": "创建ReqState - 构造最小化的ReqState测试实例",
    "append_text": "追加文本 - 将文本块追加到请求状态的缓冲区",
    "get_text": "获取文本 - 合并所有缓冲的文本块并返回",
    "get_crash_dump_output": "获取崩溃转储输出 - 返回请求状态的调试信息",
    "test_scaled_fp8_quant_per_tensor": "测试per-tensor FP8量化 - 验证动态和静态量化模式",
    "test_scaled_fp8_quant_per_token_dynamic": "测试per-token动态FP8量化 - 验证按token动态量化",
    "test_scaled_fp8_quant_with_padding": "测试带填充的FP8量化 - 验证token填充后的量化结果",
    "test_fused_moe_wn16": "测试WNA16融合MoE - 验证W4A16/W8A16量化MoE内核",
}

# get file comment
def get_file_comment(filename):
    desc = FILE_DESCRIPTIONS.get(filename, "")
    if not desc:
        name = filename.replace(".py", "").replace("test_", "").replace("_", " ")
        desc = f"{name}测试"
    return f"# 文件名: {filename} - {desc}"

# get func comment
def get_func_comment(func_name, indent=""):
    if func_name in GENERIC_FUNC_DESC:
        return f"{indent}# {GENERIC_FUNC_DESC[func_name]}"
    for prefix, desc in sorted(GENERIC_FUNC_DESC.items(), key=lambda x: -len(x[0])):
        if func_name.startswith(prefix):
            return f"{indent}# {desc}"
    if func_name.startswith("test_"):
        test_name = func_name[5:].replace("_", " ")
        return f"{indent}# 测试{test_name}"
    if func_name.startswith("_"):
        clean = func_name.lstrip("_").replace("_", " ")
        return f"{indent}# 内部方法: {clean}"
    return f"{indent}# {func_name.replace('_', ' ')}"

# process file
def process_file(filepath):
    filename = os.path.basename(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")

    file_comment = get_file_comment(filename)

    # Check if file already has our Chinese comment
    if lines and lines[0].startswith(f"# 文件名: {filename}"):
        print(f"  SKIP (already annotated): {filename}")
        return False

    new_lines = []

    # Find where to insert the file comment
    # Skip shebang and encoding lines, insert after them
    insert_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#!") or stripped.startswith("# -*- coding:") or stripped.startswith("#!/"):
            insert_idx = i + 1
        elif stripped == "":
            # Allow blank lines before module docstring
            if insert_idx == i:
                insert_idx = i + 1
            else:
                break
        elif stripped.startswith('"""') or stripped.startswith("'''"):
            # Docstring - insert before it
            break
        elif stripped.startswith("#") and not stripped.startswith("# 文件名:"):
            # Other comments (copyright, SPDX, etc.) - skip past them
            insert_idx = i + 1
        elif stripped.startswith("import ") or stripped.startswith("from "):
            break
        else:
            break

    # Insert file comment
    if insert_idx == 0:
        new_lines = [file_comment] + lines
    else:
        # Check if there's already a comment block at the top
        # Insert after shebang/encoding but before docstring or imports
        new_lines = lines[:insert_idx] + [file_comment] + lines[insert_idx:]

    # Now add function comments
    result = []
    i = 0
    while i < len(new_lines):
        line = new_lines[i]
        stripped = line.strip()

        # Detect function/method definitions
        match = re.match(r'^(\s*)(async\s+)?def\s+(\w+)\s*\(', line)
        if match:
            indent = match.group(1)
            func_name = match.group(3)

            # Check if previous line is already a comment (English or Chinese)
            prev_line = result[-1] if result else ""
            prev_stripped = prev_line.strip()

            # Don't add if previous line is already a Chinese comment
            if prev_stripped.startswith("#") and any('\u4e00' <= c <= '\u9fff' for c in prev_stripped):
                result.append(line)
                i += 1
                continue

            # Don't add if previous line is a docstring opener on the same def
            # (we'll let docstrings serve as the comment)
            # But do add if previous line is an English comment

            func_comment = get_func_comment(func_name, indent)

            # If previous line is a comment, add Chinese after it
            if prev_stripped.startswith("#") and not any('\u4e00' <= c <= '\u9fff' for c in prev_stripped):
                # Add Chinese comment after the English comment
                result.append(line)
                # Add inline Chinese comment if the def line doesn't already have one
                if not any('\u4e00' <= c <= '\u9fff' for c in line):
                    # Append Chinese description to the def line
                    pass
                i += 1
                continue
            else:
                # Add Chinese comment before the def
                result.append(func_comment)

        result.append(line)
        i += 1

    # Write back
    new_content = "\n".join(result)
    # Ensure file ends with newline
    if not new_content.endswith("\n"):
        new_content += "\n"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"  DONE: {filename}")
    return True

# main
def main():
    base_dir = r"E:\code\ai_infra\sglang\test\manual"

    # Collect all target files
    target_dirs = [
        os.path.join(base_dir, "prefill_only"),
        os.path.join(base_dir, "quant"),
        os.path.join(base_dir, "scheduler"),
        os.path.join(base_dir, "spec"),
        os.path.join(base_dir, "spec", "eagle"),
        os.path.join(base_dir, "vlm"),
    ]

    files = []

    # Root-level .py files
    for f in os.listdir(base_dir):
        if f.endswith(".py"):
            files.append(os.path.join(base_dir, f))

    # Subdirectory .py files
    for d in target_dirs:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".py"):
                    files.append(os.path.join(d, f))

    print(f"Found {len(files)} .py files to process")

    annotated = 0
    for filepath in sorted(files):
        try:
            if process_file(filepath):
                annotated += 1
        except Exception as e:
            print(f"  ERROR: {filepath}: {e}")

    print(f"\nAnnotated {annotated} files out of {len(files)}")

if __name__ == "__main__":
    main()
