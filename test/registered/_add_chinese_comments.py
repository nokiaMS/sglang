#!/usr/bin/env python3
"""Script to add Chinese comments to all .py files in specified directories."""

import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIRS = [
    "attention",
    "backends",
    "bench_fn",
    "breakable_cuda_graph",
    "constrained_decoding",
    "core",
    "cp",
    "cpu",
    "debug_utils",
    "disaggregation",
    "dllm",
    "dp_attn",
    "dp_engine",
    "ep",
    "eval",
    "function_call",
]

FILENAME_DESC = {
    "test_triton": "Triton内核测试",
    "test_torch_native": "PyTorch原生实现测试",
    "test_flashinfer": "FlashInfer后端测试",
    "test_trtllm_mla": "TensorRT-LLM MLA注意力测试",
    "test_tokenspeed_mla": "TokenSpeed MLA注意力测试",
    "test_flashmla": "FlashMLA注意力测试",
    "test_cutlass_mla": "CUTLASS MLA注意力测试",
    "test_mamba2": "Mamba2状态空间模型测试",
    "test_dual_chunk_flash_attn": "双块Flash注意力测试",
    "test_deepseek_v4": "DeepSeek V4注意力测试",
    "test_dsa": "DSA注意力测试",
    "test_trtllm_mha": "TensorRT-LLM MHA注意力测试",
    "test_tbo": "TBO(令牌批量优化)测试",
    "test_hybrid_attn": "混合注意力测试",
    "test_flex_attention": "FlexAttention测试",
    "test_fa4": "Flash Attention 4测试",
    "test_fa3": "Flash Attention 3测试",
    "conftest": "Pytest配置和共享夹具",
    "test_wave_attention_kernels": "Wave注意力内核测试",
    "test_triton_sliding_window": "Triton滑动窗口注意力测试",
    "test_triton_attention_kernels": "Triton注意力内核测试",
    "test_triton_attention_backend": "Triton注意力后端测试",
    "test_torch_native_attention_backend": "PyTorch原生注意力后端测试",
    "test_qwen3_next_deterministic": "Qwen3 Next确定性测试",
    "test_normal_decode_set_metadata": "正常解码设置元数据测试",
    "test_kda_kernels": "KDA内核测试",
    "test_hybrid_attn_backend": "混合注意力后端测试",
    "test_gemma4_swa_triton_oob_regression": "Gemma4 SWA Triton越界回归测试",
    "test_gdn_prefill_cutedsl": "GDN预填充CuteDSL测试",
    "test_gdn_noncontiguous_stride": "GDN非连续步幅测试",
    "test_flash_attention_4": "Flash Attention 4测试",
    "test_deterministic": "确定性测试",
    "test_deepseek_v3_deterministic": "DeepSeek V3确定性测试",
    "test_create_kvindices": "KV索引创建测试",
    "test_chunk_gated_delta_rule": "分块门控Delta规则测试",
    "test_torch_compile": "Torch编译后端测试",
    "test_qwen3_fp4_trtllm_gen_moe": "Qwen3 FP4 TRT-LLM生成MoE测试",
    "test_flashinfer_trtllm_gen_moe_backend": "FlashInfer TRT-LLM生成MoE后端测试",
    "test_flashinfer_trtllm_gen_attn_backend": "FlashInfer TRT-LLM生成注意力后端测试",
    "test_flashinfer_fusion_preflight": "FlashInfer融合预检测试",
    "test_deepseek_v3_fp4_cutlass_moe": "DeepSeek V3 FP4 CUTLASS MoE测试",
    "test_deepseek_v3_fp4_cutedsl_moe": "DeepSeek V3 FP4 CuteDSL MoE测试",
    "test_deepseek_r1_fp8_trtllm_backend": "DeepSeek R1 FP8 TRT-LLM后端测试",
    "test_bench_serving_reasoning_stream": "推理流式服务基准测试",
    "test_benchmark_datasets_api": "数据集API基准测试",
    "test_bench_serving_functionality": "服务功能基准测试",
    "test_breakable_cuda_graph": "可中断CUDA图测试",
    "test_constrained_decoding": "约束解码测试",
    "test_srt_engine": "SRT引擎测试",
    "test_srt_endpoint": "SRT端点测试",
    "test_request_queue_validation": "请求队列验证测试",
    "test_hidden_states": "隐藏状态测试",
    "test_engine_child_pids": "引擎子进程PID测试",
    "test_basic_sanity_eagle3": "Eagle3基本健全性测试",
    "test_basic_sanity": "基本健全性测试",
    "test_qwen3_30b": "Qwen3 30B上下文并行测试",
    "test_deepseek_v4_flash_fp4_b200_cp": "DeepSeek V4 Flash FP4 B200上下文并行测试",
    "test_deepseek_v3_cp_single_node": "DeepSeek V3单节点上下文并行测试",
    "test_deepseek_v32_cp_single_node": "DeepSeek V3.2单节点上下文并行测试",
    "utils": "CPU工具函数",
    "test_topk": "TopK算子测试",
    "test_store_cache": "存储缓存测试",
    "test_shared_expert": "共享专家测试",
    "test_server_args_backend": "服务器参数后端测试",
    "test_rope": "旋转位置编码测试",
    "test_qwen3": "Qwen3 CPU测试",
    "test_qkv_proj_with_rope": "带RoPE的QKV投影测试",
    "test_norm": "归一化算子测试",
    "test_moe": "MoE混合专家测试",
    "test_mla": "MLA注意力测试",
    "test_mamba": "Mamba状态空间模型测试",
    "test_intel_amx_attention_backend_c": "Intel AMX注意力后端C测试",
    "test_intel_amx_attention_backend_b": "Intel AMX注意力后端B测试",
    "test_intel_amx_attention_backend_a": "Intel AMX注意力后端A测试",
    "test_gemm": "通用矩阵乘法测试",
    "test_flash_attn": "Flash注意力测试",
    "test_extend": "扩展操作测试",
    "test_decode": "解码操作测试",
    "test_cpu_graph": "CPU图测试",
    "test_causal_conv1d": "因果1D卷积测试",
    "test_bmm": "批量矩阵乘法测试",
    "test_binding": "算子绑定测试",
    "test_activation": "激活函数测试",
    "test_tensor_dump_forward_hook": "张量转储前向钩子测试",
    "test_soft_watchdog": "软看门狗测试",
    "test_schedule_simulator": "调度模拟器测试",
    "test_engine_dumper_comparator_e2e": "引擎转储比较器端到端测试",
    "test_dumper": "转储器测试",
    "test_dump_loader": "转储加载器测试",
    "test_dump_comparator": "转储比较器测试",
    "test_crash_dump": "崩溃转储测试",
    "test_source_editor": "源代码编辑器测试",
    "test_dumper_integration": "转储器集成测试",
    "test_code_patcher": "代码补丁器测试",
    "test_visualizer": "可视化器测试",
    "testing_helpers": "测试辅助工具",
    "test_utils": "工具函数测试",
    "test_preset": "预设配置测试",
    "test_per_token_visualizer": "逐令牌可视化器测试",
    "test_output_types": "输出类型测试",
    "test_model_validation": "模型验证测试",
    "test_meta_overrider": "元数据覆盖器测试",
    "test_manually_verify": "手动验证测试",
    "test_log_sink": "日志接收器测试",
    "test_entrypoint": "入口点测试",
    "test_e2e_demo": "端到端演示测试",
    "test_dp_utils": "数据并行工具测试",
    "test_display": "显示模块测试",
    "test_bundle_matcher": "束匹配器测试",
    "test_bundle_comparator": "束比较器测试",
    "test_types": "类型定义测试",
    "test_formatter": "格式化器测试",
    "test_comparator": "比较器测试",
    "test_tensor_naming": "张量命名测试",
    "test_dim_parser": "维度解析器测试",
    "test_dims_parser": "多维度解析器测试",
    "test_planner": "规划器测试",
    "test_parallel_info": "并行信息测试",
    "test_executor": "执行器测试",
    "test_thd_seq_lens_loader": "THD序列长度加载器测试",
    "test_concat_steps": "拼接步骤测试",
    "test_aux_plugins": "辅助插件测试",
    "test_aux_loader": "辅助加载器测试",
    "test_axis_aligner": "轴对齐器测试",
    "test_specv2_kvcache_offloading": "SpecV2 KV缓存卸载测试",
    "test_epd_disaggregation": "EPD分离式部署测试",
    "test_disaggregation_xpu": "XPU分离式部署测试",
    "test_disaggregation_pp": "流水线并行分离式部署测试",
    "test_disaggregation_hybrid_attention": "混合注意力分离式部署测试",
    "test_disaggregation_dsv4": "DeepSeek V4分离式部署测试",
    "test_disaggregation_dp_attention": "数据并行注意力分离式部署测试",
    "test_disaggregation_different_tp": "不同TP分离式部署测试",
    "test_disaggregation_decode_radix_cache": "解码基数缓存分离式部署测试",
    "test_disaggregation_decode_offload": "解码卸载分离式部署测试",
    "test_disaggregation_basic": "基础分离式部署测试",
    "test_disaggregation_aarch64": "ARM64分离式部署测试",
    "test_llada2_mini_amd": "LLaDA2 Mini AMD测试",
    "test_llada2_mini": "LLaDA2 Mini测试",
    "test_dp_attention": "数据并行注意力测试",
    "test_data_parallelism": "数据并行测试",
    "test_mooncake_ep_small": "Mooncake专家并行小规模测试",
    "test_deepep_small": "DeepEP小规模测试",
    "test_deepep_large": "DeepEP大规模测试",
    "test_vlms_mmmu_eval": "视觉语言模型MMMU评估测试",
    "test_text_models_gsm8k_eval": "文本模型GSM8K评估测试",
    "test_kimik2_detector": "Kimik2检测器测试",
}

FUNC_DESC_PATTERNS = [
    (r"^test_(.+)$", lambda m: f"测试{m.group(1).replace('_', '')}"),
    (r"^main$", "主函数"),
    (r"^setup$", "初始化设置"),
    (r"^teardown$", "清理操作"),
    (r"^setUp$", "初始化设置"),
    (r"^tearDown$", "清理操作"),
    (r"^run_(.+)$", lambda m: f"运行{m.group(1).replace('_', '')}"),
    (r"^create_(.+)$", lambda m: f"创建{m.group(1).replace('_', '')}"),
    (r"^generate_(.+)$", lambda m: f"生成{m.group(1).replace('_', '')}"),
    (r"^load_(.+)$", lambda m: f"加载{m.group(1).replace('_', '')}"),
    (r"^save_(.+)$", lambda m: f"保存{m.group(1).replace('_', '')}"),
    (r"^parse_(.+)$", lambda m: f"解析{m.group(1).replace('_', '')}"),
    (r"^build_(.+)$", lambda m: f"构建{m.group(1).replace('_', '')}"),
    (r"^init_(.+)$", lambda m: f"初始化{m.group(1).replace('_', '')}"),
    (r"^compute_(.+)$", lambda m: f"计算{m.group(1).replace('_', '')}"),
    (r"^get_(.+)$", lambda m: f"获取{m.group(1).replace('_', '')}"),
    (r"^set_(.+)$", lambda m: f"设置{m.group(1).replace('_', '')}"),
    (r"^check_(.+)$", lambda m: f"检查{m.group(1).replace('_', '')}"),
    (r"^validate_(.+)$", lambda m: f"验证{m.group(1).replace('_', '')}"),
    (r"^convert_(.+)$", lambda m: f"转换{m.group(1).replace('_', '')}"),
    (r"^compare_(.+)$", lambda m: f"比较{m.group(1).replace('_', '')}"),
    (r"^process_(.+)$", lambda m: f"处理{m.group(1).replace('_', '')}"),
    (r"^forward$", "前向传播"),
    (r"^backward$", "反向传播"),
]

FUNC_SPECIFIC = {
    "test_correctness": "测试正确性",
    "test_numerical_correctness": "测试数值正确性",
    "test_output": "测试输出",
    "test_shape": "测试形状",
    "test_dtype": "测试数据类型",
    "test_memory": "测试内存",
    "test_performance": "测试性能",
    "test_latency": "测试延迟",
    "test_throughput": "测试吞吐量",
    "test_accuracy": "测试精度",
    "test_equality": "测试相等性",
    "test_default": "测试默认行为",
    "test_basic": "测试基本功能",
    "test_empty": "测试空输入",
    "test_single": "测试单个输入",
    "test_batch": "测试批量输入",
    "test_error": "测试错误处理",
    "test_exception": "测试异常处理",
    "test_invalid": "测试无效输入",
    "test_edge_case": "测试边界情况",
    "test_overflow": "测试溢出",
    "test_underflow": "测试下溢",
    "test_nan": "测试NaN处理",
    "test_inf": "测试无穷大处理",
    "test_gradient": "测试梯度",
    "test_backward": "测试反向传播",
    "test_deterministic": "测试确定性",
    "test_reproducibility": "测试可复现性",
    "test_concurrent": "测试并发",
    "test_parallel": "测试并行",
    "test_serial": "测试串行",
    "test_async": "测试异步",
    "test_sync": "测试同步",
    "test_timeout": "测试超时",
    "test_retry": "测试重试",
    "test_cleanup": "测试清理",
    "test_config": "测试配置",
    "test_args": "测试参数",
    "test_kwargs": "测试关键字参数",
    "test_fixture": "测试夹具",
    "test_parametrize": "测试参数化",
    "test_skip": "测试跳过条件",
    "test_xfail": "测试预期失败",
    "test_benchmark": "测试基准性能",
    "test_stress": "测试压力",
    "test_regression": "测试回归",
    "test_integration": "测试集成",
    "test_unit": "测试单元",
    "test_e2e": "端到端测试",
    "test_end_to_end": "端到端测试",
}


def get_file_desc(filename):
    stem = os.path.splitext(filename)[0]
    if stem in FILENAME_DESC:
        return FILENAME_DESC[stem]
    if stem.startswith("test_"):
        name = stem[5:].replace("_", " ")
        return f"{name}测试"
    return f"{stem}模块"


def get_func_desc(func_name):
    if func_name in FUNC_SPECIFIC:
        return FUNC_SPECIFIC[func_name]
    for pattern, desc in FUNC_DESC_PATTERNS:
        m = re.match(pattern, func_name)
        if m:
            if callable(desc):
                return desc(m)
            return desc
    return f"执行{func_name.replace('_', '')}"


def is_init_only_imports(content):
    lines = content.strip().split('\n')
    non_empty_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith('#')]
    if not non_empty_lines:
        return True
    for line in non_empty_lines:
        if not (line.startswith('import ') or line.startswith('from ') or
                line.startswith('__all__') or line.startswith('"""') or line.startswith("'''")):
            return False
    return True


def has_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def has_file_comment(content, filename):
    first_lines = content[:300]
    return f"# 文件名: {filename}" in first_lines


def find_comment_pos(line):
    in_single = False
    in_double = False
    in_triple_single = False
    in_triple_double = False
    i = 0
    while i < len(line):
        if i + 2 < len(line):
            triple = line[i:i+3]
            if triple == '"""' and not in_single and not in_triple_single:
                in_triple_double = not in_triple_double
                i += 3
                continue
            elif triple == "'''" and not in_double and not in_triple_double:
                in_triple_single = not in_triple_single
                i += 3
                continue
        if not in_triple_single and not in_triple_double:
            if line[i] == '"' and not in_single:
                in_double = not in_double
            elif line[i] == "'" and not in_double:
                in_single = not in_single
            elif line[i] == '#' and not in_single and not in_double:
                return i
        i += 1
    return -1


INLINE_PATTERNS = [
    (r'\.cuda\(\)', "转移到GPU"),
    (r'\.cpu\(\)', "转移到CPU"),
    (r'\.to\(["\']cuda["\']', "转移到GPU"),
    (r'torch\.cuda\.synchronize', "同步CUDA操作"),
    (r'\.backward\(\)', "反向传播"),
    (r'torch\.no_grad\(\)', "禁用梯度计算"),
    (r'\.zero_grad\(\)', "清零梯度"),
    (r'\.step\(\)', "执行优化步骤"),
    (r'\.item\(\)', "获取标量值"),
    (r'\.detach\(\)', "分离梯度计算图"),
    (r'subprocess\.run\(', "运行子进程"),
    (r'subprocess\.Popen\(', "启动子进程"),
    (r'os\.environ', "访问环境变量"),
    (r'torch\.compile\(', "编译模型"),
    (r'\.half\(\)', "转换为半精度"),
    (r'\.float\(\)', "转换为单精度"),
    (r'\.bfloat16\(\)', "转换为BF16精度"),
    (r'\.to\(dtype', "转换数据类型"),
]


def get_inline_comment(line):
    stripped = line.strip()
    if not stripped or stripped.startswith('#') or stripped.startswith('@') or stripped.startswith('"""') or stripped.startswith("'''"):
        return None
    if stripped.startswith('def ') or stripped.startswith('class ') or stripped.startswith('async def '):
        return None
    for pattern, comment in INLINE_PATTERNS:
        if re.search(pattern, stripped):
            return comment
    return None


def add_chinese_comments(content, filepath):
    filename = os.path.basename(filepath)
    lines = content.split('\n')
    result = []

    file_desc = get_file_desc(filename)
    file_comment = f"# 文件名: {filename} - {file_desc}"

    insert_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#!') or stripped.startswith('# -*- coding') or stripped.startswith('# coding'):
            insert_idx = i + 1
        else:
            break

    if not has_file_comment(content, filename):
        lines.insert(insert_idx, file_comment)

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        func_match = re.match(r'^(\s*)(async\s+)?def\s+(\w+)\s*\(', line)
        if func_match:
            indent = func_match.group(1)
            func_name = func_match.group(3)

            has_preceding_chinese = False
            if result:
                last_result = result[-1].strip()
                if last_result.startswith('#') and has_chinese(last_result):
                    has_preceding_chinese = True

            if not has_preceding_chinese:
                func_desc = get_func_desc(func_name)
                chinese_comment = f"{indent}# {func_desc}"
                result.append(chinese_comment)

            result.append(line)
            i += 1
            continue

        if stripped and not has_chinese(stripped):
            inline_comment = get_inline_comment(stripped)
            if inline_comment:
                comment_pos = find_comment_pos(line)
                if comment_pos >= 0:
                    existing = line[comment_pos:].rstrip()
                    if not has_chinese(existing):
                        line = line[:comment_pos] + existing + f"  # {inline_comment}"
                else:
                    line = line.rstrip() + f"  # {inline_comment}"

        result.append(line)
        i += 1

    return '\n'.join(result)


def process_file(filepath):
    filename = os.path.basename(filepath)

    if filename == '__init__.py':
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            if is_init_only_imports(content):
                return False
        except:
            return False

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  ERROR reading {filepath}: {e}")
        return False

    if has_file_comment(content, filename):
        print(f"  SKIP (already annotated): {filepath}")
        return False

    if content.count('\n') > 5000:
        print(f"  SKIP (too large): {filepath}")
        return False

    modified = add_chinese_comments(content, filepath)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(modified)
        return True
    except Exception as e:
        print(f"  ERROR writing {filepath}: {e}")
        return False


def main():
    processed = 0
    skipped = 0
    errors = 0
    processed_files = []

    for dir_name in DIRS:
        dir_path = os.path.join(BASE_DIR, dir_name)
        if not os.path.exists(dir_path):
            print(f"SKIP: Directory not found: {dir_path}")
            continue

        print(f"\nProcessing: {dir_name}/")

        for root, dirs, files in os.walk(dir_path):
            for filename in sorted(files):
                if not filename.endswith('.py'):
                    continue

                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, BASE_DIR)

                try:
                    if process_file(filepath):
                        print(f"  OK: {rel_path}")
                        processed += 1
                        processed_files.append(rel_path)
                    else:
                        skipped += 1
                except Exception as e:
                    print(f"  ERROR: {rel_path}: {e}")
                    errors += 1

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {errors}")
    print(f"  Total:     {processed + skipped + errors}")
    print(f"\nProcessed files:")
    for f in processed_files:
        print(f"  - {f}")


if __name__ == '__main__':
    main()
