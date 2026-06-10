import logging  # 导入日志模块
from typing import Callable, Dict, List, Optional, Tuple  # 导入类型提示工具

import torch  # 导入PyTorch核心库
import torch.distributed as dist  # 导入分布式通信模块
from torch import nn  # 导入神经网络模块

from sglang.srt.distributed import get_tp_group  # 导入获取张量并行组的函数
from sglang.srt.layers.dp_attention import (  # 导入数据并行注意力相关模块
    get_attention_tp_group,  # 获取注意力张量并行组
    is_dp_attention_enabled,  # 检查是否启用了数据并行注意力
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput  # 导入logits处理器输出类
from sglang.srt.layers.utils.hash import murmur_hash32  # 导入murmur哈希函数，用于确定性采样
from sglang.srt.layers.utils.logprob import get_token_ids_logprobs, get_top_logprobs  # 导入logprob提取工具函数
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo  # 导入采样批量信息类
from sglang.srt.sampling.sampling_params import TOP_K_ALL  # 导入TOP_K_ALL常量（表示不限制top-k）
from sglang.srt.server_args import get_global_server_args  # 导入获取全局服务器参数的函数
from sglang.srt.utils.common import (  # 导入通用工具函数
    get_bool_env_var,  # 获取布尔型环境变量
    is_cuda,  # 检查是否为CUDA平台
    is_hip,  # 检查是否为HIP（AMD ROCm）平台
    is_musa,  # 检查是否为摩尔线程MUSA平台
    is_npu,  # 检查是否为昇腾NPU平台
)

if is_cuda():  # 如果是CUDA平台
    from flashinfer.sampling import (  # 从flashinfer导入采样函数
        min_p_sampling_from_probs,  # 基于min-p的概率采样
        top_k_top_p_sampling_from_probs,  # 基于top-k和top-p的概率采样
    )
    from sgl_kernel import (  # 从sgl_kernel导入概率重归一化函数
        top_k_renorm_prob,  # top-k概率重归一化
        top_p_renorm_prob,  # top-p概率重归一化
    )

if is_musa():  # 如果是摩尔线程MUSA平台
    from sgl_kernel import (  # 从sgl_kernel导入采样和重归一化函数
        min_p_sampling_from_probs,  # 基于min-p的概率采样
        top_k_renorm_prob,  # top-k概率重归一化
        top_k_top_p_sampling_from_probs,  # 基于top-k和top-p的概率采样
        top_p_renorm_prob,  # top-p概率重归一化
    )

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()  # 是否使用AMD aiter库（仅HIP平台）
if _use_aiter:  # 如果启用了aiter
    from aiter import greedy_sample as _aiter_greedy_sample  # 导入aiter的贪心采样函数

if is_npu():  # 如果是昇腾NPU平台
    import torch_npu  # 导入昇腾NPU扩展库

logger = logging.getLogger(__name__)  # 创建当前模块的日志记录器

SYNC_TOKEN_IDS_ACROSS_TP = get_bool_env_var("SYNC_TOKEN_IDS_ACROSS_TP")  # 是否在TP组间同步token ID
SGLANG_RETURN_ORIGINAL_LOGPROB = get_bool_env_var("SGLANG_RETURN_ORIGINAL_LOGPROB")  # 是否返回原始（温度缩放前的）logprob
_CUSTOM_SAMPLER_FACTORIES: Dict[str, Callable[[], "Sampler"]] = {}  # 自定义采样器工厂注册表
_BUILT_IN_SAMPLING_BACKENDS = {"flashinfer", "pytorch", "ascend"}  # 内置采样后端集合


# 采样器模块：实现LLM推理中的token采样逻辑，包括贪心采样、top-k/top-p/min-p采样、
# 确定性采样等，支持多种后端（flashinfer/pytorch/ascend），并计算和附加logprob信息。
class Sampler(nn.Module):  # 采样器类，继承自nn.Module
    def __init__(self):  # 初始化采样器
        super().__init__()  # 调用父类初始化
        self.tp_sync_group = get_tp_group().device_group  # 获取张量并行的设备通信组
        if is_dp_attention_enabled():  # 如果启用了数据并行注意力
            self.tp_sync_group = get_attention_tp_group().device_group  # 使用注意力TP组替代

        self.rl_on_policy_target = get_global_server_args().rl_on_policy_target  # 获取RL在线策略目标
        # In RL on-policy mode, deterministic inference is automatically enabled.
        # 在RL在线策略模式下，自动启用确定性推理。
        self.enable_deterministic = (  # 是否启用确定性推理
            get_global_server_args().enable_deterministic_inference  # 从全局参数获取
        )
        # In RL on-policy mode, we use log_softmax to compute logprobs to match the trainer.
        # 在RL在线策略模式下，使用log_softmax计算logprob以匹配训练器。
        self.use_log_softmax_logprob = self.rl_on_policy_target is not None  # 是否使用log_softmax计算logprob
        self.use_ascend_backend = get_global_server_args().sampling_backend == "ascend"  # 是否使用昇腾后端

    def _preprocess_logits(  # 预处理logits
        self, logits: torch.Tensor, sampling_info: SamplingBatchInfo  # 输入logits和采样信息
    ) -> torch.Tensor:  # 返回处理后的logits
        """Apply custom logit processors."""
        """应用自定义logit处理器。"""
        if sampling_info.has_custom_logit_processor:  # 如果有自定义logit处理器
            apply_custom_logit_processor(logits, sampling_info)  # 应用自定义logit处理器
        return logits  # 返回处理后的logits

    def forward(  # 前向传播：执行采样并计算logprob
        self,
        logits_output: LogitsProcessorOutput,  # 模型前向传播的logits输出
        sampling_info: SamplingBatchInfo,  # 采样的元数据信息
        return_logprob: bool,  # 是否返回logprob信息
        top_logprobs_nums: List[int],  # 每个序列返回的top logprob数量
        token_ids_logprobs: List[List[int]],  # 每个序列需要提取logprob的指定token ID列表
        positions: torch.Tensor,  # token在序列中的位置，用于确定性采样生成唯一种子
    ):
        """Run a sampler & compute logprobs and update logits_output accordingly.
        运行采样器并计算logprob，然后相应地更新logits_output。

        Args:
            logits_output: The logits from the model forward
                模型前向传播的logits
            sampling_info: Metadata for sampling
                采样的元数据
            return_logprob: If set, store the output logprob information to
                logits_output
                如果设置，将输出logprob信息存储到logits_output
            top_logprobs_nums: Number of top lobprobs per sequence in a batch
                批次中每个序列的top logprob数量
            token_ids_logprobs: Per-sequence list of specific token IDs to retrieve
                logprobs for. Each element is a list of token IDs (or None) for one
                sequence in the batch. This is used in speculative decoding.
                每个序列需要提取logprob的指定token ID列表。每个元素是批次中一个序列的
                token ID列表（或None），用于推测解码。
            positions: The positions of the tokens in the sequence. Used for deterministic sampling
                to get the unique seed for each position.
                token在序列中的位置，用于确定性采样获取每个位置的唯一种子。
        """
        logits = logits_output.next_token_logits  # 获取下一个token的logits

        # Preprocess logits (custom processors and NaN handling)
        # 预处理logits（自定义处理器和NaN处理）
        logits = self._preprocess_logits(logits, sampling_info)

        if sampling_info.is_all_greedy:  # 如果所有请求都是贪心采样
            if _use_aiter:  # 如果使用aiter加速库
                batch_next_token_ids = torch.empty(  # 创建空张量存放采样结果
                    logits.shape[0], device=logits.device, dtype=torch.int32  # 形状为[批次大小]，int32类型
                )
                _aiter_greedy_sample(batch_next_token_ids, logits)  # 使用aiter贪心采样
            else:  # 不使用aiter
                batch_next_token_ids = torch.argmax(logits, -1)  # 取logits最大值索引作为采样结果
            if return_logprob:  # 如果需要返回logprob
                original_logprobs = logprobs = torch.nn.functional.log_softmax(  # 计算log_softmax得到logprob
                    logits, dim=-1  # 在词汇表维度上计算
                )
        else:  # 非全贪心采样的情况
            simple_sampling_case = (  # 判断是否为简单采样（无需top-k/top-p/min-p过滤）
                not sampling_info.need_top_p_sampling  # 不需要top-p采样
                and not sampling_info.need_top_k_sampling  # 不需要top-k采样
                and not sampling_info.need_min_p_sampling  # 不需要min-p采样
            )

            # If requested, cache original logprobs before temperature scaling.
            # 如果需要，在温度缩放前缓存原始logprob。
            if return_logprob and SGLANG_RETURN_ORIGINAL_LOGPROB:  # 如果需要返回原始logprob
                original_logprobs = torch.log_softmax(logits, dim=-1)  # 在温度缩放前计算logprob

            # In RL on-policy mode, we use log_softmax to compute logprobs to match the trainer.
            # 在RL在线策略模式下，使用log_softmax计算logprob以匹配训练器。
            logprobs_via_logsoftmax_kernel = None  # 初始化通过log_softmax计算的logprob为None
            if self.rl_on_policy_target is not None:  # 如果处于RL在线策略模式
                # TODO: use more inplace ops to save memory
                # TODO: 使用更多原地操作以节省内存
                logits_div_temperature = (  # logits除以温度
                    logits.bfloat16().div(sampling_info.temperatures).bfloat16()  # 用bfloat16精度进行温度缩放
                )
                logprobs_via_logsoftmax_kernel = torch.log_softmax(  # 对温度缩放后的logits计算log_softmax
                    logits_div_temperature, dim=-1  # 在词汇表维度上计算
                )
                del logits_div_temperature  # 删除中间变量以释放内存

            if self.use_ascend_backend:  # 如果使用昇腾后端
                # Ascend backend: sample from logits directly.
                # 昇腾后端：直接从logits采样。
                batch_next_token_ids, logprobs = self._forward_ascend_backend(  # 调用昇腾后端采样方法
                    logits, sampling_info, simple_sampling_case, return_logprob  # 传入logits、采样信息、是否简单采样、是否返回logprob
                )
            elif (  # 否则，如果是RL在线策略 + 确定性推理 + 简单采样的情况
                self.use_log_softmax_logprob  # 使用log_softmax计算logprob
                and self.enable_deterministic  # 启用了确定性推理
                and simple_sampling_case  # 是简单采样情况
            ):
                # RL on-policy path: sample from logprobs to match the trainer.
                # RL在线策略路径：从logprob采样以匹配训练器。
                batch_next_token_ids = self._sample_from_logprobs(  # 从logprob进行采样
                    logprobs_via_logsoftmax_kernel,  # 通过log_softmax计算的logprob
                    sampling_info,  # 采样信息
                    positions,  # token位置
                )
                if return_logprob and not SGLANG_RETURN_ORIGINAL_LOGPROB:  # 如果需要logprob且不需要原始logprob
                    logprobs = logprobs_via_logsoftmax_kernel  # 使用log_softmax计算的logprob
            else:  # 标准路径
                # Standard path: do softmax and sample from probs.
                # 标准路径：先softmax再从概率分布采样。
                logits.div_(sampling_info.temperatures)  # 原地除以温度进行温度缩放

                # In-place op to save memory
                # 原地操作以节省内存
                logits[:] = torch.softmax(logits, dim=-1)  # 原地softmax得到概率分布
                probs = logits  # logits现在存储的是概率值

                batch_next_token_ids = self._sample_from_probs(  # 从概率分布进行采样
                    probs, sampling_info, positions, simple_sampling_case  # 传入概率、采样信息、位置、是否简单采样
                )
                if return_logprob and not SGLANG_RETURN_ORIGINAL_LOGPROB:  # 如果需要logprob且不需要原始logprob
                    logprobs = (  # 计算logprob
                        logprobs_via_logsoftmax_kernel  # 优先使用log_softmax计算的logprob
                        if logprobs_via_logsoftmax_kernel is not None  # 如果存在
                        else torch.log(probs)  # 否则对概率取对数
                    )
                del probs  # 删除概率张量以释放内存

        # Attach logprobs to logits_output (in-place modification)
        # 将logprob附加到logits_output（原地修改）
        if return_logprob:  # 如果需要返回logprob
            if SGLANG_RETURN_ORIGINAL_LOGPROB:  # 如果需要返回原始logprob
                logprobs = original_logprobs  # 使用温度缩放前的原始logprob
            self._attach_logprobs_to_output(  # 将logprob信息附加到输出
                logits_output,  # logits输出对象
                logprobs,  # logprob张量
                top_logprobs_nums,  # top logprob数量
                token_ids_logprobs,  # 需要提取logprob的token ID列表
                sampling_info,  # 采样信息
                batch_next_token_ids,  # 采样的token ID
            )

        self._sync_token_ids_across_tp(batch_next_token_ids, sampling_info)  # 在TP组间同步token ID

        return batch_next_token_ids  # 返回采样得到的token ID

    def _sample_from_probs(  # 从概率分布采样的方法
        self,
        probs: torch.Tensor,  # 概率分布张量
        sampling_info: SamplingBatchInfo,  # 采样信息
        positions: torch.Tensor,  # token位置
        simple_sampling_case: bool,  # 是否为简单采样
    ) -> torch.Tensor:  # 返回采样得到的token ID
        """Sample from probability distribution (after softmax).
        从概率分布（softmax后）中采样。

        Used for standard sampling with flashinfer/pytorch backends.
        Handles both simple (direct multinomial) and complex (top-k/top-p/min-p) cases.
        用于flashinfer/pytorch后端的标准采样。
        处理简单（直接多项式）和复杂（top-k/top-p/min-p）两种情况。
        """
        if simple_sampling_case:  # 如果是简单采样（无需top-k/top-p/min-p过滤）
            batch_next_token_ids = sampling_from_probs_torch(  # 使用PyTorch原生操作进行采样
                probs,  # 概率分布
                sampling_seed=sampling_info.sampling_seed,  # 采样种子（用于确定性推理）
                positions=positions,  # token位置
            )
        else:  # 需要top-k/top-p/min-p过滤的复杂采样
            backend = get_global_server_args().sampling_backend  # 获取当前采样后端
            if backend == "flashinfer":  # 如果使用flashinfer后端
                assert (  # 断言flashinfer后端不支持采样种子
                    sampling_info.sampling_seed is None
                ), "Sampling seed is not supported for flashinfer backend"  # flashinfer后端不支持采样种子
                if sampling_info.need_min_p_sampling:  # 如果需要min-p采样
                    probs = top_k_renorm_prob(probs, sampling_info.top_ks)  # 先用top-k重归一化概率
                    probs = top_p_renorm_prob(probs, sampling_info.top_ps)  # 再用top-p重归一化概率
                    batch_next_token_ids = min_p_sampling_from_probs(  # 使用min-p从概率采样
                        probs, sampling_info.min_ps  # 传入概率和min-p阈值
                    )
                else:  # 不需要min-p采样，只需要top-k和top-p
                    batch_next_token_ids = top_k_top_p_sampling_from_probs(  # 使用top-k和top-p从概率采样
                        probs.contiguous(),  # 确保概率张量是连续的
                        sampling_info.top_ks,  # top-k值
                        sampling_info.top_ps,  # top-p值
                        filter_apply_order="joint",  # 联合应用top-k和top-p过滤
                    )
            elif backend == "pytorch":  # 如果使用pytorch后端
                # A slower fallback implementation with torch native operations.
                # 使用PyTorch原生操作的较慢回退实现。
                batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs_torch(  # 调用PyTorch原生top-k/top-p/min-p采样
                    probs,  # 概率分布
                    sampling_info.top_ks,  # top-k值
                    sampling_info.top_ps,  # top-p值
                    sampling_info.min_ps,  # min-p值
                    sampling_info.need_min_p_sampling,  # 是否需要min-p采样
                    sampling_info.sampling_seed,  # 采样种子
                    positions,  # token位置
                )
            else:  # 未知的采样后端
                raise ValueError(f"Invalid sampling backend: {backend}")  # 抛出无效后端错误
        return batch_next_token_ids  # 返回采样得到的token ID

    def _sample_from_logprobs(  # 从logprob采样的方法（使用Gumbel技巧）
        self,
        logprobs: torch.Tensor,  # log概率张量
        sampling_info: SamplingBatchInfo,  # 采样信息
        positions: torch.Tensor,  # token位置
    ) -> torch.Tensor:  # 返回采样得到的token ID
        """Sample from log-probabilities using the Gumbel trick.
        使用Gumbel技巧从log概率中采样。

        Used for deterministic sampling with simple cases (no top-k/top-p/min-p).
        Requires sampling_seed to be set in sampling_info.
        用于简单情况（无top-k/top-p/min-p）的确定性采样。
        要求sampling_info中设置sampling_seed。
        """
        assert (  # 断言必须设置采样种子
            sampling_info.sampling_seed is not None
        ), "sampling_seed is required for sampling from logprobs"  # 从logprob采样需要sampling_seed
        sampled_index = multinomial_with_seed(  # 使用带种子的多项式采样（Gumbel技巧）
            logprobs, sampling_info.sampling_seed, positions  # 传入logprob、种子和位置
        )
        return sampled_index.view(-1).to(torch.int32)  # 展平并转换为int32类型返回

    def _sample_from_logits(  # 从logits直接采样的方法（昇腾后端专用）
        self,
        logits: torch.Tensor,  # 温度缩放后的logits
        sampling_info: SamplingBatchInfo,  # 采样信息
        simple_sampling_case: bool,  # 是否为简单采样
    ) -> torch.Tensor:  # 返回采样得到的token ID
        """Sample from temperature-scaled logits without softmax.
        从温度缩放后的logits直接采样，无需先做softmax。

        Used for the Ascend NPU backend which handles softmax internally.
        用于昇腾NPU后端，该后端内部处理softmax。
        """
        if simple_sampling_case:  # 如果是简单采样
            probs = torch.softmax(logits, dim=-1)  # 对logits做softmax得到概率
            batch_next_token_ids = torch.multinomial(probs, num_samples=1).view(-1)  # 从概率分布中多项式采样
            return batch_next_token_ids.to(torch.int32)  # 转换为int32类型返回
        else:  # 需要top-k/top-p/min-p过滤的复杂采样
            assert (  # 断言只有昇腾后端支持从logits采样
                self.use_ascend_backend
            ), "Only ascend backend supports sampling from logits"  # 只有昇腾后端支持从logits采样
            batch_next_token_ids = top_k_top_p_min_p_sampling_from_logits_ascend(  # 调用昇腾专用的top-k/top-p/min-p采样
                logits,  # logits
                sampling_info.top_ks,  # top-k值
                sampling_info.top_ps,  # top-p值
                sampling_info.min_ps,  # min-p值
                sampling_info.need_min_p_sampling,  # 是否需要min-p采样
            )
            return batch_next_token_ids.to(torch.int32)  # 转换为int32类型返回

    def _forward_ascend_backend(  # 昇腾后端的完整采样路径
        self,
        logits: torch.Tensor,  # logits张量
        sampling_info: SamplingBatchInfo,  # 采样信息
        simple_sampling_case: bool,  # 是否为简单采样
        return_logprob: bool,  # 是否返回logprob
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:  # 返回token ID和logprob的元组
        """Handle the full Ascend backend sampling path.
        处理完整的昇腾后端采样路径。

        Ascend backend has fused kernels that handle softmax internally,
        so we sample directly from temperature-scaled logits.
        昇腾后端拥有内部处理softmax的融合内核，
        因此我们直接从温度缩放后的logits采样。

        Returns:
            A tuple of (batch_next_token_ids, logprobs). logprobs is None
            when return_logprob is False or SGLANG_RETURN_ORIGINAL_LOGPROB is set.
            返回(batch_next_token_ids, logprobs)的元组。当return_logprob为False
            或设置了SGLANG_RETURN_ORIGINAL_LOGPROB时，logprobs为None。
        """
        logits.div_(sampling_info.temperatures)  # 原地除以温度进行温度缩放
        batch_next_token_ids = self._sample_from_logits(  # 从温度缩放后的logits采样
            logits, sampling_info, simple_sampling_case  # 传入logits、采样信息、是否简单采样
        )
        logprobs = None  # 初始化logprob为None
        if return_logprob and not SGLANG_RETURN_ORIGINAL_LOGPROB:  # 如果需要返回logprob且不需要原始logprob
            logprobs = torch.log_softmax(logits, dim=-1)  # 计算log_softmax得到logprob
        return batch_next_token_ids, logprobs  # 返回采样结果和logprob

    def _attach_logprobs_to_output(  # 将logprob信息附加到输出对象
        self,
        logits_output: LogitsProcessorOutput,  # logits输出对象
        logprobs: torch.Tensor,  # logprob张量
        top_logprobs_nums: List[int],  # top logprob数量列表
        token_ids_logprobs: List[List[int]],  # 需要提取logprob的token ID列表
        sampling_info: SamplingBatchInfo,  # 采样信息
        batch_next_token_ids: torch.Tensor,  # 采样得到的token ID
    ):
        # clamp to avoid -inf values
        # 钳制以避免-inf值
        logprobs.clamp_(min=torch.finfo(logprobs.dtype).min)  # 将logprob值钳制到数据类型最小值以上

        # Attach logprobs to logits_output (in-place modification)
        # 将logprob附加到logits_output（原地修改）
        if any(x > 0 for x in top_logprobs_nums):  # 如果有任何请求需要top logprob
            (  # 计算top logprob并存入输出
                logits_output.next_token_top_logprobs_val,  # top logprob的值
                logits_output.next_token_top_logprobs_idx,  # top logprob的索引
            ) = get_top_logprobs(logprobs, top_logprobs_nums, no_copy_to_cpu=True)  # 获取top logprob，不拷贝到CPU

        if any(x is not None for x in token_ids_logprobs):  # 如果有任何请求需要指定token的logprob
            (  # 计算指定token的logprob并存入输出
                logits_output.next_token_token_ids_logprobs_val,  # 指定token的logprob值
                logits_output.next_token_token_ids_logprobs_idx,  # 指定token的索引
            ) = get_token_ids_logprobs(  # 获取指定token ID的logprob
                logprobs, token_ids_logprobs, no_copy_to_cpu=True  # 不拷贝到CPU
            )

        logits_output.next_token_logprobs = logprobs[  # 提取每个采样token对应的logprob
            torch.arange(len(batch_next_token_ids), device=sampling_info.device),  # 行索引：0到批次大小-1
            batch_next_token_ids,  # 列索引：采样得到的token ID
        ]

    def _sync_token_ids_across_tp(  # 在TP组间同步token ID
        self, batch_next_token_ids: torch.Tensor, sampling_info: SamplingBatchInfo  # 采样结果和采样信息
    ):
        if SYNC_TOKEN_IDS_ACROSS_TP or sampling_info.grammars:  # 如果需要跨TP同步或使用了语法约束
            # For performance reasons, SGLang does not sync the final token IDs across TP ranks by default.
            # This saves one all-reduce, but the correctness of this approach depends on the determinism of several operators:
            # the last all-reduce, the last lm_head matmul, and all sampling kernels.
            # These kernels are deterministic in most cases, but there are some rare instances where they are not deterministic.
            # In such cases, enable this env variable to prevent hanging due to TP ranks becoming desynchronized.
            # When using xgrammar, this becomes more likely so we also do the sync when grammar is used.
            # 出于性能考虑，SGLang默认不在TP组间同步最终的token ID。
            # 这节省了一次all-reduce，但此方法的正确性取决于多个算子的确定性：
            # 最后一次all-reduce、最后一次lm_head矩阵乘法以及所有采样内核。
            # 这些内核在大多数情况下是确定性的，但在少数情况下可能不确定。
            # 在这种情况下，启用此环境变量可防止TP组因不同步而挂起。
            # 使用xgrammar时，这种情况更可能发生，因此在使用语法约束时也会进行同步。

            torch.distributed.all_reduce(  # 执行all-reduce操作
                batch_next_token_ids,  # 需要同步的token ID张量
                op=dist.ReduceOp.MIN,  # 使用MIN操作（因为正确的token ID在所有rank上相同，MIN可确保一致性）
                group=self.tp_sync_group,  # TP同步通信组
            )

    def compute_logprobs_only(  # 仅计算logprob而不执行采样的方法
        self,
        logits_output: LogitsProcessorOutput,  # logits输出对象
        sampling_info: SamplingBatchInfo,  # 采样信息
        return_logprob: bool,  # 是否返回logprob
        top_logprobs_nums: List[int],  # top logprob数量列表
        token_ids_logprobs: List[List[int]],  # 需要提取logprob的token ID列表
    ) -> None:
        """
        Compute logprobs for requested token IDs without performing sampling.
        仅计算请求token ID的logprob，不执行采样。

        Optimized for prefill-only scoring requests that need token probabilities
        but don't require next token generation.
        针对仅需token概率而无需生成下一个token的预填充评分请求进行优化。
        """

        if logits_output.next_token_logits is None:  # 如果没有可用的logits
            logger.warning("No logits available for logprob computation")  # 记录警告日志
            return  # 直接返回

        # Check if any requests actually need logprobs computation
        # 检查是否有请求确实需要logprob计算
        needs_token_ids_logprobs = any(  # 是否需要指定token的logprob
            token_ids is not None and len(token_ids) > 0  # token_ids非空
            for token_ids in token_ids_logprobs  # 遍历每个请求的token_ids
        )
        needs_top_logprobs = any(x > 0 for x in top_logprobs_nums)  # 是否需要top logprob

        if not (needs_token_ids_logprobs or needs_top_logprobs):  # 如果都不需要
            return  # 直接返回

        # Preprocess logits (custom processors and NaN handling)
        # 预处理logits（自定义处理器和NaN处理）
        logits = self._preprocess_logits(logits_output.next_token_logits, sampling_info)  # 应用预处理

        # Compute logprobs
        # 计算logprob
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)  # 使用log_softmax计算logprob

        # Handle top logprobs if requested
        # 如果请求了top logprob则处理
        if needs_top_logprobs:  # 如果需要top logprob
            (  # 计算top logprob并存入输出
                logits_output.next_token_top_logprobs_val,  # top logprob的值
                logits_output.next_token_top_logprobs_idx,  # top logprob的索引
            ) = get_top_logprobs(logprobs, top_logprobs_nums, no_copy_to_cpu=True)  # 获取top logprob

        # Handle token_ids logprobs if requested
        # 如果请求了指定token的logprob则处理
        if needs_token_ids_logprobs:  # 如果需要指定token的logprob
            (  # 计算指定token的logprob并存入输出
                logits_output.next_token_token_ids_logprobs_val,  # 指定token的logprob值
                logits_output.next_token_token_ids_logprobs_idx,  # 指定token的索引
            ) = get_token_ids_logprobs_batch_optimized(logprobs, token_ids_logprobs)  # 使用批量优化的方法获取


def register_sampler_backend(backend: str, factory: Callable[[], "Sampler"]) -> None:  # 注册自定义采样器后端
    """Register a custom sampler factory for a backend string."""
    """为后端字符串注册自定义采样器工厂。"""

    if not backend:  # 如果后端名称为空
        raise ValueError("backend must be a non-empty string")  # 抛出值错误

    from sglang.srt.server_args import SAMPLING_BACKEND_CHOICES  # 导入采样后端选项集合

    if backend in _CUSTOM_SAMPLER_FACTORIES:  # 如果该后端已注册过
        logger.warning("Overriding existing sampler factory for backend '%s'", backend)  # 记录覆盖警告
    SAMPLING_BACKEND_CHOICES.add(backend)  # 将新后端添加到选项集合
    _CUSTOM_SAMPLER_FACTORIES[backend] = factory  # 注册工厂函数


def create_sampler(backend: Optional[str] = None) -> "Sampler":  # 创建采样器实例
    """Create a sampler honoring custom backend registrations."""
    """创建采样器，遵循自定义后端注册。"""

    server_args = get_global_server_args()  # 获取全局服务器参数
    backend = backend or (server_args.sampling_backend if server_args else None)  # 确定使用的后端

    if backend in _CUSTOM_SAMPLER_FACTORIES:  # 如果是自定义注册的后端
        sampler = _CUSTOM_SAMPLER_FACTORIES[backend]()  # 调用工厂函数创建采样器
        if not isinstance(sampler, Sampler):  # 检查返回的是否为Sampler实例
            raise TypeError(  # 抛出类型错误
                f"Custom sampler factory for backend '{backend}' must return a Sampler"  # 工厂函数必须返回Sampler实例
            )
        return sampler  # 返回自定义采样器

    if backend is None or backend in _BUILT_IN_SAMPLING_BACKENDS:  # 如果是内置后端或未指定后端
        return Sampler()  # 返回默认Sampler实例

    raise ValueError(  # 未知后端
        f"Unknown sampling backend '{backend}'. Register it via register_sampler_backend()."  # 提示通过register_sampler_backend注册
    )


def top_k_top_p_min_p_sampling_from_probs_torch(  # 使用PyTorch原生操作实现top-k/top-p/min-p采样
    probs: torch.Tensor,  # 概率分布张量
    top_ks: torch.Tensor,  # top-k值张量
    top_ps: torch.Tensor,  # top-p值张量
    min_ps: torch.Tensor,  # min-p值张量
    need_min_p_sampling: bool,  # 是否需要min-p采样
    sampling_seed: Optional[torch.Tensor],  # 采样种子（可选）
    positions: torch.Tensor,  # token位置
):
    """
    A top-k, top-p and min-p sampling implementation with native pytorch operations.
    When sampling_seed is not None, deterministic inference will be enabled, it will sample
    with the sampling_seed of each request.
    使用PyTorch原生操作实现的top-k、top-p和min-p采样。
    当sampling_seed不为None时，将启用确定性推理，使用每个请求的采样种子进行采样。
    """
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)  # 按概率降序排序，获取排序值和原始索引
    probs_sum = torch.cumsum(probs_sort, dim=-1)  # 计算累积概率和
    probs_sort[  # 将超出top-k范围的概率置零
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1)  # 列索引：0到词汇表大小-1
        >= top_ks.view(-1, 1)  # 与top-k值比较
    ] = 0.0  # 超出top-k范围的位置概率置零
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0  # 将超出top-p范围的概率置零（累积和减去当前值即之前所有值的和）

    if need_min_p_sampling:  # 如果需要min-p采样
        # TODO: probs_sort should be re-normalized for the use of multinomial_with_seed
        # TODO: 使用multinomial_with_seed时probs_sort应该重新归一化
        assert (  # 断言：使用采样种子时min-p采样结果不正确
            sampling_seed is None
        ), "With sampling seed, multinomial_with_seed will provide wrong results"  # 带采样种子时multinomial_with_seed会给出错误结果
        min_p_thresholds = probs_sort[:, 0] * min_ps  # 计算min-p阈值：最大概率乘以min-p值
        probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0  # 将低于min-p阈值的概率置零

    if sampling_seed is None:  # 如果没有采样种子（非确定性推理）
        sampled_index = torch.multinomial(probs_sort, num_samples=1)  # 使用标准多项式采样
    else:  # 有采样种子（确定性推理）
        # NOTE: when using top-k/top-p/min-p sampling, we need to modify probs before we
        # apply log to get logprobs. Therefore, we cannot use log_softmax directly.
        # For now, we use log to the modified probs to get logprobs, but for numerical
        # stability, we'd better come up with a solution to use log_softmax.
        # 注意：使用top-k/top-p/min-p采样时，需要在取log获取logprob之前修改概率。
        # 因此不能直接使用log_softmax。
        # 目前我们对修改后的概率取log，但为了数值稳定性，最好想办法使用log_softmax。
        logprobs = probs_sort.to(torch.float64)  # Using float64 for numerical stability  # 转为float64以保证数值稳定性
        del probs_sort  # 删除概率排序张量以释放内存
        logprobs.log_()  # 原地取对数得到logprob
        sampled_index = multinomial_with_seed(logprobs, sampling_seed, positions)  # 使用带种子的多项式采样

    # int32 range is enough to represent the token ids
    # int32范围足以表示token ID
    probs_idx = probs_idx.to(torch.int32)  # 将索引转换为int32
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)  # 根据采样索引收集原始token ID并展平
    return batch_next_token_ids  # 返回采样得到的token ID


def top_k_top_p_min_p_sampling_from_logits_ascend(  # 昇腾NPU专用的top-k/top-p/min-p采样实现
    logits: torch.Tensor,  # 温度缩放后的logits
    top_ks: torch.Tensor,  # top-k值张量
    top_ps: torch.Tensor,  # top-p值张量
    min_ps: torch.Tensor,  # min-p值张量
    need_min_p_sampling: bool,  # 是否需要min-p采样
):
    """A top-k, top-p and min-p sampling implementation for ascend npu with torch_npu interface.
    使用torch_npu接口为昇腾NPU实现的top-k、top-p和min-p采样。

    Takes temperature-scaled logits as input (softmax is applied internally).
    以温度缩放后的logits作为输入（softmax在内部应用）。
    """
    # torch_npu.npu_top_k_top_p requires top_k value range in [1, 1024]
    # torch_npu.npu_top_k_top_p要求top_k值在[1, 1024]范围内
    if hasattr(torch_npu, "npu_top_k_top_p") and torch.all(  # 如果支持NPU专用算子且top_k值在合法范围内
        (top_ks <= 1024) & (top_ks >= 1)
    ):
        logits_top_k_top_p = torch_npu.npu_top_k_top_p(logits, top_ps, top_ks)  # 使用NPU融合算子做top-k和top-p过滤
        probs_top_k_top_p = logits_top_k_top_p.softmax(dim=-1)  # 对过滤后的logits做softmax得到概率

        if need_min_p_sampling:  # 如果需要min-p采样
            min_p_thresholds = probs_top_k_top_p.max(dim=-1) * min_ps  # 计算min-p阈值：最大概率乘以min-p值
            min_p_mask = probs_top_k_top_p < min_p_thresholds.view(-1, 1)  # 生成低于min-p阈值的掩码
            probs_top_k_top_p.masked_fill_(min_p_mask, 0.0)  # 将低于阈值的概率置零

        batch_next_token_ids = torch.multinomial(probs_top_k_top_p, num_samples=1)  # 从过滤后的概率分布中采样
    else:  # top_k超出范围或不支持NPU融合算子，使用回退实现
        probs = torch.softmax(logits, dim=-1)  # 对logits做softmax得到概率
        probs_sort, probs_idx = probs.sort(dim=-1, descending=True)  # 按概率降序排序

        # when top_k is -1 (in which sglang turns it to TOP_K_ALL), make it explicitly equal to logit's size
        # 当top_k为-1时（SGLang将其转为TOP_K_ALL），显式设为等于logits的大小
        topk_all_mask = top_ks == TOP_K_ALL  # 生成top_k为TOP_K_ALL的掩码
        top_ks.masked_fill_(topk_all_mask, probs.shape[1])  # 将TOP_K_ALL替换为词汇表大小
        top_k_mask = torch.arange(0, probs.shape[-1], device=probs.device).view(  # 生成top-k掩码
            1, -1
        ) >= top_ks.view(-1, 1)  # 列索引大于等于top_k值的位置
        probs_sort.masked_fill_(top_k_mask, 0.0)  # 将超出top-k范围的概率置零

        probs_sum = torch.cumsum(probs_sort, dim=-1)  # 计算累积概率和
        top_p_mask = probs_sum - probs_sort > top_ps.view(-1, 1)  # 生成top-p掩码：之前累积和超过top_p的位置
        probs_sort.masked_fill_(top_p_mask, 0.0)  # 将超出top-p范围的概率置零

        if need_min_p_sampling:  # 如果需要min-p采样
            min_p_thresholds = probs_sort[:, 0] * min_ps  # 计算min-p阈值：最大概率乘以min-p值
            min_p_mask = probs_sort < min_p_thresholds.view(-1, 1)  # 生成低于min-p阈值的掩码
            probs_sort.masked_fill_(min_p_mask, 0.0)  # 将低于阈值的概率置零

        sampled_index = torch.multinomial(probs_sort, num_samples=1)  # 从过滤后的概率分布中采样
        probs_idx = probs_idx.to(torch.int32)  # 将索引转换为int32
        batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index)  # 根据采样索引收集原始token ID

    return batch_next_token_ids.view(-1)  # 展平并返回采样得到的token ID


@torch.compile(dynamic=True)  # 使用torch.compile编译优化，启用动态形状
def multinomial_with_seed(  # 带种子的多项式采样（使用Gumbel技巧实现确定性）
    logprobs: torch.Tensor, seed: torch.Tensor, positions: torch.Tensor  # log概率、种子、位置
) -> torch.Tensor:  # 返回采样索引
    """
    Samples n elements from an input tensor `inputs` of shape (n, m) using
    a unique random seed for each row. This is a deterministic batched alternative to
    `torch.multinomial`.
    使用每行唯一的随机种子，从形状为(n, m)的输入张量`inputs`中采样n个元素。
    这是`torch.multinomial`的确定性批量替代方案。

    Args:
        inputs: A float tensor of shape (n, m) representing n categorical
                distributions with m categories each. The values are treated
                as weights and do not need to sum to 1.
                形状为(n, m)的浮点张量，表示n个具有m个类别的分类分布。
                值被视为权重，不需要总和为1。
        seed:   An integer tensor of shape (n,) containing the random seed
                for each corresponding row in `inputs`.
                形状为(n,)的整数张量，包含`inputs`中每行对应的随机种子。
        positions: The positions of the tokens in the sequence. Used for deterministic sampling
                to get the unique seed for each position.
                token在序列中的位置，用于确定性采样获取每个位置的唯一种子。

    Returns:
        A tensor of shape (n,) where the i-th element is an index sampled
        from the distribution in `inputs[i]` using `seed[i]`.
        形状为(n,)的张量，其中第i个元素是使用`seed[i]`从`inputs[i]`分布中采样的索引。
    """
    n, m = logprobs.shape  # 获取批次大小和词汇表大小
    seed = seed.to(torch.uint64)  # 将种子转换为uint64类型
    col_indices = torch.arange(m, device=logprobs.device)  # 创建列索引：0到词汇表大小-1
    hashed = murmur_hash32(seed, positions, col_indices)  # 使用murmur哈希生成确定性随机值

    # NOTE (sehoon): it is critical to keep gumbel noise calculation in float64 to avoid numerical instability.
    # keeping logprobs in float64 is less critical, but we found it's still safer to keep it in float64.
    # 注意(sehoon)：将gumbel噪声计算保持在float64对避免数值不稳定性至关重要。
    # 将logprob保持在float64不那么关键，但我们发现保持在float64仍然更安全。
    x = hashed.to(torch.float64) / torch.iinfo(torch.uint32).max  # 将哈希值归一化到[0,1]范围

    # x is a uniform sample in [0, 1]. get gumbel noise from it.
    # which is equivalent to -log(-log(x))
    # keep everything in in-place operations to avoid unnecessary memory allocations.
    # x是[0,1]上的均匀采样。从中生成gumbel噪声。
    # 等价于-log(-log(x))
    # 所有操作保持原地操作以避免不必要的内存分配。
    x.log_().clamp_(min=torch.finfo(x.dtype).min).neg_()  # -log(x)  # 先取log，钳制避免log(0)，再取反
    x.log_().neg_()  # -log(-log(x)) == gumbel noise  # 再取log后取反，得到gumbel噪声

    # add gumbel noise to logprobs
    # 将gumbel噪声加到logprob上
    x.add_(logprobs.to(torch.float64))  # 原地加上logprob

    return torch.argmax(x, dim=1, keepdim=True)  # 取argmax得到采样索引，保持维度


def sampling_from_probs_torch(  # 使用PyTorch原生操作进行简单采样（无top-k/top-p/min-p过滤）
    probs: torch.Tensor,  # 概率分布张量
    sampling_seed: Optional[torch.Tensor] = None,  # 采样种子（可选）
    positions: Optional[torch.Tensor] = None,  # token位置（可选）
):
    """A sampling implementation with native pytorch operations, without
    top-k, top-p, or min-p filtering.
    使用PyTorch原生操作的采样实现，不含top-k、top-p或min-p过滤。

    Note: For deterministic sampling from logprobs, use Sampler._sample_from_logprobs instead.
    注意：对于从logprob的确定性采样，请改用Sampler._sample_from_logprobs。
    """
    if sampling_seed is None:  # 如果没有采样种子
        sampled_index = torch.multinomial(probs, num_samples=1)  # 使用标准多项式采样
    else:  # 有采样种子
        # Deterministic sampling: convert probs to logprobs and use gumbel trick
        # 确定性采样：将概率转为logprob并使用gumbel技巧
        sampled_index = multinomial_with_seed(  # 使用带种子的多项式采样
            torch.log(probs), sampling_seed, positions  # 对概率取对数后传入
        )
    batch_next_token_ids = sampled_index.view(-1).to(torch.int32)  # 展平并转换为int32
    return batch_next_token_ids  # 返回采样得到的token ID


def top_p_normalize_probs_torch(  # 使用PyTorch原生操作进行top-p归一化
    probs: torch.Tensor,  # 概率分布张量
    top_ps: torch.Tensor,  # top-p值张量
):
    # See also top_k_top_p_min_p_sampling_from_probs_torch
    # 另见top_k_top_p_min_p_sampling_from_probs_torch
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)  # 按概率降序排序
    probs_sum = torch.cumsum(probs_sort, dim=-1)  # 计算累积概率和
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0  # 将超出top-p范围的概率置零
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))  # 原地归一化：除以剩余概率之和
    return torch.zeros_like(probs_sort).scatter_(-1, probs_idx, probs_sort)  # 将排序后的概率散布回原始位置


def get_token_ids_logprobs_batch_optimized(  # 批量优化的token ID logprob提取函数
    logprobs: torch.Tensor,  # logprob张量 [批次大小, 词汇表大小]
    token_ids_logprobs: List[List[int]],  # 需要提取logprob的token ID列表
) -> Tuple[List, List]:  # 返回logprob值列表和索引列表
    """
    Vectorized batch processing for token ID logprobs extraction.
    向量化的批量token ID logprob提取处理。

    Uses a single GPU kernel call for the entire batch instead of multiple
    separate calls, significantly improving performance for large batches.
    对整个批次使用单次GPU内核调用而非多次独立调用，显著提升大批量的性能。

    Args:
        logprobs: Log probabilities tensor [batch_size, vocab_size]
            log概率张量 [批次大小, 词汇表大小]
        token_ids_logprobs: List of token IDs to extract logprobs for
            需要提取logprob的token ID列表

    Example:
        # Input: batch_size=3, vocab_size=5
        logprobs = torch.tensor([
            [-1.2, -2.1, -0.8, -3.0, -1.5],  # batch 0
            [-0.5, -1.8, -2.2, -1.1, -2.7],  # batch 1
            [-2.0, -0.9, -1.4, -2.8, -1.6],  # batch 2
        ])
        token_ids_logprobs = [[1, 3], [2], [0, 2, 4]]

        # Output:
        # values = [tensor([-2.1, -3.0]), tensor([-2.2]), tensor([-2.0, -1.4, -1.6])]
        # indices = [[1, 3], [2], [0, 2, 4]]
    """
    batch_size = len(token_ids_logprobs)  # 获取批次大小
    device = logprobs.device  # 获取设备类型

    # Step 1: Calculate lengths for each request, treating None as empty list
    # Example: [[1, 3], [2], [0, 2, 4]] -> token_lengths = tensor([2, 1, 3])
    # 步骤1：计算每个请求的token ID数量，None视为空列表
    # 示例：[[1, 3], [2], [0, 2, 4]] -> token_lengths = tensor([2, 1, 3])
    token_lengths = torch.tensor(  # 创建每个请求的token ID数量张量
        [len(token_ids or []) for token_ids in token_ids_logprobs], device=device  # None视为空列表
    )
    total_tokens = int(token_lengths.sum().item())  # 2 + 1 + 3 = 6  # 计算总token数

    # Handle edge case where no tokens are requested
    # 处理没有请求任何token的边界情况
    if total_tokens == 0:  # 如果总token数为0
        return [logprobs.new_empty(0) for _ in token_ids_logprobs], [  # 返回空张量列表
            [] for _ in token_ids_logprobs  # 返回空列表
        ]

    # Step 2: Build flattened indices using torch operations
    # Example: row_indices = [0, 0, 1, 2, 2, 2] (batch indices repeated by their lengths)
    # 步骤2：使用torch操作构建扁平化索引
    # 示例：row_indices = [0, 0, 1, 2, 2, 2]（按长度重复的批次索引）
    row_indices = torch.repeat_interleave(  # 按每个请求的token数量重复行索引
        torch.arange(batch_size, device=device), token_lengths  # 行索引和重复次数
    )
    # Example: col_indices = [1, 3, 2, 0, 2, 4] (flattened token IDs from all requests)
    # 示例：col_indices = [1, 3, 2, 0, 2, 4]（所有请求的扁平化token ID）
    col_indices = torch.tensor(  # 创建扁平化的列索引（token ID）
        [
            token_id  # 每个token ID
            for token_ids in token_ids_logprobs  # 遍历每个请求的token ID列表
            for token_id in (token_ids or [])  # 遍历token ID列表中的每个ID
        ],
        device=device,  # 设备
        dtype=torch.long,  # long类型
    )

    # Step 3: Single vectorized gather operation
    # Example: logprobs[row_indices, col_indices] -> [-2.1, -3.0, -2.2, -2.0, -1.4, -1.6]
    # 步骤3：单次向量化gather操作
    # 示例：logprobs[row_indices, col_indices] -> [-2.1, -3.0, -2.2, -2.0, -1.4, -1.6]
    gathered_logprobs = logprobs[row_indices, col_indices]  # 一次提取所有请求的logprob

    # Step 4: Split results back per request using torch operations
    # Example: split tensor [6] into chunks of sizes [2, 1, 3] -> [tensor(2), tensor(1), tensor(3)]
    # 步骤4：使用torch操作按请求分割结果
    # 示例：将张量[6]按大小[2, 1, 3]分块 -> [tensor(2), tensor(1), tensor(3)]
    split_logprobs = torch.split_with_sizes(  # 按各请求的token数量分割
        gathered_logprobs, token_lengths.tolist(), dim=0  # 传入总logprob和各请求长度
    )

    # Step 5: Format output to match expected return structure
    # Example: Convert split tensors back to list format with proper empty handling
    # i=0: [1,3] -> append split_logprobs[0] and [1,3]
    # i=1: [2] -> append split_logprobs[1] and [2]
    # i=2: [0,2,4] -> append split_logprobs[2] and [0,2,4]
    # 步骤5：格式化输出以匹配预期的返回结构
    # 示例：将分割后的张量转换回列表格式，正确处理空值
    # i=0: [1,3] -> 附加split_logprobs[0]和[1,3]
    # i=1: [2] -> 附加split_logprobs[1]和[2]
    # i=2: [0,2,4] -> 附加split_logprobs[2]和[0,2,4]
    output_token_ids_logprobs_val = []  # 初始化输出logprob值列表
    output_token_ids_logprobs_idx = []  # 初始化输出token ID索引列表

    for i, token_ids in enumerate(token_ids_logprobs):  # 遍历每个请求
        if token_ids is not None and len(token_ids) > 0:  # 如果token_ids非空
            output_token_ids_logprobs_val.append(split_logprobs[i])  # 附加该请求的logprob值
            output_token_ids_logprobs_idx.append(token_ids)  # 附加该请求的token ID
        else:  # token_ids为空
            output_token_ids_logprobs_val.append(logprobs.new_empty(0))  # 附加空张量
            output_token_ids_logprobs_idx.append([])  # 附加空列表

    return output_token_ids_logprobs_val, output_token_ids_logprobs_idx  # 返回logprob值和索引


def apply_custom_logit_processor(  # 应用自定义logit处理器的函数
    logits: torch.Tensor,  # logits张量
    sampling_batch_info: SamplingBatchInfo,  # 采样批量信息
    num_tokens_in_batch: int = 1,  # 每个批次中的token数量，默认1（推测解码时可能大于1）
):
    """Apply custom logit processors to the logits.
    This function will modify the logits in-place.
    num_tokens_in_batch is needed to support spec decoding, where each batch can contain multiple
    tokens. By default, we assume each batch contains only 1 token.
    将自定义logit处理器应用到logits上。
    此函数会原地修改logits。
    num_tokens_in_batch用于支持推测解码，其中每个批次可能包含多个token。默认每个批次仅含1个token。
    """

    assert logits.shape[0] == len(sampling_batch_info) * num_tokens_in_batch, (  # 断言logits批次大小与采样信息匹配
        f"The batch size of logits ({logits.shape[0]}) does not match the batch size of "  # logits的批次大小
        f"sampling_batch_info ({len(sampling_batch_info)}) x num_tokens_in_batch "  # 采样信息的批次大小
        f"({num_tokens_in_batch})"  # 每批token数
    )

    for _, (  # 遍历每个自定义logit处理器
        processor,  # 处理器函数
        batch_mask,  # 批次掩码，标记哪些请求使用该处理器
    ) in sampling_batch_info.custom_logit_processor.items():
        # Get the batch indices that need to be processed
        # 获取需要处理的批次索引
        batch_indices = batch_mask.nonzero(as_tuple=True)[0]  # 获取掩码中非零位置的索引

        assert batch_mask.shape[0] == len(sampling_batch_info), (  # 断言掩码大小与采样信息匹配
            f"The number of batch mask ({batch_mask.shape[0]}) does not match the number of "  # 掩码数量
            f"sampling_batch_info ({len(sampling_batch_info)})"  # 采样信息数量
        )
        batch_mask = torch.repeat_interleave(batch_mask, num_tokens_in_batch)  # 按每批token数重复掩码

        # Apply the processor to the logits
        # 将处理器应用到logits上
        logits[batch_mask] = processor(  # 对掩码选中的logits应用处理器
            logits[batch_mask],  # 被选中的logits
            [sampling_batch_info.custom_params[i] for i in batch_indices],  # 对应的自定义参数
        )

        logger.debug(  # 记录调试日志
            f"Custom logit processor {processor.__class__.__name__} is applied."  # 打印已应用的处理器名称
        )
