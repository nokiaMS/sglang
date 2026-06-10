# 文件说明：HashTopK 模块的实现，用于基于哈希的混合专家(MoE)路由选择。
# 该模块通过 token ID 查找表（tid2eid）将 token 映射到专家，而非传统的门控网络评分。
# 支持融合共享专家、DeepEP 水填充（waterfill）负载均衡以及融合 JIT 内核加速。
from __future__ import annotations  # 启用延迟类型注解评估

import logging  # 导入日志模块
from typing import Optional, Tuple  # 导入类型提示

import torch  # 导入PyTorch
from torch import nn  # 导入神经网络模块

from sglang.srt.environ import envs  # 导入环境变量配置
from sglang.srt.eplb.expert_distribution import (  # 导入专家分布记录器
    get_global_expert_distribution_recorder,
)
from sglang.srt.eplb.expert_location_dispatch import (  # 导入专家位置调度信息
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
)
from sglang.srt.layers.moe.topk import (  # 导入TopK输出和掩码工具
    StandardTopKOutput,
    _mask_topk_ids_padded_region,
)
from sglang.srt.utils import is_hip  # 导入HIP平台检测工具

logger = logging.getLogger(__name__)  # 创建日志记录器


class HashTopK(nn.Module):  # 基于哈希的TopK路由模块
    def __init__(  # 初始化HashTopK模块
        self,
        topk,  # 每个token选择的专家数量
        num_experts,  # 专家总数
        num_fused_shared_experts,  # 融合共享专家数量
        vocab_size,  # 词表大小
        scoring_func="sqrtsoftplus",  # 评分函数类型，默认为sqrtsoftplus
        routed_scaling_factor=1.5,  # 路由缩放因子
        apply_routed_scaling_factor_on_output=False,  # 是否在输出上应用路由缩放因子
    ):
        super().__init__()  # 调用父类初始化
        self.layer_id = None  # 层ID，后续设置
        from sglang.srt.server_args import get_global_server_args  # 导入服务器参数获取函数

        self.enable_deepep_waterfill = (  # 是否启用DeepEP水填充负载均衡
            num_fused_shared_experts > 0  # 需要有融合共享专家
            and get_global_server_args().enable_deepep_waterfill  # 且服务器参数启用
        )
        self.deepep_waterfill_balancer = None  # DeepEP水填充均衡器，后续初始化

        if self.enable_deepep_waterfill:  # 如果启用DeepEP水填充
            # Waterfill appends the shared expert after EPLB maps routed IDs.  # 水填充在EPLB映射路由ID后追加共享专家
            topk -= num_fused_shared_experts  # 从topk中减去融合共享专家数量
            num_fused_shared_experts = 0  # 将融合共享专家数置零

        self.num_experts = num_experts  # 保存专家总数
        self.topk = topk  # 保存topk值
        self.routed_scaling_factor = routed_scaling_factor  # 保存路由缩放因子
        self.num_fused_shared_experts = num_fused_shared_experts  # 保存融合共享专家数
        self.score_func = scoring_func  # 保存评分函数
        self.tid2eid = nn.Parameter(  # token ID到专家ID的查找表参数
            torch.empty(vocab_size, topk - num_fused_shared_experts, dtype=torch.int32),  # 形状为(vocab_size, topk-融合共享专家数)
            requires_grad=False,  # 不需要梯度
        )
        self._init_default_tid2eid()  # 初始化默认的token到专家映射

        assert not apply_routed_scaling_factor_on_output, "not implemented"  # 断言不支持在输出上应用路由缩放因子

    def _init_default_tid2eid(self) -> None:  # 初始化默认的token到专家映射表
        topk = self.tid2eid.shape[1]  # 获取topk维度
        if topk == 0:  # 如果topk为0
            return  # 直接返回

        # DummyModelLoader only initializes floating tensors, so keep this int  # DummyModelLoader只初始化浮点张量，所以保持这个整型
        # lookup table valid until real checkpoints overwrite it.  # 查找表在真实检查点覆写前保持有效
        token_ids = torch.arange(  # 生成token ID序列
            self.tid2eid.shape[0], dtype=self.tid2eid.dtype, device=self.tid2eid.device  # 与查找表相同的dtype和device
        ).unsqueeze(1)  # 增加维度用于广播
        expert_offsets = torch.arange(  # 生成专家偏移量序列
            topk, dtype=self.tid2eid.dtype, device=self.tid2eid.device  # 与查找表相同的dtype和device
        ).unsqueeze(0)  # 增加维度用于广播
        tid2eid = (token_ids + expert_offsets) % self.num_experts  # 通过取模生成循环映射
        with torch.no_grad():  # 在无梯度上下文中
            self.tid2eid.copy_(tid2eid.to(self.tid2eid.dtype))  # 将计算结果复制到参数中

    def empty_topk_output(self, device: torch.device):  # 生成空的TopK输出（用于零token情况）
        topk = self.topk - self.num_fused_shared_experts  # 计算实际topk值（去除融合共享专家）
        topk_weights = torch.empty((0, topk), dtype=torch.float32, device=device)  # 创建空权重张量
        topk_ids = torch.full((0, topk), -1, dtype=torch.int32, device=device)  # 创建填充-1的ID张量
        router_logits = torch.empty((0, topk), dtype=torch.float32, device=device)  # 创建空路由logits张量
        return self._apply_deepep_waterfill(  # 应用DeepEP水填充并返回
            StandardTopKOutput(topk_weights, topk_ids, router_logits),  # 构造标准TopK输出
            num_tokens=0,  # token数为0
        )

    def _apply_deepep_waterfill(  # 应用DeepEP水填充负载均衡
        self, topk_output: StandardTopKOutput, num_tokens: int  # TopK输出和token数量
    ) -> StandardTopKOutput:
        if self.enable_deepep_waterfill and self.deepep_waterfill_balancer is None:  # 如果启用了水填充但均衡器未初始化
            raise RuntimeError(  # 抛出运行时错误
                "DeepEP waterfill HashTopK must be prepared by ModelRunner before forward."  # DeepEP水填充HashTopK必须在forward前由ModelRunner准备
            )
        if self.deepep_waterfill_balancer is None:  # 如果均衡器为None（未启用水填充）
            return topk_output  # 直接返回原始输出
        return self.deepep_waterfill_balancer.expand_topk(topk_output, num_tokens)  # 使用均衡器扩展TopK输出

    def _forward_torch(  # PyTorch实现的forward逻辑
        self, router_logits: torch.Tensor, input_ids: torch.Tensor  # 路由logits和输入token ID
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.score_func == "softmax":  # 如果评分函数为softmax
            scores = router_logits.softmax(dim=-1)  # 对最后一维做softmax
        elif self.score_func == "sigmoid":  # 如果评分函数为sigmoid
            scores = router_logits.sigmoid()  # 对logits做sigmoid
        else:  # 默认使用sqrtsoftplus
            scores = torch.nn.functional.softplus(router_logits).sqrt()  # 先softplus再开方

        num_token = scores.shape[0]  # 获取token数量

        topk_ids = torch.zeros(  # 初始化topk ID张量
            (num_token, self.topk), dtype=torch.int32, device=scores.device  # 形状为(num_token, topk)
        )
        topk_weights = torch.zeros(  # 初始化topk权重张量
            (num_token, self.topk), dtype=scores.dtype, device=scores.device  # 形状为(num_token, topk)
        )

        if self.num_fused_shared_experts == 1:  # 如果有1个融合共享专家
            topk_ids[:, :-1] = self.tid2eid[input_ids]  # 查找表获取路由专家ID（不含最后一位）
            topk_weights[:, :-1] = scores.gather(1, topk_ids[:, :-1])  # 根据ID收集对应权重

            if self.score_func != "softmax":  # 如果评分函数不是softmax，需要归一化
                topk_weights[:, :-1] /= topk_weights[:, :-1].sum(dim=-1, keepdim=True)  # 归一化路由专家权重

            topk_ids[:, -1] = torch.randint(  # 为共享专家生成ID
                low=self.num_experts,  # 下界为专家总数
                high=self.num_experts + self.num_fused_shared_experts,  # 上界为专家总数+融合共享专家数
                size=(num_token,),  # 大小为token数量
                dtype=topk_ids.dtype,  # 与topk_ids同类型
                device=topk_ids.device,  # 与topk_ids同设备
            )

            topk_weights[:, -1] = (  # 设置共享专家权重
                topk_weights[:, :-1].sum(dim=-1) / self.routed_scaling_factor  # 路由专家权重之和除以缩放因子
            )
        else:  # 没有融合共享专家的情况
            topk_ids[:, :] = self.tid2eid[input_ids]  # 查找表获取所有专家ID
            topk_weights[:, :] = scores.gather(1, topk_ids[:, :])  # 根据ID收集所有权重
            if self.score_func != "softmax":  # 如果评分函数不是softmax，需要归一化
                topk_weights[:, :] /= topk_weights[:, :].sum(dim=-1, keepdim=True)  # 归一化所有权重

        return topk_weights, topk_ids  # 返回权重和ID

    def forward(  # 前向传播主入口
        self,
        hidden_states: torch.Tensor,  # 隐藏状态张量
        router_logits: torch.Tensor,  # 路由logits张量
        input_ids: torch.Tensor,  # 输入token ID张量
        num_token_non_padded: Optional[torch.Tensor] = None,  # 非填充token数量
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,  # 专家位置调度信息
    ):
        assert (  # 断言各输入的第一维大小一致
            input_ids.shape[0] == hidden_states.shape[0] == router_logits.shape[0]
        ), f"{input_ids.shape=} {hidden_states.shape=} {router_logits.shape=}"

        if envs.SGLANG_OPT_USE_FUSED_HASH_TOPK.get():  # 如果启用了融合HashTopK内核
            from sglang.jit_kernel.dsv4 import hash_topk  # 导入融合内核

            topk_weights, topk_ids = hash_topk(  # 使用融合内核计算
                router_logits=router_logits,  # 路由logits
                input_ids=input_ids,  # 输入token ID
                tid2eid=self.tid2eid,  # token到专家映射表
                num_fused_shared_experts=self.num_fused_shared_experts,  # 融合共享专家数
                routed_scaling_factor=self.routed_scaling_factor,  # 路由缩放因子
                scoring_func=self.score_func,  # 评分函数
            )
        else:  # 否则使用PyTorch实现
            topk_weights, topk_ids = self._forward_torch(router_logits, input_ids)  # 调用PyTorch实现

        if is_hip():  # 如果是HIP平台（AMD GPU）
            topk_weights = topk_weights.to(torch.float32)  # 将权重转为float32

        topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)  # 逻辑ID转物理ID
        _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)  # 掩码填充区域的topk ID
        get_global_expert_distribution_recorder().on_select_experts(topk_ids=topk_ids)  # 记录专家选择分布
        topk_output = StandardTopKOutput(  # 构造标准TopK输出
            topk_weights=topk_weights, topk_ids=topk_ids, router_logits=router_logits  # 包含权重、ID和logits
        )
        return self._apply_deepep_waterfill(topk_output, hidden_states.shape[0])  # 应用水填充并返回
