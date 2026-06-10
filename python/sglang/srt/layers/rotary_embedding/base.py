# 旋转位置编码（RoPE）基类及线性缩放旋转位置编码实现，支持CUDA/NPU/CPU/XPU等多平台前向计算
"""RotaryEmbedding base class + LinearScalingRotaryEmbedding.  # RotaryEmbedding基类 + LinearScalingRotaryEmbedding（线性缩放旋转位置编码）"""

from __future__ import annotations  # 启用延迟注解求值 # 启用延迟注解

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union  # 导入类型注解 # 导入类型提示

import torch  # 导入PyTorch # 导入PyTorch框架

from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb  # 导入旋转编码应用工具 # 导入旋转编码应用函数
from sglang.srt.layers.utils import MultiPlatformOp  # 导入多平台操作基类 # 导入多平台操作基类
from sglang.srt.server_args import get_global_server_args  # 导入全局服务器参数 # 导入全局服务器参数
from sglang.srt.utils import (  # 导入平台检测和环境变量工具 # 导入工具函数
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_hip,
    is_mps,
    is_musa,
    is_npu,
    is_xpu,
)

if TYPE_CHECKING:  # 仅类型检查时导入 # 仅类型检查时导入
    from sglang.jit_kernel.rope import FusedSetKVBufferArg  # For type check-only  # 仅用于类型检查

_is_cuda = is_cuda()  # 是否为CUDA平台 # 是否为CUDA平台
_is_hip = is_hip()  # 是否为HIP(AMD GPU)平台 # 是否为HIP平台
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip  # 是否使用AITER且为HIP平台 # 是否使用AITER
_is_npu = is_npu()  # 是否为NPU(华为)平台 # 是否为NPU平台
_is_cpu_amx_available = cpu_has_amx_support()  # CPU是否支持AMX指令集 # CPU是否支持AMX
_is_cpu = is_cpu()  # 是否为CPU平台 # 是否为CPU平台
_is_xpu = is_xpu()  # 是否为XPU(Intel GPU)平台 # 是否为XPU平台
_is_musa = is_musa()  # 是否为MUSA(摩尔线程)平台 # 是否为MUSA平台
_is_mps = is_mps()  # 是否为MPS(Apple)平台 # 是否为MPS平台

if _is_cuda:  # CUDA平台导入JIT核函数 # CUDA平台导入JIT核函数
    from sglang.jit_kernel.rope import apply_rope_with_cos_sin_cache_inplace  # 导入原地应用RoPE的CUDA核函数 # 导入CUDA原地RoPE核函数

if _is_npu:  # NPU平台导入专用算子 # NPU平台导入专用算子
    import torch_npu  # 导入华为NPU扩展 # 导入torch_npu
    from sgl_kernel_npu.norm.fused_rope_qk_mqa import fused_rope_qk_mqa  # 导入NPU融合RoPE算子 # 导入NPU融合RoPE

if _is_hip:  # HIP平台导入融合算子 # HIP平台导入融合算子
    from sglang.srt.layers.attention.utils import (
        fused_qk_rope_reshape_and_cache,  # 导入QK RoPE重塑和缓存融合算子 # 导入融合重塑缓存算子
    )


class RotaryEmbedding(MultiPlatformOp):  # 旋转位置编码基类，继承多平台操作基类 # 旋转位置编码基类
    """Original rotary positional embedding.  # 原始旋转位置编码。"""

    def __init__(  # 初始化旋转位置编码 # 初始化方法
        self,
        head_size: int,  # 注意力头大小 # 注意力头维度
        rotary_dim: int,  # 旋转维度 # 旋转编码维度
        max_position_embeddings: int,  # 最大位置编码数 # 最大位置嵌入数
        base: int,  # 旋转基频 # 旋转基频
        is_neox_style: bool,  # 是否为NeoX风格 # 是否NeoX风格
        dtype: torch.dtype,  # 数据类型 # 数据类型
    ) -> None:
        super().__init__()  # 调用父类初始化 # 调用父类初始化
        self.head_size = head_size  # 保存注意力头大小 # 保存头大小
        self.rotary_dim = rotary_dim  # 保存旋转维度 # 保存旋转维度
        self.max_position_embeddings = max_position_embeddings  # 保存最大位置数 # 保存最大位置数
        self.base = base  # 保存旋转基频 # 保存基频
        self.is_neox_style = is_neox_style  # 保存NeoX风格标志 # 保存NeoX风格标志
        self.dtype = dtype  # 保存数据类型 # 保存数据类型

        cache = self._compute_cos_sin_cache()  # 计算cos/sin缓存 # 计算cos/sin缓存
        # NOTE(ByronHsu): cache needs to be in FP32 for numerical stability  # 注意(ByronHsu)：缓存需要保持FP32以确保数值稳定性
        if not _is_cuda:  # 非CUDA平台将缓存转为指定数据类型 # 非CUDA平台转换缓存数据类型
            cache = cache.to(dtype)

        if (  # 判断是否需要使用回退内核 # 判断是否使用回退内核
            (not (_is_cuda) or self.head_size not in [64, 128, 256, 512])  # 非CUDA或头大小不在支持列表中 # 非CUDA或头大小不在支持范围
            and not (_is_cpu)  # 非CPU # 非CPU
            and not (_is_xpu)  # 非XPU # 非XPU
            and not (_is_npu)  # 非NPU # 非NPU
            and not (_is_musa)  # 非MUSA # 非MUSA
            and not (_is_mps)  # 非MPS # 非MPS
        ):
            # rotary_embedding from sglang.jit_kernel.rope and vllm._custom_ops has the same implementation.  # sglang.jit_kernel.rope和vllm._custom_ops的rotary_embedding实现相同。
            # TODO: Test on different devices and remove this conditional.  # 待办：在不同设备上测试并移除此条件判断。
            if _is_cuda:  # CUDA平台使用sglang JIT核函数 # CUDA平台使用sglang JIT核函数
                from sglang.jit_kernel.rope import rotary_embedding
            elif _is_hip:  # HIP平台使用sgl_kernel # HIP平台使用sgl_kernel
                from sgl_kernel import rotary_embedding
            else:  # 其他平台使用vLLM自定义算子 # 其他平台使用vLLM自定义算子
                from vllm._custom_ops import rotary_embedding

            self.use_fallback_kernel = True  # 标记使用回退内核 # 标记使用回退内核
            self.fallback_rotary_embedding = rotary_embedding  # 保存回退旋转编码函数 # 保存回退函数
        else:
            self.use_fallback_kernel = False  # 不使用回退内核 # 不使用回退内核

        self.cos_sin_cache: torch.Tensor  # 声明cos/sin缓存张量 # 声明缓存类型
        self.register_buffer("cos_sin_cache", cache, persistent=False)  # 注册为不持久化的缓冲区 # 注册为缓冲区

        self._apply_rotary_emb_wrapped = apply_rotary_emb  # 保存旋转编码应用函数 # 保存旋转编码应用函数

        # XXX (MUSA): Implement sgl_kernel.rotary_embedding support for MUSA backend  # XXX (MUSA)：为MUSA后端实现sgl_kernel.rotary_embedding支持
        if get_global_server_args().rl_on_policy_target is not None or _is_musa:  # RL训练模式或MUSA平台 # RL训练模式或MUSA平台
            self._forward_method = self.forward_native  # 使用原生前向方法 # 使用原生前向方法
            self._apply_rotary_emb_wrapped = torch.compile(dynamic=True)(  # 编译优化旋转编码函数 # 编译优化旋转编码函数
                apply_rotary_emb
            )
        self.position_cos, self.position_sin = None, None  # 初始化位置cos/sin为None # 初始化位置cos/sin

    def _match_cos_sin_cache_dtype(self, query: torch.Tensor) -> None:  # 匹配cos/sin缓存的数据类型和设备与query一致 # 匹配缓存数据类型和设备
        # __setattr__ in nn.Module (called by `self.cos_sin_cache = ...`)  # nn.Module的__setattr__（由`self.cos_sin_cache = ...`调用）
        # is expensive, so avoid calling it if possible  # 开销较大，因此尽可能避免调用
        if (  # 如果设备或数据类型不匹配 # 如果设备或数据类型不匹配
            self.cos_sin_cache.device != query.device  # 设备不匹配 # 设备不同
            or self.cos_sin_cache.dtype != query.dtype  # 数据类型不匹配 # 数据类型不同
        ):
            self.cos_sin_cache = self.cos_sin_cache.to(query.device, dtype=query.dtype)  # 转换缓存到相同设备和数据类型 # 转换缓存

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:  # 计算逆频率张量 # 计算逆频率
        """Compute the inverse frequency.  # 计算逆频率。"""
        # NOTE(woosuk): To exactly match the HF implementation, we need to  # 注意(woosuk)：为了完全匹配HuggingFace实现，我们需要
        # use CPU to compute the cache and then move it to GPU. However, we  # 使用CPU计算缓存然后移动到GPU。然而我们
        # create the cache on GPU for faster initialization. This may cause  # 在GPU上创建缓存以加速初始化。这可能导致
        # a slight numerical difference between the HF implementation and ours.  # HF实现和我们的实现之间有轻微数值差异。
        init_device = (  # 确定初始化设备 # 确定初始化设备
            "cpu" if get_global_server_args().rl_on_policy_target is not None else None  # RL训练模式使用CPU # RL模式使用CPU
        )
        inv_freq = 1.0 / (  # 计算逆频率 1/base^(2i/d) # 计算逆频率
            base
            ** (
                torch.arange(
                    0, self.rotary_dim, 2, dtype=torch.float, device=init_device  # 生成0到rotary_dim步长为2的序列 # 生成等差序列
                )
                / self.rotary_dim  # 除以旋转维度 # 归一化
            )
        )
        if get_global_server_args().rl_on_policy_target is not None:  # RL训练模式将逆频率移到GPU # RL模式移到GPU
            inv_freq = inv_freq.cuda()
        return inv_freq  # 返回逆频率 # 返回逆频率

    def _compute_cos_sin_cache(self) -> torch.Tensor:  # 计算cos/sin缓存 # 计算cos/sin缓存
        """Compute the cos and sin cache.  # 计算cos和sin缓存。"""
        inv_freq = self._compute_inv_freq(self.base)  # 计算逆频率 # 计算逆频率
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)  # 生成位置索引序列 # 生成位置序列

        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # 计算外积得到频率矩阵 # 计算外积
        cos = freqs.cos()  # 计算cos值 # 计算余弦
        sin = freqs.sin()  # 计算sin值 # 计算正弦
        cache = torch.cat((cos, sin), dim=-1)  # 拼接cos和sin # 拼接cos和sin
        return cache  # 返回缓存 # 返回缓存

    def _ensure_cos_sin_cache_length(self, needed_max_pos: int):  # 确保cos/sin缓存长度足够 # 确保缓存长度足够
        """Ensure cos_sin_cache length > needed_max_pos.  # 确保cos_sin_cache长度大于needed_max_pos。"""
        from sglang.srt.environ import envs  # 导入环境变量 # 导入环境变量

        cur_len = int(self.cos_sin_cache.shape[0])  # 当前缓存长度 # 当前缓存长度
        if needed_max_pos < cur_len:  # 已有缓存足够 # 缓存已足够
            return

        # Align to reduce realloc frequency  # 对齐以减少重新分配频率 # 对齐以减少重分配频率
        align = envs.SGLANG_ROPE_CACHE_ALIGN.get()  # 获取对齐参数 # 获取对齐参数
        new_len = ((needed_max_pos + align) // align) * align  # 计算新的对齐长度 # 计算对齐后长度
        device = self.cos_sin_cache.device  # 获取当前设备 # 获取设备
        dtype = self.cos_sin_cache.dtype  # 获取当前数据类型 # 获取数据类型

        # Compute inv_freq on same device  # 在相同设备上计算逆频率 # 在相同设备上计算逆频率
        inv_freq = self._compute_inv_freq(self.base).to(device=device)  # 计算并移到对应设备 # 计算逆频率

        # Incremental computation for new positions only  # 仅对新位置增量计算 # 仅增量计算新位置
        start = cur_len  # 增量计算的起始位置 # 起始位置
        t_new = torch.arange(start, new_len, dtype=inv_freq.dtype, device=device)  # 新位置序列 # 新位置序列
        if t_new.numel() == 0:  # 无新位置需要计算 # 无新位置
            return

        freqs_new = torch.einsum("i,j->ij", t_new, inv_freq)  # 计算新位置的频率 # 计算新频率
        cos_new = freqs_new.cos()  # 计算新cos值 # 计算新余弦
        sin_new = freqs_new.sin()  # 计算新sin值 # 计算新正弦
        new_rows = torch.cat((cos_new, sin_new), dim=-1).to(dtype=dtype)  # 拼接并转换数据类型 # 拼接并转换类型

        # Update cache with new rows  # 用新行更新缓存 # 用新行更新缓存
        self.cos_sin_cache = torch.cat((self.cos_sin_cache, new_rows), dim=0).to(  # 拼接新旧缓存 # 拼接新旧缓存
            device=device, dtype=dtype
        )

    def get_cos_sin_with_position(self, positions):  # 根据位置索引获取cos/sin值 # 根据位置获取cos/sin
        assert positions.ndim == 1, (  # 断言位置为一维 # 断言位置为一维
            "2D positions (multimodal RoPE) are not supported by the base "
            "RotaryEmbedding. Override this method in a subclass (e.g. MRotaryEmbedding)."
        )
        cos_sin = self.cos_sin_cache.index_select(0, positions.flatten())  # 按位置索引选择cos/sin # 按位置索引选择
        last_dim = cos_sin.size()[-1]  # 最后一维大小 # 最后一维大小
        cos, sin = (  # 分离并扩展cos和sin # 分离cos和sin
            cos_sin.reshape(-1, 2, last_dim // 2).repeat(1, 1, 2).chunk(2, dim=-2)  # 重塑、重复、分块 # 重塑重复分块
        )
        # BSNH  # BSNH格式 # BSNH维度格式
        self.position_cos, self.position_sin = (  # 保存位置cos/sin # 保存位置cos/sin
            cos.view(-1, 1, 1, last_dim).contiguous(),  # 重塑为(token,1,1,dim) # 重塑为BSNH格式
            sin.view(-1, 1, 1, last_dim).contiguous(),  # 重塑为(token,1,1,dim) # 重塑为BSNH格式
        )

    def get_cos_sin(self, seqlen: int) -> tuple[torch.Tensor, torch.Tensor]:  # 获取指定序列长度的cos/sin # 获取指定长度的cos/sin
        cos_sin = self.cos_sin_cache[:seqlen]  # 截取前seqlen行 # 截取前seqlen行
        cos, sin = cos_sin.chunk(2, dim=-1)  # 分离cos和sin # 分离cos和sin
        return cos, sin  # 返回cos和sin # 返回cos和sin

    def forward_native(  # PyTorch原生实现的前向方法 # 原生PyTorch前向方法
        self,
        positions: torch.Tensor,  # 位置索引 # 位置索引
        query: torch.Tensor,  # 查询张量 # 查询张量
        key: torch.Tensor,  # 键张量 # 键张量
        offsets: Optional[torch.Tensor] = None,  # 位置偏移 # 位置偏移量
        fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,  # 融合设置KV缓冲区参数 # 融合KV缓冲区参数
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-native implementation of forward().  # forward()的PyTorch原生实现。"""
        assert (  # 断言不支持融合KV缓冲区 # 断言不支持融合KV缓冲区
            fused_set_kv_buffer_arg is None
        ), "fused_set_kv_buffer_arg is not supported for native implementation"

        if offsets is not None:  # 有偏移则加到位置上 # 有偏移则加到位置上
            positions = positions + offsets

        positions = positions.flatten()  # 展平位置 # 展平位置
        num_tokens = positions.shape[0]  # token数量 # token数量

        if hasattr(self, "sin_cos_cache"):  # 如果有sin_cos_cache属性 # 检查sin_cos_cache属性
            cos_sin = self.sin_cos_cache  # 使用sin_cos_cache # 使用sin_cos_cache
        else:
            cos_sin = self.cos_sin_cache.index_select(0, positions)  # 按位置索引选择 # 按位置索引选择
        cos, sin = cos_sin.chunk(2, dim=-1)  # 分离cos和sin # 分离cos和sin

        query_shape = query.shape  # 保存query原始形状 # 保存原始形状
        query = query.view(num_tokens, -1, self.head_size)  # 重塑query为3D # 重塑query
        query_rot = query[..., : self.rotary_dim]  # 提取旋转部分 # 提取旋转部分
        query_pass = query[..., self.rotary_dim :]  # 提取不旋转部分 # 提取不旋转部分
        query_rot = self._apply_rotary_emb_wrapped(  # 应用旋转编码 # 应用旋转编码
            query_rot, cos, sin, self.is_neox_style
        )
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)  # 拼接并恢复形状 # 拼接并恢复形状

        key_shape = key.shape  # 保存key原始形状 # 保存原始形状
        key = key.view(num_tokens, -1, self.head_size)  # 重塑key为3D # 重塑key
        key_rot = key[..., : self.rotary_dim]  # 提取旋转部分 # 提取旋转部分
        key_pass = key[..., self.rotary_dim :]  # 提取不旋转部分 # 提取不旋转部分
        key_rot = self._apply_rotary_emb_wrapped(key_rot, cos, sin, self.is_neox_style)  # 应用旋转编码 # 应用旋转编码
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)  # 拼接并恢复形状 # 拼接并恢复形状
        return query, key  # 返回旋转后的query和key # 返回旋转后的query和key

    def forward_npu(  # NPU平台的前向方法 # NPU平台前向方法
        self,
        positions: torch.Tensor,  # 位置索引 # 位置索引
        query: torch.Tensor,  # 查询张量 # 查询张量
        key: torch.Tensor,  # 键张量 # 键张量
        offsets: Optional[torch.Tensor] = None,  # 位置偏移 # 位置偏移量
        fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,  # 融合KV缓冲区参数 # 融合KV缓冲区参数
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-npu implementation of forward().  # forward()的NPU PyTorch实现。"""
        assert (  # 断言不支持融合KV缓冲区 # 断言不支持融合KV缓冲区
            fused_set_kv_buffer_arg is None
        ), "fused_set_kv_buffer_arg is not supported for npu implementation"
        if (  # 判断是否需要使用原生方法 # 判断是否使用原生方法
            query.dtype == torch.bfloat16  # query为bf16类型 # query为bf16
            and self.cos_sin_cache.dtype == torch.float  # 缓存为float类型 # 缓存为float
            or key.ndim == 3  # 或key为3D # 或key为3D
        ):
            if hasattr(self, "sin_cos_cache"):  # 检查sin_cos_cache属性 # 检查属性
                cos_sin = self.sin_cos_cache  # 使用sin_cos_cache # 使用缓存
            else:
                cos_sin = self.cos_sin_cache.index_select(0, positions)  # 按位置索引选择 # 按位置索引选择

            if query.shape[0] * query.shape[1] < 65535:  # 序列长度小于阈值时使用融合算子 # 小序列使用融合算子
                return fused_rope_qk_mqa(  # 调用NPU融合RoPE算子 # 调用NPU融合算子
                    query,
                    key,
                    cos_sin,
                    self.rotary_dim,
                    self.is_neox_style,
                )
            else:  # 大序列回退到原生方法 # 大序列回退原生方法
                return self.forward_native(positions, query, key, offsets)
        if self.is_neox_style:  # NeoX风格设置旋转模式 # NeoX风格设置旋转模式
            rotary_mode = "half"  # 半旋转模式 # 半旋转模式
        else:
            rotary_mode = "interleave"  # 交错旋转模式 # 交错旋转模式

        mrope_section = [0, 0, 0]  # 多模态RoPE段设为0 # 多模态RoPE段
        # The npu_mrope kernel only supports 1D or 2D tensors for query and key.  # npu_mrope核函数仅支持1D或2D的query和key张量。
        # Therefore, when their dimensions exceed 2D, we flatten query and key to 2D tensors before computation  # 因此，当维度超过2D时，我们在计算前将query和key展平为2D张量
        # and reshape their original shapes afterward.  # 并在之后恢复原始形状。
        query_shape = query.shape  # 保存query原始形状 # 保存原始形状
        key_shape = key.shape  # 保存key原始形状 # 保存原始形状
        query = query.reshape(query.shape[0], -1)  # 展平query为2D # 展平query
        key = key.reshape(key.shape[0], -1)  # 展平key为2D # 展平key

        query_out, key_out = torch_npu.npu_mrope(  # 调用NPU多模态RoPE算子 # 调用NPU多模态RoPE
            positions,
            query,
            key,
            self.cos_sin_cache,
            self.head_size,
            mrope_section=mrope_section,
            rotary_mode=rotary_mode,
        )

        query_out = query_out.reshape(query_shape)  # 恢复query原始形状 # 恢复原始形状
        key_out = key_out.reshape(key_shape)  # 恢复key原始形状 # 恢复原始形状
        return query_out, key_out  # 返回旋转后的query和key # 返回结果

    def forward_cpu(  # CPU平台的前向方法 # CPU平台前向方法
        self,
        positions: torch.Tensor,  # 位置索引 # 位置索引
        query: torch.Tensor,  # 查询张量 # 查询张量
        key: torch.Tensor,  # 键张量 # 键张量
        offsets: Optional[torch.Tensor] = None,  # 位置偏移 # 位置偏移量
        fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,  # 融合KV缓冲区参数 # 融合KV缓冲区参数
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (  # 断言不支持融合KV缓冲区 # 断言不支持融合KV缓冲区
            fused_set_kv_buffer_arg is None
        ), "fused_set_kv_buffer_arg is not supported for cpu implementation"

        positions = torch.add(positions, offsets) if offsets is not None else positions  # 有偏移则加到位置上 # 有偏移则加偏移
        if _is_cpu_amx_available:  # CPU支持AMX指令集 # CPU支持AMX
            return torch.ops.sgl_kernel.rotary_embedding_cpu(  # 使用CPU专用旋转编码算子 # 使用CPU旋转编码算子
                positions,
                query,
                key,
                self.head_size,
                self.cos_sin_cache,
                self.is_neox_style,
            )
        else:  # 不支持AMX则使用原生方法 # 不支持AMX则回退原生
            return self.forward_native(
                positions, query, key, offsets, fused_set_kv_buffer_arg
            )

    def forward_cuda(  # CUDA平台的前向方法 # CUDA平台前向方法
        self,
        positions: torch.Tensor,  # 位置索引 # 位置索引
        query: torch.Tensor,  # 查询张量 # 查询张量
        key: torch.Tensor,  # 键张量 # 键张量
        offsets: Optional[torch.Tensor] = None,  # 位置偏移 # 位置偏移量
        fused_set_kv_buffer_arg: Optional[Union[FusedSetKVBufferArg, dict]] = None,  # 融合KV缓冲区参数 # 融合KV缓冲区参数
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_fallback_kernel:  # 使用JIT核函数路径 # 使用JIT核函数路径
            batch_size = positions.size(0)  # 批量大小 # 批量大小
            q_rope = query.view(batch_size, -1, self.head_size)  # 重塑query # 重塑query
            k_rope = key.view(batch_size, -1, self.head_size)  # 重塑key # 重塑key
            if self.head_size != self.rotary_dim:  # 头大小与旋转维度不同时截取 # 头大小与旋转维度不同时截取
                q_rope = q_rope[..., : self.rotary_dim]  # 截取旋转维度部分 # 截取旋转部分
                k_rope = k_rope[..., : self.rotary_dim]  # 截取旋转维度部分 # 截取旋转部分
            apply_rope_with_cos_sin_cache_inplace(  # 原地应用RoPE # 原地应用RoPE
                positions=positions,
                q=q_rope,
                k=k_rope,
                cos_sin_cache=self.cos_sin_cache,
                is_neox=self.is_neox_style,
                fused_args=fused_set_kv_buffer_arg,
            )
        else:  # 使用回退内核路径 # 使用回退内核路径

            if fused_set_kv_buffer_arg is not None and _is_hip:  # HIP平台且有融合参数 # HIP平台且有融合参数
                extra_args = fused_set_kv_buffer_arg  # 提取额外参数 # 提取额外参数

                k_cache_shape = fused_set_kv_buffer_arg["key_cache"].shape  # key缓存形状 # key缓存形状
                qk_head_dim = k_cache_shape[-1]  # QK头维度 # QK头维度
                tp_k_head_num = k_cache_shape[-2]  # 张量并行K头数 # TP K头数

                key = key.view(-1, tp_k_head_num, qk_head_dim)  # 重塑key # 重塑key

                tokens = key.shape[0]  # token数量 # token数量

                query = query.view(tokens, -1, qk_head_dim)  # 重塑query # 重塑query

                query, key, k_cache, v_cache = fused_qk_rope_reshape_and_cache(  # 融合QK RoPE重塑和缓存 # 融合重塑和缓存
                    q=query,
                    k=key,
                    pos=positions,
                    cos_sin=self.cos_sin_cache,
                    is_neox=self.is_neox_style,
                    flash_layout=True,  # 使用Flash布局 # 使用Flash布局
                    offs=None,  # 无偏移 # 无偏移
                    q_out=query,  # 输出query # 输出query
                    k_out=key,  # 输出key # 输出key
                    output_zeros=False,  # 不输出零 # 不输出零
                    **extra_args,  # 额外参数 # 额外参数
                )
            else:  # 非HIP平台或无融合参数 # 非HIP平台或无融合参数
                assert (  # 断言不支持融合KV缓冲区 # 断言不支持融合KV缓冲区
                    fused_set_kv_buffer_arg is None
                ), "save kv cache is not supported for fallback_rotary_embedding."
                self.cos_sin_cache = self.cos_sin_cache.to(  # 转换缓存数据类型和设备 # 转换缓存
                    query.device, dtype=query.dtype
                )
                self.fallback_rotary_embedding(  # 调用回退旋转编码函数 # 调用回退旋转编码
                    positions,
                    query,
                    key,
                    self.head_size,
                    self.cos_sin_cache,
                    self.is_neox_style,
                )
        return query, key  # 返回旋转后的query和key # 返回结果

    def extra_repr(self) -> str:  # 返回模块的额外表示字符串 # 返回额外表示字符串
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"  # 头大小和旋转维度 # 头大小和旋转维度
        s += f", max_position_embeddings={self.max_position_embeddings}"  # 最大位置数 # 最大位置数
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"  # 基频和NeoX风格 # 基频和NeoX风格
        return s  # 返回表示字符串 # 返回字符串

    def forward_xpu(  # XPU平台的前向方法 # XPU平台前向方法
        self,
        positions: torch.Tensor,  # 位置索引 # 位置索引
        query: torch.Tensor,  # 查询张量 # 查询张量
        key: torch.Tensor,  # 键张量 # 键张量
        offsets: Optional[torch.Tensor] = None,  # 位置偏移 # 位置偏移量
        fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,  # 融合KV缓冲区参数 # 融合KV缓冲区参数
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (  # 断言不支持融合KV缓冲区 # 断言不支持融合KV缓冲区
            fused_set_kv_buffer_arg is None
        ), "fused_set_kv_buffer_arg is not supported for xpu implementation"
        positions = torch.add(positions, offsets) if offsets is not None else positions  # 有偏移则加到位置上 # 有偏移则加偏移

        self._match_cos_sin_cache_dtype(query)  # 匹配缓存数据类型 # 匹配缓存数据类型
        return torch.ops.sgl_kernel.rotary_embedding(  # 调用XPU旋转编码算子 # 调用XPU旋转编码算子
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
            self.is_neox_style,
        )


class LinearScalingRotaryEmbedding(RotaryEmbedding):  # 线性缩放旋转位置编码，继承RotaryEmbedding # 线性缩放旋转位置编码
    """RotaryEmbedding extended with linear scaling.  # 带线性缩放扩展的旋转位置编码。

    It supports multiple scaling factors. Since multiple LoRA adapters may have  # 支持多个缩放因子。由于多个LoRA适配器可能有
    different scaling factors, we need multiple cos/sin caches. In this way,  # 不同的缩放因子，我们需要多个cos/sin缓存。这样，
    instead of running rotary embedding kernel per lora, we can run multiple  # 我们可以批量运行多个LoRA，
    lora in a batched way.  # 而不是每个LoRA单独运行旋转编码核函数。

    In addition to that, we also keep the cos/sin cache for the scaling factor  # 此外，我们始终保持缩放因子
    of 1 (default) at all times.  # 为1（默认）的cos/sin缓存。

    Exemplary for two scaling factors x=1, y and z with embeddings  # 以两个缩放因子x=1、y和z为例
    [[x11, x12, ... x1m], ..., [xn1, xn2, ..., xnm]] and  # 嵌入矩阵x
    [[y11, y12, ... y1o], ..., [yn1, yn2, ..., yno]], and  # 嵌入矩阵y
    [[z11, z12, ... z1p], ..., [zn1, zn2, ..., znp]],  # 嵌入矩阵z

    we construct the cos/sin cache as follows:  # 我们按如下方式构造cos/sin缓存：
    [[x11, x12, ... x1m, y11, y12, ... y1o, z11, z12, ... z1p],  # 拼接缓存行
        ...
     [xn1, xn2, ... xnm, yn1, yn2, ... yno, zn1, zn2, ... znp]]  # 拼接缓存行

    We then use offsets to index into the cos/sin cache for  # 然后我们使用偏移量索引到cos/sin缓存中
    the respective scaling factors.  # 以获取对应的缩放因子。

    The offset to cache can be accessed via `scaling_factor_to_offset` API.  # 可以通过`scaling_factor_to_offset` API访问缓存偏移量。

    Credits to the Reddit user /u/kaiokendev  # 致谢Reddit用户/u/kaiokendev
    """

    def __init__(  # 初始化线性缩放旋转位置编码 # 初始化方法
        self,
        head_size: int,  # 注意力头大小 # 注意力头维度
        rotary_dim: int,  # 旋转维度 # 旋转编码维度
        max_position_embeddings: int,  # 最大位置编码数 # 最大位置嵌入数
        base: int,  # 旋转基频 # 旋转基频
        is_neox_style: bool,  # 是否为NeoX风格 # 是否NeoX风格
        scaling_factors: Union[List[float], float],  # 缩放因子列表 # 缩放因子
        dtype: torch.dtype,  # 数据类型 # 数据类型
    ) -> None:
        if isinstance(scaling_factors, float):  # 如果缩放因子是浮点数，转为列表 # 如果是单个浮点数转为列表
            scaling_factors = [scaling_factors]
        self.scaling_factors: List[float] = scaling_factors  # noqa  # 保存缩放因子列表 # 保存缩放因子
        super().__init__(  # 调用父类初始化 # 调用父类初始化
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )
        # Lazy initialized.  # 延迟初始化。# 延迟初始化
        self._scaling_factor_to_offset: Dict[float, int]  # 缩放因子到偏移量的映射 # 缩放因子到偏移量映射

    def _compute_cos_sin_cache(self) -> torch.Tensor:  # 计算多缩放因子的cos/sin缓存 # 计算多缩放因子的cos/sin缓存
        inv_freq = self._compute_inv_freq(self.base)  # 计算逆频率 # 计算逆频率
        cache_list: List[torch.Tensor] = []  # 缓存列表 # 缓存列表
        # offsets to the next cache in a tensor.  # 张量中下一个缓存的偏移量。
        # Each offset corresponds to the same index in scaling_factors.  # 每个偏移量对应scaling_factors中相同索引的缩放因子。
        offsets: List[int] = []  # 偏移量列表 # 偏移量列表
        for scaling_factor in self.scaling_factors:  # 遍历每个缩放因子 # 遍历缩放因子
            # NOTE(woosuk): self.max_position_embeddings is the original  # 注意(woosuk)：self.max_position_embeddings是原始的
            # maximum length before applying the rope scaling.  # 应用RoPE缩放前的最大长度。
            # Thus, the maximum length after applying the rope scaling is  # 因此，应用RoPE缩放后的最大长度为
            # self.max_position_embeddings * self.scaling_factor.  # self.max_position_embeddings * self.scaling_factor。
            max_len = self.max_position_embeddings * scaling_factor  # 缩放后的最大长度 # 缩放后最大长度
            t = torch.arange(max_len, dtype=torch.float)  # 生成位置序列 # 生成位置序列
            t = t / scaling_factor  # 除以缩放因子 # 除以缩放因子

            freqs = torch.einsum("i,j -> ij", t, inv_freq)  # 计算外积 # 计算外积
            cos = freqs.cos()  # 计算cos值 # 计算余弦
            sin = freqs.sin()  # 计算sin值 # 计算正弦
            cache = torch.cat((cos, sin), dim=-1)  # 拼接cos和sin # 拼接cos和sin
            if not cache_list:  # 第一个缓存，偏移为0 # 第一个缓存偏移为0
                offset = 0
            else:  # 后续缓存，偏移量累加 # 后续缓存偏移累加
                last_offset = offsets[-1]  # 上一个偏移量 # 上一个偏移量
                next_max_len = cache_list[-1].shape[0]  # 上一个缓存的长度 # 上一个缓存长度
                offset = last_offset + next_max_len  # 计算新偏移量 # 计算新偏移量
            offsets.append(offset)  # 记录偏移量 # 记录偏移量
            cache_list.append(cache)  # 添加到缓存列表 # 添加缓存
        self._scaling_factor_to_offset = {  # 构建缩放因子到偏移量的映射 # 构建映射
            float(scaling_factor): offsets[i]
            for i, scaling_factor in enumerate(self.scaling_factors)
        }
        assert len(self.scaling_factors) == len(offsets)  # 断言缩放因子和偏移量数量一致 # 断言数量一致
        return torch.cat(cache_list, dim=0)  # 拼接所有缓存并返回 # 拼接返回

    @property
    def scaling_factor_to_offset(self) -> Dict[float, int]:  # 获取缩放因子到偏移量的映射 # 缩放因子到偏移量的映射属性
        return self._scaling_factor_to_offset  # 返回映射字典 # 返回映射
