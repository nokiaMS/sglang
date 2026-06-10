# DeepSeek V4压缩状态池模块
# 实现KV缓存与注意力分数的联合存储结构（KVAndScore），以及压缩状态池（CompressStatePool）
# 用于DeepSeek V4模型的多级KV缓存压缩，支持在线压缩和环形缓冲区管理

from __future__ import annotations  # 启用延迟类型注解评估

import dataclasses  # 导入数据类装饰器模块
from contextlib import nullcontext  # 导入空上下文管理器

import torch  # 导入PyTorch张量库

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE  # 导入GPU内存类型常量
from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool  # 导入自定义内存池初始化工具
from sglang.srt.utils import is_hip  # 导入AMD HIP平台检测函数
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter  # 导入内存节省适配器

_is_hip = is_hip()  # 检测当前是否为AMD HIP平台


@dataclasses.dataclass  # 使用数据类装饰器定义KVAndScore
class KVAndScore:  # KV缓存与注意力分数的联合存储结构
    kv_score: torch.Tensor  # KV和score拼接在一起的张量

    @property  # 属性：提取KV部分
    def kv(self) -> torch.Tensor:  # 获取KV张量
        return self.kv_score[..., : self._item_size]  # 取前半部分作为KV

    @property  # 属性：提取score部分
    def score(self) -> torch.Tensor:  # 获取score张量
        return self.kv_score[..., self._item_size :]  # 取后半部分作为score

    @property  # 属性：获取形状
    def shape(self):  # 获取张量形状
        return self.kv_score.shape  # 返回kv_score的形状

    def __post_init__(self):  # 数据类初始化后处理
        self._item_size = self.kv_score.shape[-1] // 2  # 计算单个项的大小（KV和score各占一半）

    @staticmethod  # 静态方法：从分离的KV和score创建KVAndScore
    def from_kv_score(*, kv: torch.Tensor, score: torch.Tensor) -> KVAndScore:  # 从KV和score构建联合结构
        assert kv.shape == score.shape  # 断言KV和score形状一致
        return KVAndScore(torch.cat([kv, score], dim=-1))  # 沿最后一维拼接KV和score

    def new_empty(self, new_shape) -> KVAndScore:  # 创建具有新形状的空KVAndScore
        assert new_shape[-1] == self._item_size  # 断言最后一维等于单项大小
        new_shape = list(new_shape)  # 将形状转为列表
        new_shape[-1] = 2 * self._item_size  # 最后一维扩展为2倍（包含KV和score）
        return KVAndScore(self.kv_score.new_empty(new_shape, requires_grad=False))  # 创建不需要梯度的空张量

    def __getitem__(self, index) -> KVAndScore:  # 索引访问操作
        return KVAndScore(self.kv_score[index])  # 对kv_score进行索引并包装返回

    def __setitem__(self, index, value: KVAndScore):  # 索引赋值操作
        self.kv_score[index] = value.kv_score  # 将value的kv_score赋值到指定索引

    def clear(self):  # 清空KV和score
        self.kv.zero_()  # 将KV部分清零
        self.score.fill_(float("-inf"))  # 将score部分填充为负无穷（表示无效分数）

    def view(self, *args):  # 视图重塑操作
        args = list(args)  # 将参数转为列表
        if isinstance(args[-1], int) and args[-1] != -1:  # 如果最后一个参数是整数且不为-1
            args[-1] = 2 * self._item_size  # 将最后一维调整为2倍单项大小
        return KVAndScore(self.kv_score.view(*args))  # 对kv_score进行视图重塑并包装返回

    def clone(self) -> KVAndScore:  # 深拷贝操作
        return KVAndScore(self.kv_score.clone())  # 克隆kv_score并包装返回

    @staticmethod  # 静态方法：沿指定维度拼接多个KVAndScore
    def cat(tensors: list[KVAndScore], dim: int) -> KVAndScore:  # 拼接多个KVAndScore
        assert dim != -1, "Concatenation along last dim is not supported."  # 不支持沿最后一维拼接
        assert len(tensors) > 0, "At least one tensor is required for concatenation."  # 至少需要一个张量
        item_size = tensors[0]._item_size  # 获取第一个张量的单项大小
        for v in tensors:  # 遍历所有张量
            assert (
                v._item_size == item_size
            ), "All tensors must have the same item size."  # 断言所有张量的单项大小一致

        return KVAndScore(torch.cat([v.kv_score for v in tensors], dim=dim))  # 沿指定维度拼接kv_score


class CompressStatePool:  # 压缩状态池，管理KV缓存的压缩状态
    def __init__(  # 初始化压缩状态池
        self,
        size: int,  # 状态池大小
        ring_size: int,  # 环形缓冲区大小
        overlap: bool,  # 是否启用重叠压缩
        head_dim: int,  # 注意力头维度
        dtype: torch.dtype,  # 数据类型
        device: str,  # 设备类型
        enable_memory_saver: bool,  # 是否启用内存节省
        ratio: int,  # 压缩比率
        online: bool = False,  # 是否启用在线压缩模式
        swa_page_size: int = 0,  # 滑动窗口注意力页面大小
    ):
        self.ring_size = ring_size  # 保存环形缓冲区大小
        self.swa_page_size = swa_page_size  # 保存SWA页面大小
        self.enable_memory_saver = enable_memory_saver  # 保存内存节省标志

        if online:  # 在线压缩模式
            assert ring_size == 1, "online compress requires ring_size=1"  # 在线压缩要求ring_size=1
            self._size = size + self.ring_size + 1  # 计算总大小（含额外空间）
            last_dim = 3 * head_dim  # 在线模式下最后一维为3倍head_dim（max, sum, kv）
        else:  # 离线压缩模式
            self._size = size + self.ring_size + 1  # 计算总大小（含额外空间）
            self._size = (self._size + ratio - 1) // ratio * ratio  # 向上对齐到ratio的整数倍
            last_dim = 2 * (1 + overlap) * head_dim  # 离线模式下最后一维取决于重叠设置

        if _is_hip:  # AMD HIP平台处理
            self.kv_score_buffer = KVAndScore(  # 创建KV-score缓冲区
                torch.empty((self._size, last_dim), dtype=dtype, device=device)  # 分配空张量
            )
            if not online:  # 非在线模式下
                self.kv_score_buffer[-1].clear()  # 清空最后一个槽位作为哨兵
        else:  # NVIDIA CUDA平台处理
            self.memory_saver_adapter = TorchMemorySaverAdapter.create(  # 创建内存节省适配器
                enable=enable_memory_saver  # 根据配置决定是否启用
            )
            self.enable_custom_mem_pool, self.custom_mem_pool, _ = (  # 初始化自定义内存池
                maybe_init_custom_mem_pool(device=device)  # 根据设备类型初始化内存池
            )

            with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):  # 进入KV缓存内存区域
                with (  # 使用自定义内存池分配
                    torch.cuda.use_mem_pool(self.custom_mem_pool)
                    if self.custom_mem_pool  # 如果有自定义内存池
                    else nullcontext()  # 否则使用空上下文
                ):
                    self.kv_score_buffer = KVAndScore(  # 创建KV-score缓冲区
                        torch.empty(  # 分配空张量
                            (self._size, last_dim),  # 形状为(大小, 最后一维)
                            dtype=dtype,  # 数据类型
                            device=device,  # 设备
                        )
                    )
                    if not online:  # 非在线模式下
                        self.kv_score_buffer[-1].clear()  # 清空最后一个槽位作为哨兵

    def translate_from_swa_loc_to_state_loc(  # 将SWA位置转换为压缩状态位置
        self, swa_loc: torch.Tensor  # SWA位置张量
    ) -> torch.Tensor:
        swa_pages = swa_loc // self.swa_page_size  # 计算SWA页码
        state_loc = swa_pages * self.ring_size + (swa_loc % self.ring_size)  # 计算状态位置
        state_loc = torch.where(swa_loc < 0, -1, state_loc)  # 负值位置标记为-1（无效）
        return state_loc  # 返回转换后的状态位置

    def get_state_by_state_loc(self, state_loc: torch.Tensor) -> KVAndScore:  # 根据状态位置获取压缩状态
        return self.kv_score_buffer[state_loc]  # 返回指定位置的KVAndScore

    def set_state_by_state_loc(self, state_loc: torch.Tensor, value: KVAndScore):  # 根据状态位置设置压缩状态
        self.kv_score_buffer[state_loc] = value  # 将值写入指定位置
        self.kv_score_buffer[-1].clear()  # 清空哨兵位置
