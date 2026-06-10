# FlashInfer通信后端工具模块
# 提供基于torch.distributed的通信后端实现，替代MPI用于FlashInfer MNNVL工作空间初始化
import torch.distributed as dist  # PyTorch分布式通信模块

from sglang.srt.utils import is_flashinfer_available  # 检查FlashInfer是否可用的工具函数

if is_flashinfer_available():  # 如果FlashInfer可用
    from flashinfer.comm.mnnvl import CommBackend  # 导入FlashInfer的通信后端基类
else:  # 否则使用占位基类

    class CommBackend:
        """
        Placeholder base class when flashinfer is not available
        """  # FlashInfer不可用时的占位基类
        # FlashInfer不可用时的占位基类

        pass  # 占位实现


class TorchDistributedCommBackend(CommBackend):
    """
    Use torch distributed instead of MPI to set up flashinfer MNNVL workspaces during initialization
    """  # 使用torch.distributed替代MPI来初始化FlashInfer MNNVL工作空间
    # 使用torch.distributed替代MPI来初始化FlashInfer MNNVL工作空间

    def __init__(self, group: dist.ProcessGroup):  # 初始化通信后端
        self._group = group  # 保存进程组引用

    def Get_rank(self) -> int:  # 获取当前进程在组内的排名
        return self._group.rank()  # 返回进程组中的排名

    def Get_size(self) -> int:  # 获取进程组的大小
        return self._group.size()  # 返回进程组中的进程数量

    def allgather(self, data: int):  # 全收集操作，所有进程共享数据
        gathered = [None] * self.Get_size()  # 初始化收集列表
        dist.all_gather_object(gathered, data, group=self._group)  # 执行全收集
        return gathered  # 返回收集到的数据列表

    def bcast(self, data, root: int = 0):  # 广播操作，从root进程向所有进程广播数据
        obj_list = [data]  # 包装为列表
        # broadcast_object_list mutates obj_list in-place # broadcast_object_list会原地修改obj_list
        # broadcast_object_list会原地修改obj_list
        dist.broadcast_object_list(obj_list, src=root, group=self._group)  # 执行广播
        return obj_list[0]  # 返回广播后的数据

    def Split(self, color: int, key: int):  # 分裂通信组（此处无需分裂，直接返回自身）
        # No need to split, we already use the proper group # 无需分裂，我们已使用正确的进程组
        # 无需分裂，我们已使用正确的进程组
        return self  # 直接返回当前后端实例

    def barrier(self):  # 屏障同步，等待所有进程到达
        dist.barrier(group=self._group)  # 在进程组内执行屏障同步
