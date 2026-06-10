# 基于轮询的屏障同步原语，用于分布式推理中多进程到达同步
# 通过CPU侧的all_reduce操作检测所有进程是否都已到达屏障
import torch  # 导入PyTorch张量库

from sglang.srt.distributed import get_world_group  # 导入获取世界通信组的函数


class PollBasedBarrier:  # 基于轮询的屏障同步类
    def __init__(self, noop: bool = False):  # 初始化屏障
        self._noop = noop  # 是否为空操作模式（跳过同步）
        self._local_arrived = False  # 本地进程是否已到达屏障

    def local_arrive(self):  # 标记本地进程已到达屏障
        assert not self._local_arrived  # 断言本地进程尚未到达
        self._local_arrived = True  # 设置本地到达标志为True

    def poll_global_arrived(self) -> bool:  # 轮询检查所有进程是否都已到达屏障
        global_arrived = self._compute_global_arrived()  # 计算全局到达状态
        output = self._local_arrived and global_arrived  # 本地和全局都已到达才返回True
        if output:  # 如果所有进程都已到达
            self._local_arrived = False  # 重置本地到达标志以便下次使用
        return output  # 返回是否所有进程都已到达

    def _compute_global_arrived(self) -> bool:  # 计算所有进程是否都已到达屏障
        local_arrived = self._noop or self._local_arrived  # 空操作模式视为已到达
        global_arrived = torch.tensor(local_arrived)  # 将布尔值转为张量
        # Can optimize if bottleneck  # 如果成为瓶颈可以优化
        torch.distributed.all_reduce(  # 执行全归约操作
            global_arrived,  # 输入张量
            torch.distributed.ReduceOp.MIN,  # 使用MIN操作：所有进程都为True才返回True
            group=get_world_group().cpu_group,  # 使用CPU通信组
        )
        return global_arrived.item()  # 返回归约结果的标量值
