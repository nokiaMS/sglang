# 调度器接收跳过器 - 控制调度器从TokenizerManager接收请求的频率，减少不必要的接收操作以提升性能

from sglang.srt.environ import envs  # 导入环境变量配置
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # 导入前向模式枚举
from sglang.srt.server_args import ServerArgs  # 导入服务器参数类


class SchedulerRecvSkipper:  # 调度器接收跳过器类，用于控制接收频率
    @staticmethod
    def maybe_create(server_args: ServerArgs):  # 工厂方法：根据配置决定是否创建跳过器实例
        if server_args.scheduler_recv_interval <= 1:  # 如果接收间隔小于等于1，则不需要跳过
            return None  # 返回None表示不启用跳过器
        return SchedulerRecvSkipper(server_args)  # 创建并返回跳过器实例

    def __init__(self, server_args: ServerArgs):  # 初始化跳过器
        # Can be supported if needed, but may need e.g. `global_forward_mode`  # 如有需要可以支持，但可能需要例如 `global_forward_mode`
        assert not server_args.enable_dp_attention  # 断言：不支持DP注意力模式
        self._counter = 0  # 计数器，用于累积权重值
        self._threshold = server_args.scheduler_recv_interval  # 阈值，达到后触发接收
        # All can be tuned if needed  # 所有权重参数都可以根据需要调整
        self._default_weight = envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DEFAULT.get()  # 默认权重值
        self._weight_of_forward_mode = {  # 不同前向模式对应的权重值映射
            ForwardMode.DECODE: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DECODE.get(),  # 解码模式权重
            ForwardMode.TARGET_VERIFY: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_TARGET_VERIFY.get(),  # 目标验证模式权重
            None: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_NONE.get(),  # 无前向模式时的权重
        }

    def handle(self, last_forward_mode: ForwardMode):  # 处理是否应该接收新请求
        should_recv = False  # 默认不接收

        last_weight = self._weight_of_forward_mode.get(  # 获取上次前向模式对应的权重
            last_forward_mode, self._default_weight  # 如果没有匹配则使用默认权重
        )
        self._counter += last_weight  # 累加权重到计数器

        if self._counter >= self._threshold:  # 如果计数器达到阈值
            self._counter = 0  # 重置计数器
            should_recv = True  # 标记需要接收

        return should_recv  # 返回是否应该接收
