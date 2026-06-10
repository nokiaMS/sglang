# 空闲休眠器
# 在调度器空闲时通过ZMQ Poller实现低功耗等待，避免CPU空转。
# 同时支持定期清理GPU缓存以释放闲置资源。

import zmq

from sglang.srt.environ import envs
from sglang.srt.observability.req_time_stats import real_time
from sglang.srt.platforms import current_platform


class IdleSleeper:
    """空闲休眠器，在长空闲期间降低系统功耗。

    在具有长非活动周期的设置中，当sglang空闲时降低系统功耗是理想的。
    这不仅节省功耗，还当请求最终到来时提供更多CPU散热空间。
    当多个GPU连接时这很重要，否则每个GPU会固定一个线程100% CPU使用率。

    最简单的解决方案是在所有可能接收需要立即处理的数据的socket上使用zmq.Poller。
    """

    def __init__(self, sockets):
        """初始化ZMQ Poller并注册所有需要监听的socket"""
        self.poller = zmq.Poller()
        self.last_empty_time = real_time()
        for s in sockets:
            self.poller.register(s, zmq.POLLIN)

        self.empty_cache_interval = envs.SGLANG_EMPTY_CACHE_INTERVAL.get()

    def maybe_sleep(self):
        """可能进入休眠等待，同时检查是否需要清理GPU缓存"""
        # 阻塞等待最多1秒，直到有消息到达
        self.poller.poll(1000)
        # 定期清理GPU缓存
        if (
            self.empty_cache_interval > 0
            and real_time() - self.last_empty_time > self.empty_cache_interval
        ):
            self.last_empty_time = real_time()
            current_platform.empty_cache()
