# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# LoRA 注册表模块：管理所有可用的 LoRA 适配器及其请求计数，支持并发推理和动态适配器加载/卸载。


import asyncio  # 导入异步IO库，用于协程和并发操作
from collections import OrderedDict  # 导入有序字典，用于维护LRU顺序
from dataclasses import dataclass, field, fields  # 导入数据类相关装饰器和函数
from typing import Dict, List, Optional, Union  # 导入类型注解
from uuid import NAMESPACE_URL, uuid4, uuid5  # 导入UUID生成函数，用于生成唯一标识符

from sglang.srt.utils import ConcurrentCounter  # 导入并发计数器，用于跟踪LoRA使用情况
from sglang.srt.utils.aio_rwlock import RWLock  # 导入异步读写锁，用于保护注册表的并发访问


@dataclass(frozen=True)  # 冻结的数据类，实例创建后不可修改
class LoRARef:
    """
    Reference record for a LoRA model.

    This object guarantees a unique ``lora_id`` and may include ``lora_name``, ``lora_path``, and ``pinned``.
    The ID eliminates conflicts from reused LoRA names or paths and can be used to generate deterministic cache
    keys (e.g., radix cache).
    """
    # LoRA 适配器的引用记录。
    # 保证唯一的 lora_id，可包含 lora_name、lora_path 和 pinned 标志。
    # 该 ID 消除了因 LoRA 名称或路径重复而产生的冲突，并可用于生成确定性缓存键（如 radix cache）。

    lora_id: str = field(default_factory=lambda: uuid4().hex)  # LoRA的唯一标识符，默认使用uuid4生成随机hex字符串
    lora_name: Optional[str] = None  # LoRA适配器的名称，可选
    lora_path: Optional[str] = None  # LoRA适配器的文件路径，可选
    pinned: Optional[bool] = None  # 是否固定（不被LRU淘汰），可选

    def __post_init__(self):  # 数据类初始化后的校验钩子
        if self.lora_id is None:  # 如果lora_id为None
            raise ValueError("lora_id cannot be None")  # 抛出值错误异常

    @staticmethod
    def deterministic_id(lora_name: str, lora_path: str) -> str:
        """Stable ``lora_id`` for ``--lora-paths`` adapters.

        Each node in a multi-node launch parses ``--lora-paths`` independently;
        ``uuid4`` would mint a different id per node for the same adapter,
        breaking cross-node lookups when the master broadcasts a request id.
        """
        # 为 --lora-paths 适配器生成稳定的 lora_id。
        # 在多节点启动时，每个节点独立解析 --lora-paths；
        # uuid4 会为同一适配器在不同节点生成不同的id，
        # 导致主节点广播请求id时跨节点查找失败。
        return uuid5(NAMESPACE_URL, f"{lora_name}\0{lora_path}").hex  # 使用uuid5基于名称和路径生成确定性hex字符串

    def __str__(self) -> str:  # 自定义字符串表示，用于调试和日志输出
        parts = [  # 收集所有非None字段的键值对
            f"{f.name}={value}"  # 格式化为"字段名=值"
            for f in fields(self)  # 遍历数据类的所有字段
            if (value := getattr(self, f.name)) is not None  # 仅包含值不为None的字段
        ]
        return f"{self.__class__.__name__}({', '.join(parts)})"  # 拼接为"LoRARef(field1=val1, field2=val2)"格式


class LoRARegistry:
    """
    The central registry to keep track of available LoRA adapters and ongoing LoRA requests.

    The `LoRARegistry` resides in the tokenizer manager process and acts as the single source of truth for all
    available LoRA adapters. It supports concurrent inference and dynamic adapter updates through a two-phase
    update / eventual consistency model between the tokenizer manager process and the scheduler processes.
    """
    # LoRA 注册表：跟踪可用 LoRA 适配器和正在进行的 LoRA 请求的中央注册表。
    # LoRARegistry 位于 tokenizer manager 进程中，是所有可用 LoRA 适配器的唯一事实来源。
    # 它通过 tokenizer manager 进程和调度器进程之间的两阶段更新/最终一致性模型，
    # 支持并发推理和动态适配器更新。

    def __init__(self, lora_paths: Optional[List[LoRARef]] = None):  # 初始化注册表，可选传入预加载的LoRA列表
        assert lora_paths is None or all(  # 断言：lora_paths为None或所有元素都是LoRARef实例
            isinstance(lora, LoRARef) for lora in lora_paths
        ), (
            "server_args.lora_paths should have been normalized to LoRARef objects during server initialization. "
            "Please file an issue if you see this error."
        )

        # A read-write lock to ensure adapters loading / unloading operations are exclusive.
        # Please note that the counter increment/decrement operations are not synchronized through this
        # lock, as they are designed to be non-blocking and can be performed concurrently.
        # 读写锁：确保适配器加载/卸载操作互斥。
        # 注意：计数器的增减操作不通过此锁同步，因为它们设计为非阻塞且可并发执行。
        self._registry_lock = RWLock()  # 创建异步读写锁实例
        # An ordered dictionary to hold LoRARef objects, mapping from LoRA name to LoRARef.
        # The LoRARefs are stored in LRU order, such that LoRA adapters that have been
        # most recently used are stored at the end. Note that lookups count for accesses.
        # Ties are broken arbitrarily.
        # 有序字典：保存 LoRARef 对象，从 LoRA 名称映射到 LoRARef。
        # LoRARef 按LRU顺序存储，最近使用的适配器排在末尾。查找也计为访问。
        # 顺序相同的情况下顺序不定。
        self._registry: OrderedDict[str, LoRARef] = OrderedDict()  # 创建有序字典作为注册表
        # Counters for ongoing requests, mapping from LoRA ID to ConcurrentCounter.
        # 正在进行的请求的计数器，从 LoRA ID 映射到 ConcurrentCounter。
        self._counters: Dict[str, ConcurrentCounter] = {}  # 创建字典存储每个LoRA的并发计数器

        # Initialize the registry with provided LoRA paths, if present.
        # 如果提供了 LoRA 路径，则初始化注册表。
        if lora_paths:  # 如果传入了lora_paths列表
            for lora_ref in lora_paths:  # 遍历每个LoRARef
                self._register_adapter(lora_ref)  # 注册该适配器

    async def register(self, lora_ref: LoRARef):  # 异步注册新的LoRA适配器
        """
        Register a new LoRARef object in the registry.

        Args:
            lora_ref (LoRARef): The LoRARef object to register.
        """
        # 在注册表中注册一个新的 LoRARef 对象。
        # 参数：
        #     lora_ref (LoRARef): 要注册的 LoRARef 对象。
        async with self._registry_lock.writer_lock:  # 获取写锁，确保独占访问
            self._register_adapter(lora_ref)  # 调用内部方法注册适配器

    async def unregister(self, lora_name: str) -> str:  # 异步注销LoRA适配器，返回被移除的LoRA ID
        """
        Unregister a LoRARef object from the registry and returns the removed LoRA ID.

        Args:
            lora_name (str): The name of the LoRA model to unregister.
        """
        # 从注册表中注销 LoRARef 对象并返回被移除的 LoRA ID。
        # 参数：
        #     lora_name (str): 要注销的 LoRA 模型名称。
        async with self._registry_lock.writer_lock:  # 获取写锁，确保独占访问
            lora_ref = self._registry.get(lora_name, None)  # 根据名称查找LoRARef，不存在则为None
            if lora_ref is None:  # 如果未找到该适配器
                raise ValueError(  # 抛出值错误异常
                    f"LoRA with name {lora_name} does not exist. Loaded LoRAs: {self._registry.keys()}"
                )
            del self._registry[lora_name]  # 从注册表中删除该适配器

        return lora_ref.lora_id  # 返回被移除适配器的ID

    async def acquire(self, lora_name: Union[str, List[str]]) -> Union[str, List[str]]:  # 异步获取LoRA ID并增加使用计数
        """
        Queries registry for LoRA IDs based on LoRA names and start tracking the usage of the corresponding LoRA adapters
        by incrementing its counter.
        """
        # 根据LoRA名称查询注册表获取LoRA ID，并通过递增计数器开始跟踪对应LoRA适配器的使用情况。

        def _lookup(name: str) -> str:  # 内部查找函数，根据名称查找LoRARef并返回其ID
            if name is None:  # 如果名称为None
                return None  # 直接返回None

            lora_ref = self._registry.get(name, None)  # 根据名称查找LoRARef
            if lora_ref is None:  # 如果未找到
                raise ValueError(  # 抛出值错误异常
                    f"The following requested LoRA adapters are not loaded: {name}\n"
                    f"Loaded adapters: {self._registry.keys()}."
                )
            self._registry.move_to_end(name)  # 将该适配器移到有序字典末尾（标记为最近使用）
            return lora_ref.lora_id  # 返回LoRA的ID

        if isinstance(lora_name, str):  # 如果传入的是单个字符串名称
            async with self._registry_lock.writer_lock:  # 获取写锁（因为move_to_end需要修改顺序）
                lora_id = _lookup(lora_name)  # 查找并获取LoRA ID

            await self._counters[lora_id].increment(notify_all=False)  # 异步递增该LoRA的计数器，不通知等待者
            return lora_id  # 返回LoRA ID
        elif isinstance(lora_name, list):  # 如果传入的是名称列表
            async with self._registry_lock.writer_lock:  # 获取写锁
                lora_ids = [_lookup(name) for name in lora_name]  # 批量查找所有名称对应的LoRA ID

            # Increment the counters only after all IDs are looked up.
            # 仅在所有ID查找完成后才递增计数器。
            await asyncio.gather(  # 并发递增所有非None LoRA的计数器
                *[
                    self._counters[id].increment(notify_all=False)  # 异步递增计数器
                    for id in lora_ids  # 遍历所有LoRA ID
                    if id is not None  # 跳过None值
                ]
            )
            return lora_ids  # 返回LoRA ID列表
        else:  # 其他类型
            raise TypeError("lora_name must be either a string or a list of strings.")  # 抛出类型错误

    async def release(self, lora_id: Union[str, List[str]]):  # 异步释放LoRA适配器，递减使用计数
        """
        Decrements the usage counter for a LoRA adapter, indicating that it is no longer in use.
        """
        # 递减 LoRA 适配器的使用计数，表示该适配器不再被使用。
        async with self._registry_lock.reader_lock:  # 获取读锁（不需要修改注册表，只操作计数器）
            if isinstance(lora_id, str):  # 如果传入的是单个字符串ID
                await self._counters[lora_id].decrement()  # 异步递减该LoRA的计数器
            elif isinstance(lora_id, list):  # 如果传入的是ID列表
                await asyncio.gather(  # 并发递减所有非None LoRA的计数器
                    *[
                        self._counters[id].decrement()  # 异步递减计数器
                        for id in lora_id  # 遍历所有LoRA ID
                        if id is not None  # 跳过None值
                    ]
                )
            else:  # 其他类型
                raise TypeError("lora_id must be either a string or a list of strings.")  # 抛出类型错误

    async def wait_for_unload(self, lora_id: str):  # 异步等待LoRA适配器的使用计数归零
        """
        Waits until the usage counter for a LoRA adapter reaches zero, indicating that it is no longer in use.
        This is useful for ensuring that a LoRA adapter can be safely unloaded.

        This method itself is not synchronized, which is safe because it should only be called during LoRA unloading,
        which itself is guaranteed to be sequential.
        """
        # 等待 LoRA 适配器的使用计数归零，表示该适配器不再被使用。
        # 这对于确保 LoRA 适配器可以安全卸载非常有用。
        # 此方法本身不进行同步，这是安全的，因为它只在 LoRA 卸载期间被调用，
        # 而卸载操作本身保证是顺序执行的。
        assert (  # 断言：该LoRA已从注册表中移除
            lora_id not in self._registry
        ), "wait_for_unload should only be called after the LoRA adapter has been unregistered. "
        assert (  # 断言：该LoRA仍有计数器记录
            lora_id in self._counters
        ), "The LoRA ID should still have a counter if it has been registered before."

        # Wait until no requests are using this LoRA adapter.
        # 等待直到没有请求正在使用此 LoRA 适配器。
        await self._counters[lora_id].wait_for_zero()  # 异步等待计数器归零
        del self._counters[lora_id]  # 删除该LoRA的计数器记录

    async def get_unregistered_loras(self, lora_name: set[str]):  # 异步获取未注册的LoRA名称列表
        """
        Returns all LoRA adapters in lora_name that are not found in self._registry.
        """
        # 返回 lora_name 中未在 self._registry 中找到的所有 LoRA 适配器名称。
        async with self._registry_lock.writer_lock:  # 获取写锁（因为move_to_end需要修改顺序）
            unregistered_loras = []  # 初始化未注册LoRA名称列表

            for name in lora_name:  # 遍历每个查询的名称
                if name in self._registry:  # 如果该名称已在注册表中
                    # This counts as a lookup, so we want to update the cache
                    # 这算作一次查找，因此我们需要更新缓存顺序
                    self._registry.move_to_end(name)  # 将其移到末尾（标记为最近使用）
                else:  # 如果该名称不在注册表中
                    unregistered_loras.append(name)  # 添加到未注册列表

            return unregistered_loras  # 返回未注册的LoRA名称列表

    async def lru_lora_name(self, exclude_pinned=False):  # 异步获取最近最少使用的LoRA名称
        """
        Returns the least recently used LoRA adapter.
        If exclude_pinned is True, then return the LRU LoRA adapter that isn't pinned.
        """
        # 返回最近最少使用的 LoRA 适配器名称。
        # 如果 exclude_pinned 为 True，则返回未被固定的最近最少使用的 LoRA 适配器。
        async with self._registry_lock.reader_lock:  # 获取读锁
            if not exclude_pinned:  # 如果不排除固定的适配器
                return next(iter(self._registry), None)  # 返回有序字典中第一个（最久未使用的）名称，空则返回None

            for lora_name, lora_ref in self._registry.items():  # 遍历所有注册的适配器（按LRU顺序）
                if not lora_ref.pinned:  # 如果该适配器未被固定
                    return lora_name  # 返回第一个未固定的名称（即最久未使用的非固定适配器）
            else:  # 所有适配器都被固定
                return None  # 返回None

    def _register_adapter(self, lora_ref: LoRARef):  # 内部方法：注册LoRA适配器（不加锁）
        """
        Internal helper method to register a LoRA adapter.
        """
        # 内部辅助方法：注册 LoRA 适配器。

        if lora_ref.lora_name in self._registry:  # 如果该名称已被注册
            raise ValueError(  # 抛出值错误异常
                f"LoRA with name {lora_ref.lora_name} already exists. Loaded LoRAs: {self._registry.keys()}"
            )
        self._registry[lora_ref.lora_name] = lora_ref  # 将LoRARef存入注册表
        self._counters[lora_ref.lora_id] = ConcurrentCounter()  # 为该LoRA创建新的并发计数器
        return lora_ref  # 返回注册的LoRARef

    @property
    def num_registered_loras(self) -> int:  # 属性：当前注册的LoRA适配器数量
        """
        Returns the total number of LoRA adapters currently registered.
        """
        # 返回当前注册的 LoRA 适配器总数。
        return len(self._registry)  # 返回注册表中的条目数

    def get_all_adapters(self) -> Dict[str, LoRARef]:  # 获取所有已注册的LoRA适配器
        """
        Returns a dictionary of all registered LoRA adapters.

        Returns:
            Dict[str, LoRARef]: A dictionary mapping LoRA names to LoRARef objects.
        """
        # 返回所有已注册 LoRA 适配器的字典。
        # 返回值：
        #     Dict[str, LoRARef]: 从 LoRA 名称映射到 LoRARef 对象的字典。
        return dict(self._registry)  # 将有序字典转换为普通字典返回
