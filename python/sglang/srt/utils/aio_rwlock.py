# 异步读写锁模块
# 实现基于asyncio的读写锁（RWLock），支持多读者单写者并发控制
# 提供公平性保证：写者等待时新读者也会被阻塞

import asyncio  # 异步IO模块


class RWLock:  # 异步读写锁，支持多读者单写者并发
    def __init__(self):  # 初始化读写锁
        # Protects internal state
        # 保护内部状态的锁
        self._lock = asyncio.Lock()  # 基础互斥锁

        # Condition variable used to wait for state changes
        # 用于等待状态变化的条件变量
        self._cond = asyncio.Condition(self._lock)  # 基于互斥锁的条件变量

        # Number of readers currently holding the lock
        # 当前持有锁的读者数量
        self._readers = 0  # 读者计数器

        # Whether a writer is currently holding the lock
        # 写者是否当前持有锁
        self._writer_active = False  # 写者活跃标志

        # How many writers are queued waiting for a turn
        # 等待轮次的写者数量
        self._waiting_writers = 0  # 等待中的写者计数器

    @property
    def reader_lock(self):  # 获取读者锁上下文管理器
        """
        A context manager for acquiring a shared (reader) lock.

        Example:
            async with rwlock.reader_lock:
                # read-only access
        """
        # 用于获取共享（读者）锁的上下文管理器。
        # 示例：
        #     async with rwlock.reader_lock:
        #         # 只读访问
        return _ReaderLock(self)  # 返回读者锁实例

    @property
    def writer_lock(self):  # 获取写者锁上下文管理器
        """
        A context manager for acquiring an exclusive (writer) lock.

        Example:
            async with rwlock.writer_lock:
                # exclusive access
        """
        # 用于获取排他（写者）锁的上下文管理器。
        # 示例：
        #     async with rwlock.writer_lock:
        #         # 排他访问
        return _WriterLock(self)  # 返回写者锁实例

    async def acquire_reader(self):  # 获取读者锁
        async with self._lock:  # 获取内部互斥锁
            # Wait until there is no active writer or waiting writer
            # to ensure fairness.
            # 等待直到没有活跃的写者或等待的写者，以确保公平性。
            while self._writer_active or self._waiting_writers > 0:  # 有写者活跃或等待时阻塞
                await self._cond.wait()  # 等待条件变量通知
            self._readers += 1  # 增加读者计数

    async def release_reader(self):  # 释放读者锁
        async with self._lock:  # 获取内部互斥锁
            self._readers -= 1  # 减少读者计数
            # If this was the last reader, wake up anyone waiting
            # (potentially a writer or new readers).
            # 如果这是最后一个读者，唤醒所有等待者（可能是写者或新读者）。
            if self._readers == 0:  # 如果没有读者了
                self._cond.notify_all()  # 通知所有等待者

    async def acquire_writer(self):  # 获取写者锁
        async with self._lock:  # 获取内部互斥锁
            # Increment the count of writers waiting
            # 增加等待中的写者计数
            self._waiting_writers += 1  # 标记有写者在等待
            try:  # 确保最终减少等待计数
                # Wait while either a writer is active or readers are present
                # 当写者活跃或有读者存在时等待
                while self._writer_active or self._readers > 0:  # 有活跃写者或读者时阻塞
                    await self._cond.wait()  # 等待条件变量通知
                self._writer_active = True  # 标记写者活跃
            finally:  # 无论是否异常都执行
                # Decrement waiting writers only after we've acquired the writer lock
                # 只有在获取写者锁后才减少等待写者计数
                self._waiting_writers -= 1  # 减少等待中的写者计数

    async def release_writer(self):  # 释放写者锁
        async with self._lock:  # 获取内部互斥锁
            self._writer_active = False  # 标记写者不再活跃
            # Wake up anyone waiting (readers or writers)
            # 唤醒所有等待者（读者或写者）
            self._cond.notify_all()  # 通知所有等待者

    async def is_locked(self):  # 检查锁是否被持有
        async with self._lock:  # 获取内部互斥锁
            return self._writer_active or self._readers > 0  # 写者活跃或有读者时返回True


class _ReaderLock:  # 读者锁上下文管理器
    def __init__(self, rwlock: RWLock):  # 初始化读者锁
        self._rwlock = rwlock  # 保存读写锁引用

    async def __aenter__(self):  # 进入异步上下文，获取读者锁
        await self._rwlock.acquire_reader()  # 获取读者锁
        return self  # 返回自身

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # 退出异步上下文，释放读者锁
        await self._rwlock.release_reader()  # 释放读者锁


class _WriterLock:  # 写者锁上下文管理器
    def __init__(self, rwlock: RWLock):  # 初始化写者锁
        self._rwlock = rwlock  # 保存读写锁引用

    async def __aenter__(self):  # 进入异步上下文，获取写者锁
        await self._rwlock.acquire_writer()  # 获取写者锁
        return self  # 返回自身

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # 退出异步上下文，释放写者锁
        await self._rwlock.release_writer()  # 释放写者锁
