# 统一视频解码器，优先使用torchcodec后端，decord作为回退
# 支持CPU和CUDA GPU解码，提供多线程并行解码和帧提取功能
"""Unified video decoder: torchcodec preferred, decord as fallback."""  # 统一视频解码器：优先torchcodec，decord作为回退

import logging  # 导入日志记录模块
import os  # 导入操作系统接口模块

import numpy as np  # 导入NumPy数组库

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

try:  # 尝试导入torchcodec
    from torchcodec.decoders import VideoDecoder  # 导入torchcodec视频解码器

    _BACKEND = "torchcodec"  # 设置后端为torchcodec
except (ImportError, RuntimeError):  # 如果导入失败
    _BACKEND = "decord"  # 回退到decord后端


_cuda_backend_enabled: bool | None = None  # CUDA后端启用状态（None表示未检查）


def _try_cuda_backend() -> bool:  # 尝试启用torchcodec的CUDA后端，首次调用后缓存结果
    """Try to enable torchcodec CUDA backend. Caches result after first call."""  # 尝试启用torchcodec CUDA后端，首次调用后缓存结果
    global _cuda_backend_enabled  # 声明使用全局变量
    if _cuda_backend_enabled is not None:  # 如果已经检查过
        return _cuda_backend_enabled  # 返回缓存结果
    try:  # 尝试启用CUDA后端
        from torchcodec.decoders import set_cuda_backend  # 导入设置CUDA后端函数

        set_cuda_backend("beta")  # 启用beta版CUDA后端
        _cuda_backend_enabled = True  # 标记CUDA后端已启用
    except Exception:  # 如果启用失败
        _cuda_backend_enabled = False  # 标记CUDA后端不可用
    return _cuda_backend_enabled  # 返回CUDA后端启用状态


class VideoDecoderWrapper:  # 统一视频解码器包装类
    """Unified video decoder that uses torchcodec when available, decord as fallback.  # 统一视频解码器，torchcodec可用时使用，否则回退到decord

    All frames are returned in NHWC uint8 numpy format for consistency.  # 所有帧以NHWC uint8 numpy格式返回以保持一致性
    """

    def __init__(self, source, device: str = "cpu", num_decode_threads: int = 0):  # 初始化视频解码器
        """source: file path (str) or video bytes.  # source: 文件路径（字符串）或视频字节
        device: "cpu" or "cuda". GPU decoding only supported with torchcodec.  # device: "cpu"或"cuda"。GPU解码仅torchcodec支持
        num_decode_threads: number of parallel decoder instances for frame  # num_decode_threads: 帧提取的并行解码器实例数
            extraction (torchcodec only). 0 = auto (capped at 16),  # 仅torchcodec。0=自动（上限16）
            1 = single decoder. Set > 1 to split frame indices across  # 1=单解码器。设>1以在
            multiple decoders in parallel threads.  # 多个并行线程的解码器间分割帧索引
        """
        self._source = source  # 保存视频源
        self._num_decode_threads = num_decode_threads  # 保存并行解码线程数
        self._source_bytes = source if isinstance(source, bytes) else None  # 如果源是字节则保存
        self._source_path = source if isinstance(source, str) else None  # 如果源是路径则保存
        self._tmp_path = None  # 临时文件路径（decord需要文件路径）
        if _BACKEND == "torchcodec":  # 如果使用torchcodec后端
            kwargs = {"dimension_order": "NHWC"}  # 设置维度顺序为NHWC
            if device == "cuda" and _try_cuda_backend():  # 如果请求CUDA且CUDA后端可用
                kwargs["device"] = "cuda"  # 设置CUDA设备
            self._tc_kwargs = kwargs  # 保存torchcodec参数
            try:  # 尝试创建torchcodec解码器
                self._decoder = VideoDecoder(source, **kwargs)  # 使用指定参数创建解码器
            except RuntimeError:  # 如果创建失败
                if "device" in kwargs:  # 如果失败与设备有关
                    logger.warning("CUDA video decoding failed, falling back to CPU.")  # 记录CUDA解码失败回退到CPU
                    kwargs.pop("device")  # 移除CUDA设备参数
                    self._tc_kwargs = kwargs  # 更新参数
                    self._decoder = VideoDecoder(source, **kwargs)  # 使用CPU参数重新创建解码器
                else:
                    raise  # 抛出非设备相关的运行时错误
        else:  # 使用decord后端
            from decord import VideoReader, cpu  # 导入decord视频读取器

            if isinstance(source, bytes):  # 如果源是字节数据
                import tempfile  # 导入临时文件模块

                fd, tmp_path = tempfile.mkstemp(suffix=".mp4")  # 创建临时MP4文件
                try:
                    os.write(fd, source)  # 将字节数据写入临时文件
                finally:
                    os.close(fd)  # 关闭文件描述符
                self._tmp_path = tmp_path  # 保存临时文件路径
                self._decoder = VideoReader(tmp_path, ctx=cpu(0))  # 从临时文件创建解码器
            else:  # 源是文件路径
                self._decoder = VideoReader(source, ctx=cpu(0))  # 从文件路径创建解码器

    def __len__(self):  # 返回视频帧数
        return len(self._decoder)  # 委托给底层解码器

    def __getitem__(self, idx):  # 获取指定索引的单帧
        """Return single frame as numpy NHWC uint8."""  # 返回单帧的NHWC uint8 numpy数组
        if _BACKEND == "torchcodec":  # 如果使用torchcodec
            return self._decoder[idx].numpy()  # 转为numpy数组
        else:  # 使用decord
            frame = self._decoder[idx]  # 获取帧
            return frame.asnumpy() if hasattr(frame, "asnumpy") else np.array(frame)  # 转为numpy数组

    @property
    def avg_fps(self) -> float:  # 获取视频平均帧率
        if _BACKEND == "torchcodec":  # 如果使用torchcodec
            return self._decoder.metadata.average_fps  # 从元数据获取帧率
        else:  # 使用decord
            return self._decoder.get_avg_fps()  # 调用get_avg_fps获取帧率

    def get_frames_at(self, indices: list) -> np.ndarray:  # 获取指定索引的帧，返回NHWC uint8 numpy数组
        """Return frames at given indices as numpy array with shape (N, H, W, C)."""  # 返回指定索引的帧，形状为(N, H, W, C)的numpy数组
        if _BACKEND == "torchcodec":  # 如果使用torchcodec
            batch = self._decoder.get_frames_at(indices)  # 批量获取帧
            return batch.data.numpy()  # 转为numpy数组
        else:  # 使用decord
            return self._decoder.get_batch(indices).asnumpy()  # 批量获取并转为numpy数组

    def get_frames_as_tensor(self, indices: list):  # 获取指定索引的帧，返回固定内存的PyTorch张量
        """Return frames at given indices as a torch tensor (NHWC, uint8, pinned memory)."""  # 返回指定索引的帧，NHWC uint8固定内存张量
        import torch  # 导入PyTorch

        if (  # 如果满足多线程解码条件
            _BACKEND == "torchcodec"  # 使用torchcodec后端
            and self._num_decode_threads != 1  # 非单线程模式
            and len(indices) > 1  # 请求多于1帧
        ):
            num_threads = self._num_decode_threads  # 获取线程数
            if num_threads <= 0:  # 如果为自动模式
                num_threads = min(os.cpu_count() or 8, 16)  # 自动设置为CPU核心数或16中的较小值
            num_threads = min(num_threads, len(indices))  # 线程数不超过帧数
            if num_threads > 1:  # 如果需要多线程
                return self._parallel_decode(indices, num_threads)  # 使用并行解码

        if _BACKEND == "torchcodec":  # 单线程torchcodec路径
            batch = self._decoder.get_frames_at(indices)  # 批量获取帧
            return batch.data.pin_memory()  # 固定内存张量
        else:  # 单线程decord路径
            arr = self._decoder.get_batch(indices).asnumpy()  # 批量获取并转为numpy
            return torch.from_numpy(arr).pin_memory()  # 转为固定内存张量

    def _parallel_decode(self, indices, num_threads):  # 使用多个VideoDecoder实例并行解码帧
        """Decode frames using multiple VideoDecoder instances in parallel threads."""  # 使用多个VideoDecoder实例在并行线程中解码帧
        from concurrent.futures import ThreadPoolExecutor, as_completed  # 导入线程池执行器

        import torch  # 导入PyTorch

        chunks = [list(c) for c in np.array_split(indices, num_threads) if len(c) > 0]  # 将帧索引分割为多个块
        source = self._source  # 获取视频源
        kwargs = self._tc_kwargs  # 获取torchcodec参数

        def _decode_chunk(chunk):  # 解码单个帧索引块
            d = VideoDecoder(source, **kwargs)  # 创建新的解码器实例
            return d.get_frames_at(chunk).data  # 获取帧数据张量

        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:  # 创建线程池
            future_to_idx = {  # 映射未来任务到块索引
                executor.submit(_decode_chunk, chunk): idx  # 提交解码任务
                for idx, chunk in enumerate(chunks)  # 遍历每个块
            }
            results = [None] * len(chunks)  # 初始化结果列表
            for future in as_completed(future_to_idx):  # 等待任务完成
                idx = future_to_idx[future]  # 获取块索引
                results[idx] = future.result()  # 保存解码结果

        return torch.cat(results, dim=0).pin_memory()  # 拼接所有结果并固定内存

    @property
    def source_bytes(self) -> bytes | None:  # 获取原始视频字节数据
        """Return raw video bytes if available (needed for audio extraction)."""  # 如果可用则返回原始视频字节（音频提取需要）
        if self._source_bytes is not None:  # 如果源是字节数据
            return self._source_bytes  # 直接返回
        path = self._tmp_path or self._source_path  # 获取文件路径（临时或原始）
        if path is not None:  # 如果有路径
            if os.path.isfile(path):  # 如果文件存在
                with open(path, "rb") as f:  # 以二进制模式打开
                    return f.read()  # 读取并返回所有字节
        return None  # 无法获取字节数据

    def close(self):  # 显式清理临时文件
        """Explicitly clean up temporary files."""  # 显式清理临时文件
        if self._tmp_path is not None:  # 如果有临时文件
            if os.path.exists(self._tmp_path):  # 如果临时文件存在
                os.unlink(self._tmp_path)  # 删除临时文件
            self._tmp_path = None  # 清除临时文件路径

    def __del__(self):  # 析构函数
        self.close()  # 调用清理方法

    def __enter__(self):  # 上下文管理器入口
        return self  # 返回自身

    def __exit__(self, *args):  # 上下文管理器退出
        self.close()  # 调用清理方法
