# 文件说明：TBO（Two-Batch Overlap）注意力后端实现
# 该模块实现了两批次重叠调度策略下的注意力机制后端，
# 将主后端和子后端组合在一起，支持CUDA图捕获和重放时的批次拆分。

from typing import TYPE_CHECKING, Callable, List, Optional  # 导入类型提示相关模块 # 导入类型检查、可调用对象、列表和可选类型

import torch  # 导入PyTorch深度学习框架 # 导入PyTorch框架

from sglang.srt.batch_overlap import two_batch_overlap  # 导入两批次重叠调度工具 # 导入两批次重叠工具模块
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend  # 导入注意力后端基类 # 导入注意力后端基类
from sglang.srt.speculative.spec_info import SpecInput  # 导入投机解码规格信息 # 导入投机解码输入规格类

if TYPE_CHECKING:  # 如果是类型检查阶段 # 类型检查时才导入
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode  # 导入前向批次信息和前向模式 # 导入前向批次和前向模式类型


class TboAttnBackend(AttentionBackend):  # TBO注意力后端类，继承自注意力后端基类 # TBO注意力后端类
    def __init__(self, primary: AttentionBackend, children: List[AttentionBackend]):  # 初始化方法，接收主后端和子后端列表 # 初始化TBO后端
        super().__init__()  # 调用父类初始化 # 调用基类初始化
        self.primary = primary  # 保存主注意力后端 # 保存主后端实例
        self.children = children  # 保存子注意力后端列表 # 保存子后端列表
        # Dispatcher aliases the primary's pool refs so get_attn_backend()
        # reads through TboAttnBackend resolve to the underlying pool.
        # 调度器别名为primary的池引用，使get_attn_backend()通过TboAttnBackend读取底层池。
        self.token_to_kv_pool = primary.token_to_kv_pool  # 别名引用主后端的token到KV池 # 引用主后端的KV缓存池
        self.req_to_token_pool = primary.req_to_token_pool  # 别名引用主后端的请求到token池 # 引用主后端的请求-token映射池

    @classmethod  # 类方法装饰器 # 类方法
    def init_new(cls, creator: Callable[[], AttentionBackend]):  # 工厂方法，创建TBO后端实例 # 创建新的TBO后端实例
        return cls(  # 返回新创建的TBO后端 # 返回TBO实例
            primary=creator(),  # 创建主后端 # 创建主后端
            children=[creator() for _ in range(2)],  # 创建2个子后端 # 创建2个子后端
        )

    def init_forward_metadata(self, forward_batch: "ForwardBatch"):  # 初始化前向元数据 # 初始化前向传播元数据
        self.primary.init_forward_metadata(forward_batch=forward_batch)  # 初始化主后端的前向元数据 # 初始化主后端元数据
        if forward_batch.tbo_children is not None:  # 如果TBO子批次存在 # 检查是否有TBO子批次
            for child, forward_batch_child in zip(  # 遍历子后端和对应子批次 # 遍历子后端和子批次
                self.children, forward_batch.tbo_children, strict=True  # 严格匹配子后端和子批次 # 严格模式配对
            ):
                if forward_batch_child.batch_size > 0:  # 如果子批次的批次大小大于0 # 检查子批次非空
                    child.init_forward_metadata(forward_batch=forward_batch_child)  # 初始化子后端的前向元数据 # 初始化子后端元数据

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):  # 初始化CUDA图状态 # 初始化CUDA图捕获状态
        self.primary.init_cuda_graph_state(max_bs=max_bs, max_num_tokens=max_num_tokens)  # 初始化主后端的CUDA图状态 # 初始化主后端CUDA图状态
        for item in self.children:  # 遍历所有子后端 # 遍历子后端
            # TODO for children, maybe can provide *smaller* max_bs to optimize
            # 待办：对于子后端，可以提供更小的max_bs来优化
            item.init_cuda_graph_state(max_bs=max_bs, max_num_tokens=max_num_tokens)  # 初始化子后端的CUDA图状态 # 初始化子后端CUDA图状态

    def init_forward_metadata_capture_cuda_graph(  # CUDA图捕获时初始化前向元数据 # CUDA图捕获阶段初始化前向元数据
        self,
        bs: int,  # 批次大小 # 批次大小
        num_tokens: int,  # token数量 # token数量
        req_pool_indices: torch.Tensor,  # 请求池索引 # 请求池索引张量
        seq_lens: torch.Tensor,  # 序列长度 # 序列长度张量
        encoder_lens: Optional[torch.Tensor],  # 编码器长度 # 编码器长度（可选）
        forward_mode: "ForwardMode",  # 前向模式 # 前向传播模式
        spec_info: Optional[SpecInput],  # 投机解码信息 # 投机解码输入信息（可选）
    ):
        self.primary.init_forward_metadata_capture_cuda_graph(  # 初始化主后端的CUDA图捕获元数据 # 初始化主后端CUDA图捕获元数据
            bs=bs,  # 批次大小 # 传入批次大小
            num_tokens=num_tokens,  # token数量 # 传入token数量
            req_pool_indices=req_pool_indices,  # 请求池索引 # 传入请求池索引
            seq_lens=seq_lens,  # 序列长度 # 传入序列长度
            encoder_lens=encoder_lens,  # 编码器长度 # 传入编码器长度
            forward_mode=forward_mode,  # 前向模式 # 传入前向模式
            spec_info=spec_info,  # 投机解码信息 # 传入投机解码信息
        )

        self._init_forward_metadata_cuda_graph_children(  # 初始化子后端的CUDA图元数据 # 初始化子后端CUDA图元数据
            fn_name="init_forward_metadata_capture_cuda_graph",  # 函数名称 # 指定函数名
            bs=bs,  # 批次大小 # 传入批次大小
            req_pool_indices=req_pool_indices,  # 请求池索引 # 传入请求池索引
            seq_lens=seq_lens,  # 序列长度 # 传入序列长度
            encoder_lens=encoder_lens,  # 编码器长度 # 传入编码器长度
            forward_mode=forward_mode,  # 前向模式 # 传入前向模式
            spec_info=spec_info,  # 投机解码信息 # 传入投机解码信息
            capture_num_tokens=num_tokens,  # 捕获token数量 # 传入捕获token数量
        )

    def init_forward_metadata_replay_cuda_graph(  # CUDA图重放时初始化前向元数据 # CUDA图重放阶段初始化前向元数据
        self,
        bs: int,  # 批次大小 # 批次大小
        req_pool_indices: torch.Tensor,  # 请求池索引 # 请求池索引张量
        seq_lens: torch.Tensor,  # 序列长度 # 序列长度张量
        seq_lens_sum: int,  # 序列长度总和 # 序列长度总和
        encoder_lens: Optional[torch.Tensor],  # 编码器长度 # 编码器长度（可选）
        forward_mode: "ForwardMode",  # 前向模式 # 前向传播模式
        spec_info: Optional[SpecInput],  # 投机解码信息 # 投机解码输入信息（可选）
        seq_lens_cpu: Optional[torch.Tensor],  # CPU上的序列长度 # CPU端序列长度（可选）
    ):
        self.primary.init_forward_metadata_replay_cuda_graph(  # 初始化主后端的CUDA图重放元数据 # 初始化主后端CUDA图重放元数据
            bs=bs,  # 批次大小 # 传入批次大小
            req_pool_indices=req_pool_indices,  # 请求池索引 # 传入请求池索引
            seq_lens=seq_lens,  # 序列长度 # 传入序列长度
            seq_lens_sum=seq_lens_sum,  # 序列长度总和 # 传入序列长度总和
            encoder_lens=encoder_lens,  # 编码器长度 # 传入编码器长度
            forward_mode=forward_mode,  # 前向模式 # 传入前向模式
            spec_info=spec_info,  # 投机解码信息 # 传入投机解码信息
            seq_lens_cpu=seq_lens_cpu,  # CPU上的序列长度 # 传入CPU端序列长度
        )

        self._init_forward_metadata_cuda_graph_children(  # 初始化子后端的CUDA图元数据 # 初始化子后端CUDA图元数据
            fn_name="init_forward_metadata_replay_cuda_graph",  # 函数名称 # 指定函数名
            bs=bs,  # 批次大小 # 传入批次大小
            req_pool_indices=req_pool_indices,  # 请求池索引 # 传入请求池索引
            seq_lens=seq_lens,  # 序列长度 # 传入序列长度
            encoder_lens=encoder_lens,  # 编码器长度 # 传入编码器长度
            forward_mode=forward_mode,  # 前向模式 # 传入前向模式
            spec_info=spec_info,  # 投机解码信息 # 传入投机解码信息
            replay_seq_lens_sum=seq_lens_sum,  # 重放序列长度总和 # 传入重放序列长度总和
            replay_seq_lens_cpu=seq_lens_cpu,  # 重放CPU序列长度 # 传入重放CPU端序列长度
        )

    def _init_forward_metadata_cuda_graph_children(  # 初始化子后端的CUDA图前向元数据（内部方法） # 初始化子后端CUDA图元数据的内部方法
        self,
        fn_name: str,  # 函数名称 # 要调用的函数名
        # common args
        # 公共参数
        # 公共参数
        bs: int,  # 批次大小 # 批次大小
        req_pool_indices: torch.Tensor,  # 请求池索引 # 请求池索引张量
        seq_lens: torch.Tensor,  # 序列长度 # 序列长度张量
        encoder_lens: Optional[torch.Tensor],  # 编码器长度 # 编码器长度（可选）
        forward_mode: "ForwardMode",  # 前向模式 # 前向传播模式
        spec_info: Optional[SpecInput],  # 投机解码信息 # 投机解码输入信息（可选）
        # capture args
        # 捕获参数
        # 捕获阶段参数
        capture_num_tokens: int = None,  # 捕获token数量 # 捕获阶段的token数量
        # replay args
        # 重放参数
        # 重放阶段参数
        replay_seq_lens_sum: int = None,  # 重放序列长度总和 # 重放阶段的序列长度总和
        replay_seq_lens_cpu: Optional[torch.Tensor] = None,  # 重放CPU序列长度 # 重放阶段的CPU端序列长度
    ):
        token_num_per_seq = two_batch_overlap.get_token_num_per_seq(  # 获取每个序列的token数量 # 获取每序列token数
            forward_mode=forward_mode, spec_info=spec_info  # 传入前向模式和投机信息 # 传入前向模式和投机信息
        )
        if fn_name == "init_forward_metadata_capture_cuda_graph":  # 如果是CUDA图捕获函数 # 判断是否为捕获阶段
            assert (  # 断言检查 # 断言
                capture_num_tokens == bs * token_num_per_seq  # 捕获token数应等于批次大小乘以每序列token数 # 验证token数量一致性
            ), "For target-verify or decode mode, num_tokens should be equal to token_num_per_seq * bs"  # 错误提示信息 # 错误提示
        num_tokens = bs * token_num_per_seq  # 计算总token数量 # 计算总token数

        tbo_split_seq_index, tbo_split_token_index = (  # 获取TBO拆分的序列和token索引 # 获取拆分索引
            two_batch_overlap.compute_split_indices_for_cuda_graph_replay(  # 计算CUDA图重放的拆分索引 # 计算拆分索引
                forward_mode=forward_mode,  # 前向模式 # 传入前向模式
                cuda_graph_num_tokens=num_tokens,  # CUDA图token数量 # 传入CUDA图token数
                spec_info=spec_info,  # 投机解码信息 # 传入投机信息
            )
        )

        num_tokens_child_left = tbo_split_token_index  # 左侧子后端的token数量 # 左子后端token数
        num_tokens_child_right = num_tokens - tbo_split_token_index  # 右侧子后端的token数量 # 右子后端token数
        bs_child_left = tbo_split_seq_index  # 左侧子后端的批次大小 # 左子后端批次大小
        bs_child_right = bs - bs_child_left  # 右侧子后端的批次大小 # 右子后端批次大小

        assert (  # 断言检查 # 断言
            num_tokens_child_left > 0 and num_tokens_child_right > 0  # 确保两侧子后端都有token # 确保两侧都有token
        ), f"{num_tokens_child_left=} {num_tokens_child_right=} {forward_mode=} {num_tokens=}"  # 错误提示信息 # 错误提示

        common_pre_split_args = dict(  # 拆分前的公共参数字典 # 公共拆分前参数
            fn_name=fn_name,  # 函数名称 # 函数名
            bs=bs,  # 批次大小 # 批次大小
            req_pool_indices=req_pool_indices,  # 请求池索引 # 请求池索引
            seq_lens=seq_lens,  # 序列长度 # 序列长度
            encoder_lens=encoder_lens,  # 编码器长度 # 编码器长度
            forward_mode=forward_mode,  # 前向模式 # 前向模式
            spec_info=spec_info,  # 投机解码信息 # 投机信息
            capture_num_tokens=capture_num_tokens,  # 捕获token数量 # 捕获token数
            replay_seq_lens_sum=replay_seq_lens_sum,  # 重放序列长度总和 # 重放序列长度总和
            replay_seq_lens_cpu=replay_seq_lens_cpu,  # 重放CPU序列长度 # 重放CPU序列长度
        )

        args_left = _init_forward_metadata_cuda_graph_split(  # 构建左侧子后端的参数 # 构建左子后端参数
            output_bs=bs_child_left,  # 左侧输出批次大小 # 左子后端批次大小
            seq_slice=slice(None, tbo_split_seq_index),  # 左侧序列切片 # 左侧序列切片
            **common_pre_split_args,  # 公共参数 # 传入公共参数
        )
        args_right = _init_forward_metadata_cuda_graph_split(  # 构建右侧子后端的参数 # 构建右子后端参数
            output_bs=bs_child_right,  # 右侧输出批次大小 # 右子后端批次大小
            seq_slice=slice(tbo_split_seq_index, None),  # 右侧序列切片 # 右侧序列切片
            **common_pre_split_args,  # 公共参数 # 传入公共参数
        )

        child_left, child_right = self.children  # 获取左右子后端 # 获取左右子后端
        getattr(child_left, fn_name)(**args_left)  # 调用左侧子后端的指定函数 # 调用左子后端函数
        getattr(child_right, fn_name)(**args_right)  # 调用右侧子后端的指定函数 # 调用右子后端函数

    def get_cuda_graph_seq_len_fill_value(self):  # 获取CUDA图序列长度填充值 # 获取CUDA图序列长度填充值
        ans = self.primary.get_cuda_graph_seq_len_fill_value()  # 从主后端获取填充值 # 从主后端获取填充值
        for child in self.children:  # 遍历所有子后端 # 遍历子后端
            assert ans == child.get_cuda_graph_seq_len_fill_value()  # 断言子后端的填充值与主后端一致 # 验证子后端填充值一致
        return ans  # 返回填充值 # 返回填充值

    def forward(self, *args, **kwargs):  # 通用前向传播方法 # 通用前向传播
        return self.primary.forward(*args, **kwargs)  # 委托给主后端执行 # 委托给主后端

    def forward_extend(self, *args, **kwargs):  # 扩展阶段前向传播 # 扩展阶段前向传播
        return self.primary.forward_extend(*args, **kwargs)  # 委托给主后端执行 # 委托给主后端

    def forward_decode(self, *args, **kwargs):  # 解码阶段前向传播 # 解码阶段前向传播
        return self.primary.forward_decode(*args, **kwargs)  # 委托给主后端执行 # 委托给主后端

    def get_indexer_metadata(self, layer_id: int, forward_batch: "ForwardBatch"):  # 获取索引器元数据 # 获取索引器元数据
        return self.primary.get_indexer_metadata(layer_id, forward_batch)  # 委托给主后端获取 # 委托给主后端


def _init_forward_metadata_cuda_graph_split(  # 为CUDA图拆分构建子后端参数（模块级函数） # 构建CUDA图拆分子后端参数的辅助函数
    fn_name: str,  # 函数名称 # 要调用的函数名
    seq_slice: slice,  # 序列切片 # 序列切片范围
    output_bs: int,  # 输出批次大小 # 输出批次大小
    # common args
    # 公共参数
    # 公共参数
    bs: int,  # 原始批次大小 # 原始批次大小
    req_pool_indices: torch.Tensor,  # 请求池索引 # 请求池索引张量
    seq_lens: torch.Tensor,  # 序列长度 # 序列长度张量
    encoder_lens: Optional[torch.Tensor],  # 编码器长度 # 编码器长度（可选）
    forward_mode: "ForwardMode",  # 前向模式 # 前向传播模式
    spec_info: Optional[SpecInput],  # 投机解码信息 # 投机解码输入信息（可选）
    # capture args
    # 捕获参数
    # 捕获阶段参数
    capture_num_tokens: int = None,  # 捕获token数量 # 捕获阶段的token数量
    # replay args
    # 重放参数
    # 重放阶段参数
    replay_seq_lens_sum: int = None,  # 重放序列长度总和 # 重放阶段的序列长度总和
    replay_seq_lens_cpu: Optional[torch.Tensor] = None,  # 重放CPU序列长度 # 重放阶段的CPU端序列长度
):
    token_num_per_seq = two_batch_overlap.get_token_num_per_seq(  # 获取每个序列的token数量 # 获取每序列token数
        forward_mode=forward_mode, spec_info=spec_info  # 传入前向模式和投机信息 # 传入前向模式和投机信息
    )
    assert encoder_lens is None, "encoder_lens is not supported yet"  # 断言编码器长度不支持 # 编码器长度暂不支持
    if spec_info is not None:  # 如果投机解码信息存在 # 检查投机信息是否存在
        output_spec_info = two_batch_overlap.split_spec_info(  # 拆分投机解码信息 # 拆分投机信息
            spec_info=spec_info,  # 原始投机信息 # 原始投机信息
            start_seq_index=seq_slice.start if seq_slice.start is not None else 0,  # 起始序列索引 # 起始序列索引
            end_seq_index=seq_slice.stop if seq_slice.stop is not None else bs,  # 结束序列索引 # 结束序列索引
            start_token_index=(  # 起始token索引 # 起始token索引
                seq_slice.start * token_num_per_seq  # 起始序列索引乘以每序列token数 # 计算起始token位置
                if seq_slice.start is not None  # 如果起始索引不为空 # 检查起始索引
                else 0  # 否则为0 # 默认为0
            ),
            end_token_index=(  # 结束token索引 # 结束token索引
                seq_slice.stop * token_num_per_seq  # 结束序列索引乘以每序列token数 # 计算结束token位置
                if seq_slice.stop is not None  # 如果结束索引不为空 # 检查结束索引
                else bs * token_num_per_seq  # 否则为批次大小乘以每序列token数 # 默认末尾位置
            ),
        )

    else:  # 否则（无投机信息） # 无投机信息时
        output_spec_info = None  # 输出投机信息为空 # 设为空
    ans = dict(  # 构建返回参数字典 # 构建参数字典
        bs=output_bs,  # 输出批次大小 # 输出批次大小
        req_pool_indices=req_pool_indices[seq_slice],  # 切片后的请求池索引 # 切片后的请求池索引
        seq_lens=seq_lens[seq_slice],  # 切片后的序列长度 # 切片后的序列长度
        # directly forward
        # 直接转发
        # 直接转发
        forward_mode=forward_mode,  # 前向模式 # 前向模式
        # ignore
        # 忽略
        # 忽略
        encoder_lens=None,  # 编码器长度设为空 # 设为空
        spec_info=output_spec_info,  # 输出投机信息 # 输出投机信息
    )

    if fn_name == "init_forward_metadata_capture_cuda_graph":  # 如果是CUDA图捕获函数 # 判断是否为捕获阶段
        assert (  # 断言检查 # 断言
            capture_num_tokens == bs * token_num_per_seq  # 捕获token数应等于批次大小乘以每序列token数 # 验证token数量一致性
        ), "Only support num_tokens==bs * token_num_per_seq for target-verify or decode mode"  # 错误提示信息 # 错误提示
        ans.update(  # 更新参数字典 # 更新参数
            dict(
                num_tokens=output_bs * token_num_per_seq,  # 输出token数量 # 输出token数量
            )
        )
    elif fn_name == "init_forward_metadata_replay_cuda_graph":  # 如果是CUDA图重放函数 # 判断是否为重放阶段
        output_seq_lens_cpu = replay_seq_lens_cpu[seq_slice]  # 切片后的CPU序列长度 # 切片CPU端序列长度
        ans.update(  # 更新参数字典 # 更新参数
            dict(
                seq_lens_sum=output_seq_lens_cpu.sum().item(),  # 序列长度总和 # 序列长度总和
                seq_lens_cpu=output_seq_lens_cpu,  # CPU序列长度 # CPU端序列长度
            )
        )
    else:  # 其他情况 # 其他情况
        raise NotImplementedError  # 抛出未实现错误 # 抛出未实现异常

    return ans  # 返回参数字典 # 返回参数字典
