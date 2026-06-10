# 分词器控制混入类 - 为TokenizerManager提供控制面操作（权重更新、缓存管理、LoRA、性能分析、内部状态等）

from __future__ import annotations  # 启用延迟类型注解评估

import asyncio  # 导入异步IO模块
import hashlib  # 导入哈希模块
import logging  # 导入日志模块
import time  # 导入时间模块
import uuid  # 导入UUID模块
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple  # 导入类型提示

import fastapi  # 导入FastAPI框架

from sglang.srt.managers.communicator import FanOutCommunicator  # 导入扇出通信器
from sglang.srt.managers.io_struct import (  # 导入各种请求/响应结构体
    AddExternalCorpusReqInput,  # 添加外部语料库请求输入
    AddExternalCorpusReqOutput,  # 添加外部语料库请求输出
    AttachHiCacheStorageReqInput,  # 附加HiCache存储请求输入
    AttachHiCacheStorageReqOutput,  # 附加HiCache存储请求输出
    CheckWeightsReqInput,  # 检查权重请求输入
    CheckWeightsReqOutput,  # 检查权重请求输出
    ClearHiCacheReqInput,  # 清除HiCache请求输入
    ClearHiCacheReqOutput,  # 清除HiCache请求输出
    CloseSessionReqInput,  # 关闭会话请求输入
    DestroyWeightsUpdateGroupReqInput,  # 销毁权重更新组请求输入
    DestroyWeightsUpdateGroupReqOutput,  # 销毁权重更新组请求输出
    DetachHiCacheStorageReqInput,  # 分离HiCache存储请求输入
    DetachHiCacheStorageReqOutput,  # 分离HiCache存储请求输出
    DumperControlReqInput,  # 转储控制请求输入
    DumperControlReqOutput,  # 转储控制请求输出
    ExpertDistributionReq,  # 专家分布请求
    ExpertDistributionReqOutput,  # 专家分布请求输出
    ExpertDistributionReqType,  # 专家分布请求类型
    FlushCacheReqInput,  # 刷新缓存请求输入
    FlushCacheReqOutput,  # 刷新缓存请求输出
    GetInternalStateReq,  # 获取内部状态请求
    GetInternalStateReqOutput,  # 获取内部状态请求输出
    GetLoadsReqOutput,  # 获取负载请求输出
    GetWeightsByNameReqInput,  # 按名称获取权重请求输入
    GetWeightsByNameReqOutput,  # 按名称获取权重请求输出
    InitWeightsSendGroupForRemoteInstanceReqInput,  # 初始化远程实例权重发送组请求输入
    InitWeightsSendGroupForRemoteInstanceReqOutput,  # 初始化远程实例权重发送组请求输出
    InitWeightsUpdateGroupReqInput,  # 初始化权重更新组请求输入
    InitWeightsUpdateGroupReqOutput,  # 初始化权重更新组请求输出
    ListExternalCorporaReqInput,  # 列出外部语料库请求输入
    ListExternalCorporaReqOutput,  # 列出外部语料库请求输出
    LoadLoRAAdapterFromTensorsReqInput,  # 从张量加载LoRA适配器请求输入
    LoadLoRAAdapterFromTensorsReqOutput,  # 从张量加载LoRA适配器请求输出
    LoadLoRAAdapterReqInput,  # 加载LoRA适配器请求输入
    LoadLoRAAdapterReqOutput,  # 加载LoRA适配器请求输出
    LoRAUpdateOutput,  # LoRA更新输出
    OpenSessionReqInput,  # 打开会话请求输入
    ProfileReq,  # 性能分析请求
    ProfileReqOutput,  # 性能分析请求输出
    ProfileReqType,  # 性能分析请求类型
    ReleaseMemoryOccupationReqInput,  # 释放内存占用请求输入
    ReleaseMemoryOccupationReqOutput,  # 释放内存占用请求输出
    RemoveExternalCorpusReqInput,  # 移除外部语料库请求输入
    RemoveExternalCorpusReqOutput,  # 移除外部语料库请求输出
    ResumeMemoryOccupationReqInput,  # 恢复内存占用请求输入
    ResumeMemoryOccupationReqOutput,  # 恢复内存占用请求输出
    SendWeightsToRemoteInstanceReqInput,  # 发送权重到远程实例请求输入
    SendWeightsToRemoteInstanceReqOutput,  # 发送权重到远程实例请求输出
    SetInternalStateReq,  # 设置内部状态请求
    SetInternalStateReqOutput,  # 设置内部状态请求输出
    SlowDownReqInput,  # 减速请求输入
    SlowDownReqOutput,  # 减速请求输出
    UnloadLoRAAdapterReqInput,  # 卸载LoRA适配器请求输入
    UnloadLoRAAdapterReqOutput,  # 卸载LoRA适配器请求输出
    UpdateWeightsFromDistributedReqInput,  # 从分布式更新权重请求输入
    UpdateWeightsFromDistributedReqOutput,  # 从分布式更新权重请求输出
    UpdateWeightsFromIPCReqInput,  # 从IPC更新权重请求输入
    UpdateWeightsFromIPCReqOutput,  # 从IPC更新权重请求输出
    UpdateWeightsFromTensorReqInput,  # 从张量更新权重请求输入
    UpdateWeightsFromTensorReqOutput,  # 从张量更新权重请求输出
)
from sglang.srt.managers.load_snapshot import LoadSnapshot  # 导入负载快照类
from sglang.srt.server_args import LoRARef, ServerArgs  # 导入LoRA引用和服务器参数
from sglang.srt.utils import get_bool_env_var  # 导入布尔环境变量获取工具
from sglang.utils import TypeBasedDispatcher  # 导入基于类型的分发器

if TYPE_CHECKING:  # 仅用于类型检查时导入
    from sglang.srt.managers.tokenizer_manager import TokenizerManager  # 导入分词器管理器

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

# Declarative spec: (attr_name_prefix, response_type[, mode])  声明式规范：(属性名前缀, 响应类型[, 模式])
# Each entry creates self.{prefix}_communicator and registers  每个条目创建 self.{prefix}_communicator 并注册
# response_type -> communicator.handle_recv in the dispatch table.  response_type -> communicator.handle_recv 到分发表中。
_COMMUNICATOR_SPECS = [  # 通信器规范列表
    ("init_weights_update_group", InitWeightsUpdateGroupReqOutput),  # 初始化权重更新组
    ("destroy_weights_update_group", DestroyWeightsUpdateGroupReqOutput),  # 销毁权重更新组
    ("update_weights_from_distributed", UpdateWeightsFromDistributedReqOutput),  # 从分布式更新权重
    (
        "init_weights_send_group_for_remote_instance",  # 初始化远程实例权重发送组
        InitWeightsSendGroupForRemoteInstanceReqOutput,
    ),
    ("send_weights_to_remote_instance", SendWeightsToRemoteInstanceReqOutput),  # 发送权重到远程实例
    ("update_weights_from_tensor", UpdateWeightsFromTensorReqOutput),  # 从张量更新权重
    ("update_weights_from_ipc", UpdateWeightsFromIPCReqOutput),  # 从IPC更新权重
    ("get_weights_by_name", GetWeightsByNameReqOutput),  # 按名称获取权重
    ("release_memory_occupation", ReleaseMemoryOccupationReqOutput),  # 释放内存占用
    ("resume_memory_occupation", ResumeMemoryOccupationReqOutput),  # 恢复内存占用
    ("check_weights", CheckWeightsReqOutput),  # 检查权重
    ("slow_down", SlowDownReqOutput),  # 减速
    ("flush_cache", FlushCacheReqOutput),  # 刷新缓存
    ("add_external_corpus", AddExternalCorpusReqOutput),  # 添加外部语料库
    ("remove_external_corpus", RemoveExternalCorpusReqOutput),  # 移除外部语料库
    ("list_external_corpora", ListExternalCorporaReqOutput),  # 列出外部语料库
    ("clear_hicache_storage", ClearHiCacheReqOutput),  # 清除HiCache存储
    ("attach_hicache_storage", AttachHiCacheStorageReqOutput),  # 附加HiCache存储
    ("detach_hicache_storage", DetachHiCacheStorageReqOutput),  # 分离HiCache存储
    ("profile", ProfileReqOutput),  # 性能分析
    ("get_internal_state", GetInternalStateReqOutput),  # 获取内部状态
    ("set_internal_state", SetInternalStateReqOutput),  # 设置内部状态
    ("expert_distribution", ExpertDistributionReqOutput),  # 专家分布
    ("update_lora_adapter", LoRAUpdateOutput),  # 更新LoRA适配器
    ("get_loads", GetLoadsReqOutput, "watching"),  # 获取负载（watching模式）
    ("dumper_control", DumperControlReqOutput),  # 转储控制
]


class TokenizerControlMixin:  # 分词器控制混入类
    """Mixin for TokenizerManager's control-plane operations (weights, cache, lora,
    profile, internal state, etc.) -- everything that talks to the scheduler via
    FanOutCommunicator, as opposed to data-plane inference requests multiplexed by rid.
    TokenizerManager控制面操作的混入类（权重、缓存、LoRA、性能分析、内部状态等）——
    所有通过FanOutCommunicator与调度器通信的操作，而非由rid多路复用的数据面推理请求。
    """

    def init_communicators(self: TokenizerManager, server_args: ServerArgs):  # 初始化所有通信器
        dispatch_pairs = []  # 分发对列表
        for spec in _COMMUNICATOR_SPECS:  # 遍历通信器规范
            name, resp_type = spec[0], spec[1]  # 获取名称和响应类型
            mode = spec[2] if len(spec) > 2 else "queueing"  # 获取模式，默认为queueing
            comm = FanOutCommunicator(self.send_to_scheduler, server_args.dp_size, mode)  # 创建通信器
            setattr(self, f"{name}_communicator", comm)  # 动态设置通信器属性
            dispatch_pairs.append((resp_type, comm.handle_recv))  # 添加到分发对列表
        self._result_dispatcher += TypeBasedDispatcher(dispatch_pairs)  # 注册到结果分发器

    async def add_external_corpus(  # 添加外部语料库
        self: TokenizerManager, obj: AddExternalCorpusReqInput  # 请求输入对象
    ) -> AddExternalCorpusReqOutput:  # 返回添加结果
        self.auto_create_handle_loop()  # 确保处理循环已启动
        if self.server_args.speculative_algorithm != "NGRAM":  # 如果不是NGRAM推测解码
            return AddExternalCorpusReqOutput(  # 返回失败结果
                success=False,  # 成功标志为False
                message="Ngram speculative decoding is not enabled.",  # Ngram推测解码未启用
            )
        truncated = False  # 是否截断标志
        try:
            if not obj.corpus_id:  # 如果没有提供语料库ID
                import uuid  # 导入UUID模块

                obj.corpus_id = uuid.uuid4().hex  # 生成新的UUID作为语料库ID
            if obj.file_path is not None:  # 如果提供了文件路径
                from sglang.srt.speculative.cpp_ngram.external_corpus import (
                    iter_external_corpus_chunks,  # 导入外部语料块迭代器
                )

                max_tokens = (  # 获取最大token数限制
                    self.server_args.speculative_ngram_external_corpus_max_tokens
                )
                obj.token_chunks = list(  # 将文件分块为token序列
                    iter_external_corpus_chunks(
                        obj.file_path, self.tokenizer, max_tokens  # 文件路径、分词器和最大token数
                    )
                )
            elif obj.documents is not None:  # 如果提供了文档列表
                from sglang.srt.speculative.cpp_ngram.external_corpus import (
                    SEPARATOR_TOKEN,  # 导入分隔符token
                )

                max_tokens = (  # 获取最大token数限制
                    self.server_args.speculative_ngram_external_corpus_max_tokens
                )
                token_chunks = []  # token块列表
                total_tokens = 0  # 总token计数
                has_prev = False  # 是否有前一个文档标志
                for doc in obj.documents:  # 遍历所有文档
                    if not doc:  # 跳过空文档
                        continue
                    token_ids = list(  # 编码文档为token ID列表
                        self.tokenizer.encode(doc, add_special_tokens=False)  # 不添加特殊token
                    )
                    if not token_ids:  # 跳过编码后为空的文档
                        continue
                    if has_prev:  # 如果有前一个文档
                        token_ids = [SEPARATOR_TOKEN] + token_ids  # 在前面添加分隔符
                    if total_tokens + len(token_ids) > max_tokens:  # 超出最大token数限制
                        truncated = True  # 标记为截断
                        break  # 跳出循环
                    token_chunks.append(token_ids)  # 添加到token块列表
                    total_tokens += len(token_ids)  # 累加token数
                    has_prev = True  # 标记已有文档
                obj.token_chunks = token_chunks  # 设置token块
            else:  # 既没提供文件路径也没提供文档
                return AddExternalCorpusReqOutput(  # 返回失败结果
                    success=False,  # 成功标志为False
                    message="Either file_path or documents must be provided.",  # 必须提供文件路径或文档
                )
            obj.file_path = None  # 清空文件路径（不再需要）
            obj.documents = None  # 清空文档列表（不再需要）
            results = await self.add_external_corpus_communicator(obj)  # 发送请求到调度器
            all_success, all_message = FanOutCommunicator.merge_results(results)  # 合并所有DP rank的结果
            if truncated and all_success:  # 如果截断且成功
                all_message += f" (truncated: exceeded {max_tokens} token limit)"  # 添加截断提示
            return AddExternalCorpusReqOutput(  # 返回结果
                success=all_success,  # 成功标志
                corpus_id=results[0].corpus_id if all_success else "",  # 语料库ID
                message=all_message,  # 消息
                loaded_token_count=results[0].loaded_token_count if all_success else 0,  # 加载的token数量
            )
        except Exception as e:  # 捕获异常
            return AddExternalCorpusReqOutput(success=False, message=str(e))  # 返回失败结果

    async def remove_external_corpus(  # 移除外部语料库
        self: TokenizerManager, corpus_id: str  # 要移除的语料库ID
    ) -> RemoveExternalCorpusReqOutput:  # 返回移除结果
        self.auto_create_handle_loop()  # 确保处理循环已启动
        if self.server_args.speculative_algorithm != "NGRAM":  # 如果不是NGRAM推测解码
            return RemoveExternalCorpusReqOutput(  # 返回失败结果
                success=False,  # 成功标志为False
                message="Ngram speculative decoding is not enabled.",  # Ngram推测解码未启用
            )
        results = await self.remove_external_corpus_communicator(  # 发送移除请求
            RemoveExternalCorpusReqInput(corpus_id=corpus_id)  # 构建请求输入
        )
        all_success, all_message = FanOutCommunicator.merge_results(results)  # 合并结果
        return RemoveExternalCorpusReqOutput(success=all_success, message=all_message)  # 返回结果

    async def list_external_corpora(  # 列出所有外部语料库
        self: TokenizerManager,
    ) -> ListExternalCorporaReqOutput:  # 返回语料库列表
        self.auto_create_handle_loop()  # 确保处理循环已启动
        if self.server_args.speculative_algorithm != "NGRAM":  # 如果不是NGRAM推测解码
            return ListExternalCorporaReqOutput(  # 返回失败结果
                success=False,  # 成功标志为False
                message="Ngram speculative decoding is not enabled.",  # Ngram推测解码未启用
            )
        results = await self.list_external_corpora_communicator(  # 发送列出请求
            ListExternalCorporaReqInput()  # 构建请求输入
        )
        all_success, all_message = FanOutCommunicator.merge_results(results)  # 合并结果
        # Merge corpus token counts from all DP ranks (each rank loads the same set).  合并所有DP rank的语料库token计数（每个rank加载相同的集合）
        corpus_token_counts = results[0].corpus_token_counts if all_success else {}  # 取第一个rank的结果
        return ListExternalCorporaReqOutput(  # 返回结果
            success=all_success,  # 成功标志
            corpus_token_counts=corpus_token_counts,  # 语料库token计数
            message=all_message,  # 消息
        )

    async def flush_cache(  # 刷新缓存
        self: TokenizerManager, timeout_s: Optional[float] = None  # 超时时间（秒）
    ) -> FlushCacheReqOutput:  # 返回刷新结果
        self.auto_create_handle_loop()  # 确保处理循环已启动
        return (
            await self.flush_cache_communicator(FlushCacheReqInput(timeout_s=timeout_s))  # 发送刷新请求
        )[0]  # 取第一个结果

    async def clear_hicache_storage(self: TokenizerManager) -> ClearHiCacheReqOutput:  # 清除HiCache存储
        """Clear the hierarchical cache storage."""  # 清除分层缓存存储
        self.auto_create_handle_loop()  # 确保处理循环已启动
        # Delegate to the scheduler to handle HiCacheStorage clearing  委托调度器处理HiCacheStorage清除
        return (await self.clear_hicache_storage_communicator(ClearHiCacheReqInput()))[
            0  # 取第一个结果
        ]

    async def attach_hicache_storage(  # 附加（启用）HiCache存储后端
        self: TokenizerManager,
        hicache_storage_backend: str,  # 存储后端类型
        hicache_storage_backend_extra_config_json: Optional[str] = None,  # 额外配置JSON
        hicache_storage_prefetch_policy: Optional[str] = None,  # 预取策略
        hicache_write_policy: Optional[str] = None,  # 写入策略
    ) -> AttachHiCacheStorageReqOutput:  # 返回附加结果
        """Attach (enable) HiCache storage backend at runtime."""  # 在运行时附加（启用）HiCache存储后端
        self.auto_create_handle_loop()  # 确保处理循环已启动
        results = await self.attach_hicache_storage_communicator(  # 发送附加请求
            AttachHiCacheStorageReqInput(
                hicache_storage_backend=hicache_storage_backend,  # 存储后端
                hicache_storage_backend_extra_config_json=hicache_storage_backend_extra_config_json,  # 额外配置
                hicache_storage_prefetch_policy=hicache_storage_prefetch_policy,  # 预取策略
                hicache_write_policy=hicache_write_policy,  # 写入策略
            )
        )

        all_success, all_message = FanOutCommunicator.merge_results(results)  # 合并结果
        out = AttachHiCacheStorageReqOutput(success=all_success, message=all_message)  # 构建输出
        # TODO: partial rollback if failed  TODO: 失败时部分回滚
        if all_success:  # 如果全部成功
            # Keep tokenizer side server_info consistent with scheduler side.  保持分词器端的server_info与调度器端一致
            self.server_args.hicache_storage_backend = hicache_storage_backend  # 更新存储后端配置
            if hicache_storage_backend_extra_config_json is not None:  # 如果有额外配置
                self.server_args.hicache_storage_backend_extra_config = (
                    hicache_storage_backend_extra_config_json  # 更新额外配置
                )
            if hicache_storage_prefetch_policy is not None:  # 如果有预取策略
                self.server_args.hicache_storage_prefetch_policy = (
                    hicache_storage_prefetch_policy  # 更新预取策略
                )
            if hicache_write_policy is not None:  # 如果有写入策略
                self.server_args.hicache_write_policy = (
                    hicache_write_policy  # 更新写入策略
                )
        return out  # 返回输出

    async def detach_hicache_storage(  # 分离（禁用）HiCache存储后端
        self: TokenizerManager,
    ) -> DetachHiCacheStorageReqOutput:  # 返回分离结果
        """Detach (disable) HiCache storage backend at runtime."""  # 在运行时分离（禁用）HiCache存储后端
        self.auto_create_handle_loop()  # 确保处理循环已启动
        results = await self.detach_hicache_storage_communicator(  # 发送分离请求
            DetachHiCacheStorageReqInput()  # 构建请求输入
        )

        all_success, all_message = FanOutCommunicator.merge_results(results)  # 合并结果
        out = DetachHiCacheStorageReqOutput(success=all_success, message=all_message)  # 构建输出
        # TODO: partial rollback if failed  TODO: 失败时部分回滚
        if all_success:  # 如果全部成功
            self.server_args.hicache_storage_backend = None  # 清空存储后端配置
            self.server_args.hicache_storage_backend_extra_config = None  # 清空额外配置
        return out  # 返回输出

    async def start_profile(  # 启动性能分析
        self: TokenizerManager,
        output_dir: Optional[str] = None,  # 输出目录
        start_step: Optional[int] = None,  # 起始步数
        num_steps: Optional[int] = None,  # 步数
        activities: Optional[List[str]] = None,  # 活动类型列表
        with_stack: Optional[bool] = None,  # 是否记录调用栈
        record_shapes: Optional[bool] = None,  # 是否记录张量形状
        profile_by_stage: bool = False,  # 是否按阶段分析
        merge_profiles: bool = False,  # 是否合并分析结果
        profile_prefix: Optional[str] = None,  # 分析文件前缀
        profile_stages: Optional[List[str]] = None,  # 分析阶段列表
    ):
        self.auto_create_handle_loop()  # 确保处理循环已启动
        env_with_stack: bool = get_bool_env_var("SGLANG_PROFILE_WITH_STACK", "true")  # 读取环境变量
        with_stack = False if with_stack is False or env_with_stack is False else True  # 合并环境和参数设置
        env_record_shapes: bool = get_bool_env_var(  # 读取环境变量
            "SGLANG_PROFILE_RECORD_SHAPES", "true"  # 是否记录形状
        )
        record_shapes = (record_shapes is not False) and env_record_shapes  # 合并环境和参数设置
        req = ProfileReq(  # 构建性能分析请求
            type=ProfileReqType.START_PROFILE,  # 请求类型为启动分析
            output_dir=output_dir,  # 输出目录
            start_step=start_step,  # 起始步数
            num_steps=num_steps,  # 步数
            activities=activities,  # 活动类型
            with_stack=with_stack,  # 是否记录调用栈
            record_shapes=record_shapes,  # 是否记录形状
            profile_by_stage=profile_by_stage,  # 是否按阶段分析
            profile_id=str(time.time()),  # 使用时间戳作为分析ID
            merge_profiles=merge_profiles,  # 是否合并
            profile_prefix=profile_prefix,  # 前缀
            profile_stages=profile_stages,  # 阶段
        )
        return await self._execute_profile(req)  # 执行性能分析

    async def stop_profile(self: TokenizerManager):  # 停止性能分析
        self.auto_create_handle_loop()  # 确保处理循环已启动
        req = ProfileReq(type=ProfileReqType.STOP_PROFILE)  # 构建停止分析请求
        return await self._execute_profile(req)  # 执行

    async def _execute_profile(self: TokenizerManager, req: ProfileReq):  # 执行性能分析请求
        result = (await self.profile_communicator(req))[0]  # 发送请求并获取第一个结果
        if not result.success:  # 如果失败
            raise RuntimeError(result.message)  # 抛出运行时异常
        return result  # 返回结果

    async def start_expert_distribution_record(self: TokenizerManager):  # 开始记录专家分布
        self.auto_create_handle_loop()  # 确保处理循环已启动
        req = ExpertDistributionReq(action=ExpertDistributionReqType.START_RECORD)  # 构建开始记录请求
        await self.expert_distribution_communicator(req)  # 发送请求

    async def stop_expert_distribution_record(self: TokenizerManager):  # 停止记录专家分布
        self.auto_create_handle_loop()  # 确保处理循环已启动
        req = ExpertDistributionReq(action=ExpertDistributionReqType.STOP_RECORD)  # 构建停止记录请求
        await self.expert_distribution_communicator(req)  # 发送请求

    async def dump_expert_distribution_record(self: TokenizerManager):  # 导出专家分布记录
        self.auto_create_handle_loop()  # 确保处理循环已启动
        req = ExpertDistributionReq(action=ExpertDistributionReqType.DUMP_RECORD)  # 构建导出记录请求
        await self.expert_distribution_communicator(req)  # 发送请求

    async def init_weights_update_group(  # 初始化权重更新组
        self: TokenizerManager,
        obj: InitWeightsUpdateGroupReqInput,  # 请求输入对象
        request: Optional[fastapi.Request] = None,  # 可选的FastAPI请求对象
    ) -> Tuple[bool, str]:  # 返回（成功标志，消息）
        self.auto_create_handle_loop()  # 确保处理循环已启动
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for update weights from distributed"  # 断言：dp_size必须为1或启用DP注意力

        results = await self.init_weights_update_group_communicator(obj)  # 发送请求
        return FanOutCommunicator.merge_results(results)  # 合并并返回结果

    async def destroy_weights_update_group(  # 销毁权重更新组
        self: TokenizerManager,
        obj: DestroyWeightsUpdateGroupReqInput,  # 请求输入对象
        request: Optional[fastapi.Request] = None,  # 可选的FastAPI请求对象
    ) -> Tuple[bool, str]:  # 返回（成功标志，消息）
        self.auto_create_handle_loop()  # 确保处理循环已启动
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for destroy parameter update group"  # 断言：dp_size必须为1或启用DP注意力

        results = await self.destroy_weights_update_group_communicator(obj)  # 发送请求
        return FanOutCommunicator.merge_results(results)  # 合并并返回结果

    async def update_weights_from_distributed(  # 从分布式更新权重
        self: TokenizerManager,
        obj: UpdateWeightsFromDistributedReqInput,  # 请求输入对象
        request: Optional[fastapi.Request] = None,  # 可选的FastAPI请求对象
    ) -> Tuple[bool, str]:  # 返回（成功标志，消息）
        self.auto_create_handle_loop()  # 确保处理循环已启动
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for update weights from distributed"  # 断言：dp_size必须为1或启用DP注意力

        if obj.abort_all_requests:  # 如果需要中止所有请求
            self.abort_request(abort_all=True)  # 中止所有请求

        # Hold is_pause_cond while updating to prevent unpause from racing.  更新时持有is_pause_cond以防止取消暂停的竞态条件
        async with self.is_pause_cond:  # 获取暂停条件锁
            is_paused = self.is_pause  # 读取暂停状态
            if is_paused:  # 如果已暂停
                results = await self.update_weights_from_distributed_communicator(obj)  # 直接更新

        if not is_paused:  # 如果未暂停
            async with self.model_update_lock.writer_lock:  # 获取模型更新写锁
                results = await self.update_weights_from_distributed_communicator(obj)  # 更新权重

        success, message = FanOutCommunicator.merge_results(results)  # 合并结果
        if success and obj.weight_version is not None:  # 如果成功且提供了权重版本
            self._update_weight_version_if_provided(obj.weight_version)  # 更新权重版本
            message += f" Weight version updated to {obj.weight_version}."  # 添加版本更新信息

        return success, message  # 返回成功标志和消息

    async def init_weights_send_group_for_remote_instance(  # 为远程实例初始化权重发送组
        self: TokenizerManager,
        obj: InitWeightsSendGroupForRemoteInstanceReqInput,  # 请求输入对象
        request: Optional[fastapi.Request] = None,  # 可选的FastAPI请求对象
    ) -> Tuple[bool, str]:  # 返回（成功标志，消息）
        self.auto_create_handle_loop()  # 确保处理循环已启动
        # TODO: support DP  TODO: 支持DP
        assert (
            self.server_args.dp_size == 1
        ), "dp_size must be 1 for init_weights_send_group_for_remote_instance"  # 断言：dp_size必须为1
        result = (
            await self.init_weights_send_group_for_remote_instance_communicator(obj)  # 发送请求
        )[0]  # 取第一个结果
        return result.success, result.message  # 返回成功标志和消息

    async def send_weights_to_remote_instance(  # 发送权重到远程实例
        self: TokenizerManager,
        obj: SendWeightsToRemoteInstanceReqInput,  # 请求输入对象
        request: Optional[fastapi.Request] = None,  # 可选的FastAPI请求对象
    ) -> Tuple[bool, str]:  # 返回（成功标志，消息）
        self.auto_create_handle_loop()  # 确保处理循环已启动
        # TODO: support DP  TODO: 支持DP
        assert (
            self.server_args.dp_size == 1
        ), "dp_size must be 1 for send_weights_to_remote_instance"  # 断言：dp_size必须为1
        result = (await self.send_weights_to_remote_instance_communicator(obj))[0]  # 发送请求并取第一个结果
        return result.success, result.message  # 返回成功标志和消息

    async def update_weights_from_tensor(  # 从张量更新权重
        self: TokenizerManager,
        obj: UpdateWeightsFromTensorReqInput,  # 请求输入对象
        request: Optional[fastapi.Request] = None,  # 可选的FastAPI请求对象
    ) -> Tuple[bool, str]:  # 返回（成功标志，消息）
        self.auto_create_handle_loop()  # 确保处理循环已启动
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for update weights from tensor"  # 断言：dp_size必须为1或启用DP注意力

        if obj.abort_all_requests:  # 如果需要中止所有请求
            self.abort_request(abort_all=True)  # 中止所有请求

        async with self.is_pause_cond:  # 获取暂停条件锁
            is_paused = self.is_pause  # 读取暂停状态
            if is_paused:  # 如果已暂停
                results = await self.update_weights_from_tensor_communicator(obj)  # 直接更新

        if not is_paused:  # 如果未暂停
            async with self.model_update_lock.writer_lock:  # 获取模型更新写锁
                results = await self.update_weights_from_tensor_communicator(obj)  # 更新权重

        success, message = FanOutCommunicator.merge_results(results)  # 合并结果
        if success and obj.weight_version is not None:  # 如果成功且提供了权重版本
            self._update_weight_version_if_provided(obj.weight_version)  # 更新权重版本
            message += f" Weight version updated to {obj.weight_version}."  # 添加版本更新信息

        return success, message  # 返回成功标志和消息

    async def update_weights_from_ipc(  # 通过IPC更新权重
        self: TokenizerManager,
        obj: UpdateWeightsFromIPCReqInput,  # 请求输入对象
        request: Optional[fastapi.Request] = None,  # 可选的FastAPI请求对象
    ) -> Tuple[bool, str]:  # 返回（成功标志，消息）
        """Update weights via IPC for checkpoint-engine integration."""  # 通过IPC更新权重，用于检查点引擎集成
        self.auto_create_handle_loop()  # 确保处理循环已启动
        try:
            # For now, we only support single data parallel instance  目前仅支持单个数据并行实例
            assert (
                self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
            ), "dp_size must be 1 or dp attention must be enabled for update weights from IPC"  # 断言：dp_size必须为1或启用DP注意力
            logger.info("Starting IPC weight update")  # 记录开始IPC权重更新

            async with self.is_pause_cond:  # 获取暂停条件锁
                is_paused = self.is_pause  # 读取暂停状态
                if is_paused:  # 如果已暂停
                    result = (await self.update_weights_from_ipc_communicator(obj))[0]  # 直接更新
                    success, message = result.success, result.message  # 提取结果

            if not is_paused:  # 如果未暂停
                async with self.model_update_lock.writer_lock:  # 获取模型更新写锁
                    result = (await self.update_weights_from_ipc_communicator(obj))[0]  # 更新权重
                    success, message = result.success, result.message  # 提取结果
        except Exception as e:  # 捕获异常
            error_msg = f"IPC weight update failed: {str(e)}"  # 构建错误消息
            logger.error(error_msg)  # 记录错误
            success, message = False, error_msg  # 设置失败标志和消息

        if success and obj.weight_version is not None:  # 如果成功且提供了权重版本
            self._update_weight_version_if_provided(obj.weight_version)  # 更新权重版本
            message += f" Weight version updated to {obj.weight_version}."  # 添加版本更新信息

        return success, message  # 返回成功标志和消息

    async def _unload_lora_adapter_locked(  # 在持有锁的情况下卸载LoRA适配器
        self: TokenizerManager,
        obj: UnloadLoRAAdapterReqInput,  # 请求输入对象
    ) -> UnloadLoRAAdapterReqOutput:  # 返回卸载结果
        assert (
            self.lora_update_lock.locked()
        ), "self.lora_update_lock must be locked in order for self._unload_lora_adapter_locked() to be called"  # 断言：必须持有锁

        # Unregister the LoRA adapter from the registry to stop new requests for this adapter  从注册表注销LoRA适配器以停止该适配器的新请求
        # from being started.  防止新请求启动。
        lora_id = await self.lora_registry.unregister(obj.lora_name)  # 从注册表注销
        obj.lora_id = lora_id  # 设置lora_id

        # Initiate the actual unloading operation at the backend processes only after all  仅在所有
        # ongoing requests using this LoRA adapter are finished.  使用此LoRA适配器的进行中请求完成后，才启动实际的卸载操作。
        await self.lora_registry.wait_for_unload(lora_id)  # 等待卸载完成
        result = (await self.update_lora_adapter_communicator(obj))[0]  # 发送卸载请求

        return result  # 返回结果

    async def load_lora_adapter(  # 加载LoRA适配器
        self: TokenizerManager,
        obj: LoadLoRAAdapterReqInput,  # 请求输入对象
        _: Optional[fastapi.Request] = None,  # 可选的FastAPI请求对象（未使用）
    ) -> LoadLoRAAdapterReqOutput:  # 返回加载结果
        self.auto_create_handle_loop()  # 确保处理循环已启动

        try:
            if not self.server_args.enable_lora:  # 如果未启用LoRA
                raise ValueError(
                    "LoRA is not enabled. Please set `--enable-lora` to enable LoRA."  # LoRA未启用，请设置 --enable-lora
                )

            # TODO (lifuhuang): Remove this after we verify that dynamic lora loading works  TODO：验证动态LoRA加载在dp_size>1下可用后移除
            # with dp_size > 1.
            assert (
                self.server_args.dp_size == 1
            ), "dp_size must be 1 for dynamic lora loading"  # 断言：dp_size必须为1
            logger.info(
                "Start load Lora adapter. Lora name=%s, path=%s",  # 开始加载LoRA适配器
                obj.lora_name,  # LoRA名称
                obj.lora_path,  # LoRA路径
            )

            async with self.lora_update_lock:  # 获取LoRA更新锁
                # Generate new uniquely identifiable LoRARef object.  生成新的唯一标识的LoRARef对象
                new_adapter = LoRARef(
                    lora_name=obj.lora_name,  # LoRA名称
                    lora_path=obj.lora_path,  # LoRA路径
                    pinned=obj.pinned,  # 是否固定
                )

                # Trigger the actual loading operation at the backend processes.  在后端进程触发实际加载操作
                obj.lora_id = new_adapter.lora_id  # 设置lora_id
                result = (await self.update_lora_adapter_communicator(obj))[0]  # 发送加载请求

                # Register the LoRA adapter only after loading is successful.  仅在加载成功后注册LoRA适配器
                if result.success:  # 如果加载成功
                    await self.lora_registry.register(new_adapter)  # 注册到LoRA注册表
                    self.lora_ref_cache[obj.lora_name] = new_adapter  # 更新缓存

                if self.server_args.max_loaded_loras is not None:  # 如果设置了最大LoRA数量限制
                    while (
                        self.lora_registry.num_registered_loras  # 已注册的LoRA数量
                        > self.server_args.max_loaded_loras  # 超过最大限制
                    ):
                        lru_lora_name = await self.lora_registry.lru_lora_name(
                            exclude_pinned=True  # 排除固定的适配器
                        )
                        if lru_lora_name is None:  # 没有可驱逐的适配器
                            raise ValueError(
                                "Didn't find any LoRA adapters when trying to evict LRU LoRA adapter. "  # 未找到可驱逐的LRU LoRA适配器
                                f"LoRA registry is: {self.lora_registry._registry}"  # LoRA注册表内容
                            )

                        logger.info(
                            f"Unloading least recently used LoRA adapter '{lru_lora_name}' "  # 卸载最近最少使用的LoRA适配器
                            f"(current number of adapters: {self.lora_registry.num_registered_loras}, "  # 当前适配器数量
                            f"max allowed: {self.server_args.max_loaded_loras})"  # 最大允许数量
                        )

                        unload_result = await self._unload_lora_adapter_locked(
                            UnloadLoRAAdapterReqInput(lora_name=lru_lora_name)  # 构建卸载请求
                        )
                        if not unload_result.success:  # 如果卸载失败
                            raise ValueError(
                                f"Error while unloading LRU LoRA adapter '{lru_lora_name}': "  # 卸载LRU LoRA适配器时出错
                                f"{unload_result.error_message}"  # 错误消息
                            )
                        del result.loaded_adapters[lru_lora_name]  # 从已加载适配器列表中删除

                return result  # 返回结果
        except ValueError as e:  # 捕获值错误
            return LoadLoRAAdapterReqOutput(
                success=False,  # 成功标志为False
                error_message=str(e),  # 错误消息
            )

    async def load_lora_adapter_from_tensors(  # 从张量加载LoRA适配器
        self: TokenizerManager,
        obj: LoadLoRAAdapterFromTensorsReqInput,  # 请求输入对象
        _: Optional[fastapi.Request] = None,  # 可选的FastAPI请求对象（未使用）
    ) -> LoadLoRAAdapterFromTensorsReqOutput:  # 返回加载结果
        self.auto_create_handle_loop()  # 确保处理循环已启动

        try:
            if not self.server_args.enable_lora:  # 如果未启用LoRA
                raise ValueError(
                    "LoRA is not enabled. Please set `--enable-lora` to enable LoRA."  # LoRA未启用
                )

            assert (
                self.server_args.dp_size == 1
            ), "dp_size must be 1 for dynamic lora loading"  # 断言：dp_size必须为1
            logger.info(
                "Start load Lora adapter from tensors. Lora name=%s",  # 开始从张量加载LoRA适配器
                obj.lora_name,  # LoRA名称
            )

            async with self.lora_update_lock:  # 获取LoRA更新锁
                new_adapter = LoRARef(
                    lora_name=obj.lora_name,  # LoRA名称
                    lora_path="__tensor__",  # 路径标记为张量
                    pinned=obj.pinned,  # 是否固定
                )
                obj.lora_id = new_adapter.lora_id  # 设置lora_id
                result = (await self.update_lora_adapter_communicator(obj))[0]  # 发送加载请求

                if result.success:  # 如果加载成功
                    await self.lora_registry.register(new_adapter)  # 注册到LoRA注册表
                    self.lora_ref_cache[obj.lora_name] = new_adapter  # 更新缓存
                if self.server_args.max_loaded_loras is not None:  # 如果设置了最大LoRA数量限制
                    while (
                        self.lora_registry.num_registered_loras  # 已注册的LoRA数量
                        > self.server_args.max_loaded_loras  # 超过最大限制
                    ):
                        lru_lora_name = await self.lora_registry.lru_lora_name(
                            exclude_pinned=True  # 排除固定的适配器
                        )
                        if lru_lora_name is None:  # 没有可驱逐的适配器
                            raise ValueError(
                                "Didn't find any LoRA adapters when trying to evict LRU LoRA adapter. "  # 未找到可驱逐的LRU LoRA适配器
                                f"LoRA registry is: {self.lora_registry._registry}"  # LoRA注册表内容
                            )

                        logger.info(
                            f"Unloading least recently used LoRA adapter '{lru_lora_name}' "  # 卸载最近最少使用的LoRA适配器
                            f"(current number of adapters: {self.lora_registry.num_registered_loras}, "  # 当前适配器数量
                            f"max allowed: {self.server_args.max_loaded_loras})"  # 最大允许数量
                        )

                        unload_result = await self._unload_lora_adapter_locked(
                            UnloadLoRAAdapterReqInput(lora_name=lru_lora_name)  # 构建卸载请求
                        )
                        if not unload_result.success:  # 如果卸载失败
                            raise ValueError(
                                f"Error while unloading LRU LoRA adapter '{lru_lora_name}': "  # 卸载LRU LoRA适配器时出错
                                f"{unload_result.error_message}"  # 错误消息
                            )
                        del result.loaded_adapters[lru_lora_name]  # 从已加载适配器列表中删除

                return result  # 返回结果
        except ValueError as e:  # 捕获值错误
            return LoadLoRAAdapterFromTensorsReqOutput(
                success=False,  # 成功标志为False
                error_message=str(e),  # 错误消息
            )

    async def unload_lora_adapter(  # 卸载LoRA适配器
        self: TokenizerManager,
        obj: UnloadLoRAAdapterReqInput,  # 请求输入对象
        _: Optional[fastapi.Request] = None,  # 可选的FastAPI请求对象（未使用）
    ) -> UnloadLoRAAdapterReqOutput:  # 返回卸载结果
        self.auto_create_handle_loop()  # 确保处理循环已启动

        try:
            if not self.server_args.enable_lora:  # 如果未启用LoRA
                raise ValueError(
                    "LoRA is not enabled. Please set `--enable-lora` to enable LoRA."  # LoRA未启用
                )

            assert (
                obj.lora_name is not None
            ), "lora_name must be provided to unload LoRA adapter"  # 断言：必须提供lora_name

            # TODO (lifuhuang): Remove this after we verify that dynamic lora loading works  TODO：验证动态LoRA加载在dp_size>1下可用后移除
            # with dp_size > 1.
            assert (
                self.server_args.dp_size == 1
            ), "dp_size must be 1 for dynamic lora loading"  # 断言：dp_size必须为1
            logger.info(
                "Start unload Lora adapter. Lora name=%s",  # 开始卸载LoRA适配器
                obj.lora_name,  # LoRA名称
            )

            async with self.lora_update_lock:  # 获取LoRA更新锁
                return await self._unload_lora_adapter_locked(obj)  # 调用带锁卸载方法
        except ValueError as e:  # 捕获值错误
            return UnloadLoRAAdapterReqOutput(success=False, error_message=str(e))  # 返回失败结果

    async def get_weights_by_name(  # 按名称获取权重
        self: TokenizerManager,
        obj: GetWeightsByNameReqInput,  # 请求输入对象
        request: Optional[fastapi.Request] = None,  # 可选的FastAPI请求对象
    ):
        self.auto_create_handle_loop()  # 确保处理循环已启动
        results = await self.get_weights_by_name_communicator(obj)  # 发送请求
        all_parameters = [r.parameter for r in results]  # 提取所有参数
        if self.server_args.dp_size == 1:  # 如果是单DP
            return all_parameters[0]  # 返回第一个结果
        else:  # 多DP
            return all_parameters  # 返回所有结果

    async def release_memory_occupation(  # 释放内存占用
        self: TokenizerManager,
        obj: ReleaseMemoryOccupationReqInput,  # 请求输入对象
        request: Optional[fastapi.Request] = None,  # 可选的FastAPI请求对象
    ):
        self.auto_create_handle_loop()  # 确保处理循环已启动
        await self.release_memory_occupation_communicator(obj)  # 发送请求

    async def resume_memory_occupation(  # 恢复内存占用
        self: TokenizerManager,
        obj: ResumeMemoryOccupationReqInput,  # 请求输入对象
        request: Optional[fastapi.Request] = None,  # 可选的FastAPI请求对象
    ):
        self.auto_create_handle_loop()  # 确保处理循环已启动
        await self.resume_memory_occupation_communicator(obj)  # 发送请求

    async def check_weights(  # 检查权重一致性
        self: TokenizerManager,
        obj: CheckWeightsReqInput,  # 请求输入对象
        request: Optional[fastapi.Request] = None,  # 可选的FastAPI请求对象
    ) -> Tuple[bool, str, Optional[List[Dict]], Optional[str]]:  # 返回（成功标志，消息，各rank校验和，引擎校验和）
        self.auto_create_handle_loop()  # 确保处理循环已启动
        results = await self.check_weights_communicator(obj)  # 发送请求
        success, message = FanOutCommunicator.merge_results(results)  # 合并结果
        ranks: Optional[List[Dict]] = None  # 各rank的校验信息
        per_engine_checksum: Optional[str] = None  # 引擎级别的校验和
        if any(r.payload is not None for r in results):  # 如果有结果包含payload
            ranks = []  # 初始化rank列表
            for r in results:  # 遍历结果
                if isinstance(r.payload, list):  # 如果payload是列表
                    ranks.extend(r.payload)  # 扩展到ranks
                else:  # 如果payload是单个对象
                    ranks.append(r.payload)  # 添加到ranks
            h = hashlib.sha256()  # 创建SHA256哈希对象
            for rank in ranks:  # 遍历各rank
                h.update(rank["per_gpu_checksum"].encode())  # 更新哈希
            per_engine_checksum = h.hexdigest()  # 获取十六进制摘要
        return success, message, ranks, per_engine_checksum  # 返回所有结果

    async def slow_down(  # 减速处理
        self: TokenizerManager,
        obj: SlowDownReqInput,  # 请求输入对象
        request: Optional[fastapi.Request] = None,  # 可选的FastAPI请求对象
    ):
        self.auto_create_handle_loop()  # 确保处理循环已启动
        await self.slow_down_communicator(obj)  # 发送请求

    async def get_internal_state(self: TokenizerManager) -> List[Dict[Any, Any]]:  # 获取内部状态
        self.auto_create_handle_loop()  # 确保处理循环已启动
        req = GetInternalStateReq()  # 构建请求
        responses: List[GetInternalStateReqOutput] = (
            await self.get_internal_state_communicator(req)  # 发送请求
        )
        # Many DP ranks  多个DP rank
        return [res.internal_state for res in responses]  # 提取所有内部状态

    async def set_internal_state(  # 设置内部状态
        self: TokenizerManager, obj: SetInternalStateReq  # 请求输入对象
    ) -> List[bool]:  # 返回各rank的更新状态
        self.auto_create_handle_loop()  # 确保处理循环已启动
        responses: List[SetInternalStateReqOutput] = (
            await self.set_internal_state_communicator(obj)  # 发送请求
        )
        return [res.updated for res in responses]  # 提取更新状态

    async def dumper_control(  # 转储控制
        self: TokenizerManager, obj: DumperControlReqInput  # 请求输入对象
    ) -> List[DumperControlReqOutput]:  # 返回转储控制结果列表
        self.auto_create_handle_loop()  # 确保处理循环已启动
        return await self.dumper_control_communicator(obj)  # 发送请求并返回结果

    async def get_loads(  # 获取负载信息
        self: TokenizerManager,
        include: Optional[List[str]] = None,  # 要包含的部分：core, memory, spec, lora, disagg, queues, all
        dp_rank: Optional[int] = None,  # 可选的DP rank过滤
    ) -> List[LoadSnapshot]:  # 返回负载快照列表
        """
        Get load snapshots for /v1/loads endpoint.  获取 /v1/loads 端点的负载快照

        Args:
            include: List of sections to include. Options: core, memory, spec, lora, disagg, queues, all  要包含的部分列表。选项：core, memory, spec, lora, disagg, queues, all
            dp_rank: Optional filter for specific DP rank  可选的特定DP rank过滤

        Returns:
            List of LoadSnapshot, one per scheduler (filtered by dp_rank if specified)  LoadSnapshot列表，每个调度器一个（如果指定dp_rank则过滤）
        """
        self.auto_create_handle_loop()  # 确保处理循环已启动
        if dp_rank is not None and (dp_rank < 0 or dp_rank >= self.server_args.dp_size):  # 检查dp_rank范围
            return []  # 返回空列表

        reader = self.load_snapshot_reader  # 获取负载快照读取器
        if dp_rank is not None:  # 如果指定了dp_rank
            load = reader.read(dp_rank)  # 读取指定rank的负载
            results = [load] if load is not None else []  # 包装为列表
        else:  # 未指定dp_rank
            results = reader.read_all()  # 读取所有rank的负载

        return results  # 返回结果

    async def open_session(  # 打开会话
        self: TokenizerManager,
        obj: OpenSessionReqInput,  # 请求输入对象
        request: Optional[fastapi.Request] = None,  # 可选的FastAPI请求对象
    ):
        self.auto_create_handle_loop()  # 确保处理循环已启动
        if obj.streaming:  # 如果是流式会话
            if not self.server_args.enable_streaming_session:  # 如果未启用流式会话
                raise ValueError(
                    "Streaming sessions are disabled. "  # 流式会话已禁用
                    "Please relaunch with --enable-streaming-session."  # 请使用 --enable-streaming-session 重新启动
                )

        if obj.session_id is None:  # 如果没有提供会话ID
            obj.session_id = uuid.uuid4().hex  # 生成新的UUID作为会话ID
        elif obj.session_id in self.session_futures:  # 如果会话ID已存在
            return None  # 返回None表示会话已存在

        future = asyncio.Future()  # 创建异步Future
        self.session_futures[obj.session_id] = future  # 存储到会话Future字典
        self.send_to_scheduler.send_pyobj(obj)  # 发送请求到调度器

        try:
            return await future  # 等待Future结果
        finally:
            self.session_futures.pop(obj.session_id, None)  # 清理Future

    async def close_session(  # 关闭会话
        self: TokenizerManager,
        obj: CloseSessionReqInput,  # 请求输入对象
        request: Optional[fastapi.Request] = None,  # 可选的FastAPI请求对象
    ):
        await self.send_to_scheduler.send_pyobj(obj)  # 发送关闭会话请求到调度器

    def _update_weight_version_if_provided(  # 如果提供了权重版本则更新
        self: TokenizerManager, weight_version: Optional[str]  # 权重版本字符串
    ) -> None:  # 无返回值
        """Update weight version if provided."""  # 如果提供了权重版本则更新
        if weight_version is not None:  # 如果权重版本不为None
            self.server_args.weight_version = weight_version  # 更新服务器参数中的权重版本
