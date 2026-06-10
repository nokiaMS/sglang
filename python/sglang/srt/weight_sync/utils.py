# 权重同步工具模块
# 提供跨张量并行rank更新推理引擎权重的异步函数
# 支持FSDP和Megatron等分布式训练框架的权重预处理

from typing import Optional  # 可选类型注解

import torch  # PyTorch深度学习框架
import torch.distributed as dist  # PyTorch分布式通信模块
from torch.distributed.device_mesh import DeviceMesh  # 设备网格
from torch.distributed.tensor import DTensor  # 分布式张量

from sglang.srt.entrypoints.engine import Engine  # 推理引擎
from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput  # 张量权重更新请求
from sglang.srt.model_executor.model_runner import LocalSerializedTensor  # 本地序列化张量
from sglang.srt.utils import MultiprocessingSerializer  # 多进程序列化工具


async def update_weights(  # 异步更新推理引擎权重
    engine: Engine,  # 推理引擎实例
    params_batch: list[tuple[str, torch.Tensor]],  # (名称, 张量)元组列表，批处理以减少CPU调用开销
    device_mesh_key: str,  # 设备网格键，通常为"tp"或"infer_tp"
    device_mesh: DeviceMesh,  # 设备网格
    load_format: Optional[str] = None,  # 权重格式
):
    """
    Update weights for the inference engine.
    This function is designed to be stateless, so that the caller process could keep the stateful engine.
    Example Use Case:
        - Multiple Producer Process will call this function in a SPMD style

    Args:
        engine: The inference engine created by the caller process.
        params_batch: A list of (name, tensor) tuples. We batched the tensors to avoid the overhead of cpu call.
        device_mesh_key: The key of the device mesh. Typically "tp" or "infer_tp"
        device_mesh: The device mesh.
        load_format: The format of the weights.
    """
    # 更新推理引擎的权重。
    # 此函数设计为无状态，以便调用者进程可以保持有状态的引擎。
    # 示例用例：
    #   - 多个生产者进程以SPMD方式调用此函数
    # 参数：
    #   engine: 调用者进程创建的推理引擎
    #   params_batch: (名称, 张量)元组列表，批处理以减少CPU调用开销
    #   device_mesh_key: 设备网格键，通常为"tp"或"infer_tp"
    #   device_mesh: 设备网格
    #   load_format: 权重格式
    infer_tp_size = device_mesh[device_mesh_key].mesh.size()[0]  # 获取推理TP大小
    infer_tp_rank = device_mesh[device_mesh_key].get_local_rank()  # 获取本地推理TP rank
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions  # 延迟导入torch补丁

    monkey_patch_torch_reductions()  # 应用torch归约操作的monkey补丁

    # [
    #   (name0, ipc_tensor0_tp0),
    #   (name1, ipc_tensor1_tp0),
    # ]
    # 序列化每个rank的命名张量批次
    named_tensors_batch = [  # 构建序列化后的命名张量批次
        (
            name,  # 张量名称
            MultiprocessingSerializer.serialize(  # 序列化张量数据
                _preprocess_tensor_for_update_weights(tensor.detach())  # 预处理并分离梯度
            ),
        )
        for name, tensor in params_batch  # 遍历参数批次
    ]

    if infer_tp_rank == 0:  # 如果是rank 0
        gathered_serialized_batches = [None for _ in range(infer_tp_size)]  # 分配收集列表
    else:  # 非rank 0
        gathered_serialized_batches = None  # 不需要收集列表

    # [
    #   [ (name0, ipc_tensor0_tp0), (name1, ipc_tensor1_tp0) ],
    #   [ (name0, ipc_tensor0_tp1), (name1, ipc_tensor1_tp1) ],
    # ]
    # 跨TP rank收集序列化的张量批次
    dist.gather_object(  # 执行分布式对象收集
        obj=named_tensors_batch,  # 本地序列化批次
        object_gather_list=gathered_serialized_batches,  # 收集目标列表
        dst=device_mesh[device_mesh_key].mesh.tolist()[0],  # 目标rank
        group=device_mesh[device_mesh_key].get_group(),  # 通信组
    )

    if infer_tp_rank == 0:  # 如果是rank 0，负责组装和发送更新请求
        # Use zip(*) to "transpose" the data structure.
        # After transpose, the data structure is like:
        # [
        #   ( (name0, ipc_tensor0_tp0), (name0, ipc_tensor0_tp1) ),
        #   ( (name1, ipc_tensor1_tp0), (name1, ipc_tensor1_tp1) ),
        # ]
        # 使用zip(*)转置数据结构。
        # 转置后的数据结构为：
        # [
        #   ( (name0, ipc_tensor0_tp0), (name0, ipc_tensor0_tp1) ),
        #   ( (name1, ipc_tensor1_tp0), (name1, ipc_tensor1_tp1) ),
        # ]
        logical_tensors = zip(*gathered_serialized_batches, strict=True)  # 转置：按名称分组

        named_tensors = [  # 构建本地序列化张量列表
            # [
            #   (name0, LocalSerializedTensor(values=[ipc_tensor0_tp0, ipc_tensor0_tp1])),
            #   (name1, LocalSerializedTensor(values=[ipc_tensor1_tp0, ipc_tensor1_tp1])),
            # ]
            (
                tensor_group[0][0],  # 取第一个rank的名称（所有rank名称相同）
                LocalSerializedTensor(  # 创建本地序列化张量
                    values=[rank_part[1] for rank_part in tensor_group]  # 收集所有rank的序列化数据
                ),
            )
            for tensor_group in logical_tensors  # 遍历转置后的张量组
        ]

        update_weights_request = UpdateWeightsFromTensorReqInput(  # 创建权重更新请求
            serialized_named_tensors=[  # 为每个TP rank序列化完整的命名张量列表
                MultiprocessingSerializer.serialize(named_tensors)  # 序列化命名张量列表
                for _ in range(infer_tp_size)  # 为每个rank创建一份
            ],
            load_format=load_format,  # 权重格式
        )

        return await engine.update_weights_from_tensor(update_weights_request)  # 异步调用引擎更新权重


def _preprocess_tensor_for_update_weights(tensor: torch.Tensor):  # 预处理张量用于权重更新
    """
    Preprocess the tensor for update weights.
    Example Use Case:
        - FSDP: we gather tensor by calling full_tensor in _preprocess_tensor_for_update_weights
        - Megatron: we do nothing here, assuming it is gathered when feed into this func

    Args:
        tensor: The tensor to be preprocessed.

    Returns:
        The full tensor if it is a DTensor, otherwise the original tensor.
    """
    # 预处理张量用于权重更新。
    # 示例用例：
    #   - FSDP：通过调用full_tensor收集张量
    #   - Megatron：此处不做处理，假设传入时已收集
    # 参数：
    #   tensor: 待预处理的张量
    # 返回：
    #   如果是DTensor则返回完整张量，否则返回原始张量。
    if isinstance(tensor, DTensor):  # 如果是分布式张量
        return tensor.full_tensor()  # 收集为完整张量
    return tensor  # 非分布式张量直接返回
