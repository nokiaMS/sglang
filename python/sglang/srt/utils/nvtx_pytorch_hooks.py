# PyTorch逐层NVTX性能分析的钩子模块
# 提供在PyTorch网络中注册前向钩子的功能，用于NVIDIA Nsight性能分析
# 可记录模块名称、输入张量维度、可训练参数和静态参数等性能分析信息
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
"""PyTorch hooks for layerwise NVTX profiling."""

import torch  # 导入PyTorch
import torch.cuda.nvtx as nvtx  # 导入NVTX性能分析工具


class PytHooks(object):  # PyTorch钩子类，用于在网络中注册前向钩子以支持NVTX性能分析
    """This module contains all the code needed to enable forward hooks in a pytorch network.

    To register the hooks for a given network, the user needs to instantiate a PytHook object.
    Then call the register_hooks method.

    Example:

        my_hook = PytHook()
        my_hook.register_hooks(my_network_model)
    """

    def __init__(self):  # 初始化模块变量
        """Initialize module variables

        Returns:
            None:

        Raises:
            None:
        """
        super().__init__()  # 调用父类初始化
        self.module_to_name_map = {}  # 模块对象到名称的映射字典

    @staticmethod
    def print_tensor(tensor_obj, prefix, tensor_list=None):  # 递归遍历包含张量的迭代器并打印张量维度
        """Descends iterators that contains Tensors and prints the Tensor

        Recursive function that descends iterator type arguments until
        it finds a Tensor object.

        Args:
            tensor_obj: Could be a Tensor or an iterator type that contains Tensors
            prefix: String name to assign to the Tensor
            tensor_list: List to accumulate tensor dimensions

        Returns:
            List of tensor dimensions

        Raises:
            None:
        """
        if tensor_list is None:  # 如果未提供列表
            tensor_list = []  # 创建空列表

        if isinstance(tensor_obj, list) or isinstance(tensor_obj, tuple):  # 如果是列表或元组
            for ten in tensor_obj:  # 遍历每个元素
                tensor_list = PytHooks.print_tensor(ten, prefix, tensor_list)  # 递归处理
        elif isinstance(tensor_obj, torch.Tensor):  # 如果是张量
            tensor_dims = list(tensor_obj.size())  # 获取张量维度
            tensor_list.append(tensor_dims)  # 添加到列表
        return tensor_list  # 返回张量维度列表

    def process_layer_params(self, module_obj):  # 提取LLM和VLM相关层类型的静态参数
        """Extract the static parameters from LLM and VLM relevant layer types

        Args:
            module_obj(class): Module state data structure.

        Returns:
            param_info(dict): Parameter meta_data for the given op.

        Raises:
            None

        """
        param_info = {}  # 初始化参数信息字典
        # Extract parameters for layers commonly used in LLMs and VLMs  # 提取LLM和VLM常用层的参数
        if (  # 如果是Conv1d/Conv2d/Conv3d卷积层
            isinstance(module_obj, torch.nn.Conv1d)
            or isinstance(module_obj, torch.nn.Conv2d)
            or isinstance(module_obj, torch.nn.Conv3d)
        ):
            conv_params = {}  # 卷积参数字典
            conv_params["in_chan"] = module_obj.in_channels  # 输入通道数
            conv_params["out_chan"] = module_obj.out_channels  # 输出通道数
            conv_params["filter_dim"] = module_obj.kernel_size  # 卷积核尺寸
            conv_params["stride"] = module_obj.stride  # 步幅
            conv_params["padding"] = module_obj.padding  # 填充
            conv_params["dilation"] = module_obj.dilation  # 膨胀率
            conv_params["transposed"] = module_obj.transposed  # 是否转置卷积
            conv_params["output_padding"] = module_obj.output_padding  # 输出填充
            conv_params["groups"] = module_obj.groups  # 分组数
            conv_params["padding_mode"] = module_obj.padding_mode  # 填充模式
            param_info = conv_params  # 保存卷积参数
        elif (  # 如果是转置卷积层
            isinstance(module_obj, torch.nn.ConvTranspose1d)
            or isinstance(module_obj, torch.nn.ConvTranspose2d)
            or isinstance(module_obj, torch.nn.ConvTranspose3d)
        ):
            convtranspose_params = {}  # 转置卷积参数字典
            convtranspose_params["in_chan"] = module_obj.in_channels  # 输入通道数
            convtranspose_params["out_chan"] = module_obj.out_channels  # 输出通道数
            convtranspose_params["filter_dim"] = module_obj.kernel_size  # 卷积核尺寸
            convtranspose_params["stride"] = module_obj.stride  # 步幅
            convtranspose_params["padding"] = module_obj.padding  # 填充
            convtranspose_params["dilation"] = module_obj.dilation  # 膨胀率
            convtranspose_params["transposed"] = module_obj.transposed  # 是否转置
            convtranspose_params["output_padding"] = module_obj.output_padding  # 输出填充
            convtranspose_params["groups"] = module_obj.groups  # 分组数
            convtranspose_params["padding_mode"] = module_obj.padding_mode  # 填充模式
            param_info = convtranspose_params  # 保存转置卷积参数
        elif (  # 如果是最大池化层
            isinstance(module_obj, torch.nn.MaxPool1d)
            or isinstance(module_obj, torch.nn.MaxPool2d)
            or isinstance(module_obj, torch.nn.MaxPool3d)
        ):

            def _handle_int_or_tuple(parameter):  # 将参数统一处理为列表格式
                if isinstance(parameter, tuple):  # 如果是元组
                    return list(parameter)  # 转换为列表
                elif isinstance(parameter, int):  # 如果是整数
                    return [parameter, parameter]  # 转换为两个相同值的列表

            pooling_params = {}  # 池化参数字典
            pooling_params["filter_dim"] = _handle_int_or_tuple(module_obj.kernel_size)  # 卷积核尺寸
            pooling_params["stride"] = _handle_int_or_tuple(module_obj.stride)  # 步幅
            pooling_params["padding"] = _handle_int_or_tuple(module_obj.padding)  # 填充
            pooling_params["dilation"] = _handle_int_or_tuple(module_obj.dilation)  # 膨胀率
            param_info = pooling_params  # 保存池化参数
        elif (  # 如果是平均池化层
            isinstance(module_obj, torch.nn.AvgPool1d)
            or isinstance(module_obj, torch.nn.AvgPool2d)
            or isinstance(module_obj, torch.nn.AvgPool3d)
        ):
            pooling_params = {}  # 池化参数字典
            pooling_params["filter_dim"] = [  # 卷积核尺寸
                module_obj.kernel_size,
                module_obj.kernel_size,
            ]
            pooling_params["stride"] = [module_obj.stride, module_obj.stride]  # 步幅
            pooling_params["padding"] = [module_obj.padding, module_obj.padding]  # 填充
            pooling_params["ceil_mode"] = module_obj.ceil_mode  # 向上取整模式
            pooling_params["count_include_pad"] = module_obj.count_include_pad  # 是否包含填充计算
            param_info = pooling_params  # 保存池化参数
        elif (  # 如果是自适应平均池化层
            isinstance(module_obj, torch.nn.AdaptiveAvgPool1d)
            or isinstance(module_obj, torch.nn.AdaptiveAvgPool2d)
            or isinstance(module_obj, torch.nn.AdaptiveAvgPool3d)
        ):
            pooling_params = {}  # 池化参数字典
            pooling_params["output_size"] = [  # 输出尺寸
                module_obj.output_size,
                module_obj.output_size,
            ]
            param_info = pooling_params  # 保存池化参数
        elif isinstance(module_obj, torch.nn.Linear):  # 如果是线性层
            param_info["in_features"] = module_obj.in_features  # 输入特征数
            param_info["out_features"] = module_obj.out_features  # 输出特征数
        elif (  # 如果是批归一化层
            isinstance(module_obj, torch.nn.BatchNorm1d)
            or isinstance(module_obj, torch.nn.BatchNorm2d)
            or isinstance(module_obj, torch.nn.BatchNorm3d)
        ):
            param_info["num_features"] = module_obj.num_features  # 特征数
            param_info["epsilon"] = module_obj.eps  # epsilon值
            param_info["momentum"] = module_obj.momentum  # 动量
        elif isinstance(module_obj, torch.nn.ReLU):  # 如果是ReLU激活层
            param_info["in_place"] = module_obj.inplace  # 是否原地操作
        elif isinstance(module_obj, torch.nn.Dropout):  # 如果是Dropout层
            param_info["p"] = module_obj.p  # 丢弃概率
            param_info["in_place"] = module_obj.inplace  # 是否原地操作
        elif isinstance(module_obj, torch.nn.Embedding):  # 如果是嵌入层
            param_info["num_embeddings"] = module_obj.num_embeddings  # 嵌入数量
            param_info["embedding_dim"] = module_obj.embedding_dim  # 嵌入维度
        elif isinstance(  # 如果是上采样层
            module_obj,
            (
                torch.nn.Upsample,
                torch.nn.UpsamplingNearest2d,
                torch.nn.UpsamplingBilinear2d,
            ),
        ):
            param_info["scale_factor"] = module_obj.scale_factor  # 缩放因子

        return param_info  # 返回参数信息字典

    def module_fwd_hook(self, module_obj, in_tensor, out_tensor):  # 前向钩子回调函数，结束NVTX标记
        """Callback function that ends the NVTX marker

        Records the module name and tensor information
        Called after the module executes the forward method.

        Args:
            module_obj: Pointer to the module object
            in_tensor: Input tensor or list of tensors
            out_tensor: Output tensor of the resulting forward operator

        Returns:
            None:

        Raises:
            None:
        """
        nvtx.range_pop()  # 结束NVTX范围标记
        return

    def module_fwd_pre_hook(self, module_obj, in_tensor):  # 前向预钩子回调函数，创建带模块名称的NVTX标记
        """Creates an NVTX marker with the module name in it.

        This function is called before the module executes

        Args:
            module_obj: Module object data structure - used to get unique module name
            in_tensor: Input tensor data structure

        Returns:
            None

        Raises:
            None
        """
        marker_dict = {}  # 标记信息字典
        module_name = self.module_to_name_map.get(module_obj, "unknown")  # 获取模块名称
        marker_dict["Module"] = module_name  # 记录模块名称

        ## Get trainable parameters like weights and bias  # 获取可训练参数（如权重和偏置）
        module_params = module_obj.named_parameters(recurse=False)  # 获取模块参数
        for idx, (param_name, param_obj) in enumerate(module_params):  # 遍历参数
            if idx == 0:  # 如果是第一个参数
                marker_dict["TrainableParams"] = {}  # 创建可训练参数字典
            marker_dict["TrainableParams"][param_name] = list(param_obj.size())  # 记录参数名和维度

        in_tensor_list = PytHooks.print_tensor(in_tensor, "Input")  # 提取输入张量维度
        if in_tensor_list:  # 如果有输入张量
            marker_dict["Inputs"] = in_tensor_list  # 记录输入维度

        param_info = self.process_layer_params(module_obj)  # 提取层静态参数
        if param_info:  # 如果有参数信息
            marker_dict["StaticParams"] = param_info  # 记录静态参数

        nvtx.range_push("{}".format(marker_dict))  # 开始NVTX范围标记

        return

    def register_hooks(self, network_model, module_prefix="top"):  # 用户级函数，激活所有钩子
        """User level function that activates all the hooks

        The user needs to call this method from the network source code
        The code descends all the modules in the network and registers their
        respective hooks.

        Args:
            network_model: Model object for the network
            module_prefix: (default: top)

        Returns:
            None

        Raises:
            Exception if a module instance is reused
        """
        # Module types to skip (simple operations that don't need detailed profiling)  # 需要跳过的模块类型
        skip_types = (  # 不需要详细分析的简单操作类型
            torch.nn.Identity,
            torch.nn.Dropout,
            torch.nn.Dropout1d,
            torch.nn.Dropout2d,
            torch.nn.Dropout3d,
        )

        for name, module in network_model.named_modules(prefix=module_prefix):  # 遍历网络中所有模块
            # Skip certain module types to reduce profiling overhead  # 跳过特定模块类型以减少性能分析开销
            if isinstance(module, skip_types):  # 如果是跳过类型
                continue  # 跳过

            module.register_forward_pre_hook(self.module_fwd_pre_hook)  # 注册前向预钩子
            module.register_forward_hook(self.module_fwd_hook)  # 注册前向钩子
            if module not in self.module_to_name_map:  # 如果模块尚未在映射中
                self.module_to_name_map[module] = name  # 添加模块到名称映射
            else:  # 如果模块已在映射中
                raise ValueError("Module instance {} is not unique ".format(module))  # 抛出异常
        return
