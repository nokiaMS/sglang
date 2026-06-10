# 基数树（Radix Tree）的Python绑定模块
# 提供C++实现的高性能基数树数据结构的Python接口，用于KV缓存的前缀匹配和淘汰管理
# 支持GPU/CPU双层级存储，包含写穿透（write-through）和加载回传（load-onboard）机制

from __future__ import annotations  # 启用延迟类型注解评估

import os  # 导入操作系统模块
from typing import TYPE_CHECKING, List, Optional, Tuple  # 导入类型提示工具

import torch  # 导入PyTorch张量库
from torch.utils.cpp_extension import load  # 导入C++扩展加载工具

_abs_path = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件的绝对路径目录
radix_tree_cpp = load(  # 加载C++扩展模块
    name="radix_tree_cpp",  # 扩展模块名称
    sources=[  # C++源文件列表
        f"{_abs_path}/tree_v2_binding.cpp",  # 绑定接口源文件
        f"{_abs_path}/tree_v2_debug.cpp",  # 调试功能源文件
        f"{_abs_path}/tree_v2.cpp",  # 基数树核心实现源文件
    ],
    extra_cflags=["-O3", "-std=c++20"],  # 额外编译选项：O3优化，C++20标准
)

if TYPE_CHECKING:  # 仅在类型检查时执行的代码块

    class TreeNodeCpp:
        """
        A placeholder for the TreeNode class. Cannot be constructed elsewhere.
        TreeNode类的占位符。不能在其他地方构造。
        """

    class IOHandle:
        """
        A placeholder for the IOHandle class. Cannot be constructed elsewhere.
        IOHandle类的占位符。不能在其他地方构造。
        """

    class RadixTreeCpp:
        def __init__(  # 初始化RadixTreeCpp实例
            self,
            disabled: bool,  # 是否禁用基数树
            host_size: Optional[int],  # CPU上基数树的大小，None表示无CPU树
            page_size: int,  # 基数树的页面大小
            write_through_threshold: int,  # 从GPU写到CPU的穿透阈值
        ):
            """
            Initializes the RadixTreeCpp instance.
            初始化RadixTreeCpp实例。
            Args:
                disabled (bool): If True, the radix tree is disabled.
                    如果为True，基数树被禁用。
                host_size (Optional[int]): Size of the radix tree on the CPU. None means no CPU tree.
                    CPU上基数树的大小。None表示无CPU树。
                page_size (int): Size of the page for the radix tree.
                    基数树的页面大小。
                write_through_threshold (int): Threshold for writing through from GPU to CPU.
                    从GPU写到CPU的穿透阈值。
            """
            self.tree = radix_tree_cpp.RadixTree(  # type: ignore  # 创建底层C++基数树对象
                disabled, host_size, page_size, write_through_threshold  # 传递初始化参数
            )

        def match_prefix(  # 在基数树中匹配前缀
            self, prefix: List[int]  # 要匹配的前缀token ID列表
        ) -> Tuple[List[torch.Tensor], int, TreeNodeCpp, TreeNodeCpp]:
            """
            Matches a prefix in the radix tree.
            在基数树中匹配前缀。
            Args:
                prefix (List[int]): The prefix to match.
                    要匹配的前缀。
            Returns:
                Tuple[List[torch.Tensor], TreeNodeCpp, TreeNodeCpp]:
                    0. A list of indices that is matched by the prefix on the GPU.
                       在GPU上被前缀匹配到的索引列表。
                    1. Sum length of the indices matched on the CPU.
                       在CPU上匹配到的索引总长度。
                    2. The last node of the prefix matched on the GPU.
                       在GPU上前缀匹配的最后一个节点。
                    3. The last node of the prefix matched on the CPU.
                       在CPU上前缀匹配的最后一个节点。
            """
            return self.tree.match_prefix(prefix)  # 调用底层C++方法匹配前缀

        def evict(self, num_tokens: int) -> List[torch.Tensor]:  # 从基数树中淘汰指定数量的token
            """
            Evicts a number of tokens from the radix tree.
            从基数树中淘汰一定数量的token。
            Args:
                num_tokens (int): The number of tokens to evict.
                    要淘汰的token数量。
            Returns:
                List[torch.Tensor]: A list of indices that were evicted.
                    被淘汰的索引列表。
            """
            return self.tree.evict(num_tokens)  # 调用底层C++方法执行淘汰

        def lock_ref(self, handle: TreeNodeCpp, lock: bool) -> None:  # 锁定或解锁树节点的引用
            """
            Locks or unlocks a reference to a tree node.
            锁定或解锁树节点的引用。
            After locking, the node will not be evicted from the radix tree.
            锁定后，该节点将不会被从基数树中淘汰。
            Args:
                handle (TreeNodeCpp): The tree node to lock or unlock.
                    要锁定或解锁的树节点。
                lock (bool): If True, locks the node; if False, unlocks it.
                    如果为True，锁定节点；如果为False，解锁节点。
            """
            return self.tree.lock_ref(handle, lock)  # 调用底层C++方法锁定/解锁引用

        def writing_through(  # 写穿透：插入键值对并执行写穿透检查
            self, key: List[int], indices: torch.Tensor  # 键列表和关联的值张量
        ) -> Tuple[List[Tuple[IOHandle, torch.Tensor, torch.Tensor]], int]:
            """
            Inserts a key-value pair into the radix tree and perform write-through check.
            向基数树插入键值对并执行写穿透检查。
            Args:
                key (List[int]): The key to insert.
                    要插入的键。
                indices (torch.Tensor): The value associated with the key.
                    与键关联的值。
            Returns:
                Tuple[List[Tuple[IOHandle, torch.Tensor, torch.Tensor]], int]:
                    0. A list of (IOHandle, device indices, host indices) tuples.
                       (IOHandle, 设备索引, 主机索引) 元组的列表。
                       These IOhandles require write-through to the CPU in python side.
                       这些IOHandle需要在Python侧执行写穿透到CPU。
                    1. The number of indices that are matched on device.
                       在设备上匹配到的索引数量。
            """
            return self.tree.writing_through(key, indices)  # 调用底层C++方法执行写穿透

        def loading_onboard(  # 加载回传：更新树节点的设备索引
            self,
            host_node: TreeNodeCpp,  # 主机上的树节点，必须是设备节点的后代
            new_device_indices: torch.Tensor,  # 要设置的新设备索引张量
        ) -> Tuple[IOHandle, List[torch.Tensor]]:
            """
            Updates the device indices of tree nodes within a range on the tree.
            更新树上一定范围内树节点的设备索引。
            Args:
                host_node (TreeNodeCpp): The tree node on the host, must be descendant of device_node.
                    主机上的树节点，必须是设备节点的后代。
                new_device_indices (torch.Tensor): The new device indices to set.
                    要设置的新设备索引。
                    The length of this tensor must be exactly host indices length.
                    此张量的长度必须与主机索引长度完全一致。
            Returns:
                Tuple[IOHandle, List[torch.Tensor]]:
                    0. An IOHandle that requires loading to the CPU in python side.
                       需要在Python侧加载到CPU的IOHandle。
                    1. A list of host indices corresponding to the new device indices.
                       与新设备索引对应的主机索引列表。
            """
            return self.tree.loading_onboard(host_node, new_device_indices)  # 调用底层C++方法执行加载回传

        def commit_writing_through(self, handle: IOHandle, success: bool) -> None:  # 提交写穿透过程
            """
            Commits the write-through process for a tree node.
            提交树节点的写穿透过程。
            Args:
                handle (IOHandle): The IOHandle to commit.
                    要提交的IOHandle。
                success (bool): If True, commits the write-through; if False, just indicates failure.
                    如果为True，提交写穿透；如果为False，仅表示失败。
            """
            return self.tree.commit_writing_through(handle, success)  # 调用底层C++方法提交写穿透

        def commit_loading_onboard(self, handle: IOHandle, success: bool) -> None:  # 提交加载回传过程
            """
            Commits the load onboard process for tree nodes within a range on the tree.
            提交树上一定范围内树节点的加载回传过程。
            Args:
                handle (IOHandle): The IOHandle to commit.
                    要提交的IOHandle。
                success (bool): If True, commits the load-onboard; if False, just indicates failure.
                    如果为True，提交加载回传；如果为False，仅表示失败。
            """
            return self.tree.commit_loading_onboard(handle, success)  # 调用底层C++方法提交加载回传

        def evictable_size(self) -> int:  # 返回基数树可淘汰部分的大小
            """
            Returns the size of the evictable part of the radix tree.
            返回基数树可淘汰部分的大小。
            This is the size of the part that can be evicted from the GPU (ref_count = 0).
            这是可以从GPU淘汰的部分的大小（引用计数 = 0）。
            Returns:
                int: The size of the evictable part.
                    可淘汰部分的大小。
            """
            return self.tree.evictable_size()  # 调用底层C++方法获取可淘汰大小

        def protected_size(self) -> int:  # 返回基数树受保护部分的大小
            """
            Returns the size of the protected part of the radix tree.
            返回基数树受保护部分的大小。
            This is the size of the part that cannot be evicted from the GPU (ref_count > 0).
            这是不可以从GPU淘汰的部分的大小（引用计数 > 0）。
            Returns:
                int: The size of the protected part.
                    受保护部分的大小。
            """
            return self.tree.protected_size()  # 调用底层C++方法获取受保护大小

        def total_size(self) -> int:  # 返回基数树的总大小（包括CPU节点）
            """
            Returns the total size of the radix tree (including CPU nodes).
            返回基数树的总大小（包括CPU节点）。
            Returns:
                int: The total size of the radix tree.
                    基数树的总大小。
            """
            return self.tree.total_size()  # 调用底层C++方法获取总大小

        def reset(self) -> None:  # 重置基数树，清除所有节点和索引
            """
            Resets the radix tree, clearing all nodes and indices.
            重置基数树，清除所有节点和索引。
            """
            return self.tree.reset()  # 调用底层C++方法执行重置

        def debug_print(self) -> None:  # 打印基数树内部状态用于调试
            """
            Prints the internal state of the radix tree for debugging purposes.
            打印基数树的内部状态，用于调试目的。
            """
            return self.tree.debug_print()  # 调用底层C++方法打印调试信息

else:
    # Real implementation of the classes for runtime
    # 运行时类的实际实现
    RadixTreeCpp = radix_tree_cpp.RadixTree  # 直接引用C++扩展中的RadixTree类
    TreeNodeCpp = object  # TreeNodeCpp设为object类型
    IOHandle = object  # IOHandle设为object类型
