# 输出发送器包装类
# 封装ZMQ socket的输出发送操作，支持多HTTP worker场景下的IPC通信处理。

from typing import Optional, Union

import zmq

from sglang.srt.managers.io_struct import BaseBatchReq, BaseReq


class SenderWrapper:
    """ZMQ输出发送器包装类，封装socket发送逻辑"""

    def __init__(self, socket: zmq.Socket):
        self.socket = socket

    def send_output(
        self,
        output: Union[BaseReq, BaseBatchReq],
        recv_obj: Optional[Union[BaseReq, BaseBatchReq]] = None,
    ):
        """发送输出到ZMQ socket，自动处理多HTTP worker的IPC关联"""
        if self.socket is None:
            return

        if (
            isinstance(recv_obj, BaseReq)
            and recv_obj.http_worker_ipc is not None
            and output.http_worker_ipc is None
        ):
            # handle communicator reqs for multi-http worker case
            # 处理多HTTP worker情况：从接收对象继承IPC信息
            output.http_worker_ipc = recv_obj.http_worker_ipc

        self.socket.send_pyobj(output)
