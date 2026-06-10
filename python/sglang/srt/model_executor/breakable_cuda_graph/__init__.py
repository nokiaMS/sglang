# 可中断CUDA图（Breakable CUDA Graph）模块初始化文件
# 该模块实现了可中断的CUDA图捕获与重放机制，
# 将模型前向传播拆分为多个CUDA图段，在段之间支持急切执行断点。
