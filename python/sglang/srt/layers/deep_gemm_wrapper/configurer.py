# DeepGEMM配置模块 - 根据硬件能力和环境变量确定是否启用JIT DeepGEMM及相关特性
import logging  # 导入日志模块

from sglang.srt.environ import envs  # 导入环境变量配置
from sglang.srt.utils import (  # 导入工具函数
    get_device_sm,  # 获取GPU SM版本
    is_blackwell_supported,  # 判断是否支持Blackwell架构
    is_cuda,  # 判断是否为CUDA平台
    is_musa,  # 判断是否为MUSA平台
)

logger = logging.getLogger(__name__)  # 创建日志记录器

_is_cuda = is_cuda()  # 判断当前是否为CUDA平台
_is_musa = is_musa()  # 判断当前是否为MUSA平台


def _compute_enable_deep_gemm():  # 计算是否启用DeepGEMM
    sm_version = get_device_sm()  # 获取GPU SM版本
    if (_is_cuda and sm_version < 90) or (_is_musa and sm_version < 31):  # CUDA需SM90+，MUSA需SM31+
        return False  # SM版本不满足要求，不启用
    # DeepGEMM requires TMEM/tcgen05 (SM100+datacenter), not available on SM120
    # DeepGEMM需要TMEM/tcgen05（SM100+数据中心），SM120不可用
    if sm_version == 120:  # SM120不支持TMEM/tcgen05
        return False  # 不启用
    if not (_is_cuda or _is_musa):  # 既不是CUDA也不是MUSA
        return False  # 不启用

    try:
        import deep_gemm  # noqa: F401  # 尝试导入deep_gemm模块
    except ImportError:  # 导入失败
        return False  # deep_gemm不可用，不启用

    return envs.SGLANG_ENABLE_JIT_DEEPGEMM.get()  # 返回环境变量配置的启用状态


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()  # 是否启用JIT DeepGEMM的全局标志

DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and is_blackwell_supported()  # 是否为Blackwell架构的DeepGEMM
DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL  # 是否使用UE8M0缩放格式（Blackwell专属）
DEEPGEMM_NEED_TMA_ALIGNED_SCALES = not (DEEPGEMM_SCALE_UE8M0 or _is_musa)  # 是否需要TMA对齐的缩放因子（非UE8M0且非MUSA时需要）
