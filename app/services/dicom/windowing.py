"""
CT 窗宽窗位处理
"""
import numpy as np
from typing import Tuple, Optional


# 常用 CT 窗口预设
CT_WINDOWS = {
    "lung": {"center": -600, "width": 1500},      # 肺窗
    "mediastinum": {"center": 40, "width": 400},  # 纵隔窗
    "bone": {"center": 400, "width": 1800},       # 骨窗
    "brain": {"center": 40, "width": 80},         # 脑窗
    "liver": {"center": 60, "width": 150},        # 肝窗
    "abdomen": {"center": 40, "width": 350},      # 腹部窗
    "soft_tissue": {"center": 50, "width": 350},  # 软组织窗
}


def apply_ct_window(
    image: np.ndarray,
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
    preset: Optional[str] = None,
    output_range: Tuple[float, float] = (0, 255)
) -> np.ndarray:
    """
    应用 CT 窗宽窗位
    
    Args:
        image: CT 图像数组 (HU 值)
        window_center: 窗位
        window_width: 窗宽
        preset: 预设窗口名称 (lung, mediastinum, bone, etc.)
        output_range: 输出值范围
        
    Returns:
        窗口化后的图像
    """
    if preset and preset in CT_WINDOWS:
        window_center = CT_WINDOWS[preset]["center"]
        window_width = CT_WINDOWS[preset]["width"]
    
    if window_center is None or window_width is None:
        raise ValueError("必须提供窗宽窗位或预设名称")
    
    # 计算窗口边界
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    
    # 应用窗口
    windowed = np.clip(image, lower, upper)
    
    # 归一化到输出范围
    windowed = (windowed - lower) / (upper - lower)
    windowed = windowed * (output_range[1] - output_range[0]) + output_range[0]
    
    return windowed.astype(np.float32)


def get_optimal_window(image: np.ndarray, percentile: Tuple[float, float] = (1, 99)) -> Tuple[float, float]:
    """
    自动计算最优窗宽窗位
    
    Args:
        image: CT 图像数组
        percentile: 百分位数范围
        
    Returns:
        (window_center, window_width)
    """
    p_low, p_high = np.percentile(image, percentile)
    window_width = p_high - p_low
    window_center = (p_high + p_low) / 2
    return window_center, window_width


