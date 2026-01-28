"""
变化检测器 - 计算差分图和变化量化
"""
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import AnalysisError
from app.schemas.analysis import ComparisonResult, RECISTResponse


class ChangeDetector:
    """变化检测器"""
    
    def __init__(
        self,
        threshold_hu: float = None,
        gaussian_sigma: float = None
    ):
        """
        初始化变化检测器
        
        Args:
            threshold_hu: HU 值变化阈值
            gaussian_sigma: 高斯平滑 sigma
        """
        self.threshold_hu = threshold_hu or settings.DIFF_THRESHOLD_HU
        self.gaussian_sigma = gaussian_sigma or settings.GAUSSIAN_SIGMA
    
    def compute_difference_map(
        self,
        followup: np.ndarray,
        warped_baseline: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算差分图
        
        Args:
            followup: 随访图像
            warped_baseline: 配准后的基线图像
            mask: 可选的掩膜 (仅在掩膜区域计算)
            
        Returns:
            (原始差分图, 阈值化后的显著差异图)
        """
        logger.info("计算差分图...")
        
        # 计算差分
        diff_map = followup.astype(np.float32) - warped_baseline.astype(np.float32)
        
        # 应用高斯平滑去噪
        diff_smoothed = ndimage.gaussian_filter(diff_map, sigma=self.gaussian_sigma)
        
        # 应用掩膜
        if mask is not None:
            diff_smoothed = diff_smoothed * mask
        
        # 阈值化 - 保留显著变化
        significant_changes = np.zeros_like(diff_smoothed)
        significant_changes[np.abs(diff_smoothed) > self.threshold_hu] = diff_smoothed[np.abs(diff_smoothed) > self.threshold_hu]
        
        return diff_map, significant_changes
    
    def generate_heatmap(
        self,
        diff_map: np.ndarray,
        background: np.ndarray,
        output_path: Path,
        slice_idx: Optional[int] = None,
        view: str = "axial"
    ) -> Path:
        """
        生成热力图
        
        Args:
            diff_map: 差分图
            background: 背景图像
            output_path: 输出路径
            slice_idx: 切片索引 (None 则自动选择最大变化切片)
            view: 视图方向 (axial, sagittal, coronal)
            
        Returns:
            热力图文件路径
        """
        # 选择切片
        if slice_idx is None:
            # 找到变化最大的切片
            if view == "axial":
                slice_changes = np.sum(np.abs(diff_map), axis=(0, 1))
            elif view == "sagittal":
                slice_changes = np.sum(np.abs(diff_map), axis=(1, 2))
            else:  # coronal
                slice_changes = np.sum(np.abs(diff_map), axis=(0, 2))
            slice_idx = np.argmax(slice_changes)
        
        # 提取切片
        if view == "axial":
            bg_slice = background[:, :, slice_idx]
            diff_slice = diff_map[:, :, slice_idx]
        elif view == "sagittal":
            bg_slice = background[slice_idx, :, :]
            diff_slice = diff_map[slice_idx, :, :]
        else:
            bg_slice = background[:, slice_idx, :]
            diff_slice = diff_map[:, slice_idx, :]
        
        # 创建图像
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # 显示背景 (灰度)
        ax.imshow(bg_slice.T, cmap='gray', origin='lower', aspect='equal')
        
        # 叠加热力图
        # 创建自定义 colormap: 蓝色(减少) -> 透明 -> 红色(增加)
        colors = [(0, 0, 1), (1, 1, 1, 0), (1, 0, 0)]
        n_bins = 256
        cmap = mcolors.LinearSegmentedColormap.from_list("diff_cmap", colors, N=n_bins)
        
        # 归一化差分图
        max_val = max(np.abs(diff_slice).max(), 1)
        diff_normalized = diff_slice / max_val
        
        # 创建透明度掩膜
        alpha = np.abs(diff_normalized)
        alpha = np.clip(alpha, 0, 1)
        
        # 叠加热力图
        heatmap = ax.imshow(
            diff_normalized.T, 
            cmap='RdBu_r', 
            origin='lower', 
            aspect='equal',
            alpha=alpha.T * 0.7,
            vmin=-1, 
            vmax=1
        )
        
        # 添加 colorbar
        cbar = plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('HU Change (normalized)', fontsize=12)
        
        ax.set_title(f'{view.capitalize()} View - Slice {slice_idx}', fontsize=14)
        ax.axis('off')
        
        # 保存
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()
        
        logger.info(f"热力图已保存: {output_path}")
        return output_path
    
    def quantify_changes(
        self,
        diff_map: np.ndarray,
        significant: np.ndarray,
        spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)
    ) -> Dict[str, Any]:
        """
        量化整体变化（不需要特定 ROI）
        
        Args:
            diff_map: 差分图
            significant: 显著变化掩膜
            spacing: 体素间距 (mm)
            
        Returns:
            变化量化结果
        """
        # 计算体素体积 (cc)
        voxel_volume_cc = np.prod(spacing) / 1000  # mm³ -> cc
        
        # 统计变化体素
        changed_voxels = (significant != 0).sum()
        total_voxels = significant.size
        change_percent = (changed_voxels / total_voxels) * 100 if total_voxels > 0 else 0
        
        # 计算变化体积
        changed_volume_cc = changed_voxels * voxel_volume_cc
        
        # 计算 HU 变化统计
        significant_values = significant[significant != 0]
        if len(significant_values) > 0:
            max_hu_increase = float(significant_values.max())
            max_hu_decrease = float(significant_values.min())
            mean_hu_change = float(significant_values.mean())
        else:
            max_hu_increase = 0.0
            max_hu_decrease = 0.0
            mean_hu_change = 0.0
        
        # 计算增加和减少的区域
        increase_mask = significant > 0
        decrease_mask = significant < 0
        
        increase_voxels = increase_mask.sum()
        decrease_voxels = decrease_mask.sum()
        
        return {
            "changed_voxels": int(changed_voxels),
            "total_voxels": int(total_voxels),
            "change_percent": float(change_percent),
            "changed_volume_cc": float(changed_volume_cc),
            "max_hu_increase": max_hu_increase,
            "max_hu_decrease": max_hu_decrease,
            "mean_hu_change": mean_hu_change,
            "increase_voxels": int(increase_voxels),
            "decrease_voxels": int(decrease_voxels),
            "increase_percent": float(increase_voxels / total_voxels * 100) if total_voxels > 0 else 0,
            "decrease_percent": float(decrease_voxels / total_voxels * 100) if total_voxels > 0 else 0
        }
    
    def quantify_roi_changes(
        self,
        followup: np.ndarray,
        warped_baseline: np.ndarray,
        segmentation: np.ndarray,
        roi_label: int,
        spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)
    ) -> Dict[str, Any]:
        """
        量化特定 ROI 区域的变化
        
        Args:
            followup: 随访图像
            warped_baseline: 配准后的基线图像
            segmentation: 分割掩膜
            roi_label: ROI 标签值
            spacing: 体素间距 (mm)
            
        Returns:
            变化量化结果
        """
        # 提取 ROI
        roi_mask = (segmentation == roi_label)
        
        if roi_mask.sum() == 0:
            return {"error": "ROI not found"}
        
        # 计算体素体积 (cc)
        voxel_volume_cc = np.prod(spacing) / 1000  # mm³ -> cc
        
        # 基线体积
        baseline_roi = warped_baseline * roi_mask
        baseline_volume = roi_mask.sum() * voxel_volume_cc
        
        # 随访体积 (使用阈值检测病灶范围)
        followup_roi = followup * roi_mask
        
        # 计算平均 HU 变化
        baseline_mean_hu = baseline_roi[roi_mask].mean()
        followup_mean_hu = followup_roi[roi_mask].mean()
        hu_change = followup_mean_hu - baseline_mean_hu
        
        # 计算最大直径 (简化: 使用边界框)
        coords = np.array(np.where(roi_mask)).T
        if len(coords) > 0:
            bbox_size = coords.max(axis=0) - coords.min(axis=0)
            baseline_diameter = max(bbox_size) * spacing[0]  # mm
            followup_diameter = baseline_diameter  # 简化处理
        else:
            baseline_diameter = 0
            followup_diameter = 0
        
        return {
            "baseline_volume_cc": float(baseline_volume),
            "followup_volume_cc": float(baseline_volume),  # 需要更精确的体积计算
            "volume_change_percent": 0.0,  # 需要更精确计算
            "baseline_diameter_mm": float(baseline_diameter),
            "followup_diameter_mm": float(followup_diameter),
            "diameter_change_percent": 0.0,
            "baseline_mean_hu": float(baseline_mean_hu),
            "followup_mean_hu": float(followup_mean_hu),
            "hu_change": float(hu_change)
        }
    
    def evaluate_recist(
        self,
        baseline_diameter: float,
        followup_diameter: float
    ) -> RECISTResponse:
        """
        根据 RECIST 1.1 标准评估疗效
        
        Args:
            baseline_diameter: 基线最大直径 (mm)
            followup_diameter: 随访最大直径 (mm)
            
        Returns:
            RECIST 评估结果
        """
        if baseline_diameter == 0:
            return RECISTResponse.SD
        
        change_percent = ((followup_diameter - baseline_diameter) / baseline_diameter) * 100
        
        if followup_diameter == 0:
            return RECISTResponse.CR  # 完全缓解
        elif change_percent <= -30:
            return RECISTResponse.PR  # 部分缓解
        elif change_percent >= 20:
            return RECISTResponse.PD  # 疾病进展
        else:
            return RECISTResponse.SD  # 疾病稳定
    
    def calculate_doubling_time(
        self,
        baseline_volume: float,
        followup_volume: float,
        days_between: int
    ) -> Optional[float]:
        """
        计算体积倍增时间
        
        Args:
            baseline_volume: 基线体积
            followup_volume: 随访体积
            days_between: 间隔天数
            
        Returns:
            倍增时间 (天) 或 None
        """
        if baseline_volume <= 0 or followup_volume <= 0 or days_between <= 0:
            return None
        
        if followup_volume <= baseline_volume:
            return None  # 体积未增加
        
        # 倍增时间 = days * ln(2) / ln(V2/V1)
        ratio = followup_volume / baseline_volume
        if ratio > 1:
            doubling_time = days_between * np.log(2) / np.log(ratio)
            return float(doubling_time)
        
        return None


