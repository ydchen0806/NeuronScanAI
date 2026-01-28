"""
ROI 提取器 - 从 3D 数据中裁剪病灶区域
"""
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import nibabel as nib
from scipy import ndimage

from app.core.logging import logger


class ROIExtractor:
    """ROI 提取器"""
    
    def __init__(
        self,
        default_roi_size: Tuple[int, int, int] = (64, 64, 64),
        padding: int = 10
    ):
        """
        初始化 ROI 提取器
        
        Args:
            default_roi_size: 默认 ROI 大小
            padding: 边界填充
        """
        self.default_roi_size = default_roi_size
        self.padding = padding
    
    def extract_roi(
        self,
        image: np.ndarray,
        center: Tuple[int, int, int],
        roi_size: Optional[Tuple[int, int, int]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        提取 ROI
        
        Args:
            image: 3D 图像
            center: ROI 中心坐标
            roi_size: ROI 大小
            
        Returns:
            (ROI 数组, 元数据字典)
        """
        if roi_size is None:
            roi_size = self.default_roi_size
        
        x, y, z = center
        sx, sy, sz = roi_size
        
        # 计算边界 (带填充)
        x1 = max(0, x - sx // 2)
        x2 = min(image.shape[0], x + sx // 2)
        y1 = max(0, y - sy // 2)
        y2 = min(image.shape[1], y + sy // 2)
        z1 = max(0, z - sz // 2)
        z2 = min(image.shape[2], z + sz // 2)
        
        # 提取 ROI
        roi = image[x1:x2, y1:y2, z1:z2]
        
        # 如果 ROI 小于目标大小，进行零填充
        if roi.shape != roi_size:
            padded_roi = np.zeros(roi_size, dtype=roi.dtype)
            # 计算填充位置
            px = (roi_size[0] - roi.shape[0]) // 2
            py = (roi_size[1] - roi.shape[1]) // 2
            pz = (roi_size[2] - roi.shape[2]) // 2
            padded_roi[px:px+roi.shape[0], py:py+roi.shape[1], pz:pz+roi.shape[2]] = roi
            roi = padded_roi
        
        metadata = {
            "center": center,
            "roi_size": roi_size,
            "original_bounds": [[x1, x2], [y1, y2], [z1, z2]],
            "shape": roi.shape
        }
        
        return roi, metadata
    
    def extract_roi_from_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        margin: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        根据掩膜提取 ROI
        
        Args:
            image: 3D 图像
            mask: 二值掩膜
            margin: 边界扩展
            
        Returns:
            (ROI 图像, ROI 掩膜, 元数据)
        """
        # 找到掩膜边界
        coords = np.array(np.where(mask > 0))
        if coords.size == 0:
            logger.warning("空掩膜，返回空 ROI")
            return np.array([]), np.array([]), {"error": "empty_mask"}
        
        min_coords = coords.min(axis=1)
        max_coords = coords.max(axis=1)
        
        # 添加边界
        x1 = max(0, min_coords[0] - margin)
        x2 = min(image.shape[0], max_coords[0] + margin)
        y1 = max(0, min_coords[1] - margin)
        y2 = min(image.shape[1], max_coords[1] + margin)
        z1 = max(0, min_coords[2] - margin)
        z2 = min(image.shape[2], max_coords[2] + margin)
        
        # 提取 ROI
        roi_image = image[x1:x2, y1:y2, z1:z2]
        roi_mask = mask[x1:x2, y1:y2, z1:z2]
        
        # 计算中心
        center = ((x1 + x2) // 2, (y1 + y2) // 2, (z1 + z2) // 2)
        
        metadata = {
            "center": center,
            "bounds": [[x1, x2], [y1, y2], [z1, z2]],
            "shape": roi_image.shape,
            "voxel_count": int((mask > 0).sum())
        }
        
        return roi_image, roi_mask, metadata
    
    def find_lesion_candidates(
        self,
        image: np.ndarray,
        organ_mask: np.ndarray,
        hu_threshold: Tuple[float, float] = (-100, 200),
        min_size_voxels: int = 50,
        max_candidates: int = 10
    ) -> List[Dict[str, Any]]:
        """
        在器官掩膜内寻找疑似病灶
        
        Args:
            image: CT 图像
            organ_mask: 器官掩膜
            hu_threshold: HU 值范围 (用于筛选异常区域)
            min_size_voxels: 最小体素数
            max_candidates: 最大候选数
            
        Returns:
            候选病灶列表
        """
        # 在器官内筛选异常 HU 值区域
        abnormal_mask = (
            (image >= hu_threshold[0]) & 
            (image <= hu_threshold[1]) & 
            (organ_mask > 0)
        )
        
        # 连通域分析
        labeled_array, num_features = ndimage.label(abnormal_mask)
        
        candidates = []
        for i in range(1, num_features + 1):
            component_mask = (labeled_array == i)
            size = component_mask.sum()
            
            if size >= min_size_voxels:
                # 计算中心
                coords = np.array(np.where(component_mask))
                center = tuple(coords.mean(axis=1).astype(int))
                
                # 计算平均 HU
                mean_hu = image[component_mask].mean()
                
                candidates.append({
                    "center": center,
                    "size_voxels": int(size),
                    "mean_hu": float(mean_hu),
                    "label": i
                })
        
        # 按大小排序，取前 N 个
        candidates.sort(key=lambda x: x["size_voxels"], reverse=True)
        return candidates[:max_candidates]
    
    def save_roi(
        self,
        roi: np.ndarray,
        output_path: Path,
        affine: np.ndarray = None
    ) -> Path:
        """保存 ROI 为 NIfTI 文件"""
        if affine is None:
            affine = np.eye(4)
        
        nii = nib.Nifti1Image(roi, affine)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nii, output_path)
        
        logger.info(f"ROI 已保存: {output_path}")
        return output_path


