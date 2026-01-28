"""
特征提取器 - 计算病灶的物理指标
"""
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy import ndimage
from skimage import measure

from app.core.logging import logger
from app.schemas.analysis import NoduleFinding


class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self, spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)):
        """
        初始化特征提取器
        
        Args:
            spacing: 体素间距 (mm)
        """
        self.spacing = spacing
        self.voxel_volume_cc = np.prod(spacing) / 1000  # mm³ -> cc
    
    def extract_features(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        nodule_id: str = "nodule_1",
        organ: str = "unknown",
        location: str = "unknown"
    ) -> NoduleFinding:
        """
        提取病灶特征
        
        Args:
            image: CT 图像
            mask: 病灶掩膜
            nodule_id: 结节ID
            organ: 所在器官
            location: 解剖位置
            
        Returns:
            NoduleFinding 对象
        """
        # 确保掩膜是二值的
        binary_mask = (mask > 0).astype(np.uint8)
        
        if binary_mask.sum() == 0:
            logger.warning(f"空掩膜: {nodule_id}")
            return NoduleFinding(
                nodule_id=nodule_id,
                location=location,
                organ=organ,
                coordinates=[0, 0, 0],
                volume_cc=0,
                max_diameter_mm=0,
                mean_hu=0,
                density_type="unknown",
                sphericity=0,
                shape="unknown"
            )
        
        # 计算体积 (cc)
        volume_cc = binary_mask.sum() * self.voxel_volume_cc
        
        # 计算质心坐标
        coords = np.array(np.where(binary_mask)).T
        centroid = coords.mean(axis=0).tolist()
        
        # 计算最大直径 (mm)
        max_diameter = self._calculate_max_diameter(binary_mask)
        
        # 计算平均 HU 值
        masked_values = image[binary_mask > 0]
        mean_hu = float(masked_values.mean())
        
        # 判断密度类型
        density_type = self._classify_density(mean_hu)
        
        # 计算球形度
        sphericity = self._calculate_sphericity(binary_mask)
        
        # 判断形态
        shape = self._classify_shape(binary_mask)
        
        # 提取特征描述
        characteristics = self._extract_characteristics(image, binary_mask, mean_hu)
        
        return NoduleFinding(
            nodule_id=nodule_id,
            location=location,
            organ=organ,
            coordinates=centroid,
            volume_cc=float(volume_cc),
            max_diameter_mm=float(max_diameter),
            mean_hu=mean_hu,
            density_type=density_type,
            sphericity=float(sphericity),
            shape=shape,
            characteristics=characteristics
        )
    
    def _calculate_max_diameter(self, mask: np.ndarray) -> float:
        """计算最大直径"""
        coords = np.array(np.where(mask)).T
        if len(coords) < 2:
            return 0.0
        
        # 简化计算: 使用边界框对角线
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        bbox_size = (max_coords - min_coords) * np.array(self.spacing)
        
        # 返回最大轴向直径
        return float(max(bbox_size))
    
    def _calculate_sphericity(self, mask: np.ndarray) -> float:
        """
        计算球形度
        球形度 = (π^(1/3) * (6V)^(2/3)) / A
        其中 V 是体积，A 是表面积
        """
        try:
            # 计算体积 (体素数)
            volume = mask.sum()
            
            if volume == 0:
                return 0.0
            
            # 使用 marching cubes 计算表面积
            verts, faces, _, _ = measure.marching_cubes(mask, level=0.5, spacing=self.spacing)
            surface_area = measure.mesh_surface_area(verts, faces)
            
            if surface_area == 0:
                return 0.0
            
            # 计算球形度
            volume_physical = volume * np.prod(self.spacing)
            sphericity = (np.pi ** (1/3) * (6 * volume_physical) ** (2/3)) / surface_area
            
            return min(sphericity, 1.0)  # 球形度最大为 1
            
        except Exception as e:
            logger.warning(f"球形度计算失败: {e}")
            return 0.0
    
    def _classify_density(self, mean_hu: float) -> str:
        """根据 HU 值分类密度类型"""
        if mean_hu < -500:
            return "ground_glass"  # 磨玻璃
        elif mean_hu < -100:
            return "part_solid"    # 部分实性
        else:
            return "solid"         # 实性
    
    def _classify_shape(self, mask: np.ndarray) -> str:
        """分类形态"""
        # 简化判断: 基于球形度
        sphericity = self._calculate_sphericity(mask)
        
        if sphericity > 0.8:
            return "regular"       # 规则
        elif sphericity > 0.5:
            return "lobulated"     # 分叶状
        else:
            return "irregular"     # 不规则
    
    def _extract_characteristics(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        mean_hu: float
    ) -> List[str]:
        """提取特征描述"""
        characteristics = []
        
        # 密度特征
        if mean_hu < -500:
            characteristics.append("磨玻璃影(GGO)")
        elif mean_hu < -100:
            characteristics.append("部分实性结节")
        else:
            characteristics.append("实性结节")
        
        # 形态特征
        shape = self._classify_shape(mask)
        if shape == "lobulated":
            characteristics.append("分叶状边缘")
        elif shape == "irregular":
            characteristics.append("边缘不规则")
        
        # 大小特征
        max_diameter = self._calculate_max_diameter(mask)
        if max_diameter < 6:
            characteristics.append("微小结节(<6mm)")
        elif max_diameter < 8:
            characteristics.append("小结节(6-8mm)")
        elif max_diameter < 30:
            characteristics.append("结节(8-30mm)")
        else:
            characteristics.append("肿块(>30mm)")
        
        return characteristics
    
    def extract_roi_features(
        self,
        image: np.ndarray,
        roi_coords: Tuple[int, int, int],
        roi_size: Tuple[int, int, int] = (64, 64, 64)
    ) -> Dict[str, Any]:
        """
        提取 ROI 区域的特征
        
        Args:
            image: CT 图像
            roi_coords: ROI 中心坐标 (x, y, z)
            roi_size: ROI 大小
            
        Returns:
            特征字典
        """
        x, y, z = roi_coords
        sx, sy, sz = roi_size
        
        # 计算边界
        x1 = max(0, x - sx // 2)
        x2 = min(image.shape[0], x + sx // 2)
        y1 = max(0, y - sy // 2)
        y2 = min(image.shape[1], y + sy // 2)
        z1 = max(0, z - sz // 2)
        z2 = min(image.shape[2], z + sz // 2)
        
        # 提取 ROI
        roi = image[x1:x2, y1:y2, z1:z2]
        
        return {
            "roi_shape": roi.shape,
            "mean_hu": float(roi.mean()),
            "std_hu": float(roi.std()),
            "min_hu": float(roi.min()),
            "max_hu": float(roi.max()),
            "coordinates": [x, y, z],
            "bounds": [[x1, x2], [y1, y2], [z1, z2]]
        }


