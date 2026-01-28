"""
器官分割服务 - 基于 MONAI Bundle
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import nibabel as nib
import torch
from monai.bundle import ConfigParser, download
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Orientation,
    Spacing,
    ScaleIntensityRange,
    AsDiscrete,
    KeepLargestConnectedComponent
)

from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import SegmentationError


# 器官标签映射 (wholeBody_ct_segmentation 的 104 个结构)
ORGAN_LABELS = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "aorta",
    8: "inferior_vena_cava",
    9: "portal_splenic_vein",
    10: "pancreas",
    11: "adrenal_gland_right",
    12: "adrenal_gland_left",
    13: "lung_upper_lobe_left",
    14: "lung_lower_lobe_left",
    15: "lung_upper_lobe_right",
    16: "lung_middle_lobe_right",
    17: "lung_lower_lobe_right",
    18: "vertebrae_L5",
    19: "vertebrae_L4",
    20: "vertebrae_L3",
    21: "vertebrae_L2",
    22: "vertebrae_L1",
    23: "vertebrae_T12",
    24: "vertebrae_T11",
    25: "vertebrae_T10",
    26: "vertebrae_T9",
    27: "vertebrae_T8",
    28: "vertebrae_T7",
    29: "vertebrae_T6",
    30: "vertebrae_T5",
    31: "vertebrae_T4",
    32: "vertebrae_T3",
    33: "vertebrae_T2",
    34: "vertebrae_T1",
    35: "vertebrae_C7",
    36: "vertebrae_C6",
    37: "vertebrae_C5",
    38: "vertebrae_C4",
    39: "vertebrae_C3",
    40: "vertebrae_C2",
    41: "vertebrae_C1",
    42: "esophagus",
    43: "trachea",
    44: "heart_myocardium",
    45: "heart_atrium_left",
    46: "heart_ventricle_left",
    47: "heart_atrium_right",
    48: "heart_ventricle_right",
    49: "pulmonary_artery",
    50: "brain",
    51: "iliac_artery_left",
    52: "iliac_artery_right",
    53: "iliac_vena_left",
    54: "iliac_vena_right",
    55: "small_bowel",
    56: "duodenum",
    57: "colon",
    58: "rib_left_1",
    59: "rib_left_2",
    60: "rib_left_3",
    61: "rib_left_4",
    62: "rib_left_5",
    63: "rib_left_6",
    64: "rib_left_7",
    65: "rib_left_8",
    66: "rib_left_9",
    67: "rib_left_10",
    68: "rib_left_11",
    69: "rib_left_12",
    70: "rib_right_1",
    71: "rib_right_2",
    72: "rib_right_3",
    73: "rib_right_4",
    74: "rib_right_5",
    75: "rib_right_6",
    76: "rib_right_7",
    77: "rib_right_8",
    78: "rib_right_9",
    79: "rib_right_10",
    80: "rib_right_11",
    81: "rib_right_12",
    82: "humerus_left",
    83: "humerus_right",
    84: "scapula_left",
    85: "scapula_right",
    86: "clavicula_left",
    87: "clavicula_right",
    88: "femur_left",
    89: "femur_right",
    90: "hip_left",
    91: "hip_right",
    92: "sacrum",
    93: "face",
    94: "gluteus_maximus_left",
    95: "gluteus_maximus_right",
    96: "gluteus_medius_left",
    97: "gluteus_medius_right",
    98: "gluteus_minimus_left",
    99: "gluteus_minimus_right",
    100: "autochthon_left",
    101: "autochthon_right",
    102: "iliopsoas_left",
    103: "iliopsoas_right",
    104: "urinary_bladder",
}


class OrganSegmentor:
    """器官分割器"""
    
    def __init__(self, device: str = None):
        """
        初始化分割器
        
        Args:
            device: 计算设备 (cuda/cpu)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.model_loaded = False
        
        # 推理器
        self.inferer = SlidingWindowInferer(
            roi_size=settings.SLIDING_WINDOW_SIZE,
            sw_batch_size=4,
            overlap=settings.SLIDING_WINDOW_OVERLAP,
            mode="gaussian"
        )
        
        # 预处理
        self.preprocess = Compose([
            ScaleIntensityRange(
                a_min=-1024,
                a_max=1024,
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),
        ])
        
        # 后处理
        self.postprocess = Compose([
            AsDiscrete(argmax=True),
            KeepLargestConnectedComponent(applied_labels=list(range(1, 105)))
        ])
    
    def download_model(self) -> Path:
        """下载 MONAI Bundle 模型"""
        model_dir = settings.MODELS_DIR / "monai_bundles"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        bundle_name = settings.SEGMENTATION_MODEL
        bundle_path = model_dir / bundle_name
        
        if not bundle_path.exists():
            logger.info(f"下载 MONAI Bundle: {bundle_name}")
            try:
                download(
                    name=bundle_name,
                    bundle_dir=str(model_dir),
                    source="monaihosting"
                )
                logger.info(f"模型下载完成: {bundle_path}")
            except Exception as e:
                logger.error(f"模型下载失败: {str(e)}")
                raise SegmentationError(f"模型下载失败: {str(e)}")
        
        return bundle_path
    
    def load_model(self):
        """加载分割模型"""
        if self.model_loaded:
            return
        
        try:
            bundle_path = self.download_model()
            
            # 解析配置
            config_path = bundle_path / "configs" / "inference.json"
            parser = ConfigParser()
            parser.read_config(str(config_path))
            
            # 获取网络
            self.model = parser.get_parsed_content("network")
            
            # 加载权重
            weights_path = bundle_path / "models" / "model.pt"
            if weights_path.exists():
                checkpoint = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            
            logger.info(f"分割模型加载成功，设备: {self.device}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise SegmentationError(f"模型加载失败: {str(e)}")
    
    def segment(
        self, 
        image: np.ndarray,
        return_all_organs: bool = False
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        执行器官分割
        
        Args:
            image: 输入 CT 图像 (3D numpy array)
            return_all_organs: 是否返回所有器官的单独掩膜
            
        Returns:
            (分割掩膜, 器官掩膜字典)
        """
        self.load_model()
        
        try:
            # 预处理
            image_tensor = torch.from_numpy(image).float()
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)  # 添加 channel 维度
            
            image_tensor = self.preprocess(image_tensor)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)  # 添加 batch 维度
            
            # 推理
            with torch.no_grad():
                output = self.inferer(image_tensor, self.model)
            
            # 后处理
            segmentation = self.postprocess(output[0]).cpu().numpy()
            
            # 提取各器官掩膜
            organ_masks = {}
            if return_all_organs:
                for label_id, organ_name in ORGAN_LABELS.items():
                    mask = (segmentation == label_id).astype(np.uint8)
                    if mask.sum() > 0:
                        organ_masks[organ_name] = mask
            
            return segmentation.astype(np.uint8), organ_masks
            
        except Exception as e:
            logger.error(f"分割失败: {str(e)}")
            raise SegmentationError(f"分割失败: {str(e)}")
    
    def segment_file(
        self, 
        nifti_path: Path,
        output_path: Optional[Path] = None,
        save_individual_organs: bool = False
    ) -> Tuple[Path, Dict[str, Path]]:
        """
        分割 NIfTI 文件
        
        Args:
            nifti_path: 输入 NIfTI 文件路径
            output_path: 输出路径
            save_individual_organs: 是否保存单独的器官掩膜
            
        Returns:
            (分割结果路径, 器官掩膜路径字典)
        """
        # 加载图像
        img = nib.load(nifti_path)
        data = img.get_fdata()
        
        # 执行分割
        segmentation, organ_masks = self.segment(data, return_all_organs=save_individual_organs)
        
        # 确定输出路径
        if output_path is None:
            output_path = nifti_path.parent / f"{nifti_path.stem.replace('.nii', '')}_seg.nii.gz"
        
        # 保存分割结果
        seg_img = nib.Nifti1Image(segmentation, img.affine, img.header)
        nib.save(seg_img, output_path)
        
        # 保存单独的器官掩膜
        organ_paths = {}
        if save_individual_organs:
            organs_dir = output_path.parent / "organs"
            organs_dir.mkdir(exist_ok=True)
            
            for organ_name, mask in organ_masks.items():
                organ_path = organs_dir / f"{organ_name}.nii.gz"
                organ_img = nib.Nifti1Image(mask, img.affine, img.header)
                nib.save(organ_img, organ_path)
                organ_paths[organ_name] = organ_path
        
        logger.info(f"分割完成: {output_path}")
        return output_path, organ_paths
    
    def get_organ_location(
        self, 
        segmentation: np.ndarray, 
        coordinates: Tuple[int, int, int]
    ) -> Optional[str]:
        """
        获取坐标所在的器官
        
        Args:
            segmentation: 分割掩膜
            coordinates: (x, y, z) 坐标
            
        Returns:
            器官名称或 None
        """
        x, y, z = coordinates
        label = segmentation[x, y, z]
        return ORGAN_LABELS.get(int(label), None)


