"""
影像处理工具 - 供 LLM Agent 调用
"""
from pathlib import Path
from typing import Dict, Any, Optional
from langchain_core.tools import tool

from app.core.config import settings
from app.core.logging import logger
from app.services.dicom import DicomLoader
from app.services.segmentation import OrganSegmentor
from app.services.registration import ImageRegistrator
from app.services.analysis import ChangeDetector, FeatureExtractor, ROIExtractor


# 初始化服务
_dicom_loader = None
_segmentor = None
_registrator = None
_change_detector = None
_feature_extractor = None
_roi_extractor = None


def get_dicom_loader():
    global _dicom_loader
    if _dicom_loader is None:
        _dicom_loader = DicomLoader()
    return _dicom_loader


def get_segmentor():
    global _segmentor
    if _segmentor is None:
        _segmentor = OrganSegmentor()
    return _segmentor


def get_registrator():
    global _registrator
    if _registrator is None:
        _registrator = ImageRegistrator()
    return _registrator


def get_change_detector():
    global _change_detector
    if _change_detector is None:
        _change_detector = ChangeDetector()
    return _change_detector


def get_feature_extractor():
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = FeatureExtractor()
    return _feature_extractor


def get_roi_extractor():
    global _roi_extractor
    if _roi_extractor is None:
        _roi_extractor = ROIExtractor()
    return _roi_extractor


@tool
def load_scan_tool(scan_id: str) -> Dict[str, Any]:
    """
    加载指定 ID 的 CT 扫描数据。
    
    Args:
        scan_id: 扫描的唯一标识符
        
    Returns:
        包含扫描元数据和文件路径的字典
    """
    logger.info(f"加载扫描: {scan_id}")
    
    # 查找 NIfTI 文件
    processed_dir = settings.PROCESSED_DATA_DIR
    nifti_files = list(processed_dir.glob(f"**/{scan_id}*.nii.gz"))
    
    if not nifti_files:
        return {"error": f"未找到扫描: {scan_id}"}
    
    nifti_path = nifti_files[0]
    
    # 加载数据
    loader = get_dicom_loader()
    data, img = loader.load_nifti(nifti_path)
    
    return {
        "scan_id": scan_id,
        "nifti_path": str(nifti_path),
        "shape": list(data.shape),
        "spacing": list(img.header.get_zooms()[:3]),
        "dtype": str(data.dtype)
    }


@tool
def segment_organ_tool(scan_id: str, save_organs: bool = False) -> Dict[str, Any]:
    """
    对 CT 扫描进行器官分割。
    
    Args:
        scan_id: 扫描 ID
        save_organs: 是否保存单独的器官掩膜
        
    Returns:
        分割结果信息
    """
    logger.info(f"执行器官分割: {scan_id}")
    
    # 查找 NIfTI 文件
    processed_dir = settings.PROCESSED_DATA_DIR
    nifti_files = list(processed_dir.glob(f"**/{scan_id}*.nii.gz"))
    
    if not nifti_files:
        return {"error": f"未找到扫描: {scan_id}"}
    
    nifti_path = nifti_files[0]
    
    # 执行分割
    segmentor = get_segmentor()
    seg_path, organ_paths = segmentor.segment_file(
        nifti_path,
        save_individual_organs=save_organs
    )
    
    return {
        "scan_id": scan_id,
        "segmentation_path": str(seg_path),
        "organ_masks": {k: str(v) for k, v in organ_paths.items()},
        "status": "completed"
    }


@tool
def extract_features_tool(
    scan_id: str,
    roi_center: list,
    roi_size: list = None
) -> Dict[str, Any]:
    """
    提取指定 ROI 区域的特征。
    
    Args:
        scan_id: 扫描 ID
        roi_center: ROI 中心坐标 [x, y, z]
        roi_size: ROI 大小 [sx, sy, sz]，默认 [64, 64, 64]
        
    Returns:
        特征提取结果
    """
    logger.info(f"提取特征: {scan_id}, center={roi_center}")
    
    if roi_size is None:
        roi_size = [64, 64, 64]
    
    # 加载数据
    loader = get_dicom_loader()
    processed_dir = settings.PROCESSED_DATA_DIR
    nifti_files = list(processed_dir.glob(f"**/{scan_id}*.nii.gz"))
    
    if not nifti_files:
        return {"error": f"未找到扫描: {scan_id}"}
    
    data, img = loader.load_nifti(nifti_files[0])
    spacing = tuple(img.header.get_zooms()[:3])
    
    # 提取特征
    extractor = get_feature_extractor()
    extractor.spacing = spacing
    
    features = extractor.extract_roi_features(
        data,
        tuple(roi_center),
        tuple(roi_size)
    )
    
    return {
        "scan_id": scan_id,
        "features": features
    }


@tool
def register_scans_tool(
    baseline_scan_id: str,
    followup_scan_id: str,
    use_deformable: bool = True
) -> Dict[str, Any]:
    """
    配准两个时间点的 CT 扫描。
    
    Args:
        baseline_scan_id: 基线扫描 ID
        followup_scan_id: 随访扫描 ID
        use_deformable: 是否使用非刚性配准
        
    Returns:
        配准结果信息
    """
    logger.info(f"配准扫描: {baseline_scan_id} -> {followup_scan_id}")
    
    processed_dir = settings.PROCESSED_DATA_DIR
    
    # 查找文件
    baseline_files = list(processed_dir.glob(f"**/{baseline_scan_id}*.nii.gz"))
    followup_files = list(processed_dir.glob(f"**/{followup_scan_id}*.nii.gz"))
    
    if not baseline_files:
        return {"error": f"未找到基线扫描: {baseline_scan_id}"}
    if not followup_files:
        return {"error": f"未找到随访扫描: {followup_scan_id}"}
    
    # 执行配准
    registrator = get_registrator()
    warped_path, transforms = registrator.register_files(
        followup_files[0],  # fixed
        baseline_files[0],  # moving
        use_deformable=use_deformable
    )
    
    return {
        "baseline_scan_id": baseline_scan_id,
        "followup_scan_id": followup_scan_id,
        "warped_baseline_path": str(warped_path),
        "registration_type": "rigid+deformable" if use_deformable else "rigid",
        "status": "completed"
    }


@tool
def compute_difference_tool(
    followup_scan_id: str,
    warped_baseline_path: str,
    generate_heatmap: bool = True
) -> Dict[str, Any]:
    """
    计算随访与配准后基线的差异。
    
    Args:
        followup_scan_id: 随访扫描 ID
        warped_baseline_path: 配准后的基线图像路径
        generate_heatmap: 是否生成热力图
        
    Returns:
        差异分析结果
    """
    logger.info(f"计算差异: {followup_scan_id}")
    
    # 加载数据
    loader = get_dicom_loader()
    processed_dir = settings.PROCESSED_DATA_DIR
    
    followup_files = list(processed_dir.glob(f"**/{followup_scan_id}*.nii.gz"))
    if not followup_files:
        return {"error": f"未找到随访扫描: {followup_scan_id}"}
    
    followup_data, followup_img = loader.load_nifti(followup_files[0])
    warped_data, _ = loader.load_nifti(Path(warped_baseline_path))
    
    # 计算差异
    detector = get_change_detector()
    diff_map, significant = detector.compute_difference_map(followup_data, warped_data)
    
    result = {
        "followup_scan_id": followup_scan_id,
        "max_increase_hu": float(significant.max()),
        "max_decrease_hu": float(significant.min()),
        "changed_voxels": int((significant != 0).sum()),
        "status": "completed"
    }
    
    # 生成热力图
    if generate_heatmap:
        heatmap_path = settings.STATIC_DIR / f"diff_map_{followup_scan_id}.png"
        detector.generate_heatmap(
            significant,
            followup_data,
            heatmap_path,
            view="axial"
        )
        result["heatmap_path"] = str(heatmap_path)
    
    return result


