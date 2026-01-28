"""
FastAPI 路由定义
"""
import os
import uuid
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse

from app.core.config import settings
from app.core.logging import logger
from app.schemas.dicom import ScanUploadResponse
from app.schemas.analysis import (
    AnalysisRequest,
    LongitudinalAnalysisRequest,
    ReportResponse,
    NoduleFinding
)
from app.services.dicom import DicomLoader
from app.services.segmentation import OrganSegmentor
from app.services.registration import ImageRegistrator
from app.services.analysis import ChangeDetector, FeatureExtractor, ROIExtractor
from app.agents import Orchestrator


router = APIRouter()

# 任务存储 (生产环境应使用数据库)
_tasks = {}
_scans = {}


@router.post("/ingest", response_model=ScanUploadResponse)
async def ingest_dicom(
    file: UploadFile = File(...),
    patient_id: Optional[str] = None
):
    """
    上传 DICOM ZIP 包
    
    - **file**: DICOM 文件的 ZIP 压缩包
    - **patient_id**: 可选的患者 ID
    """
    logger.info(f"接收 DICOM 上传: {file.filename}")
    
    # 验证文件类型
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="请上传 ZIP 格式的 DICOM 文件包")
    
    # 保存上传的文件
    temp_path = settings.CACHE_DIR / f"upload_{uuid.uuid4()}.zip"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 处理 DICOM
        loader = DicomLoader()
        scan_info = loader.process_upload(temp_path, patient_id)
        
        # 存储扫描信息
        _scans[scan_info.scan_id] = scan_info
        
        return ScanUploadResponse(
            scan_id=scan_info.scan_id,
            message="DICOM 上传成功",
            metadata=scan_info.metadata,
            nifti_path=scan_info.nifti_path
        )
        
    except Exception as e:
        logger.error(f"DICOM 处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时文件
        if temp_path.exists():
            temp_path.unlink()


@router.post("/analyze/single")
async def analyze_single(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    触发单次扫描分析
    
    - **scan_id**: 扫描 ID
    - **region_of_interest**: 感兴趣区域 (chest, abdomen, etc.)
    """
    scan_id = request.scan_id
    
    if scan_id not in _scans:
        raise HTTPException(status_code=404, detail=f"扫描不存在: {scan_id}")
    
    task_id = f"single_{uuid.uuid4()}"
    _tasks[task_id] = {"status": "pending", "scan_id": scan_id}
    
    # 后台执行分析
    background_tasks.add_task(
        _run_single_analysis,
        task_id,
        scan_id,
        request.region_of_interest
    )
    
    return {"task_id": task_id, "status": "processing"}


@router.post("/analyze/longitudinal")
async def analyze_longitudinal(
    request: LongitudinalAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    触发纵向对比分析
    
    - **baseline_scan_id**: 基线扫描 ID
    - **followup_scan_id**: 随访扫描 ID
    - **region_of_interest**: 感兴趣区域
    """
    baseline_id = request.baseline_scan_id
    followup_id = request.followup_scan_id
    
    if baseline_id not in _scans:
        raise HTTPException(status_code=404, detail=f"基线扫描不存在: {baseline_id}")
    if followup_id not in _scans:
        raise HTTPException(status_code=404, detail=f"随访扫描不存在: {followup_id}")
    
    task_id = f"longitudinal_{uuid.uuid4()}"
    _tasks[task_id] = {
        "status": "pending",
        "baseline_scan_id": baseline_id,
        "followup_scan_id": followup_id
    }
    
    # 后台执行分析
    background_tasks.add_task(
        _run_longitudinal_analysis,
        task_id,
        baseline_id,
        followup_id,
        request.region_of_interest
    )
    
    return {"task_id": task_id, "status": "processing"}


@router.get("/reports/{task_id}")
async def get_report(task_id: str):
    """
    获取分析报告
    
    - **task_id**: 任务 ID
    """
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    
    task = _tasks[task_id]
    
    return {
        "task_id": task_id,
        "status": task.get("status", "unknown"),
        "markdown_report": task.get("report", ""),
        "key_images": task.get("key_images", []),
        "error": task.get("error")
    }


@router.get("/scans/{scan_id}")
async def get_scan_info(scan_id: str):
    """
    获取扫描信息
    
    - **scan_id**: 扫描 ID
    """
    if scan_id not in _scans:
        raise HTTPException(status_code=404, detail=f"扫描不存在: {scan_id}")
    
    scan = _scans[scan_id]
    return scan.model_dump()


@router.get("/scans")
async def list_scans():
    """列出所有扫描"""
    return {
        "scans": [
            {"scan_id": k, "patient_id": v.metadata.patient_id, "status": v.status}
            for k, v in _scans.items()
        ]
    }


@router.get("/static/{filename}")
async def get_static_file(filename: str):
    """获取静态文件 (如热力图)"""
    file_path = settings.STATIC_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(file_path)


# ============ 后台任务函数 ============

async def _run_single_analysis(task_id: str, scan_id: str, region: str):
    """运行单次分析任务"""
    try:
        _tasks[task_id]["status"] = "processing"
        
        scan = _scans[scan_id]
        loader = DicomLoader()
        
        # 加载数据
        nifti_path = Path(scan.nifti_path)
        data, img = loader.load_nifti(nifti_path)
        
        # 执行分割
        segmentor = OrganSegmentor()
        seg_path, organ_paths = segmentor.segment_file(nifti_path)
        
        # 提取特征
        extractor = FeatureExtractor(spacing=tuple(img.header.get_zooms()[:3]))
        roi_extractor = ROIExtractor()
        
        # 寻找候选病灶 (简化示例)
        findings = []
        
        # 使用 Orchestrator 生成报告
        orchestrator = Orchestrator()
        
        scan_info = {
            "scan_id": scan_id,
            "study_date": scan.metadata.study_date.isoformat() if scan.metadata.study_date else "unknown",
            "body_part": region,
            "detected_organs": list(organ_paths.keys())
        }
        
        result = orchestrator.analyze_single(
            patient_id=scan.metadata.patient_id,
            scan_info=scan_info,
            findings=findings
        )
        
        _tasks[task_id]["status"] = "completed"
        _tasks[task_id]["report"] = result.markdown_report
        _tasks[task_id]["key_images"] = result.key_images
        
    except Exception as e:
        logger.error(f"分析任务失败: {e}")
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["error"] = str(e)


async def _run_longitudinal_analysis(
    task_id: str,
    baseline_id: str,
    followup_id: str,
    region: str
):
    """运行纵向分析任务"""
    try:
        _tasks[task_id]["status"] = "processing"
        
        baseline_scan = _scans[baseline_id]
        followup_scan = _scans[followup_id]
        
        loader = DicomLoader()
        registrator = ImageRegistrator()
        detector = ChangeDetector()
        
        # 加载数据
        baseline_path = Path(baseline_scan.nifti_path)
        followup_path = Path(followup_scan.nifti_path)
        
        # 配准
        warped_path, transforms = registrator.register_files(
            followup_path,
            baseline_path
        )
        
        # 计算差异
        followup_data, followup_img = loader.load_nifti(followup_path)
        warped_data, _ = loader.load_nifti(warped_path)
        
        diff_map, significant = detector.compute_difference_map(followup_data, warped_data)
        
        # 生成热力图
        heatmap_path = settings.STATIC_DIR / f"diff_{task_id}.png"
        detector.generate_heatmap(significant, followup_data, heatmap_path)
        
        # 使用 Orchestrator 生成报告
        orchestrator = Orchestrator()
        
        baseline_info = {
            "scan_id": baseline_id,
            "study_date": baseline_scan.metadata.study_date.isoformat() if baseline_scan.metadata.study_date else "unknown",
            "body_part": region
        }
        
        followup_info = {
            "scan_id": followup_id,
            "study_date": followup_scan.metadata.study_date.isoformat() if followup_scan.metadata.study_date else "unknown",
            "body_part": region
        }
        
        comparisons = [{
            "max_change_hu": float(significant.max()),
            "min_change_hu": float(significant.min()),
            "changed_voxels": int((significant != 0).sum())
        }]
        
        result = orchestrator.analyze_longitudinal(
            patient_id=baseline_scan.metadata.patient_id,
            baseline_scan=baseline_info,
            followup_scan=followup_info,
            findings=[],
            comparisons=comparisons,
            key_images=[str(heatmap_path)]
        )
        
        _tasks[task_id]["status"] = "completed"
        _tasks[task_id]["report"] = result.markdown_report
        _tasks[task_id]["key_images"] = [str(heatmap_path)]
        
    except Exception as e:
        logger.error(f"纵向分析任务失败: {e}")
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["error"] = str(e)


