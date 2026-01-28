"""
DICOM 相关数据模型
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
import uuid


class DicomMetadata(BaseModel):
    """DICOM 元数据"""
    patient_id: str = Field(..., description="患者ID")
    patient_name: Optional[str] = Field(None, description="患者姓名")
    study_date: Optional[datetime] = Field(None, description="检查日期")
    study_description: Optional[str] = Field(None, description="检查描述")
    series_description: Optional[str] = Field(None, description="序列描述")
    modality: str = Field(default="CT", description="检查模态")
    manufacturer: Optional[str] = Field(None, description="设备厂商")
    slice_thickness: Optional[float] = Field(None, description="层厚(mm)")
    pixel_spacing: Optional[List[float]] = Field(None, description="像素间距")
    image_count: int = Field(default=0, description="图像数量")
    

class ScanInfo(BaseModel):
    """扫描信息"""
    scan_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="扫描ID")
    metadata: DicomMetadata
    nifti_path: Optional[str] = Field(None, description="NIfTI文件路径")
    raw_path: Optional[str] = Field(None, description="原始DICOM路径")
    created_at: datetime = Field(default_factory=datetime.now)
    status: str = Field(default="pending", description="处理状态")
    
    class Config:
        json_schema_extra = {
            "example": {
                "scan_id": "550e8400-e29b-41d4-a716-446655440000",
                "metadata": {
                    "patient_id": "P001",
                    "study_date": "2025-01-01T00:00:00",
                    "modality": "CT"
                },
                "status": "completed"
            }
        }


class ScanUploadResponse(BaseModel):
    """扫描上传响应"""
    scan_id: str
    message: str
    metadata: DicomMetadata
    nifti_path: Optional[str] = None


