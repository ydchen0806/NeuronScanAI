"""
分析相关数据模型
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class RECISTResponse(str, Enum):
    """RECIST 1.1 疗效评价"""
    CR = "Complete Response"  # 完全缓解
    PR = "Partial Response"   # 部分缓解
    SD = "Stable Disease"     # 疾病稳定
    PD = "Progressive Disease"  # 疾病进展


class NoduleFinding(BaseModel):
    """结节发现"""
    nodule_id: str = Field(..., description="结节ID")
    location: str = Field(..., description="解剖位置")
    organ: str = Field(..., description="所在器官")
    coordinates: List[float] = Field(..., description="空间坐标 [x, y, z]")
    volume_cc: float = Field(..., description="体积(cc)")
    max_diameter_mm: float = Field(..., description="最大直径(mm)")
    mean_hu: float = Field(..., description="平均CT值(HU)")
    density_type: str = Field(default="solid", description="密度类型")
    sphericity: float = Field(default=0.0, description="球形度")
    shape: str = Field(default="regular", description="形态")
    characteristics: List[str] = Field(default_factory=list, description="特征描述")


class AnalysisRequest(BaseModel):
    """单次分析请求"""
    scan_id: str = Field(..., description="扫描ID")
    region_of_interest: str = Field(default="chest", description="感兴趣区域")
    
    
class LongitudinalAnalysisRequest(BaseModel):
    """纵向分析请求"""
    baseline_scan_id: str = Field(..., description="基线扫描ID")
    followup_scan_id: str = Field(..., description="随访扫描ID")
    region_of_interest: str = Field(default="chest", description="感兴趣区域")


class ComparisonResult(BaseModel):
    """对比结果"""
    nodule_id: str
    baseline_date: datetime
    followup_date: datetime
    baseline_volume_cc: float
    followup_volume_cc: float
    volume_change_percent: float
    baseline_diameter_mm: float
    followup_diameter_mm: float
    diameter_change_percent: float
    doubling_time_days: Optional[float] = None
    recist_response: RECISTResponse
    is_new_lesion: bool = False
    heatmap_path: Optional[str] = None


class AnalysisResult(BaseModel):
    """分析结果"""
    task_id: str
    scan_id: str
    status: str = "pending"
    findings: List[NoduleFinding] = Field(default_factory=list)
    organ_masks: Dict[str, str] = Field(default_factory=dict, description="器官掩膜路径")
    anatomical_params: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class ReportResponse(BaseModel):
    """报告响应"""
    task_id: str
    status: str
    markdown_report: Optional[str] = None
    key_images: List[str] = Field(default_factory=list)
    findings: List[NoduleFinding] = Field(default_factory=list)
    comparisons: List[ComparisonResult] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


