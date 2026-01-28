"""
应用配置管理
"""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, model_validator


# 获取项目根目录
_BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """应用配置"""
    
    # 应用基础配置
    APP_NAME: str = "NeuroScan AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # 路径配置 - 使用默认值而不是 None
    BASE_DIR: Path = _BASE_DIR
    DATA_DIR: Path = _BASE_DIR / "data"
    RAW_DATA_DIR: Path = _BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR: Path = _BASE_DIR / "data" / "processed"
    CACHE_DIR: Path = _BASE_DIR / "data" / "cache"
    MODELS_DIR: Path = _BASE_DIR / "models"
    STATIC_DIR: Path = _BASE_DIR / "static"
    
    # DICOM 处理配置
    TARGET_SPACING: tuple = (1.0, 1.0, 1.0)  # mm
    ORIENTATION: str = "RAS"
    
    # 分割模型配置
    SEGMENTATION_MODEL: str = "wholeBody_ct_segmentation"
    SLIDING_WINDOW_SIZE: tuple = (96, 96, 96)
    SLIDING_WINDOW_OVERLAP: float = 0.25
    
    # 配准配置
    REGISTRATION_ITERATIONS: int = 1000
    REGISTRATION_SAMPLING_RATE: float = 0.1
    
    # 变化检测配置
    DIFF_THRESHOLD_HU: float = 30.0
    GAUSSIAN_SIGMA: float = 1.0
    
    # LLM 配置
    LLM_MODEL: str = "llama3-70b-instruct"
    LLM_BASE_URL: str = "http://localhost:8000/v1"
    LLM_API_KEY: str = "local-key"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 4096
    
    # 数据库配置
    DATABASE_URL: str = "sqlite:///./neuroscan.db"
    
    # ChromaDB 配置
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    
    # API 配置
    API_PREFIX: str = "/api/v1"
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }
    
    @model_validator(mode='after')
    def create_directories(self):
        """确保所有目录存在"""
        for dir_path in [self.DATA_DIR, self.RAW_DATA_DIR, self.PROCESSED_DATA_DIR, 
                         self.CACHE_DIR, self.MODELS_DIR, self.STATIC_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
        return self


settings = Settings()


