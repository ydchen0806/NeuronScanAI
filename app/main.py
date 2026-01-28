"""
NeuroScan AI - 智能医学影像纵向诊疗系统
主应用入口
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.logging import logger
from app.api import router


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="""
# NeuroScan AI - 智能医学影像纵向诊疗系统

## 功能特性
- **DICOM 处理**: 自动解析和转换 DICOM 文件
- **器官分割**: 基于 MONAI 的全身 CT 分割
- **图像配准**: 两级配准策略 (刚性 + 非刚性)
- **变化检测**: 纵向时序分析和差异量化
- **智能报告**: LLM 驱动的诊断报告生成

## API 版本
v1.0.0
        """,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS 配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 挂载静态文件
    app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")
    
    # 注册路由
    app.include_router(router, prefix=settings.API_PREFIX)
    
    @app.on_event("startup")
    async def startup_event():
        logger.info(f"🏥 {settings.APP_NAME} 启动中...")
        logger.info(f"📁 数据目录: {settings.DATA_DIR}")
        logger.info(f"🤖 模型目录: {settings.MODELS_DIR}")
        logger.info(f"🔧 调试模式: {settings.DEBUG}")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info(f"🏥 {settings.APP_NAME} 关闭中...")
    
    @app.get("/")
    async def root():
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "status": "running",
            "docs": "/docs"
        }
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}
    
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8080,
        reload=settings.DEBUG
    )


