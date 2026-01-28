#!/usr/bin/env python3
"""
NeuroScan AI 服务启动脚本
确保在正确的 Python 环境中启动所有服务
"""

import os
import sys

# 确保项目路径在最前面
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 设置环境变量
os.environ['PYTHONPATH'] = PROJECT_ROOT

def main():
    import uvicorn
    from app.main import app
    
    print("=" * 50)
    print("🏥 NeuroScan AI - FastAPI 后端服务")
    print("=" * 50)
    print(f"项目路径: {PROJECT_ROOT}")
    print(f"Python: {sys.executable}")
    print("=" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main()

