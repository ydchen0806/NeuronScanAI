#!/usr/bin/env python3
"""
模型下载脚本
下载 MONAI Bundle 和其他必要的模型

使用方法:
    python scripts/download_models.py

需要先设置代理:
    export http_proxy=http://192.168.32.28:18000
    export https_proxy=http://192.168.32.28:18000
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def download_monai_bundle():
    """下载 MONAI Bundle 分割模型"""
    print("=" * 60)
    print("MONAI Bundle 模型下载")
    print("=" * 60)
    
    from monai.bundle import download
    
    models_dir = project_root / "models" / "monai_bundles"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 要下载的 bundle 列表
    bundles = [
        "wholeBody_ct_segmentation",  # 全身 CT 分割
    ]
    
    for bundle_name in bundles:
        bundle_path = models_dir / bundle_name
        
        if bundle_path.exists():
            print(f"✅ {bundle_name} 已存在，跳过下载")
            continue
        
        print(f"\n⬇️  下载 {bundle_name}...")
        
        try:
            download(
                name=bundle_name,
                bundle_dir=str(models_dir),
                source="monaihosting"
            )
            print(f"✅ {bundle_name} 下载完成")
        except Exception as e:
            print(f"❌ {bundle_name} 下载失败: {e}")
    
    print("\n" + "=" * 60)
    print("模型下载完成!")
    print(f"📁 模型保存在: {models_dir}")
    print("=" * 60)


def download_llm_model():
    """
    下载 LLM 模型 (如果使用本地部署)
    这里提供使用 Hugging Face 下载的示例
    """
    print("\n" + "=" * 60)
    print("LLM 模型配置说明")
    print("=" * 60)
    
    print("""
本系统支持以下 LLM 配置方式:

1. 使用 OpenAI API 兼容接口 (推荐):
   - 设置环境变量:
     export LLM_BASE_URL="http://your-llm-server:8000/v1"
     export LLM_API_KEY="your-api-key"
   
2. 使用 vLLM 本地部署:
   - 安装 vLLM: pip install vllm
   - 启动服务:
     python -m vllm.entrypoints.openai.api_server \\
       --model meta-llama/Llama-3-70B-Instruct \\
       --tensor-parallel-size 4

3. 使用 Ollama:
   - 安装 Ollama: curl -fsSL https://ollama.com/install.sh | sh
   - 拉取模型: ollama pull llama3:70b
   - 设置环境变量:
     export LLM_BASE_URL="http://localhost:11434/v1"

4. 使用医疗专用模型 Med42:
   - 下载: huggingface-cli download m42-health/med42-70b
   - 使用 vLLM 部署
""")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="模型下载脚本")
    parser.add_argument(
        "--model",
        choices=["monai", "llm", "all"],
        default="monai",
        help="要下载的模型类型"
    )
    
    args = parser.parse_args()
    
    if args.model in ["monai", "all"]:
        download_monai_bundle()
    
    if args.model in ["llm", "all"]:
        download_llm_model()


if __name__ == "__main__":
    main()


