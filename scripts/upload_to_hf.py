#!/usr/bin/env python3
"""
上传 NeuroScan AI 项目到 Hugging Face Hub

包括：
1. 代码仓库 (neuroscan-ai)
2. 模型仓库 (neuroscan-ai-models) 
3. 数据集仓库 (neuroscan-ai-dataset)
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder, upload_file

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# HF Token - 从环境变量获取
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("请设置环境变量 HF_TOKEN")


def get_api():
    """获取 HF API 实例"""
    return HfApi(token=HF_TOKEN)


def upload_code_repo():
    """上传代码仓库"""
    print("\n" + "=" * 60)
    print("📦 上传代码仓库: neuroscan-ai")
    print("=" * 60)
    
    api = get_api()
    repo_id = "cyd0806/neuroscan-ai"
    
    # 创建仓库
    try:
        create_repo(
            repo_id=repo_id,
            token=HF_TOKEN,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print(f"✅ 仓库已创建/存在: {repo_id}")
    except Exception as e:
        print(f"⚠️ 创建仓库: {e}")
    
    # 要排除的文件和目录
    ignore_patterns = [
        "*.pyc",
        "__pycache__",
        ".git",
        ".gitignore",
        "data/raw/*",
        "data/processed/*",
        "data/cache/*",
        "models/monai_bundles/*",
        "models/ollama/*",
        "logs/*",
        "output/*",
        "*.nii",
        "*.nii.gz",
        "*.pt",
        "*.pth",
        "*.ckpt",
        "*.bin",
        "*.safetensors",
        ".env",
        "*.log",
        "*.tmp",
    ]
    
    # 上传代码
    print("📤 上传代码文件...")
    try:
        upload_folder(
            folder_path=str(PROJECT_ROOT),
            repo_id=repo_id,
            repo_type="model",
            token=HF_TOKEN,
            ignore_patterns=ignore_patterns,
            commit_message="Upload NeuroScan AI code"
        )
        print(f"✅ 代码上传完成: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        raise


def upload_models_repo():
    """上传模型仓库"""
    print("\n" + "=" * 60)
    print("🧠 上传模型仓库: neuroscan-ai-models")
    print("=" * 60)
    
    api = get_api()
    repo_id = "cyd0806/neuroscan-ai-models"
    
    models_dir = PROJECT_ROOT / "models" / "monai_bundles"
    
    if not models_dir.exists():
        print("⚠️ 模型目录不存在，跳过")
        return
    
    # 创建仓库
    try:
        create_repo(
            repo_id=repo_id,
            token=HF_TOKEN,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print(f"✅ 仓库已创建/存在: {repo_id}")
    except Exception as e:
        print(f"⚠️ 创建仓库: {e}")
    
    # 创建 README
    readme_content = """---
license: apache-2.0
tags:
- medical-imaging
- segmentation
- monai
- ct-scan
---

# NeuroScan AI - Medical Imaging Models

This repository contains pretrained models for NeuroScan AI medical imaging analysis platform.

## Models

### wholeBody_ct_segmentation
- **Description**: Whole body CT segmentation model
- **Framework**: MONAI
- **Organs**: 104 anatomical structures
- **Input**: CT scan (NIfTI format)

## Usage

```python
from monai.bundle import download

# Download the model
download(name="wholeBody_ct_segmentation", bundle_dir="./models")
```

## License

Apache 2.0

## Citation

If you use these models, please cite NeuroScan AI project.
"""
    
    readme_path = models_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    # 上传模型
    print("📤 上传模型文件...")
    try:
        upload_folder(
            folder_path=str(models_dir),
            repo_id=repo_id,
            repo_type="model",
            token=HF_TOKEN,
            commit_message="Upload MONAI segmentation models"
        )
        print(f"✅ 模型上传完成: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        raise


def upload_dataset_repo():
    """上传数据集仓库"""
    print("\n" + "=" * 60)
    print("📊 上传数据集仓库: neuroscan-ai-dataset")
    print("=" * 60)
    
    api = get_api()
    repo_id = "cyd0806/neuroscan-ai-dataset"
    
    # 检查数据目录
    raw_dir = PROJECT_ROOT / "data" / "raw"
    
    if not raw_dir.exists():
        print("⚠️ 数据目录不存在，跳过")
        return
    
    # 创建数据集仓库
    try:
        create_repo(
            repo_id=repo_id,
            token=HF_TOKEN,
            repo_type="dataset",
            exist_ok=True,
            private=False
        )
        print(f"✅ 仓库已创建/存在: {repo_id}")
    except Exception as e:
        print(f"⚠️ 创建仓库: {e}")
    
    # 创建 README
    readme_content = """---
license: cc-by-nc-4.0
task_categories:
- image-segmentation
tags:
- medical-imaging
- ct-scan
- lung
- registration
size_categories:
- 1K<n<10K
---

# NeuroScan AI - Medical Imaging Dataset

This dataset contains sample medical imaging data for the NeuroScan AI platform.

## Dataset Description

### Learn2Reg Lung CT
- **Source**: [Learn2Reg Challenge](https://zenodo.org/record/3835682)
- **Description**: Paired inspiration and expiration lung CT scans
- **Format**: NIfTI (.nii.gz)
- **Cases**: 20 pairs
- **License**: CC BY-NC 4.0

## Usage

```python
# Download using huggingface_hub
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ydchen0806/neuroscan-ai-dataset",
    repo_type="dataset",
    local_dir="./data"
)
```

## Data Structure

```
data/
├── raw/
│   ├── training/
│   │   ├── scans/
│   │   │   ├── case_001_insp.nii.gz
│   │   │   ├── case_001_exp.nii.gz
│   │   │   └── ...
│   │   └── lungMasks/
│   │       └── ...
│   └── Learn2Reg_training.zip
└── processed/
    └── real_lung_001/
        ├── baseline.nii.gz
        ├── followup.nii.gz
        └── ...
```

## License

CC BY-NC 4.0 (Non-commercial use only)

## Citation

Please cite the original Learn2Reg challenge if you use this data.
"""
    
    readme_path = raw_dir.parent / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    # 上传数据集
    print("📤 上传数据集文件（这可能需要较长时间）...")
    try:
        upload_folder(
            folder_path=str(raw_dir.parent),
            repo_id=repo_id,
            repo_type="dataset",
            token=HF_TOKEN,
            ignore_patterns=["cache/*", "*.log", "*.tmp"],
            commit_message="Upload Learn2Reg lung CT dataset"
        )
        print(f"✅ 数据集上传完成: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        raise


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="上传 NeuroScan AI 到 Hugging Face Hub")
    parser.add_argument("--code", action="store_true", help="上传代码仓库")
    parser.add_argument("--models", action="store_true", help="上传模型仓库")
    parser.add_argument("--dataset", action="store_true", help="上传数据集仓库")
    parser.add_argument("--all", action="store_true", help="上传所有")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 NeuroScan AI -> Hugging Face Hub 上传工具")
    print("=" * 60)
    
    # 验证 Token
    try:
        api = get_api()
        user_info = api.whoami()
        print(f"✅ 已登录: {user_info['name']}")
    except Exception as e:
        print(f"❌ Token 验证失败: {e}")
        return 1
    
    if args.all or (not args.code and not args.models and not args.dataset):
        # 默认上传所有
        upload_code_repo()
        upload_models_repo()
        upload_dataset_repo()
    else:
        if args.code:
            upload_code_repo()
        if args.models:
            upload_models_repo()
        if args.dataset:
            upload_dataset_repo()
    
    print("\n" + "=" * 60)
    print("🎉 上传完成！")
    print("=" * 60)
    print("\n仓库地址:")
    print("  📦 代码: https://huggingface.co/cyd0806/neuroscan-ai")
    print("  🧠 模型: https://huggingface.co/cyd0806/neuroscan-ai-models")
    print("  📊 数据: https://huggingface.co/datasets/cyd0806/neuroscan-ai-dataset")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

