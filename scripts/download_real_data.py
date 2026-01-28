#!/usr/bin/env python3
"""
下载真实时序 CT 数据
数据源: Learn2Reg Challenge (Task 02 - Lung CT)
托管: Zenodo (稳定)

该数据集包含同一病人的 "吸气末" 和 "呼气末" CT 扫描，
具有显著的解剖形变，是测试配准算法的最佳数据。
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_with_progress(url: str, output_path: str, description: str = ""):
    """带进度条的下载"""
    print(f"⬇️  下载: {description}")
    print(f"   URL: {url}")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r   进度: {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
            sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, output_path, progress_hook)
        print("\n   ✅ 下载完成")
        return True
    except Exception as e:
        print(f"\n   ❌ 下载失败: {e}")
        return False


def download_learn2reg_lung():
    """
    从 Zenodo 下载 Learn2Reg 挑战赛的肺部 CT 数据 (Task 02)。
    
    这是一对真实的 '吸气-呼气' CT，包含显著的解剖形变，
    非常适合用于演示 '配准 (Registration)' 和 '差异分析'。
    """
    print("\n" + "="*60)
    print("Learn2Reg Lung CT 数据下载")
    print("="*60)
    
    # Zenodo 上的 Learn2Reg Task 2 (Lung) 数据集链接 (稳定)
    # 使用 API 链接格式
    url = "https://zenodo.org/api/records/3835682/files/training.zip/content"
    
    raw_dir = PROJECT_ROOT / "data" / "raw"
    zip_path = raw_dir / "Learn2Reg_training.zip"
    extract_dir = raw_dir / "Learn2Reg_Lung"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 下载
    if not zip_path.exists():
        print(f"\n📦 正在从 Zenodo 下载真实时序 CT 数据 (约 300MB)...")
        
        # 设置代理
        proxy = os.environ.get('http_proxy') or os.environ.get('https_proxy')
        if proxy:
            print(f"   使用代理: {proxy}")
            proxy_handler = urllib.request.ProxyHandler({
                'http': proxy,
                'https': proxy
            })
            opener = urllib.request.build_opener(proxy_handler)
            urllib.request.install_opener(opener)
        
        if not download_with_progress(url, str(zip_path), "Learn2Reg Task02 Lung CT"):
            return None
    else:
        print(f"\n📦 压缩包已存在: {zip_path}")
        print("   跳过下载，直接解压...")
    
    # 2. 解压
    # Learn2Reg training.zip 解压后的结构可能不同，需要检查
    task_dir = raw_dir / "training"
    if not task_dir.exists():
        print(f"\n📂 正在解压到: {raw_dir}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # 先列出内容
                namelist = zip_ref.namelist()
                print(f"   压缩包内容: {namelist[:5]}...")
                zip_ref.extractall(raw_dir)
            print("   ✅ 解压完成")
            
            # 检查解压后的目录结构
            print(f"   检查解压后的目录...")
            for item in raw_dir.iterdir():
                if item.is_dir():
                    print(f"      - {item.name}/")
        except Exception as e:
            print(f"   ❌ 解压失败: {e}")
            return None
    else:
        print(f"\n📂 数据已解压: {task_dir}")
    
    # 3. 查找图像目录 (可能是 imagesTr 或直接在 training 下)
    images_dir = None
    labels_dir = None
    
    # 尝试不同的目录结构
    possible_paths = [
        (task_dir / "imagesTr", task_dir / "labelsTr"),
        (task_dir, None),
        (raw_dir / "Task02_Lung" / "imagesTr", raw_dir / "Task02_Lung" / "labelsTr"),
        (raw_dir / "imagesTr", raw_dir / "labelsTr"),
    ]
    
    for img_path, lbl_path in possible_paths:
        if img_path.exists():
            images_dir = img_path
            labels_dir = lbl_path if lbl_path and lbl_path.exists() else None
            print(f"\n📁 找到图像目录: {images_dir}")
            break
    
    if images_dir is None:
        # 列出所有 nii.gz 文件
        print("\n🔍 搜索 .nii.gz 文件...")
        nii_files = list(raw_dir.rglob("*.nii.gz"))
        if nii_files:
            images_dir = nii_files[0].parent
            print(f"   找到 {len(nii_files)} 个 NIfTI 文件")
            print(f"   图像目录: {images_dir}")
        else:
            print("   ❌ 未找到 NIfTI 文件")
            return None
    
    # 列出可用数据
    print(f"\n📋 可用的图像文件:")
    image_files = sorted(images_dir.glob("*.nii.gz"))
    for f in image_files[:10]:  # 只显示前10个
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   - {f.name} ({size_mb:.1f} MB)")
    if len(image_files) > 10:
        print(f"   ... 共 {len(image_files)} 个文件")
    
    # 4. 配置演示病例
    demo_dir = PROJECT_ROOT / "data" / "processed" / "real_lung_001"
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔧 配置演示病例到: {demo_dir}")
    
    # Learn2Reg 命名规则:
    # case_XXX_insp.nii.gz = 吸气末 (Inspiration) -> 作为 Baseline
    # case_XXX_exp.nii.gz = 呼气末 (Expiration) -> 作为 Followup
    
    pairs_configured = 0
    
    # 查找 scans 目录
    scans_dir = raw_dir / "training" / "scans"
    masks_dir = raw_dir / "training" / "lungMasks"
    
    if not scans_dir.exists():
        scans_dir = images_dir
    
    print(f"\n📁 扫描目录: {scans_dir}")
    print(f"📁 掩码目录: {masks_dir}")
    
    # 配置多个病例
    for case_id in ["001", "002", "003", "004", "005"]:
        # 尝试不同的命名格式
        inspiration_file = scans_dir / f"case_{case_id}_insp.nii.gz"
        expiration_file = scans_dir / f"case_{case_id}_exp.nii.gz"
        
        # 备选命名格式
        if not inspiration_file.exists():
            inspiration_file = images_dir / f"lung_{case_id}_0000.nii.gz"
            expiration_file = images_dir / f"lung_{case_id}_0001.nii.gz"
        
        if inspiration_file.exists() and expiration_file.exists():
            case_dir = PROJECT_ROOT / "data" / "processed" / f"real_lung_{case_id}"
            case_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制为 baseline 和 followup
            baseline_path = case_dir / "baseline.nii.gz"
            followup_path = case_dir / "followup.nii.gz"
            
            if not baseline_path.exists():
                shutil.copy(inspiration_file, baseline_path)
                print(f"   ✅ 复制: {inspiration_file.name} -> {baseline_path}")
            if not followup_path.exists():
                shutil.copy(expiration_file, followup_path)
                print(f"   ✅ 复制: {expiration_file.name} -> {followup_path}")
            
            print(f"   ✅ Case {case_id}:")
            print(f"      - Baseline (吸气末): {baseline_path.name}")
            print(f"      - Followup (呼气末): {followup_path.name}")
            
            # 复制掩码 (如果存在)
            if masks_dir.exists():
                baseline_mask_file = masks_dir / f"case_{case_id}_insp.nii.gz"
                followup_mask_file = masks_dir / f"case_{case_id}_exp.nii.gz"
                
                if baseline_mask_file.exists():
                    mask_path = case_dir / "baseline_mask.nii.gz"
                    if not mask_path.exists():
                        shutil.copy(baseline_mask_file, mask_path)
                    print(f"      - Baseline Mask: {mask_path.name}")
                
                if followup_mask_file.exists():
                    mask_path = case_dir / "followup_mask.nii.gz"
                    if not mask_path.exists():
                        shutil.copy(followup_mask_file, mask_path)
                    print(f"      - Followup Mask: {mask_path.name}")
            
            pairs_configured += 1
        else:
            print(f"   ⚠️  Case {case_id}: 文件不存在")
    
    # 5. 创建元数据
    import json
    metadata = {
        "dataset": "Learn2Reg Challenge Task 02 (Lung CT)",
        "source": "Zenodo (https://zenodo.org/record/3835682)",
        "description": "同一患者的吸气末和呼气末 CT 扫描，包含显著的解剖形变",
        "pairs_configured": pairs_configured,
        "cases": []
    }
    
    for case_id in ["001", "002", "003"]:
        case_dir = PROJECT_ROOT / "data" / "processed" / f"real_lung_{case_id}"
        if case_dir.exists():
            metadata["cases"].append({
                "case_id": f"real_lung_{case_id}",
                "baseline": "baseline.nii.gz (吸气末/Inspiration)",
                "followup": "followup.nii.gz (呼气末/Expiration)",
                "expected_deformation": "显著的横膈膜移动和肺部形变",
                "path": str(case_dir)
            })
    
    metadata_path = PROJECT_ROOT / "data" / "processed" / "learn2reg_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # 6. 总结
    print("\n" + "="*60)
    print("✅ 真实时序数据准备完毕！")
    print("="*60)
    print(f"\n📊 数据统计:")
    print(f"   - 配置的病例对数: {pairs_configured}")
    print(f"   - 数据类型: 肺部 CT (吸气-呼气对)")
    print(f"   - 形变特点: 横膈膜大幅移动，肺部非刚性形变")
    print(f"\n📁 数据位置:")
    print(f"   - 原始数据: {task_dir}")
    print(f"   - 处理后数据: {PROJECT_ROOT / 'data' / 'processed'}")
    print(f"   - 元数据: {metadata_path}")
    print(f"\n💡 使用说明:")
    print("   这组数据模拟了极大的形变，用于测试 Registration Pipeline 的稳健性。")
    print("   如果配准后差异图显示除了横膈膜移动外肺部纹理基本对齐，则说明配准成功。")
    
    return PROJECT_ROOT / "data" / "processed" / "real_lung_001"


def download_nsclc_radiomics_sample():
    """
    下载 NSCLC-Radiomics 数据样本 (备选方案)
    需要 Kaggle API Key
    """
    print("\n" + "="*60)
    print("NSCLC-Radiomics 数据下载 (Kaggle)")
    print("="*60)
    
    try:
        import kaggle
        print("   Kaggle API 可用")
        
        # 检查数据集
        # kaggle datasets download -d 4quant/nsclc-radiomics
        
        output_dir = PROJECT_ROOT / "data" / "raw" / "NSCLC_Radiomics"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("   正在下载 NSCLC-Radiomics 数据集...")
        kaggle.api.dataset_download_files(
            "4quant/nsclc-radiomics",
            path=str(output_dir),
            unzip=True
        )
        
        print(f"   ✅ 下载完成: {output_dir}")
        return output_dir
        
    except ImportError:
        print("   ⚠️  Kaggle API 未安装")
        print("   安装方法: pip install kaggle")
        print("   然后配置 ~/.kaggle/kaggle.json")
        return None
    except Exception as e:
        print(f"   ❌ 下载失败: {e}")
        return None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="下载真实时序 CT 数据")
    parser.add_argument(
        "--dataset",
        choices=["learn2reg", "nsclc", "all"],
        default="learn2reg",
        help="要下载的数据集 (默认: learn2reg)"
    )
    
    args = parser.parse_args()
    
    results = {}
    
    if args.dataset in ["learn2reg", "all"]:
        results["learn2reg"] = download_learn2reg_lung()
    
    if args.dataset in ["nsclc", "all"]:
        results["nsclc"] = download_nsclc_radiomics_sample()
    
    # 返回结果
    print("\n" + "="*60)
    print("下载总结")
    print("="*60)
    for name, path in results.items():
        status = "✅" if path else "❌"
        print(f"  {status} {name}: {path or '失败'}")
    
    return results


if __name__ == "__main__":
    main()

