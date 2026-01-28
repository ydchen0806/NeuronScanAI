#!/usr/bin/env python3
"""
NeuroScan AI - 公开医学影像数据集下载工具

支持的数据集:
1. Learn2Reg Challenge (Lung CT) - 肺部吸气/呼气配对
2. RIDER Lung CT - 肺癌重复扫描
3. NSCLC Radiogenomics - 肺癌基因组学
4. LIDC-IDRI - 肺结节数据集
5. Longitudinal CT (autoPET) - 肿瘤纵向随访

使用方法:
    python scripts/download_datasets.py --list           # 列出可用数据集
    python scripts/download_datasets.py --dataset learn2reg  # 下载指定数据集
    python scripts/download_datasets.py --all            # 下载所有数据集
"""

import os
import sys
import json
import urllib.request
import urllib.parse
import zipfile
import tarfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============ 数据集注册表 ============

DATASETS = {
    "learn2reg": {
        "name": "Learn2Reg Lung CT",
        "description": "肺部吸气/呼气配对 CT，包含显著解剖形变，适合配准测试",
        "size": "~300 MB",
        "format": "NIfTI",
        "source": "Zenodo",
        "url": "https://zenodo.org/api/records/3835682/files/training.zip/content",
        "license": "CC BY-NC 4.0",
        "pairs": 20,
        "recommended": True
    },
    "rider_lung": {
        "name": "RIDER Lung CT",
        "description": "32例非小细胞肺癌患者的同日重复 CT 扫描",
        "size": "~43 GB",
        "format": "DICOM",
        "source": "TCIA",
        "url": "https://wiki.cancerimagingarchive.net/display/Public/RIDER+Lung+CT",
        "license": "CC BY 3.0",
        "pairs": 32,
        "requires_nbia": True  # 需要 NBIA Data Retriever
    },
    "nlst_sample": {
        "name": "NLST Sample (肺癌筛查试验样本)",
        "description": "国家肺癌筛查试验的示例数据",
        "size": "~500 MB",
        "format": "DICOM",
        "source": "TCIA",
        "url": "https://www.cancerimagingarchive.net/collection/nlst/",
        "license": "TCIA Data Usage Policy",
        "requires_registration": True
    },
    "covid19_ct": {
        "name": "COVID-19 CT Scans",
        "description": "COVID-19 患者胸部 CT 扫描公开数据集",
        "size": "~2 GB",
        "format": "NIfTI/PNG",
        "source": "Kaggle",
        "url": "https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset",
        "license": "CC BY-NC-SA 4.0"
    },
    "lungmask_sample": {
        "name": "LungMask Sample Data",
        "description": "用于测试 lungmask 分割的示例 CT 数据",
        "size": "~50 MB",
        "format": "NIfTI",
        "source": "GitHub",
        "url": "https://github.com/JoHof/lungmask",
        "license": "MIT"
    },
    "autopet_longitudinal": {
        "name": "autoPET Longitudinal CT",
        "description": "300例黑色素瘤患者的纵向 CT 数据（基线+随访）",
        "size": "~150 GB",
        "format": "NIfTI",
        "source": "FDAT",
        "url": "https://doi.org/10.57754/FDAT.qwsry-7t837",
        "license": "CC BY-NC 4.0",
        "pairs": 300,
        "external_download": True  # 需要手动下载
    }
}


def print_header(title: str):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def setup_proxy():
    """设置代理"""
    # 从环境变量或配置读取代理
    proxy = os.environ.get('http_proxy') or os.environ.get('https_proxy') or os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY')
    
    # 默认代理配置（如果环境变量未设置）
    if not proxy:
        # 可以在这里设置默认代理
        # proxy = "http://127.0.0.1:7890"  # 示例：本地代理
        pass
    
    if proxy:
        print(f"   🌐 使用代理: {proxy}")
        proxy_handler = urllib.request.ProxyHandler({
            'http': proxy,
            'https': proxy
        })
        opener = urllib.request.build_opener(proxy_handler)
        urllib.request.install_opener(opener)
        return True
    return False


def download_with_progress(url: str, output_path: str, description: str = "", use_proxy: bool = True) -> bool:
    """带进度条的下载"""
    print(f"\n⬇️  下载: {description}")
    print(f"   URL: {url[:80]}...")
    
    # 设置代理
    if use_proxy:
        setup_proxy()
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r   进度: {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
            sys.stdout.flush()
    
    try:
        # 设置超时和重试
        socket_timeout = 60  # 60秒超时
        import socket
        socket.setdefaulttimeout(socket_timeout)
        
        urllib.request.urlretrieve(url, output_path, progress_hook)
        print("\n   ✅ 下载完成")
        return True
    except Exception as e:
        print(f"\n   ❌ 下载失败: {e}")
        print("   💡 提示: 可以设置代理环境变量 http_proxy 或 https_proxy")
        return False


def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """解压压缩包"""
    print(f"\n📂 解压: {archive_path.name}")
    
    try:
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"   ⚠️ 不支持的压缩格式: {archive_path.suffix}")
            return False
        
        print(f"   ✅ 解压完成: {extract_to}")
        return True
    except Exception as e:
        print(f"   ❌ 解压失败: {e}")
        return False


def list_datasets():
    """列出可用数据集"""
    print_header("可用数据集")
    
    for key, info in DATASETS.items():
        recommended = " ⭐ 推荐" if info.get("recommended") else ""
        external = " (需手动下载)" if info.get("external_download") else ""
        nbia = " (需 NBIA)" if info.get("requires_nbia") else ""
        
        print(f"\n📦 {key}{recommended}{external}{nbia}")
        print(f"   名称: {info['name']}")
        print(f"   描述: {info['description']}")
        print(f"   大小: {info['size']}")
        print(f"   格式: {info['format']}")
        print(f"   来源: {info['source']}")
        print(f"   许可: {info['license']}")
        if "pairs" in info:
            print(f"   配对数: {info['pairs']}")


def download_learn2reg():
    """下载 Learn2Reg 肺部 CT 数据"""
    print_header("下载 Learn2Reg Lung CT")
    
    info = DATASETS["learn2reg"]
    raw_dir = PROJECT_ROOT / "data" / "raw"
    zip_path = raw_dir / "Learn2Reg_training.zip"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载
    if not zip_path.exists():
        if not download_with_progress(info["url"], str(zip_path), info["name"]):
            return None
    else:
        print(f"   压缩包已存在: {zip_path}")
    
    # 解压
    task_dir = raw_dir / "training"
    if not task_dir.exists():
        if not extract_archive(zip_path, raw_dir):
            return None
    
    # 配置病例
    scans_dir = task_dir / "scans"
    masks_dir = task_dir / "lungMasks"
    
    configured = 0
    for case_id in range(1, 21):  # 20 个病例
        case_str = f"{case_id:03d}"
        inspiration = scans_dir / f"case_{case_str}_insp.nii.gz"
        expiration = scans_dir / f"case_{case_str}_exp.nii.gz"
        
        if inspiration.exists() and expiration.exists():
            case_dir = PROJECT_ROOT / "data" / "processed" / f"real_lung_{case_str}"
            case_dir.mkdir(parents=True, exist_ok=True)
            
            baseline = case_dir / "baseline.nii.gz"
            followup = case_dir / "followup.nii.gz"
            
            if not baseline.exists():
                shutil.copy(inspiration, baseline)
            if not followup.exists():
                shutil.copy(expiration, followup)
            
            # 复制掩码
            baseline_mask_src = masks_dir / f"case_{case_str}_insp.nii.gz"
            followup_mask_src = masks_dir / f"case_{case_str}_exp.nii.gz"
            
            if baseline_mask_src.exists():
                baseline_mask = case_dir / "baseline_mask.nii.gz"
                if not baseline_mask.exists():
                    shutil.copy(baseline_mask_src, baseline_mask)
            
            if followup_mask_src.exists():
                followup_mask = case_dir / "followup_mask.nii.gz"
                if not followup_mask.exists():
                    shutil.copy(followup_mask_src, followup_mask)
            
            configured += 1
            print(f"   ✅ Case {case_str}: baseline + followup + masks")
    
    # 保存元数据
    metadata = {
        "dataset": info["name"],
        "source": info["source"],
        "license": info["license"],
        "download_date": datetime.now().isoformat(),
        "configured_pairs": configured,
        "description": "同一患者的吸气末和呼气末 CT 扫描"
    }
    
    metadata_path = PROJECT_ROOT / "data" / "processed" / "learn2reg_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Learn2Reg 数据准备完成!")
    print(f"   配置的病例数: {configured}")
    print(f"   数据位置: {PROJECT_ROOT / 'data' / 'processed'}")
    
    return PROJECT_ROOT / "data" / "processed"


def download_sample_nibabel():
    """下载 NiBabel 示例数据（用于快速测试）"""
    print_header("下载 NiBabel 示例数据")
    
    import nibabel as nib
    from nibabel import testing
    
    sample_dir = PROJECT_ROOT / "data" / "raw" / "nibabel_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取 nibabel 测试数据
    example_files = [
        "anatomical.nii",
        "example4d.nii.gz"
    ]
    
    for filename in example_files:
        src = Path(testing.data_path) / filename
        if src.exists():
            dst = sample_dir / filename
            if not dst.exists():
                shutil.copy(src, dst)
                print(f"   ✅ {filename}")
    
    print(f"\n✅ 示例数据保存至: {sample_dir}")
    return sample_dir


def generate_synthetic_longitudinal(n_cases: int = 5):
    """生成合成的纵向 CT 数据（用于演示）"""
    print_header("生成合成纵向数据")
    
    import numpy as np
    import nibabel as nib
    
    output_dir = PROJECT_ROOT / "data" / "processed" / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for case_id in range(1, n_cases + 1):
        case_dir = output_dir / f"case_{case_id:03d}"
        case_dir.mkdir(exist_ok=True)
        
        # 生成基线图像
        shape = (128, 128, 64)
        baseline = np.random.randn(*shape).astype(np.float32) * 100 - 500
        
        # 添加一些结构
        x, y, z = np.mgrid[:shape[0], :shape[1], :shape[2]]
        center = np.array(shape) // 2
        
        # 添加 "肺部" 区域
        lung_mask = ((x - center[0])**2 / 40**2 + 
                     (y - center[1])**2 / 35**2 + 
                     (z - center[2])**2 / 25**2) < 1
        baseline[lung_mask] = -800 + np.random.randn(*baseline[lung_mask].shape) * 50
        
        # 添加 "结节"
        nodule_center = center + np.array([10, 10, 5])
        nodule_mask = ((x - nodule_center[0])**2 + 
                       (y - nodule_center[1])**2 + 
                       (z - nodule_center[2])**2) < 8**2
        baseline[nodule_mask] = 50 + np.random.randn(*baseline[nodule_mask].shape) * 20
        
        # 生成随访图像（模拟变化）
        followup = baseline.copy()
        # 结节略微增大
        nodule_mask_2 = ((x - nodule_center[0])**2 + 
                         (y - nodule_center[1])**2 + 
                         (z - nodule_center[2])**2) < 10**2
        followup[nodule_mask_2] = 55 + np.random.randn(*followup[nodule_mask_2].shape) * 20
        
        # 保存
        affine = np.eye(4)
        
        baseline_img = nib.Nifti1Image(baseline, affine)
        baseline_img.header.set_zooms((1.5, 1.5, 2.0))
        nib.save(baseline_img, case_dir / "baseline.nii.gz")
        
        followup_img = nib.Nifti1Image(followup, affine)
        followup_img.header.set_zooms((1.5, 1.5, 2.0))
        nib.save(followup_img, case_dir / "followup.nii.gz")
        
        print(f"   ✅ Case {case_id:03d}: baseline + followup")
    
    print(f"\n✅ 合成数据生成完成: {output_dir}")
    return output_dir


def show_tcia_instructions():
    """显示 TCIA 数据集下载说明"""
    print_header("TCIA 数据集下载说明")
    
    print("""
The Cancer Imaging Archive (TCIA) 数据集需要使用专门的工具下载。

📋 下载步骤:

1. 安装 NBIA Data Retriever
   下载地址: https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images
   
2. 访问数据集页面获取 .tcia manifest 文件
   - RIDER Lung CT: https://www.cancerimagingarchive.net/collection/rider-lung-ct/
   - NLST: https://www.cancerimagingarchive.net/collection/nlst/
   
3. 使用 NBIA Data Retriever 打开 .tcia 文件进行下载

4. 下载完成后，将 DICOM 文件放入:
   {}/data/raw/tcia_<dataset_name>/

5. 使用 NeuroScan AI 的 DICOM 加载器处理数据
""".format(PROJECT_ROOT))


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="NeuroScan AI 数据集下载工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python download_datasets.py --list              # 列出可用数据集
  python download_datasets.py --dataset learn2reg # 下载 Learn2Reg 数据
  python download_datasets.py --synthetic 10      # 生成 10 个合成病例
  python download_datasets.py --tcia-help         # 显示 TCIA 下载说明
        """
    )
    
    parser.add_argument("--list", action="store_true", help="列出可用数据集")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), help="要下载的数据集")
    parser.add_argument("--synthetic", type=int, metavar="N", help="生成 N 个合成病例")
    parser.add_argument("--nibabel-sample", action="store_true", help="下载 NiBabel 示例数据")
    parser.add_argument("--tcia-help", action="store_true", help="显示 TCIA 下载说明")
    parser.add_argument("--all", action="store_true", help="下载所有自动下载的数据集")
    
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
        return
    
    if args.tcia_help:
        show_tcia_instructions()
        return
    
    results = {}
    
    if args.dataset == "learn2reg" or args.all:
        results["learn2reg"] = download_learn2reg()
    
    if args.nibabel_sample or args.all:
        results["nibabel_sample"] = download_sample_nibabel()
    
    if args.synthetic:
        results["synthetic"] = generate_synthetic_longitudinal(args.synthetic)
    
    if not any([args.list, args.dataset, args.synthetic, args.nibabel_sample, 
                args.tcia_help, args.all]):
        parser.print_help()
        return
    
    # 打印总结
    if results:
        print_header("下载总结")
        for name, path in results.items():
            status = "✅" if path else "❌"
            print(f"  {status} {name}: {path or '失败'}")


if __name__ == "__main__":
    main()
