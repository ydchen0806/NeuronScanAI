#!/usr/bin/env python3
"""
生成用于Demo的模拟时序疾病数据
基于 Learn2Reg 真实CT数据，注入模拟的病灶变化
"""

import sys
import numpy as np
import nibabel as nib
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))


def inject_lesion(data, center, radius, intensity=50, blur=True):
    """在CT数据中注入模拟病灶"""
    shape = data.shape
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    
    dist = np.sqrt(
        (x - center[0])**2 + 
        (y - center[1])**2 + 
        (z - center[2])**2
    )
    
    # 创建病灶掩码（高斯衰减边缘）
    lesion_mask = np.exp(-0.5 * (dist / (radius * 0.6))**2)
    lesion_mask[dist > radius * 1.5] = 0
    
    # 注入病灶
    result = data.copy()
    result += lesion_mask * intensity
    
    return result, lesion_mask > 0.1


def generate_longitudinal_case(
    baseline_path, followup_path, output_dir, case_name,
    patient_info=None
):
    """
    基于真实CT数据生成模拟的纵向时序病例
    
    模拟场景：肺部结节随访（6个月后结节略有增大）
    """
    print(f"\n🔄 生成 Demo 病例: {case_name}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载真实数据
    baseline_img = nib.load(baseline_path)
    baseline_data = baseline_img.get_fdata().astype(np.float32)
    
    followup_img = nib.load(followup_path)
    followup_data = followup_img.get_fdata().astype(np.float32)
    
    shape = baseline_data.shape
    print(f"  数据尺寸: {shape}")
    
    # 在右肺区域添加模拟结节（基线：小结节）
    lung_center = np.array(shape) // 2
    
    # 结节位置（右肺上叶）
    nodule_center = [
        lung_center[0] + int(shape[0] * 0.15),
        lung_center[1] - int(shape[1] * 0.1),
        lung_center[2] + int(shape[2] * 0.15)
    ]
    
    # 基线：8mm 结节
    baseline_with_nodule, mask_baseline = inject_lesion(
        baseline_data, nodule_center, radius=5, intensity=80
    )
    
    # 随访：12mm 结节（增大50%）
    followup_with_nodule, mask_followup = inject_lesion(
        followup_data, nodule_center, radius=7, intensity=95
    )
    
    # 添加第二个小结节（新发）
    nodule2_center = [
        lung_center[0] - int(shape[0] * 0.1),
        lung_center[1] + int(shape[1] * 0.05),
        lung_center[2] + int(shape[2] * 0.1)
    ]
    
    # 仅在随访中出现（新发结节）
    followup_with_nodule, _ = inject_lesion(
        followup_with_nodule, nodule2_center, radius=3, intensity=60
    )
    
    # 保存
    baseline_out = nib.Nifti1Image(baseline_with_nodule, baseline_img.affine, baseline_img.header)
    followup_out = nib.Nifti1Image(followup_with_nodule, followup_img.affine, followup_img.header)
    
    nib.save(baseline_out, output_dir / "baseline.nii.gz")
    nib.save(followup_out, output_dir / "followup.nii.gz")
    
    # 保存元数据
    if patient_info is None:
        patient_info = {}
    
    import json
    metadata = {
        "patient_id": patient_info.get("patient_id", case_name),
        "patient_name": patient_info.get("name", "Demo Patient"),
        "age": patient_info.get("age", 58),
        "gender": patient_info.get("gender", "M"),
        "diagnosis": patient_info.get("diagnosis", "肺部结节随访"),
        "baseline_date": patient_info.get("baseline_date", "2025-06-15"),
        "followup_date": patient_info.get("followup_date", "2026-01-20"),
        "interval_days": 219,
        "clinical_history": patient_info.get("clinical_history", 
            "体检发现右肺上叶结节，无咳嗽、胸痛等症状。既往高血压病史10年。"),
        "nodule_info": {
            "location": "右肺上叶前段",
            "baseline_diameter_mm": 8.0,
            "followup_diameter_mm": 12.0,
            "change_percent": 50.0,
            "new_nodule": "左肺下叶新发小结节 (约6mm)"
        },
        "data_shape": list(shape),
        "generated_at": datetime.now().isoformat()
    }
    
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ 基线: baseline.nii.gz")
    print(f"  ✅ 随访: followup.nii.gz")
    print(f"  ✅ 元数据: metadata.json")
    print(f"  📋 模拟场景: {metadata['diagnosis']}")
    
    return metadata


def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    demo_dir = data_dir / "processed" / "demo_cases"
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🎬 生成 Demo 展示数据")
    print("=" * 60)
    
    # 使用 Learn2Reg 数据作为基础
    source_dir = data_dir / "processed"
    
    # 病例配置
    demo_cases = [
        {
            "source": "real_lung_001",
            "case_name": "demo_lung_nodule_followup",
            "patient_info": {
                "patient_id": "DEMO-001",
                "name": "张某某",
                "age": 58,
                "gender": "男",
                "diagnosis": "右肺上叶结节 - 6个月随访",
                "baseline_date": "2025-06-15",
                "followup_date": "2026-01-20",
                "clinical_history": "体检发现右肺上叶磨玻璃结节(GGN)，直径约8mm。患者男性，58岁，吸烟史30年。\n既往高血压病史10年，规律服药。无咳嗽、咳痰、胸痛等症状。\n肿瘤标志物: CEA 3.2 ng/mL (正常), NSE 12.5 ng/mL (正常)。\n6个月后复查CT评估结节变化。"
            }
        },
        {
            "source": "real_lung_003",
            "case_name": "demo_lung_treatment_response",
            "patient_info": {
                "patient_id": "DEMO-002",
                "name": "李某某",
                "age": 65,
                "gender": "女",
                "diagnosis": "肺腺癌术后化疗疗效评估",
                "baseline_date": "2025-09-01",
                "followup_date": "2026-01-15",
                "clinical_history": "右肺上叶腺癌 (T2N1M0, IIB期)，行右肺上叶切除术+纵隔淋巴结清扫。\n术后辅助化疗4周期 (培美曲塞+卡铂)。\n现化疗结束后3个月，复查CT评估疗效。\nECOG评分1分，一般状况良好。"
            }
        },
        {
            "source": "real_lung_005",
            "case_name": "demo_lung_screening",
            "patient_info": {
                "patient_id": "DEMO-003",
                "name": "王某某",
                "age": 52,
                "gender": "男",
                "diagnosis": "肺癌高危人群年度筛查",
                "baseline_date": "2025-01-10",
                "followup_date": "2026-01-08",
                "clinical_history": "肺癌高危人群筛查入组。男性，52岁，吸烟指数600 (20支/天×30年)。\n父亲肺癌病史。上年度低剂量CT未见明显异常。\n本次年度随访复查。"
            }
        }
    ]
    
    results = []
    for case_config in demo_cases:
        source = source_dir / case_config["source"]
        baseline = source / "baseline.nii.gz"
        followup = source / "followup.nii.gz"
        
        if baseline.exists() and followup.exists():
            output = demo_dir / case_config["case_name"]
            metadata = generate_longitudinal_case(
                str(baseline), str(followup), str(output),
                case_config["case_name"], case_config["patient_info"]
            )
            results.append(metadata)
        else:
            print(f"⚠️ 跳过 {case_config['case_name']}: 源数据不存在")
    
    print(f"\n{'='*60}")
    print(f"✅ 生成 {len(results)} 个 Demo 病例")
    print(f"📁 位置: {demo_dir}")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    main()

