#!/usr/bin/env python3
"""
快速后端测试脚本
"""
import os
import sys
from pathlib import Path

# 设置项目路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

output_file = project_root / "output" / "quick_test_results.txt"
output_file.parent.mkdir(exist_ok=True)

def log(msg):
    """写入日志"""
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)

# 清空文件
with open(output_file, "w") as f:
    f.write("=== NeuroScan 快速测试 ===\n\n")

log("1. 测试模块导入...")

try:
    from app.services.dicom import DicomLoader
    log("   DicomLoader: OK")
except Exception as e:
    log(f"   DicomLoader: FAIL - {e}")

try:
    from app.services.registration import ImageRegistrator
    log("   ImageRegistrator: OK")
except Exception as e:
    log(f"   ImageRegistrator: FAIL - {e}")

try:
    from app.services.analysis import ChangeDetector
    log("   ChangeDetector: OK")
except Exception as e:
    log(f"   ChangeDetector: FAIL - {e}")

try:
    from app.services.report import ReportGenerator
    log("   ReportGenerator: OK")
except Exception as e:
    log(f"   ReportGenerator: FAIL - {e}")

log("\n2. 测试数据加载...")

try:
    loader = DicomLoader()
    data_dir = project_root / "data" / "processed" / "real_lung_001"
    if data_dir.exists():
        baseline = data_dir / "baseline.nii.gz"
        if baseline.exists():
            import nibabel as nib
            img = nib.load(baseline)
            data = img.get_fdata()
            log(f"   加载成功: shape={data.shape}")
        else:
            log(f"   baseline.nii.gz 不存在")
    else:
        log(f"   数据目录不存在: {data_dir}")
except Exception as e:
    log(f"   加载失败: {e}")

log("\n3. 测试报告生成...")

try:
    generator = ReportGenerator(llm_backend="template")
    report = generator.generate_longitudinal_report(
        patient_id="TEST001",
        baseline_date="2025-06-15",
        followup_date="2026-01-28",
        baseline_findings=[{"organ": "肺", "max_diameter_mm": 15.0, "volume_cc": 1.0, "mean_hu": -25, "shape": "圆形", "density_type": "实性"}],
        followup_findings=[{"organ": "肺", "max_diameter_mm": 12.0, "volume_cc": 0.8, "mean_hu": -20, "shape": "圆形", "density_type": "实性"}],
        registration_results={"rigid": "completed"},
        change_results={"changed_voxels": 1000, "change_percent": 0.05},
        modality="CT"
    )
    log(f"   报告生成成功: {len(report)} 字符")
    
    # 保存报告
    report_path = project_root / "output" / "test_longitudinal_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    log(f"   报告保存至: {report_path}")
except Exception as e:
    import traceback
    log(f"   报告生成失败: {e}")
    log(traceback.format_exc())

log("\n4. 测试 LLM 连接...")

try:
    import ollama
    models = ollama.list()
    model_names = [m.model for m in models.models]
    log(f"   Ollama 可用，模型: {model_names}")
except Exception as e:
    log(f"   Ollama 不可用: {e}")

log("\n=== 测试完成 ===")
