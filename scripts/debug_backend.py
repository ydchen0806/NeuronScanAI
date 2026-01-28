#!/usr/bin/env python3
"""
NeuroScan AI 后端调试脚本

这个脚本用于独立测试后端的核心功能：
1. DICOM/NIfTI 加载
2. 图像配准（刚性 + 非刚性）
3. 变化检测
4. 报告生成（模板模式 + LLM 模式）

使用方法:
    python scripts/debug_backend.py

作者: NeuroScan AI Team
日期: 2026-01-28
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import nibabel as nib
from datetime import datetime
import json
import traceback

# 设置环境变量
os.environ['PYTHONIOENCODING'] = 'utf-8'


def print_header(title: str):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_step(step: str, status: str = "running"):
    """打印步骤"""
    symbols = {
        "running": "🔄",
        "success": "✅",
        "error": "❌",
        "info": "ℹ️",
        "warning": "⚠️"
    }
    print(f"\n{symbols.get(status, '•')} {step}")


def print_dict(d: dict, indent: int = 2):
    """格式化打印字典"""
    for k, v in d.items():
        if isinstance(v, float):
            print(f"{' ' * indent}{k}: {v:.4f}")
        else:
            print(f"{' ' * indent}{k}: {v}")


class BackendDebugger:
    """后端调试器"""
    
    def __init__(self):
        self.data_dir = project_root / "data" / "processed"
        self.output_dir = project_root / "output" / "debug_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 测试结果
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {}
        }
    
    def find_test_data(self):
        """查找可用的测试数据"""
        print_step("查找测试数据", "running")
        
        # 查找 real_lung 数据
        lung_dirs = list(self.data_dir.glob("real_lung_*"))
        
        if lung_dirs:
            print(f"  找到 {len(lung_dirs)} 个肺部 CT 数据集:")
            for d in lung_dirs:
                files = list(d.glob("*.nii.gz"))
                print(f"    - {d.name}: {len(files)} 个文件")
            return lung_dirs[0]  # 返回第一个
        
        # 查找其他数据
        all_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        if all_dirs:
            print(f"  找到 {len(all_dirs)} 个数据目录")
            return all_dirs[0]
        
        return None
    
    def test_dicom_loader(self):
        """测试 DICOM/NIfTI 加载器"""
        print_header("测试 1: DICOM/NIfTI 加载器")
        
        try:
            from app.services.dicom import DicomLoader
            
            loader = DicomLoader()
            print_step("DicomLoader 初始化成功", "success")
            
            # 查找测试文件
            test_dir = self.find_test_data()
            if test_dir is None:
                print_step("未找到测试数据，跳过加载测试", "warning")
                return None, None
            
            # 加载 NIfTI 文件
            baseline_path = test_dir / "baseline.nii.gz"
            followup_path = test_dir / "followup.nii.gz"
            
            if not baseline_path.exists():
                # 尝试其他文件名
                nii_files = list(test_dir.glob("*.nii.gz"))
                if len(nii_files) >= 2:
                    baseline_path = nii_files[0]
                    followup_path = nii_files[1]
                elif len(nii_files) == 1:
                    baseline_path = nii_files[0]
                    followup_path = nii_files[0]  # 使用同一个文件进行测试
            
            print_step(f"加载基线: {baseline_path.name}", "running")
            baseline_data, baseline_img = loader.load_nifti(baseline_path)
            print(f"    形状: {baseline_data.shape}")
            print(f"    体素大小: {baseline_img.header.get_zooms()[:3]}")
            print(f"    数据范围: [{baseline_data.min():.1f}, {baseline_data.max():.1f}]")
            
            print_step(f"加载随访: {followup_path.name}", "running")
            followup_data, followup_img = loader.load_nifti(followup_path)
            print(f"    形状: {followup_data.shape}")
            print(f"    体素大小: {followup_img.header.get_zooms()[:3]}")
            
            self.results["tests"]["dicom_loader"] = {
                "status": "success",
                "baseline_shape": list(baseline_data.shape),
                "followup_shape": list(followup_data.shape)
            }
            
            print_step("DICOM/NIfTI 加载测试通过", "success")
            return (baseline_path, baseline_data, baseline_img), (followup_path, followup_data, followup_img)
            
        except Exception as e:
            print_step(f"加载测试失败: {e}", "error")
            traceback.print_exc()
            self.results["tests"]["dicom_loader"] = {
                "status": "error",
                "error": str(e)
            }
            return None, None
    
    def test_registration(self, baseline_path: Path, followup_path: Path):
        """测试图像配准"""
        print_header("测试 2: 图像配准")
        
        try:
            from app.services.registration import ImageRegistrator
            
            print_step("初始化配准器", "running")
            registrator = ImageRegistrator()
            print_step("ImageRegistrator 初始化成功", "success")
            
            print_step("执行两级配准（刚性 + 非刚性）", "running")
            print("    这可能需要 30-60 秒...")
            
            import time
            start_time = time.time()
            
            warped_path, transforms = registrator.register_files(
                followup_path,  # fixed
                baseline_path,  # moving
                use_deformable=True
            )
            
            elapsed = time.time() - start_time
            
            print(f"    配准完成！耗时: {elapsed:.1f} 秒")
            print(f"    输出文件: {warped_path}")
            print(f"    变换类型: {list(transforms.keys())}")
            
            self.results["tests"]["registration"] = {
                "status": "success",
                "elapsed_seconds": elapsed,
                "warped_path": str(warped_path),
                "transforms": list(transforms.keys())
            }
            
            print_step("配准测试通过", "success")
            return warped_path
            
        except Exception as e:
            print_step(f"配准测试失败: {e}", "error")
            traceback.print_exc()
            self.results["tests"]["registration"] = {
                "status": "error",
                "error": str(e)
            }
            return None
    
    def test_change_detection(self, followup_data: np.ndarray, warped_path: Path):
        """测试变化检测"""
        print_header("测试 3: 变化检测")
        
        try:
            from app.services.analysis import ChangeDetector
            from app.services.dicom import DicomLoader
            
            print_step("初始化变化检测器", "running")
            detector = ChangeDetector()
            loader = DicomLoader()
            
            # 加载配准后的图像
            warped_data, warped_img = loader.load_nifti(warped_path)
            spacing = tuple(warped_img.header.get_zooms()[:3])
            
            print_step("计算差分图", "running")
            diff_map, significant = detector.compute_difference_map(
                followup_data, 
                warped_data
            )
            
            print(f"    差分图范围: [{diff_map.min():.1f}, {diff_map.max():.1f}]")
            print(f"    显著变化体素: {(significant != 0).sum():,}")
            
            print_step("量化变化", "running")
            changes = detector.quantify_changes(diff_map, significant, spacing=spacing)
            print_dict(changes)
            
            # 生成热力图
            print_step("生成热力图", "running")
            heatmap_path = self.output_dir / "diff_heatmap.png"
            detector.generate_heatmap(significant, followup_data, heatmap_path)
            print(f"    热力图保存至: {heatmap_path}")
            
            self.results["tests"]["change_detection"] = {
                "status": "success",
                "changes": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in changes.items()},
                "heatmap_path": str(heatmap_path)
            }
            
            print_step("变化检测测试通过", "success")
            return changes, significant
            
        except Exception as e:
            print_step(f"变化检测测试失败: {e}", "error")
            traceback.print_exc()
            self.results["tests"]["change_detection"] = {
                "status": "error",
                "error": str(e)
            }
            return None, None
    
    def test_report_generation(self, change_results: dict = None):
        """测试报告生成"""
        print_header("测试 4: 报告生成")
        
        try:
            from app.services.report import ReportGenerator
            
            # 测试模板模式
            print_step("测试模板模式报告生成", "running")
            generator = ReportGenerator(llm_backend="template")
            
            # 构造测试数据
            baseline_findings = [{
                "organ": "右肺上叶",
                "location": "前段",
                "max_diameter_mm": 15.5,
                "volume_cc": 1.2,
                "mean_hu": -25,
                "shape": "类圆形",
                "density_type": "磨玻璃"
            }]
            
            followup_findings = [{
                "organ": "右肺上叶",
                "location": "前段",
                "max_diameter_mm": 12.3,
                "volume_cc": 0.8,
                "mean_hu": -20,
                "shape": "类圆形",
                "density_type": "磨玻璃"
            }]
            
            registration_results = {
                "rigid": "completed",
                "deformable": "completed",
                "spacing": (1.0, 1.0, 1.0)
            }
            
            if change_results is None:
                change_results = {
                    "changed_voxels": 15000,
                    "change_percent": 0.05,
                    "max_hu_increase": 50.0,
                    "max_hu_decrease": -45.0
                }
            
            # 生成纵向报告
            report = generator.generate_longitudinal_report(
                patient_id="TEST001",
                baseline_date="2025-06-15",
                followup_date="2026-01-28",
                baseline_findings=baseline_findings,
                followup_findings=followup_findings,
                registration_results=registration_results,
                change_results=change_results,
                modality="CT"
            )
            
            # 保存报告
            report_path = self.output_dir / "test_report.md"
            generator.save_report(report, report_path, format="md")
            print(f"    模板报告保存至: {report_path}")
            
            # 生成 HTML 报告
            html_path = self.output_dir / "test_report"
            generator.save_report(report, html_path, format="html")
            print(f"    HTML 报告保存至: {html_path}.html")
            
            # 测试 LLM 模式（如果可用）
            print_step("测试 LLM 模式报告生成", "running")
            try:
                llm_generator = ReportGenerator(llm_backend="ollama")
                
                llm_report = llm_generator.generate_longitudinal_report(
                    patient_id="TEST001",
                    baseline_date="2025-06-15",
                    followup_date="2026-01-28",
                    baseline_findings=baseline_findings,
                    followup_findings=followup_findings,
                    registration_results=registration_results,
                    change_results=change_results,
                    modality="CT"
                )
                
                llm_report_path = self.output_dir / "test_report_llm.md"
                llm_generator.save_report(llm_report, llm_report_path, format="md")
                print(f"    LLM 报告保存至: {llm_report_path}")
                print_step("LLM 模式测试通过", "success")
                
                self.results["tests"]["report_generation"] = {
                    "status": "success",
                    "template_report_path": str(report_path),
                    "llm_report_path": str(llm_report_path),
                    "llm_available": True
                }
                
            except Exception as llm_error:
                print_step(f"LLM 模式不可用: {llm_error}", "warning")
                self.results["tests"]["report_generation"] = {
                    "status": "success",
                    "template_report_path": str(report_path),
                    "llm_available": False,
                    "llm_error": str(llm_error)
                }
            
            print_step("报告生成测试通过", "success")
            return report_path
            
        except Exception as e:
            print_step(f"报告生成测试失败: {e}", "error")
            traceback.print_exc()
            self.results["tests"]["report_generation"] = {
                "status": "error",
                "error": str(e)
            }
            return None
    
    def test_segmentation(self, nifti_path: Path):
        """测试器官分割（可选，耗时较长）"""
        print_header("测试 5: 器官分割 (可选)")
        
        # 询问是否运行
        print("器官分割需要 GPU 且耗时较长（约 1-2 分钟）")
        
        try:
            from app.services.segmentation import OrganSegmentor
            
            print_step("初始化分割器", "running")
            segmentor = OrganSegmentor()
            
            print_step("检查 GPU 状态", "running")
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"    GPU: {gpu_name}")
                print(f"    显存: {gpu_mem:.1f} GB")
                
                # 检查可用显存
                free_mem = (torch.cuda.get_device_properties(0).total_memory - 
                           torch.cuda.memory_allocated()) / 1e9
                print(f"    可用显存: {free_mem:.1f} GB")
                
                if free_mem < 4.0:
                    print_step("显存不足（需要至少 4GB），跳过分割测试", "warning")
                    self.results["tests"]["segmentation"] = {
                        "status": "skipped",
                        "reason": "insufficient GPU memory"
                    }
                    return None
            else:
                print_step("GPU 不可用，跳过分割测试", "warning")
                self.results["tests"]["segmentation"] = {
                    "status": "skipped",
                    "reason": "GPU not available"
                }
                return None
            
            print_step("执行器官分割", "running")
            print("    这可能需要 1-2 分钟...")
            
            import time
            start_time = time.time()
            
            seg_path, organ_paths = segmentor.segment_file(
                nifti_path,
                save_individual_organs=False
            )
            
            elapsed = time.time() - start_time
            
            print(f"    分割完成！耗时: {elapsed:.1f} 秒")
            print(f"    分割结果: {seg_path}")
            
            self.results["tests"]["segmentation"] = {
                "status": "success",
                "elapsed_seconds": elapsed,
                "seg_path": str(seg_path)
            }
            
            print_step("分割测试通过", "success")
            return seg_path
            
        except Exception as e:
            print_step(f"分割测试失败: {e}", "error")
            traceback.print_exc()
            self.results["tests"]["segmentation"] = {
                "status": "error",
                "error": str(e)
            }
            return None
    
    def run_all_tests(self, skip_segmentation: bool = True):
        """运行所有测试"""
        print_header("NeuroScan AI 后端调试")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"输出目录: {self.output_dir}")
        
        # 测试 1: 加载器
        baseline_result, followup_result = self.test_dicom_loader()
        
        if baseline_result is None or followup_result is None:
            print_step("无法继续测试，需要有效的测试数据", "error")
            self.save_results()
            return
        
        baseline_path, baseline_data, baseline_img = baseline_result
        followup_path, followup_data, followup_img = followup_result
        
        # 测试 2: 配准
        warped_path = self.test_registration(baseline_path, followup_path)
        
        # 测试 3: 变化检测
        if warped_path:
            change_results, significant = self.test_change_detection(followup_data, warped_path)
        else:
            change_results = None
        
        # 测试 4: 报告生成
        self.test_report_generation(change_results)
        
        # 测试 5: 分割（可选）
        if not skip_segmentation:
            self.test_segmentation(baseline_path)
        else:
            print_step("跳过分割测试（使用 --with-segmentation 启用）", "info")
        
        # 保存结果
        self.save_results()
        
        # 打印总结
        self.print_summary()
    
    def save_results(self):
        """保存测试结果"""
        results_path = self.output_dir / "debug_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n测试结果已保存至: {results_path}")
    
    def print_summary(self):
        """打印测试总结"""
        print_header("测试总结")
        
        total = len(self.results["tests"])
        passed = sum(1 for t in self.results["tests"].values() if t.get("status") == "success")
        failed = sum(1 for t in self.results["tests"].values() if t.get("status") == "error")
        skipped = sum(1 for t in self.results["tests"].values() if t.get("status") == "skipped")
        
        print(f"\n总计: {total} 个测试")
        print(f"  ✅ 通过: {passed}")
        print(f"  ❌ 失败: {failed}")
        print(f"  ⏭️ 跳过: {skipped}")
        
        if failed == 0:
            print("\n🎉 所有测试通过！后端功能正常。")
        else:
            print("\n⚠️ 部分测试失败，请检查错误信息。")
        
        print(f"\n输出文件位于: {self.output_dir}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuroScan AI 后端调试脚本")
    parser.add_argument(
        "--with-segmentation",
        action="store_true",
        help="包含分割测试（需要 GPU，耗时较长）"
    )
    
    args = parser.parse_args()
    
    debugger = BackendDebugger()
    debugger.run_all_tests(skip_segmentation=not args.with_segmentation)


if __name__ == "__main__":
    main()
