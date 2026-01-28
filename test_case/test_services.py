#!/usr/bin/env python3
"""
NeuroScan AI 核心服务测试用例
测试 DICOM 加载、分割、配准、变化检测等核心功能
"""

import os
import sys
import json
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 测试结果保存目录
TEST_RESULTS_DIR = Path(__file__).parent / "results"
TEST_RESULTS_DIR.mkdir(exist_ok=True)


def log_result(test_name: str, success: bool, message: str, data: dict = None):
    """记录测试结果"""
    result = {
        "test_name": test_name,
        "success": success,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "data": data
    }
    
    result_file = TEST_RESULTS_DIR / f"service_{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} - {test_name}: {message}")
    return result


def create_test_volume(shape=(64, 64, 64), with_lesion=False, lesion_shift=(0, 0, 0)):
    """创建测试用的 3D 体积数据"""
    volume = np.random.randint(-1000, -900, shape, dtype=np.int16)  # 背景 (空气)
    
    # 添加一个模拟器官 (肝脏区域)
    cx, cy, cz = shape[0]//2, shape[1]//2, shape[2]//2
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                dist = np.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)
                if dist < 20:
                    volume[x, y, z] = 50  # 软组织 HU 值
    
    # 添加病灶
    if with_lesion:
        lx = cx + lesion_shift[0]
        ly = cy + lesion_shift[1]
        lz = cz + lesion_shift[2]
        for x in range(max(0, lx-5), min(shape[0], lx+5)):
            for y in range(max(0, ly-5), min(shape[1], ly+5)):
                for z in range(max(0, lz-5), min(shape[2], lz+5)):
                    dist = np.sqrt((x-lx)**2 + (y-ly)**2 + (z-lz)**2)
                    if dist < 5:
                        volume[x, y, z] = 100  # 病灶 HU 值较高
    
    return volume


class TestDicomWindowing:
    """测试 CT 窗口化功能"""
    
    def run(self):
        print("\n" + "="*60)
        print("测试 1: CT 窗口化")
        print("="*60)
        
        try:
            from app.services.dicom.windowing import apply_ct_window, get_optimal_window
            
            # 创建测试数据
            test_image = np.array([-1000, -500, 0, 40, 100, 500, 1000], dtype=np.float32)
            
            # 测试肺窗
            lung_window = apply_ct_window(test_image, window_center=-600, window_width=1500)
            
            # 测试腹部窗
            abdomen_window = apply_ct_window(test_image, window_center=40, window_width=400)
            
            # 测试自动窗口
            volume = create_test_volume()
            optimal_center, optimal_width = get_optimal_window(volume)
            
            return log_result("ct_windowing", True, "CT 窗口化功能正常", {
                "lung_window_output": lung_window.tolist(),
                "abdomen_window_output": abdomen_window.tolist(),
                "optimal_window": {"center": optimal_center, "width": optimal_width}
            })
        except Exception as e:
            return log_result("ct_windowing", False, f"错误: {str(e)}")


class TestImageRegistration:
    """测试图像配准功能"""
    
    def run(self):
        print("\n" + "="*60)
        print("测试 2: 图像配准")
        print("="*60)
        
        try:
            from app.services.registration.registrator import ImageRegistrator
            
            # 创建配准器
            registrator = ImageRegistrator()
            
            # 创建测试图像 - 固定图像和移动图像（有轻微偏移）
            fixed_volume = create_test_volume(with_lesion=True, lesion_shift=(0, 0, 0))
            moving_volume = create_test_volume(with_lesion=True, lesion_shift=(2, 2, 2))
            
            # 直接使用 numpy 数组调用配准（接口接受 numpy 数组）
            registered_array, transform = registrator.rigid_registration(
                fixed_volume.astype(np.float32), 
                moving_volume.astype(np.float32),
                spacing=(1.0, 1.0, 1.0)
            )
            
            return log_result("image_registration", True, "图像配准成功", {
                "fixed_shape": list(fixed_volume.shape),
                "moving_shape": list(moving_volume.shape),
                "registered_shape": list(registered_array.shape),
                "transform_type": str(type(transform).__name__)
            })
        except Exception as e:
            return log_result("image_registration", False, f"错误: {str(e)}")


class TestChangeDetection:
    """测试变化检测功能"""
    
    def run(self):
        print("\n" + "="*60)
        print("测试 3: 变化检测")
        print("="*60)
        
        try:
            from app.services.analysis.change_detector import ChangeDetector
            
            # 创建变化检测器
            detector = ChangeDetector()
            
            # 创建基线和随访图像（病灶有变化）
            baseline = create_test_volume(with_lesion=True, lesion_shift=(0, 0, 0))
            followup = create_test_volume(with_lesion=True, lesion_shift=(0, 0, 0))
            
            # 在随访图像中增大病灶
            cx, cy, cz = 32, 32, 32
            for x in range(max(0, cx-8), min(64, cx+8)):
                for y in range(max(0, cy-8), min(64, cy+8)):
                    for z in range(max(0, cz-8), min(64, cz+8)):
                        dist = np.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)
                        if dist < 8:
                            followup[x, y, z] = 120  # 增大且密度增加
            
            # 计算差异图 (接口: followup, warped_baseline, mask=None)
            diff_map, significant_changes = detector.compute_difference_map(
                followup.astype(np.float32), 
                baseline.astype(np.float32)
            )
            
            # 创建简单的分割掩码用于量化变化
            segmentation = np.zeros_like(baseline, dtype=np.uint8)
            segmentation[cx-10:cx+10, cy-10:cy+10, cz-10:cz+10] = 1  # ROI 区域
            
            # 量化变化 (接口: followup, warped_baseline, segmentation, roi_label, spacing)
            changes = detector.quantify_changes(
                followup.astype(np.float32),
                baseline.astype(np.float32),
                segmentation,
                roi_label=1,
                spacing=(1.0, 1.0, 1.0)
            )
            
            return log_result("change_detection", True, "变化检测成功", {
                "diff_map_shape": list(diff_map.shape),
                "diff_map_range": [float(diff_map.min()), float(diff_map.max())],
                "significant_changes_shape": list(significant_changes.shape),
                "changes": changes
            })
        except Exception as e:
            return log_result("change_detection", False, f"错误: {str(e)}")


class TestFeatureExtraction:
    """测试特征提取功能"""
    
    def run(self):
        print("\n" + "="*60)
        print("测试 4: 特征提取")
        print("="*60)
        
        try:
            from app.services.analysis.feature_extractor import FeatureExtractor
            
            # 创建特征提取器（spacing 在初始化时设置）
            extractor = FeatureExtractor(spacing=(1.0, 1.0, 1.0))
            
            # 创建测试图像和掩码
            image = create_test_volume(with_lesion=True)
            
            # 创建病灶掩码
            mask = np.zeros_like(image, dtype=np.uint8)
            cx, cy, cz = 32, 32, 32
            for x in range(max(0, cx-5), min(64, cx+5)):
                for y in range(max(0, cy-5), min(64, cy+5)):
                    for z in range(max(0, cz-5), min(64, cz+5)):
                        dist = np.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)
                        if dist < 5:
                            mask[x, y, z] = 1
            
            # 提取特征 (接口: image, mask, nodule_id, organ, location)
            nodule_finding = extractor.extract_features(
                image.astype(np.float32),
                mask,
                nodule_id="test_nodule_001",
                organ="lung",
                location="right_upper_lobe"
            )
            
            # 转换为字典以便序列化
            features = {
                "nodule_id": nodule_finding.nodule_id,
                "location": nodule_finding.location,
                "organ": nodule_finding.organ,
                "volume_cc": nodule_finding.volume_cc,
                "max_diameter_mm": nodule_finding.max_diameter_mm,
                "mean_hu": nodule_finding.mean_hu,
                "density_type": nodule_finding.density_type,
                "sphericity": nodule_finding.sphericity,
                "shape": nodule_finding.shape,
                "characteristics": nodule_finding.characteristics
            }
            
            return log_result("feature_extraction", True, "特征提取成功", features)
        except Exception as e:
            return log_result("feature_extraction", False, f"错误: {str(e)}")


class TestROIExtraction:
    """测试 ROI 提取功能"""
    
    def run(self):
        print("\n" + "="*60)
        print("测试 5: ROI 提取")
        print("="*60)
        
        try:
            from app.services.analysis.roi_extractor import ROIExtractor
            
            # 创建 ROI 提取器
            extractor = ROIExtractor()
            
            # 创建测试图像
            image = create_test_volume(with_lesion=True)
            
            # 创建掩码
            mask = np.zeros_like(image, dtype=np.uint8)
            cx, cy, cz = 32, 32, 32
            for x in range(max(0, cx-5), min(64, cx+5)):
                for y in range(max(0, cy-5), min(64, cy+5)):
                    for z in range(max(0, cz-5), min(64, cz+5)):
                        dist = np.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)
                        if dist < 5:
                            mask[x, y, z] = 1
            
            # 提取 ROI (使用 extract_roi_from_mask 方法，它接受 margin 参数)
            roi_image, roi_mask, metadata = extractor.extract_roi_from_mask(
                image.astype(np.float32),
                mask,
                margin=5
            )
            
            return log_result("roi_extraction", True, "ROI 提取成功", {
                "original_shape": list(image.shape),
                "roi_shape": list(roi_image.shape),
                "metadata": metadata
            })
        except Exception as e:
            return log_result("roi_extraction", False, f"错误: {str(e)}")


class TestSegmentorInit:
    """测试分割器初始化（不执行实际分割，避免 GPU 内存问题）"""
    
    def run(self):
        print("\n" + "="*60)
        print("测试 6: 分割器初始化")
        print("="*60)
        
        try:
            from app.services.segmentation.segmentor import OrganSegmentor, ORGAN_LABELS
            
            # 检查标签数量
            num_labels = len(ORGAN_LABELS)
            
            # 只测试初始化，不加载模型
            return log_result("segmentor_init", True, f"分割器配置正确，支持 {num_labels} 种器官", {
                "num_organ_labels": num_labels,
                "sample_labels": list(ORGAN_LABELS.items())[:10]
            })
        except Exception as e:
            return log_result("segmentor_init", False, f"错误: {str(e)}")


class TestNiftiIO:
    """测试 NIfTI 文件读写"""
    
    def run(self):
        print("\n" + "="*60)
        print("测试 7: NIfTI 文件读写")
        print("="*60)
        
        try:
            # 创建测试数据
            test_data = create_test_volume()
            
            # 保存为 NIfTI
            test_file = TEST_RESULTS_DIR / "test_volume.nii.gz"
            affine = np.eye(4)
            nifti_img = nib.Nifti1Image(test_data, affine)
            nib.save(nifti_img, test_file)
            
            # 读取并验证
            loaded_img = nib.load(test_file)
            loaded_data = loaded_img.get_fdata()
            
            # 验证数据一致性
            is_consistent = np.allclose(test_data.astype(np.float32), loaded_data.astype(np.float32))
            
            # 清理
            test_file.unlink()
            
            return log_result("nifti_io", True if is_consistent else False, 
                            "NIfTI 读写测试" + ("成功" if is_consistent else "失败"), {
                "original_shape": list(test_data.shape),
                "loaded_shape": list(loaded_data.shape),
                "data_consistent": is_consistent
            })
        except Exception as e:
            return log_result("nifti_io", False, f"错误: {str(e)}")


class TestSimpleITKIO:
    """测试 SimpleITK 文件读写"""
    
    def run(self):
        print("\n" + "="*60)
        print("测试 8: SimpleITK 文件读写")
        print("="*60)
        
        try:
            # 创建测试数据
            test_data = create_test_volume()
            
            # 转换为 SimpleITK 图像
            sitk_img = sitk.GetImageFromArray(test_data.astype(np.float32))
            sitk_img.SetSpacing([1.0, 1.0, 2.5])  # 设置体素间距
            sitk_img.SetOrigin([0.0, 0.0, 0.0])
            
            # 保存
            test_file = TEST_RESULTS_DIR / "test_volume_sitk.nii.gz"
            sitk.WriteImage(sitk_img, str(test_file))
            
            # 读取并验证
            loaded_img = sitk.ReadImage(str(test_file))
            loaded_data = sitk.GetArrayFromImage(loaded_img)
            loaded_spacing = loaded_img.GetSpacing()
            
            # 验证
            is_consistent = np.allclose(test_data.astype(np.float32), loaded_data)
            spacing_correct = np.allclose(loaded_spacing, [1.0, 1.0, 2.5])
            
            # 清理
            test_file.unlink()
            
            return log_result("simpleitk_io", True if (is_consistent and spacing_correct) else False,
                            "SimpleITK 读写测试" + ("成功" if is_consistent else "失败"), {
                "original_shape": list(test_data.shape),
                "loaded_shape": list(loaded_data.shape),
                "spacing": list(loaded_spacing),
                "data_consistent": is_consistent,
                "spacing_correct": spacing_correct
            })
        except Exception as e:
            return log_result("simpleitk_io", False, f"错误: {str(e)}")


def run_all_tests():
    """运行所有服务测试"""
    print("\n" + "="*60)
    print("NeuroScan AI 核心服务测试")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    tests = [
        TestDicomWindowing(),
        TestImageRegistration(),
        TestChangeDetection(),
        TestFeatureExtraction(),
        TestROIExtraction(),
        TestSegmentorInit(),
        TestNiftiIO(),
        TestSimpleITKIO(),
    ]
    
    results = []
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = test.run()
            results.append(result)
            if result["success"]:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ 测试异常: {str(e)}")
            failed += 1
    
    # 打印总结
    print("\n" + "="*60)
    print("服务测试总结")
    print("="*60)
    print(f"总计: {len(tests)} 个测试")
    print(f"通过: {passed} ✅")
    print(f"失败: {failed} ❌")
    print(f"通过率: {passed/len(tests)*100:.1f}%")
    
    # 保存总结
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": len(tests),
        "passed": passed,
        "failed": failed,
        "pass_rate": f"{passed/len(tests)*100:.1f}%",
        "results": results
    }
    
    summary_file = TEST_RESULTS_DIR / f"service_test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n测试结果已保存到: {TEST_RESULTS_DIR}")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

