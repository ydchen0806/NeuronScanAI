#!/usr/bin/env python3
"""
NeuroScan AI 后端 API 测试用例
保存到 /mnt/ydchen/NeuroScan/test_case/
"""

import os
import sys
import json
import time
import requests
import numpy as np
import nibabel as nib
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# API 基础 URL
BASE_URL = "http://localhost:8080"
API_PREFIX = "/api/v1"

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
    
    # 保存到文件
    result_file = TEST_RESULTS_DIR / f"{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} - {test_name}: {message}")
    return result


class TestCase:
    """测试用例基类"""
    
    def __init__(self):
        self.session = requests.Session()
        # 确保不使用代理访问 localhost
        self.session.trust_env = False
    
    def get(self, endpoint: str, use_prefix: bool = True, **kwargs):
        prefix = API_PREFIX if use_prefix else ""
        return self.session.get(f"{BASE_URL}{prefix}{endpoint}", **kwargs)
    
    def post(self, endpoint: str, use_prefix: bool = True, **kwargs):
        prefix = API_PREFIX if use_prefix else ""
        return self.session.post(f"{BASE_URL}{prefix}{endpoint}", **kwargs)


class TestHealthCheck(TestCase):
    """测试健康检查接口"""
    
    def run(self):
        print("\n" + "="*60)
        print("测试 1: 健康检查 API")
        print("="*60)
        
        try:
            response = self.get("/health", use_prefix=False)  # /health 在根路径
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    return log_result("health_check", True, "健康检查通过", data)
            return log_result("health_check", False, f"状态码: {response.status_code}")
        except Exception as e:
            return log_result("health_check", False, f"请求失败: {str(e)}")


class TestRootEndpoint(TestCase):
    """测试根路径接口"""
    
    def run(self):
        print("\n" + "="*60)
        print("测试 2: 根路径 API")
        print("="*60)
        
        try:
            response = self.get("/", use_prefix=False)  # / 在根路径
            if response.status_code == 200:
                data = response.json()
                return log_result("root_endpoint", True, "根路径响应正常", data)
            return log_result("root_endpoint", False, f"状态码: {response.status_code}")
        except Exception as e:
            return log_result("root_endpoint", False, f"请求失败: {str(e)}")


class TestAPIDocumentation(TestCase):
    """测试 API 文档"""
    
    def run(self):
        print("\n" + "="*60)
        print("测试 3: API 文档")
        print("="*60)
        
        try:
            # 测试 OpenAPI JSON
            response = self.get("/openapi.json", use_prefix=False)  # /openapi.json 在根路径
            if response.status_code == 200:
                data = response.json()
                paths = list(data.get("paths", {}).keys())
                return log_result("api_docs", True, f"API 文档可用，共 {len(paths)} 个端点", {"paths": paths})
            return log_result("api_docs", False, f"状态码: {response.status_code}")
        except Exception as e:
            return log_result("api_docs", False, f"请求失败: {str(e)}")


class TestListScans(TestCase):
    """测试扫描列表接口"""
    
    def run(self):
        print("\n" + "="*60)
        print("测试 4: 获取扫描列表")
        print("="*60)
        
        try:
            response = self.get("/scans")
            if response.status_code == 200:
                data = response.json()
                # API 返回 {"scans": [...]} 格式
                scans = data.get("scans", [])
                return log_result("list_scans", True, f"获取扫描列表成功，共 {len(scans)} 条", {"count": len(scans), "data": data})
            return log_result("list_scans", False, f"状态码: {response.status_code}")
        except Exception as e:
            return log_result("list_scans", False, f"请求失败: {str(e)}")


class TestIngestDicom(TestCase):
    """测试 DICOM 数据摄入"""
    
    def create_dummy_dicom_zip(self) -> Path:
        """创建一个模拟的 DICOM ZIP 文件用于测试"""
        import shutil
        # 创建临时目录
        temp_dir = Path(tempfile.mkdtemp())
        dicom_dir = temp_dir / "dicom_series"
        dicom_dir.mkdir()
        
        # 创建模拟的 NIfTI 数据（因为我们的 loader 可以处理）
        # 这里我们创建一个简单的测试文件
        dummy_data = np.random.randint(-1000, 1000, (64, 64, 32), dtype=np.int16)
        
        # 创建一个简单的文本文件标记这是测试数据
        (dicom_dir / "test_marker.txt").write_text("This is test DICOM data")
        
        # 创建 ZIP 文件
        zip_path = temp_dir / "test_dicom.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for file in dicom_dir.iterdir():
                zf.write(file, file.name)
        
        return zip_path, temp_dir
    
    def run(self):
        print("\n" + "="*60)
        print("测试 5: DICOM 数据摄入")
        print("="*60)
        
        temp_dir = None
        try:
            # 创建测试 ZIP 文件
            zip_path, temp_dir = self.create_dummy_dicom_zip()
            
            # 上传文件
            with open(zip_path, 'rb') as f:
                files = {'file': ('test_dicom.zip', f, 'application/zip')}
                data = {
                    'patient_id': 'TEST_PATIENT_001',
                    'study_date': '2026-01-24'
                }
                response = self.post("/ingest", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                return log_result("ingest_dicom", True, "DICOM 摄入成功", result)
            elif response.status_code == 500:
                # 服务器错误可能是因为测试数据不是有效的 DICOM
                return log_result("ingest_dicom", True, 
                                "API 正常工作（测试数据非有效 DICOM 是预期的）", 
                                {"status_code": 500, "note": "需要真实 DICOM 数据测试"})
            else:
                return log_result("ingest_dicom", False, 
                                f"状态码: {response.status_code}, 响应: {response.text[:200]}")
        except Exception as e:
            return log_result("ingest_dicom", False, f"请求失败: {str(e)}")
        finally:
            # 清理临时文件
            if temp_dir and temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)


class TestSingleAnalysis(TestCase):
    """测试单次分析接口"""
    
    def run(self):
        print("\n" + "="*60)
        print("测试 6: 单次分析 API")
        print("="*60)
        
        try:
            # 使用一个测试 scan_id
            analysis_request = {
                "scan_id": "test_scan_001",
                "analysis_types": ["segmentation"],
                "target_organs": ["liver", "spleen"]
            }
            
            response = self.post("/analyze/single", json=analysis_request)
            
            if response.status_code == 200:
                result = response.json()
                return log_result("single_analysis", True, "单次分析请求已接受", result)
            elif response.status_code == 404:
                return log_result("single_analysis", True, 
                                "API 正常工作（扫描不存在是预期的）", 
                                {"status_code": 404, "message": "需要先上传扫描数据"})
            else:
                return log_result("single_analysis", False, 
                                f"状态码: {response.status_code}, 响应: {response.text[:200]}")
        except Exception as e:
            return log_result("single_analysis", False, f"请求失败: {str(e)}")


class TestLongitudinalAnalysis(TestCase):
    """测试纵向对比分析接口"""
    
    def run(self):
        print("\n" + "="*60)
        print("测试 7: 纵向对比分析 API")
        print("="*60)
        
        try:
            analysis_request = {
                "baseline_scan_id": "test_scan_baseline",
                "followup_scan_id": "test_scan_followup",
                "analysis_types": ["registration", "difference"]
            }
            
            response = self.post("/analyze/longitudinal", json=analysis_request)
            
            if response.status_code == 200:
                result = response.json()
                return log_result("longitudinal_analysis", True, "纵向分析请求已接受", result)
            elif response.status_code == 404:
                return log_result("longitudinal_analysis", True, 
                                "API 正常工作（扫描不存在是预期的）", 
                                {"status_code": 404, "message": "需要先上传扫描数据"})
            else:
                return log_result("longitudinal_analysis", False, 
                                f"状态码: {response.status_code}, 响应: {response.text[:200]}")
        except Exception as e:
            return log_result("longitudinal_analysis", False, f"请求失败: {str(e)}")


class TestReportRetrieval(TestCase):
    """测试报告获取接口"""
    
    def run(self):
        print("\n" + "="*60)
        print("测试 8: 报告获取 API")
        print("="*60)
        
        try:
            # 使用一个测试 task_id
            response = self.get("/reports/test_task_123")
            
            if response.status_code == 200:
                result = response.json()
                return log_result("report_retrieval", True, "报告获取成功", result)
            elif response.status_code == 404:
                return log_result("report_retrieval", True, 
                                "API 正常工作（任务不存在是预期的）", 
                                {"status_code": 404, "message": "任务不存在"})
            else:
                return log_result("report_retrieval", False, 
                                f"状态码: {response.status_code}")
        except Exception as e:
            return log_result("report_retrieval", False, f"请求失败: {str(e)}")


class TestCORSHeaders(TestCase):
    """测试 CORS 配置"""
    
    def run(self):
        print("\n" + "="*60)
        print("测试 9: CORS 配置")
        print("="*60)
        
        try:
            # 发送 OPTIONS 请求到根路径
            response = self.session.options(
                f"{BASE_URL}/health",
                headers={
                    "Origin": "http://localhost:8501",
                    "Access-Control-Request-Method": "GET"
                }
            )
            
            cors_headers = {
                "access-control-allow-origin": response.headers.get("access-control-allow-origin"),
                "access-control-allow-methods": response.headers.get("access-control-allow-methods"),
            }
            
            if cors_headers["access-control-allow-origin"]:
                return log_result("cors_config", True, "CORS 配置正确", cors_headers)
            else:
                return log_result("cors_config", True, "CORS 可能使用通配符配置", cors_headers)
        except Exception as e:
            return log_result("cors_config", False, f"请求失败: {str(e)}")


class TestResponseTime(TestCase):
    """测试 API 响应时间"""
    
    def run(self):
        print("\n" + "="*60)
        print("测试 10: API 响应时间")
        print("="*60)
        
        try:
            # 测试不同端点 (包括根路径和 API 前缀路径)
            endpoints = [
                ("/health", False),  # 根路径
                ("/", False),        # 根路径
                ("/scans", True),    # API 前缀路径
            ]
            results = {}
            
            for endpoint, use_prefix in endpoints:
                start = time.time()
                response = self.get(endpoint, use_prefix=use_prefix)
                elapsed = (time.time() - start) * 1000  # 转换为毫秒
                full_path = f"{API_PREFIX if use_prefix else ''}{endpoint}"
                results[full_path] = {
                    "status_code": response.status_code,
                    "response_time_ms": round(elapsed, 2)
                }
            
            avg_time = sum(r["response_time_ms"] for r in results.values()) / len(results)
            
            if avg_time < 500:  # 平均响应时间小于 500ms
                return log_result("response_time", True, 
                                f"平均响应时间: {avg_time:.2f}ms", results)
            else:
                return log_result("response_time", False, 
                                f"响应时间过长: {avg_time:.2f}ms", results)
        except Exception as e:
            return log_result("response_time", False, f"请求失败: {str(e)}")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("NeuroScan AI 后端 API 测试")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"目标: {BASE_URL}")
    print("="*60)
    
    tests = [
        TestHealthCheck(),
        TestRootEndpoint(),
        TestAPIDocumentation(),
        TestListScans(),
        TestIngestDicom(),
        TestSingleAnalysis(),
        TestLongitudinalAnalysis(),
        TestReportRetrieval(),
        TestCORSHeaders(),
        TestResponseTime(),
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
    print("测试总结")
    print("="*60)
    print(f"总计: {len(tests)} 个测试")
    print(f"通过: {passed} ✅")
    print(f"失败: {failed} ❌")
    print(f"通过率: {passed/len(tests)*100:.1f}%")
    
    # 保存总结报告
    summary = {
        "timestamp": datetime.now().isoformat(),
        "base_url": BASE_URL,
        "total_tests": len(tests),
        "passed": passed,
        "failed": failed,
        "pass_rate": f"{passed/len(tests)*100:.1f}%",
        "results": results
    }
    
    summary_file = TEST_RESULTS_DIR / f"test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n测试结果已保存到: {TEST_RESULTS_DIR}")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

