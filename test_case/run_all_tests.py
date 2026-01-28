#!/usr/bin/env python3
"""
NeuroScan AI 测试运行器
运行所有测试用例并生成综合报告
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

# 测试目录
TEST_DIR = Path(__file__).parent
RESULTS_DIR = TEST_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_test_module(module_name: str, description: str) -> dict:
    """运行单个测试模块"""
    print(f"\n{'='*70}")
    print(f"运行: {description}")
    print(f"模块: {module_name}")
    print('='*70)
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            [sys.executable, str(TEST_DIR / module_name)],
            capture_output=True,
            text=True,
            timeout=300,  # 5分钟超时
            cwd=str(TEST_DIR.parent),
            env={**os.environ, 'http_proxy': '', 'https_proxy': ''}  # 禁用代理
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr[:500])
        
        return {
            "module": module_name,
            "description": description,
            "success": result.returncode == 0,
            "return_code": result.returncode,
            "elapsed_seconds": elapsed,
            "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
            "stderr": result.stderr[-500:] if result.stderr else None
        }
        
    except subprocess.TimeoutExpired:
        return {
            "module": module_name,
            "description": description,
            "success": False,
            "error": "测试超时 (>5分钟)"
        }
    except Exception as e:
        return {
            "module": module_name,
            "description": description,
            "success": False,
            "error": str(e)
        }


def main():
    """主函数"""
    print("\n" + "="*70)
    print("         NeuroScan AI 综合测试套件")
    print(f"         时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # 定义测试模块
    test_modules = [
        ("test_api.py", "API 接口测试"),
        ("test_services.py", "核心服务测试"),
        ("test_e2e.py", "端到端流程测试"),
        ("test_visualization.py", "可视化测试"),
    ]
    
    results = []
    total_start = datetime.now()
    
    for module, description in test_modules:
        result = run_test_module(module, description)
        results.append(result)
    
    total_elapsed = (datetime.now() - total_start).total_seconds()
    
    # 统计结果
    passed = sum(1 for r in results if r.get("success", False))
    failed = len(results) - passed
    
    # 打印总结
    print("\n" + "="*70)
    print("                    测试总结")
    print("="*70)
    print(f"总测试模块: {len(results)}")
    print(f"通过: {passed} ✅")
    print(f"失败: {failed} ❌")
    print(f"总耗时: {total_elapsed:.1f} 秒")
    print("-"*70)
    
    for result in results:
        status = "✅" if result.get("success") else "❌"
        elapsed = result.get("elapsed_seconds", 0)
        print(f"  {status} {result['description']}: {elapsed:.1f}s")
    
    print("="*70)
    
    # 保存综合报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_modules": len(results),
        "passed": passed,
        "failed": failed,
        "pass_rate": f"{passed/len(results)*100:.1f}%",
        "total_elapsed_seconds": total_elapsed,
        "results": results
    }
    
    report_file = RESULTS_DIR / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n综合报告已保存: {report_file}")
    
    # 列出所有结果文件
    print(f"\n所有测试结果文件:")
    for f in sorted(RESULTS_DIR.glob("*.json"))[-10:]:
        print(f"  - {f.name}")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

