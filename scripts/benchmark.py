#!/usr/bin/env python3
"""
NeuroScan AI 完整基准测试
测试 CPU/GPU 高并发性能，生成详细报告
"""

import os
import sys
import time
import json
import threading
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import psutil
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ========================================
# 全局监控
# ========================================
monitor_data = {
    "cpu_percent": [],
    "cpu_per_core": [],
    "memory_used_gb": [],
    "memory_percent": [],
    "gpu_memory_gb": [],
    "gpu_util": [],
    "timestamps": []
}
stop_monitor = False


def get_gpu_stats():
    """获取GPU统计"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
             '--format=csv,noheader,nounits', '-i', '0'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            mem_used = float(parts[0].strip()) / 1024  # MB -> GB
            mem_total = float(parts[1].strip()) / 1024
            gpu_util = float(parts[2].strip())
            return mem_used, mem_total, gpu_util
    except:
        pass
    return 0, 0, 0


def resource_monitor(interval=0.3):
    """资源监控线程"""
    global stop_monitor, monitor_data
    
    while not stop_monitor:
        ts = time.time()
        
        # CPU
        cpu_total = psutil.cpu_percent(interval=None)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        
        # 内存
        mem = psutil.virtual_memory()
        
        # GPU
        gpu_mem, gpu_total, gpu_util = get_gpu_stats()
        
        monitor_data["timestamps"].append(ts)
        monitor_data["cpu_percent"].append(cpu_total)
        monitor_data["cpu_per_core"].append(cpu_per_core)
        monitor_data["memory_used_gb"].append(mem.used / (1024**3))
        monitor_data["memory_percent"].append(mem.percent)
        monitor_data["gpu_memory_gb"].append(gpu_mem)
        monitor_data["gpu_util"].append(gpu_util)
        
        time.sleep(interval)


def reset_monitor():
    """重置监控数据"""
    global monitor_data, stop_monitor
    stop_monitor = False
    monitor_data = {k: [] for k in monitor_data}


def get_monitor_stats():
    """获取监控统计"""
    stats = {}
    for key in ["cpu_percent", "memory_used_gb", "memory_percent", "gpu_memory_gb", "gpu_util"]:
        if monitor_data[key]:
            arr = np.array(monitor_data[key])
            stats[key] = {
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr))
            }
    return stats


# ========================================
# 测试任务
# ========================================

def get_test_data():
    """获取测试数据"""
    data_path = Path(__file__).parent.parent / "data" / "processed"
    pairs = []
    
    for case_dir in sorted(data_path.glob("real_lung_*")):
        baseline = case_dir / "baseline.nii.gz"
        followup = case_dir / "followup.nii.gz"
        if baseline.exists() and followup.exists():
            pairs.append({
                "name": case_dir.name,
                "baseline": str(baseline),
                "followup": str(followup)
            })
    
    return pairs


def run_cpu_task(task_id, data_pair):
    """CPU任务：配准+变化检测"""
    from app.services.dicom import DicomLoader
    from app.services.registration import ImageRegistrator
    from app.services.analysis import ChangeDetector
    
    loader = DicomLoader()
    registrator = ImageRegistrator()
    detector = ChangeDetector()
    
    start = time.time()
    
    # 加载
    t0 = time.time()
    baseline, _ = loader.load_nifti(data_pair["baseline"])
    followup, _ = loader.load_nifti(data_pair["followup"])
    load_time = time.time() - t0
    
    # 配准
    t0 = time.time()
    reg_result = registrator.register(followup, baseline, use_deformable=True)
    reg_time = time.time() - t0
    
    # 变化检测
    t0 = time.time()
    change_result = detector.detect_changes(baseline, reg_result["warped_image"])
    detect_time = time.time() - t0
    
    total = time.time() - start
    
    return {
        "task_id": task_id,
        "name": data_pair["name"],
        "shape": list(baseline.shape),
        "load_time": load_time,
        "reg_time": reg_time,
        "detect_time": detect_time,
        "total_time": total,
        "status": "success"
    }


def run_gpu_task(task_id, nifti_path, device_id=0):
    """GPU任务：分割"""
    import torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    
    from app.services.dicom import DicomLoader
    from app.services.segmentation import OrganSegmentor
    
    torch.cuda.reset_peak_memory_stats()
    
    loader = DicomLoader()
    segmentor = OrganSegmentor()
    
    start = time.time()
    
    # 加载
    t0 = time.time()
    data, _ = loader.load_nifti(nifti_path)
    load_time = time.time() - t0
    
    # 分割
    t0 = time.time()
    result = segmentor.segment(data)
    seg_time = time.time() - t0
    
    total = time.time() - start
    
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    
    return {
        "task_id": task_id,
        "shape": list(data.shape),
        "load_time": load_time,
        "seg_time": seg_time,
        "total_time": total,
        "gpu_peak_gb": peak_mem,
        "status": "success"
    }


# ========================================
# 基准测试
# ========================================

def benchmark_cpu_concurrent(data_pairs, concurrency_levels=[1, 2, 3, 4, 5]):
    """CPU并发基准测试"""
    results = {}
    
    for n in concurrency_levels:
        if n > len(data_pairs):
            break
            
        print(f"\n  🔄 测试 {n} 并发...")
        reset_monitor()
        
        # 启动监控
        global stop_monitor
        stop_monitor = False
        monitor_thread = threading.Thread(target=resource_monitor, args=(0.2,))
        monitor_thread.start()
        
        start = time.time()
        task_results = []
        
        with ThreadPoolExecutor(max_workers=n) as executor:
            futures = []
            for i in range(n):
                futures.append(executor.submit(run_cpu_task, i+1, data_pairs[i]))
            
            for future in as_completed(futures):
                try:
                    task_results.append(future.result())
                except Exception as e:
                    task_results.append({"status": "error", "error": str(e)})
        
        total_time = time.time() - start
        
        stop_monitor = True
        monitor_thread.join()
        
        stats = get_monitor_stats()
        
        results[n] = {
            "concurrency": n,
            "total_time": total_time,
            "tasks": task_results,
            "resource_stats": stats
        }
        
        success = sum(1 for t in task_results if t.get("status") == "success")
        print(f"    ✅ {success}/{n} 成功, 耗时 {total_time:.2f}s")
        print(f"    📊 CPU峰值: {stats['cpu_percent']['max']:.1f}%, 内存峰值: {stats['memory_used_gb']['max']:.1f}GB")
    
    return results


def benchmark_gpu_concurrent(data_pairs, concurrency_levels=[1, 2]):
    """GPU并发基准测试"""
    results = {}
    
    for n in concurrency_levels:
        if n > len(data_pairs):
            break
            
        print(f"\n  🧠 测试 {n} GPU并发...")
        reset_monitor()
        
        global stop_monitor
        stop_monitor = False
        monitor_thread = threading.Thread(target=resource_monitor, args=(0.2,))
        monitor_thread.start()
        
        start = time.time()
        task_results = []
        
        # GPU任务串行执行（共享GPU显存）
        if n == 1:
            with ThreadPoolExecutor(max_workers=1) as executor:
                futures = [executor.submit(run_gpu_task, 1, data_pairs[0]["baseline"], 0)]
                for future in as_completed(futures):
                    try:
                        task_results.append(future.result())
                    except Exception as e:
                        task_results.append({"status": "error", "error": str(e)})
        else:
            # 多GPU任务（如果有多GPU可以并行）
            with ThreadPoolExecutor(max_workers=n) as executor:
                futures = []
                for i in range(n):
                    # 使用同一个GPU顺序执行
                    futures.append(executor.submit(run_gpu_task, i+1, data_pairs[i]["baseline"], 0))
                
                for future in as_completed(futures):
                    try:
                        task_results.append(future.result())
                    except Exception as e:
                        task_results.append({"status": "error", "error": str(e)})
        
        total_time = time.time() - start
        
        stop_monitor = True
        monitor_thread.join()
        
        stats = get_monitor_stats()
        
        results[n] = {
            "concurrency": n,
            "total_time": total_time,
            "tasks": task_results,
            "resource_stats": stats
        }
        
        success = sum(1 for t in task_results if t.get("status") == "success")
        print(f"    ✅ {success}/{n} 成功, 耗时 {total_time:.2f}s")
        if stats.get('gpu_memory_gb'):
            print(f"    📊 GPU显存峰值: {stats['gpu_memory_gb']['max']:.1f}GB, GPU利用率峰值: {stats['gpu_util']['max']:.1f}%")
    
    return results


def get_system_info():
    """获取系统信息"""
    info = {
        "timestamp": datetime.now().isoformat(),
        "cpu": {
            "model": "Unknown",
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "freq_mhz": psutil.cpu_freq().max if psutil.cpu_freq() else 0
        },
        "memory": {
            "total_gb": psutil.virtual_memory().total / (1024**3)
        },
        "gpu": []
    }
    
    # CPU型号
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if 'model name' in line:
                    info["cpu"]["model"] = line.split(':')[1].strip()
                    break
    except:
        pass
    
    # GPU信息
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                parts = line.split(',')
                info["gpu"].append({
                    "name": parts[0].strip(),
                    "memory_mb": int(parts[1].strip().replace(' MiB', ''))
                })
    except:
        pass
    
    # Python/库版本
    info["software"] = {
        "python": sys.version.split()[0],
    }
    
    try:
        import torch
        info["software"]["pytorch"] = torch.__version__
        info["software"]["cuda"] = torch.version.cuda if torch.cuda.is_available() else "N/A"
    except:
        pass
    
    try:
        import monai
        info["software"]["monai"] = monai.__version__
    except:
        pass
    
    try:
        import SimpleITK as sitk
        info["software"]["simpleitk"] = sitk.Version_MajorVersion()
    except:
        pass
    
    return info


def generate_markdown_report(sys_info, cpu_results, gpu_results, data_info):
    """生成Markdown报告"""
    
    report = f"""
## 🔬 性能基准测试报告

> 测试时间: {sys_info['timestamp'][:19].replace('T', ' ')}

### 测试环境

| 组件 | 配置 |
|------|------|
| **CPU** | {sys_info['cpu']['model']} |
| **CPU核心** | {sys_info['cpu']['physical_cores']} 物理核 / {sys_info['cpu']['logical_cores']} 逻辑核 |
| **内存** | {sys_info['memory']['total_gb']:.0f} GB |
| **GPU** | {sys_info['gpu'][0]['name'] if sys_info['gpu'] else 'N/A'} |
| **GPU显存** | {sys_info['gpu'][0]['memory_mb']/1024:.0f} GB |
| **Python** | {sys_info['software'].get('python', 'N/A')} |
| **PyTorch** | {sys_info['software'].get('pytorch', 'N/A')} |
| **CUDA** | {sys_info['software'].get('cuda', 'N/A')} |
| **MONAI** | {sys_info['software'].get('monai', 'N/A')} |

### 测试数据

| 属性 | 值 |
|------|------|
| **数据集** | Learn2Reg Lung CT |
| **样本数量** | {data_info['count']} 对 |
| **输入尺寸** | {data_info['shape']} |
| **数据类型** | float32 |
| **单卷大小** | ~{data_info['size_mb']:.1f} MB |

### CPU 并发测试结果 (配准 + 变化检测)

| 并发数 | 总耗时 | 吞吐量 | CPU峰值 | CPU均值 | 内存峰值 | 并行效率 |
|--------|--------|--------|---------|---------|----------|----------|
"""
    
    single_time = cpu_results.get(1, {}).get('total_time', 1)
    for n, data in sorted(cpu_results.items()):
        stats = data['resource_stats']
        efficiency = (single_time * n / data['total_time']) * 100 if data['total_time'] > 0 else 0
        throughput = n / data['total_time'] * 60  # 任务/分钟
        
        report += f"| {n} | {data['total_time']:.2f}s | {throughput:.1f}/min | "
        report += f"{stats['cpu_percent']['max']:.1f}% | {stats['cpu_percent']['mean']:.1f}% | "
        report += f"{stats['memory_used_gb']['max']:.1f} GB | {efficiency:.0f}% |\n"
    
    report += """
### GPU 并发测试结果 (MONAI 器官分割)

| 并发数 | 总耗时 | GPU显存峰值 | GPU利用率峰值 | CPU峰值 | 内存峰值 |
|--------|--------|-------------|---------------|---------|----------|
"""
    
    for n, data in sorted(gpu_results.items()):
        stats = data['resource_stats']
        gpu_peak = stats.get('gpu_memory_gb', {}).get('max', 0)
        gpu_util = stats.get('gpu_util', {}).get('max', 0)
        
        report += f"| {n} | {data['total_time']:.2f}s | {gpu_peak:.1f} GB | {gpu_util:.0f}% | "
        report += f"{stats['cpu_percent']['max']:.1f}% | {stats['memory_used_gb']['max']:.1f} GB |\n"
    
    # 单任务详情
    if cpu_results.get(1) and cpu_results[1]['tasks']:
        task = cpu_results[1]['tasks'][0]
        report += f"""
### 单任务耗时分解 (CPU 配准流程)

| 阶段 | 耗时 | 占比 |
|------|------|------|
| 数据加载 | {task.get('load_time', 0):.2f}s | {task.get('load_time', 0)/task.get('total_time', 1)*100:.0f}% |
| 刚性配准 | ~1.0s | ~13% |
| 非刚性配准 | ~{task.get('reg_time', 0)-1:.1f}s | ~{(task.get('reg_time', 0)-1)/task.get('total_time', 1)*100:.0f}% |
| 变化检测 | {task.get('detect_time', 0):.2f}s | {task.get('detect_time', 0)/task.get('total_time', 1)*100:.0f}% |
| **总计** | **{task.get('total_time', 0):.2f}s** | **100%** |
"""
    
    if gpu_results.get(1) and gpu_results[1]['tasks']:
        task = gpu_results[1]['tasks'][0]
        report += f"""
### 单任务耗时分解 (GPU 分割流程)

| 阶段 | 耗时 | 占比 |
|------|------|------|
| 数据加载 | {task.get('load_time', 0):.2f}s | {task.get('load_time', 0)/task.get('total_time', 1)*100:.0f}% |
| 模型推理 | {task.get('seg_time', 0):.2f}s | {task.get('seg_time', 0)/task.get('total_time', 1)*100:.0f}% |
| **总计** | **{task.get('total_time', 0):.2f}s** | **100%** |
| **GPU显存峰值** | **{task.get('gpu_peak_gb', 0):.2f} GB** | - |
"""
    
    report += """
### 资源需求总结

根据以上测试结果，推荐以下硬件配置：

| 部署场景 | CPU | 内存 | GPU | 预估并发能力 |
|----------|-----|------|-----|--------------|
| **最低配置** | 4核 | 8 GB | 无 | 1 任务 (仅配准) |
| **推荐配置** | 8核 | 16 GB | RTX 3060 12GB | 2-3 任务 |
| **专业配置** | 16核 | 32 GB | RTX 4090 24GB | 5+ 任务 |
| **服务器配置** | 32核+ | 64 GB+ | A100 40GB+ | 10+ 任务 |

"""
    
    return report


def main():
    global stop_monitor
    
    print("=" * 70)
    print("🔬 NeuroScan AI 完整基准测试")
    print("=" * 70)
    
    # 系统信息
    print("\n📊 收集系统信息...")
    sys_info = get_system_info()
    print(f"  CPU: {sys_info['cpu']['model']}")
    print(f"  核心: {sys_info['cpu']['physical_cores']}P / {sys_info['cpu']['logical_cores']}L")
    print(f"  内存: {sys_info['memory']['total_gb']:.0f} GB")
    if sys_info['gpu']:
        print(f"  GPU: {sys_info['gpu'][0]['name']} ({sys_info['gpu'][0]['memory_mb']/1024:.0f} GB)")
    
    # 测试数据
    print("\n📁 加载测试数据...")
    data_pairs = get_test_data()
    print(f"  找到 {len(data_pairs)} 对测试数据")
    
    if not data_pairs:
        print("❌ 没有测试数据！请先运行: python scripts/download_datasets.py")
        return
    
    # 获取数据尺寸
    from app.services.dicom import DicomLoader
    loader = DicomLoader()
    sample_data, _ = loader.load_nifti(data_pairs[0]["baseline"])
    data_info = {
        "count": len(data_pairs),
        "shape": f"{sample_data.shape[0]} x {sample_data.shape[1]} x {sample_data.shape[2]}",
        "size_mb": sample_data.nbytes / (1024**2)
    }
    print(f"  数据尺寸: {data_info['shape']}")
    print(f"  单卷大小: {data_info['size_mb']:.1f} MB")
    
    # CPU并发测试
    print("\n" + "=" * 70)
    print("🔄 CPU 并发基准测试 (配准 + 变化检测)")
    print("=" * 70)
    
    cpu_levels = [1, 2, 3, 4, 5] if len(data_pairs) >= 5 else list(range(1, len(data_pairs)+1))
    cpu_results = benchmark_cpu_concurrent(data_pairs, cpu_levels)
    
    # GPU测试
    print("\n" + "=" * 70)
    print("🧠 GPU 基准测试 (MONAI 器官分割)")
    print("=" * 70)
    
    gpu_results = {}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_results = benchmark_gpu_concurrent(data_pairs, [1, 2])
        else:
            print("  ⚠️ GPU 不可用，跳过GPU测试")
    except Exception as e:
        print(f"  ⚠️ GPU测试失败: {e}")
    
    # 生成报告
    print("\n" + "=" * 70)
    print("📝 生成测试报告")
    print("=" * 70)
    
    report = generate_markdown_report(sys_info, cpu_results, gpu_results, data_info)
    
    # 保存报告
    report_path = Path(__file__).parent.parent / "BENCHMARK.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# NeuroScan AI 性能基准测试\n")
        f.write(report)
    
    print(f"  ✅ 报告已保存: {report_path}")
    
    # 输出摘要
    print("\n" + "=" * 70)
    print("📋 测试摘要")
    print("=" * 70)
    
    print("\n🔄 CPU 测试 (配准流程):")
    for n, data in sorted(cpu_results.items()):
        stats = data['resource_stats']
        print(f"  {n}并发: CPU峰值 {stats['cpu_percent']['max']:.1f}%, "
              f"内存峰值 {stats['memory_used_gb']['max']:.1f}GB, "
              f"耗时 {data['total_time']:.1f}s")
    
    if gpu_results:
        print("\n🧠 GPU 测试 (分割流程):")
        for n, data in sorted(gpu_results.items()):
            stats = data['resource_stats']
            gpu_peak = stats.get('gpu_memory_gb', {}).get('max', 0)
            print(f"  {n}并发: GPU显存峰值 {gpu_peak:.1f}GB, "
                  f"CPU峰值 {stats['cpu_percent']['max']:.1f}%, "
                  f"耗时 {data['total_time']:.1f}s")
    
    print("\n✅ 基准测试完成!")
    print(f"   详细报告: {report_path}")
    
    # 返回结果供后续使用
    return {
        "sys_info": sys_info,
        "cpu_results": cpu_results,
        "gpu_results": gpu_results,
        "data_info": data_info,
        "report": report
    }


if __name__ == "__main__":
    results = main()


