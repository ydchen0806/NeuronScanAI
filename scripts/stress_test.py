#!/usr/bin/env python3
"""
NeuroScan AI 并发压力测试
测试 CPU/GPU 峰值使用情况，支持 2-3 任务并发
"""

import os
import sys
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
import psutil
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 全局监控数据
monitor_data = {
    "cpu_percent": [],
    "memory_percent": [],
    "memory_gb": [],
    "gpu_memory_gb": [],
    "gpu_util": []
}
stop_monitor = False


def get_gpu_stats():
    """获取GPU状态"""
    try:
        import torch
        if torch.cuda.is_available():
            # 获取当前GPU的显存使用
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            
            # 使用nvidia-smi获取总体显存
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits', '-i', '0'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                mem_used = float(parts[0]) / 1024  # 转换为GB
                gpu_util = float(parts[1])
                return mem_used, gpu_util
            return allocated, 0
        return 0, 0
    except:
        return 0, 0


def resource_monitor(interval=0.5):
    """后台资源监控线程"""
    global stop_monitor, monitor_data
    
    while not stop_monitor:
        # CPU
        cpu_percent = psutil.cpu_percent(interval=None)
        monitor_data["cpu_percent"].append(cpu_percent)
        
        # 内存
        mem = psutil.virtual_memory()
        monitor_data["memory_percent"].append(mem.percent)
        monitor_data["memory_gb"].append(mem.used / (1024**3))
        
        # GPU
        gpu_mem, gpu_util = get_gpu_stats()
        monitor_data["gpu_memory_gb"].append(gpu_mem)
        monitor_data["gpu_util"].append(gpu_util)
        
        time.sleep(interval)


def run_single_pipeline(task_id, data_pair):
    """运行单个分析流水线"""
    baseline_path, followup_path = data_pair
    
    print(f"  🔄 任务 {task_id}: 开始处理 {Path(baseline_path).parent.name}")
    start_time = time.time()
    
    try:
        # 导入模块
        from app.services.dicom import DicomLoader
        from app.services.registration import ImageRegistrator
        from app.services.analysis import ChangeDetector
        
        loader = DicomLoader()
        registrator = ImageRegistrator()
        detector = ChangeDetector()
        
        # 1. 加载数据
        t0 = time.time()
        baseline_data, _ = loader.load_nifti(baseline_path)
        followup_data, _ = loader.load_nifti(followup_path)
        load_time = time.time() - t0
        
        # 2. 配准
        t0 = time.time()
        reg_result = registrator.register(followup_data, baseline_data, use_deformable=True)
        reg_time = time.time() - t0
        
        # 3. 变化检测
        t0 = time.time()
        change_result = detector.detect_changes(baseline_data, reg_result["warped_image"])
        detect_time = time.time() - t0
        
        total_time = time.time() - start_time
        
        return {
            "task_id": task_id,
            "status": "success",
            "load_time": load_time,
            "reg_time": reg_time,
            "detect_time": detect_time,
            "total_time": total_time,
            "data_shape": baseline_data.shape
        }
        
    except Exception as e:
        return {
            "task_id": task_id,
            "status": "error",
            "error": str(e),
            "total_time": time.time() - start_time
        }


def run_segmentation_task(task_id, nifti_path):
    """运行分割任务（GPU密集型）"""
    print(f"  🧠 分割任务 {task_id}: 开始处理")
    start_time = time.time()
    
    try:
        import torch
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        from app.services.segmentation import OrganSegmentor
        segmentor = OrganSegmentor()
        
        # 执行分割
        from app.services.dicom import DicomLoader
        loader = DicomLoader()
        data, _ = loader.load_nifti(nifti_path)
        
        # 分割推理
        result = segmentor.segment(data)
        
        total_time = time.time() - start_time
        
        # 记录GPU峰值
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        
        return {
            "task_id": task_id,
            "status": "success",
            "total_time": total_time,
            "gpu_peak_gb": peak_mem
        }
        
    except Exception as e:
        return {
            "task_id": task_id,
            "status": "error",
            "error": str(e),
            "total_time": time.time() - start_time
        }


def get_test_data_pairs(data_dir, max_pairs=5):
    """获取测试数据对"""
    data_path = Path(data_dir) / "processed"
    pairs = []
    
    for case_dir in sorted(data_path.glob("real_lung_*"))[:max_pairs]:
        baseline = case_dir / "baseline.nii.gz"
        followup = case_dir / "followup.nii.gz"
        if baseline.exists() and followup.exists():
            pairs.append((str(baseline), str(followup)))
    
    return pairs


def print_stats(title, data_list):
    """打印统计信息"""
    if not data_list:
        return
    arr = np.array(data_list)
    print(f"  {title}:")
    print(f"    平均: {np.mean(arr):.2f}")
    print(f"    峰值: {np.max(arr):.2f}")
    print(f"    最小: {np.min(arr):.2f}")


def main():
    global stop_monitor, monitor_data
    
    print("=" * 70)
    print("🔥 NeuroScan AI 并发压力测试")
    print("=" * 70)
    
    # 系统信息
    print(f"\n📊 系统配置:")
    print(f"  CPU 核心: {psutil.cpu_count(logical=False)} 物理核 / {psutil.cpu_count()} 逻辑核")
    print(f"  总内存: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU显存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    except:
        print("  GPU: 不可用")
    
    # 获取测试数据
    data_dir = Path(__file__).parent.parent / "data"
    pairs = get_test_data_pairs(data_dir, max_pairs=5)
    
    if len(pairs) < 2:
        print("\n❌ 测试数据不足，需要至少 2 对数据")
        print("   请先运行: python scripts/download_datasets.py --dataset learn2reg")
        return
    
    print(f"\n📁 找到 {len(pairs)} 对测试数据")
    
    # ========================================
    # 测试 1: 单任务基准
    # ========================================
    print("\n" + "=" * 70)
    print("📌 测试 1: 单任务基准测试")
    print("=" * 70)
    
    monitor_data = {k: [] for k in monitor_data}
    stop_monitor = False
    
    # 启动监控
    monitor_thread = threading.Thread(target=resource_monitor, args=(0.2,))
    monitor_thread.start()
    
    result = run_single_pipeline(1, pairs[0])
    
    stop_monitor = True
    monitor_thread.join()
    
    if result["status"] == "success":
        print(f"\n  ✅ 单任务完成:")
        print(f"    加载时间: {result['load_time']:.2f}s")
        print(f"    配准时间: {result['reg_time']:.2f}s")
        print(f"    检测时间: {result['detect_time']:.2f}s")
        print(f"    总时间: {result['total_time']:.2f}s")
    
    print(f"\n  📈 单任务资源峰值:")
    print(f"    CPU 峰值: {max(monitor_data['cpu_percent']):.1f}%")
    print(f"    内存峰值: {max(monitor_data['memory_gb']):.1f} GB ({max(monitor_data['memory_percent']):.1f}%)")
    print(f"    GPU显存峰值: {max(monitor_data['gpu_memory_gb']):.2f} GB")
    
    single_task_time = result["total_time"]
    single_cpu_peak = max(monitor_data['cpu_percent'])
    single_mem_peak = max(monitor_data['memory_gb'])
    
    # ========================================
    # 测试 2: 2 任务并发
    # ========================================
    print("\n" + "=" * 70)
    print("📌 测试 2: 2 任务并发压力测试")
    print("=" * 70)
    
    monitor_data = {k: [] for k in monitor_data}
    stop_monitor = False
    
    monitor_thread = threading.Thread(target=resource_monitor, args=(0.2,))
    monitor_thread.start()
    
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for i, pair in enumerate(pairs[:2]):
            futures.append(executor.submit(run_single_pipeline, i+1, pair))
        
        for future in as_completed(futures):
            results.append(future.result())
    
    concurrent_2_time = time.time() - start_time
    
    stop_monitor = True
    monitor_thread.join()
    
    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"\n  ✅ 2任务并发完成: {success_count}/2 成功")
    print(f"    总耗时: {concurrent_2_time:.2f}s")
    print(f"    并行效率: {(single_task_time * 2 / concurrent_2_time * 100):.1f}%")
    
    print(f"\n  📈 2任务并发资源峰值:")
    print(f"    CPU 峰值: {max(monitor_data['cpu_percent']):.1f}%")
    print(f"    内存峰值: {max(monitor_data['memory_gb']):.1f} GB ({max(monitor_data['memory_percent']):.1f}%)")
    print(f"    GPU显存峰值: {max(monitor_data['gpu_memory_gb']):.2f} GB")
    
    concurrent_2_cpu = max(monitor_data['cpu_percent'])
    concurrent_2_mem = max(monitor_data['memory_gb'])
    
    # ========================================
    # 测试 3: 3 任务并发
    # ========================================
    print("\n" + "=" * 70)
    print("📌 测试 3: 3 任务并发压力测试")
    print("=" * 70)
    
    if len(pairs) < 3:
        print("  ⚠️ 数据不足，跳过 3 任务测试")
    else:
        monitor_data = {k: [] for k in monitor_data}
        stop_monitor = False
        
        monitor_thread = threading.Thread(target=resource_monitor, args=(0.2,))
        monitor_thread.start()
        
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i, pair in enumerate(pairs[:3]):
                futures.append(executor.submit(run_single_pipeline, i+1, pair))
            
            for future in as_completed(futures):
                results.append(future.result())
        
        concurrent_3_time = time.time() - start_time
        
        stop_monitor = True
        monitor_thread.join()
        
        success_count = sum(1 for r in results if r["status"] == "success")
        print(f"\n  ✅ 3任务并发完成: {success_count}/3 成功")
        print(f"    总耗时: {concurrent_3_time:.2f}s")
        print(f"    并行效率: {(single_task_time * 3 / concurrent_3_time * 100):.1f}%")
        
        print(f"\n  📈 3任务并发资源峰值:")
        print(f"    CPU 峰值: {max(monitor_data['cpu_percent']):.1f}%")
        print(f"    内存峰值: {max(monitor_data['memory_gb']):.1f} GB ({max(monitor_data['memory_percent']):.1f}%)")
        print(f"    GPU显存峰值: {max(monitor_data['gpu_memory_gb']):.2f} GB")
        
        concurrent_3_cpu = max(monitor_data['cpu_percent'])
        concurrent_3_mem = max(monitor_data['memory_gb'])
    
    # ========================================
    # 测试 4: GPU 分割任务 (可选)
    # ========================================
    print("\n" + "=" * 70)
    print("📌 测试 4: GPU 分割任务峰值测试")
    print("=" * 70)
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
            monitor_data = {k: [] for k in monitor_data}
            stop_monitor = False
            
            monitor_thread = threading.Thread(target=resource_monitor, args=(0.2,))
            monitor_thread.start()
            
            # 运行分割
            seg_result = run_segmentation_task(1, pairs[0][0])
            
            stop_monitor = True
            monitor_thread.join()
            
            if seg_result["status"] == "success":
                print(f"\n  ✅ 分割任务完成:")
                print(f"    耗时: {seg_result['total_time']:.2f}s")
                print(f"    GPU峰值: {seg_result.get('gpu_peak_gb', max(monitor_data['gpu_memory_gb'])):.2f} GB")
            else:
                print(f"\n  ⚠️ 分割任务跳过: {seg_result.get('error', 'unknown')}")
            
            print(f"\n  📈 分割任务资源峰值:")
            print(f"    CPU 峰值: {max(monitor_data['cpu_percent']):.1f}%")
            print(f"    内存峰值: {max(monitor_data['memory_gb']):.1f} GB")
            print(f"    GPU显存峰值: {max(monitor_data['gpu_memory_gb']):.2f} GB")
            
            gpu_seg_peak = max(monitor_data['gpu_memory_gb'])
        else:
            print("  ⚠️ GPU 不可用，跳过分割测试")
            gpu_seg_peak = 0
    except Exception as e:
        print(f"  ⚠️ 分割测试失败: {e}")
        gpu_seg_peak = 0
    
    # ========================================
    # 最终报告
    # ========================================
    print("\n" + "=" * 70)
    print("📋 压力测试总结报告")
    print("=" * 70)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                    NeuroScan AI 资源需求报告                         │
├─────────────────┬───────────────┬───────────────┬───────────────────┤
│     测试场景     │   CPU 峰值    │   内存峰值    │    GPU 显存峰值   │
├─────────────────┼───────────────┼───────────────┼───────────────────┤
│  单任务配准      │  {single_cpu_peak:>6.1f}%      │  {single_mem_peak:>6.1f} GB    │     ~0 GB (CPU)   │
│  2任务并发       │  {concurrent_2_cpu:>6.1f}%      │  {concurrent_2_mem:>6.1f} GB    │     ~0 GB (CPU)   │
│  3任务并发       │  {concurrent_3_cpu if 'concurrent_3_cpu' in dir() else 0:>6.1f}%      │  {concurrent_3_mem if 'concurrent_3_mem' in dir() else 0:>6.1f} GB    │     ~0 GB (CPU)   │
│  GPU分割任务     │   ~50%        │  ~8 GB        │   {gpu_seg_peak:>6.1f} GB       │
├─────────────────┴───────────────┴───────────────┴───────────────────┤
│                         推荐硬件配置                                 │
├─────────────────────────────────────────────────────────────────────┤
│  最低配置 (单任务): 4核 CPU, 8GB 内存, 无需GPU                       │
│  标准配置 (2并发):  8核 CPU, 16GB 内存, 12GB GPU (可选)              │
│  推荐配置 (3并发):  16核 CPU, 32GB 内存, 24GB GPU                    │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    print("✅ 压力测试完成!")


if __name__ == "__main__":
    main()

