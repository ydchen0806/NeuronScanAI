#!/usr/bin/env python3
"""
项目清理脚本

清理不需要的临时文件和缓存
"""

import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def clean_python_cache():
    """清理 Python 缓存"""
    print("🧹 清理 Python 缓存...")
    
    count = 0
    for pycache in PROJECT_ROOT.rglob("__pycache__"):
        if pycache.is_dir():
            shutil.rmtree(pycache)
            count += 1
    
    for pyc in PROJECT_ROOT.rglob("*.pyc"):
        pyc.unlink()
        count += 1
    
    print(f"   ✅ 删除 {count} 个缓存文件/目录")


def clean_temp_uploads():
    """清理临时上传文件"""
    print("🧹 清理临时上传...")
    
    cache_dir = PROJECT_ROOT / "data" / "cache"
    if cache_dir.exists():
        for f in cache_dir.iterdir():
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                shutil.rmtree(f)
        print(f"   ✅ 清理 cache 目录")
    
    # 清理临时 Patient 目录
    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    
    for d in list(raw_dir.glob("Patient_*")):
        shutil.rmtree(d)
        print(f"   删除: {d.name}")
    
    for d in list(processed_dir.glob("Patient_*")):
        shutil.rmtree(d)
        print(f"   删除: {d.name}")


def clean_logs():
    """清理日志文件"""
    print("🧹 清理日志...")
    
    logs_dir = PROJECT_ROOT / "logs"
    if logs_dir.exists():
        for f in logs_dir.glob("*.log"):
            f.unlink()
            print(f"   删除: {f.name}")


def clean_output():
    """清理输出目录"""
    print("🧹 清理输出目录...")
    
    output_dir = PROJECT_ROOT / "output"
    if output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        print("   ✅ 清理 output 目录")


def show_disk_usage():
    """显示磁盘使用"""
    print("\n📊 磁盘使用统计:")
    
    def get_size(path):
        total = 0
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total
    
    dirs = [
        ("data/raw", PROJECT_ROOT / "data" / "raw"),
        ("data/processed", PROJECT_ROOT / "data" / "processed"),
        ("models", PROJECT_ROOT / "models"),
        ("logs", PROJECT_ROOT / "logs"),
        ("output", PROJECT_ROOT / "output"),
    ]
    
    for name, path in dirs:
        if path.exists():
            size_mb = get_size(path) / (1024 * 1024)
            print(f"   {name}: {size_mb:.1f} MB")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="项目清理脚本")
    parser.add_argument("--all", action="store_true", help="清理所有")
    parser.add_argument("--cache", action="store_true", help="清理 Python 缓存")
    parser.add_argument("--temp", action="store_true", help="清理临时文件")
    parser.add_argument("--logs", action="store_true", help="清理日志")
    parser.add_argument("--output", action="store_true", help="清理输出")
    parser.add_argument("--stats", action="store_true", help="显示磁盘统计")
    
    args = parser.parse_args()
    
    if args.all or args.cache:
        clean_python_cache()
    
    if args.all or args.temp:
        clean_temp_uploads()
    
    if args.all or args.logs:
        clean_logs()
    
    if args.all or args.output:
        clean_output()
    
    if args.stats or args.all:
        show_disk_usage()
    
    if not any([args.all, args.cache, args.temp, args.logs, args.output, args.stats]):
        parser.print_help()


if __name__ == "__main__":
    main()
