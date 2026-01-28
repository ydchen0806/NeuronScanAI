#!/usr/bin/env python3
"""
中文字体配置脚本

为 matplotlib 和系统配置中文字体支持
"""

import os
import sys
import subprocess
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def install_chinese_fonts_linux():
    """在 Linux 系统上安装中文字体"""
    print("📦 安装中文字体...")
    
    # 尝试使用 apt 安装
    try:
        subprocess.run([
            "apt-get", "update"
        ], check=False, capture_output=True)
        
        subprocess.run([
            "apt-get", "install", "-y", 
            "fonts-noto-cjk",  # Google Noto CJK 字体
            "fonts-wqy-microhei",  # 文泉驿微米黑
            "fonts-wqy-zenhei"  # 文泉驿正黑
        ], check=True, capture_output=True)
        
        print("   ✅ 系统字体安装成功")
        return True
    except Exception as e:
        print(f"   ⚠️ apt 安装失败: {e}")
    
    # 尝试使用 yum
    try:
        subprocess.run([
            "yum", "install", "-y",
            "google-noto-sans-cjk-fonts",
            "wqy-microhei-fonts"
        ], check=True, capture_output=True)
        print("   ✅ 系统字体安装成功 (yum)")
        return True
    except:
        pass
    
    return False


def download_font_file():
    """下载字体文件到项目目录"""
    fonts_dir = PROJECT_ROOT / "assets" / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载开源中文字体 (思源黑体)
    font_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
    font_path = fonts_dir / "NotoSansCJKsc-Regular.otf"
    
    if font_path.exists():
        print(f"   字体已存在: {font_path}")
        return font_path
    
    print(f"   下载字体: {font_url}")
    
    try:
        import urllib.request
        
        # 设置代理
        proxy = os.environ.get('http_proxy') or os.environ.get('https_proxy')
        if proxy:
            proxy_handler = urllib.request.ProxyHandler({
                'http': proxy,
                'https': proxy
            })
            opener = urllib.request.build_opener(proxy_handler)
            urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(font_url, font_path)
        print(f"   ✅ 字体下载成功: {font_path}")
        return font_path
    except Exception as e:
        print(f"   ❌ 下载失败: {e}")
        return None


def configure_matplotlib():
    """配置 matplotlib 中文字体"""
    print("\n🔧 配置 matplotlib 中文字体...")
    
    # 方法1: 使用系统字体
    system_fonts = [
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Hiragino Sans GB",
        "DejaVu Sans"
    ]
    
    # 查找可用字体
    from matplotlib import font_manager
    available_fonts = set([f.name for f in font_manager.fontManager.ttflist])
    
    selected_font = None
    for font in system_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        print(f"   使用系统字体: {selected_font}")
        plt.rcParams['font.sans-serif'] = [selected_font] + system_fonts
        plt.rcParams['axes.unicode_minus'] = False
        return True
    
    # 方法2: 使用下载的字体
    fonts_dir = PROJECT_ROOT / "assets" / "fonts"
    font_files = list(fonts_dir.glob("*.otf")) + list(fonts_dir.glob("*.ttf"))
    
    if font_files:
        font_path = font_files[0]
        print(f"   使用项目字体: {font_path}")
        
        # 注册字体
        font_manager.fontManager.addfont(str(font_path))
        prop = font_manager.FontProperties(fname=str(font_path))
        
        plt.rcParams['font.family'] = prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False
        return True
    
    print("   ⚠️ 未找到中文字体，使用默认字体")
    return False


def create_matplotlib_config():
    """创建 matplotlib 配置文件"""
    config_dir = Path(matplotlib.get_configdir())
    config_file = config_dir / "matplotlibrc"
    
    config_content = """# NeuroScan AI matplotlib 配置
# 中文字体支持

font.family: sans-serif
font.sans-serif: Noto Sans CJK SC, WenQuanYi Micro Hei, SimHei, DejaVu Sans, sans-serif
axes.unicode_minus: False

# 图表样式
figure.facecolor: white
figure.edgecolor: white
axes.facecolor: white
axes.edgecolor: black
axes.labelcolor: black
xtick.color: black
ytick.color: black
text.color: black
"""
    
    try:
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            f.write(config_content)
        print(f"   ✅ matplotlib 配置保存至: {config_file}")
        return True
    except Exception as e:
        print(f"   ⚠️ 配置保存失败: {e}")
        return False


def test_chinese_font():
    """测试中文字体显示"""
    print("\n🧪 测试中文字体...")
    
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "中文字体测试\nNeuroScan AI\n智能医学影像系统", 
                ha='center', va='center', fontsize=16)
        ax.set_title("中文标题测试")
        ax.set_xlabel("X轴标签")
        ax.set_ylabel("Y轴标签")
        
        test_path = PROJECT_ROOT / "output" / "font_test.png"
        test_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(test_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ 测试图片保存至: {test_path}")
        return True
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 50)
    print("🔤 NeuroScan AI 中文字体配置")
    print("=" * 50)
    
    # 1. 尝试安装系统字体
    if sys.platform.startswith('linux'):
        install_chinese_fonts_linux()
    
    # 2. 下载字体文件（备选）
    download_font_file()
    
    # 3. 配置 matplotlib
    configure_matplotlib()
    
    # 4. 创建配置文件
    create_matplotlib_config()
    
    # 5. 清除字体缓存
    print("\n🔄 清除字体缓存...")
    cache_dir = Path(matplotlib.get_cachedir())
    font_cache = cache_dir / "fontlist-v330.json"
    if font_cache.exists():
        font_cache.unlink()
        print(f"   已删除: {font_cache}")
    
    # 6. 测试
    test_chinese_font()
    
    print("\n" + "=" * 50)
    print("✅ 字体配置完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
