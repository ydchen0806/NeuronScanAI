#!/bin/bash
# ============================================================
# NeuroScan AI 离线安装脚本
# 使用已下载的 packages/ 目录进行离线安装
# ============================================================

set -e

echo "============================================================"
echo "🚀 NeuroScan AI 离线环境安装"
echo "============================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGES_DIR="$SCRIPT_DIR/packages"
VENV_DIR="$SCRIPT_DIR/venv"

# 检查离线包是否存在
if [ ! -d "$PACKAGES_DIR" ]; then
    echo "❌ 错误: 未找到离线包目录 $PACKAGES_DIR"
    echo "请先在有网络的环境运行: ./download_packages.sh"
    exit 1
fi

echo -e "\n[1/4] 创建虚拟环境..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
echo "✓ 虚拟环境: $VENV_DIR"

echo -e "\n[2/4] 升级 pip..."
pip install --upgrade pip --no-index --find-links="$PACKAGES_DIR" -q 2>/dev/null || \
pip install --upgrade pip -q
echo "✓ pip 升级完成"

echo -e "\n[3/4] 安装 PyTorch..."
pip install torch torchvision torchaudio \
    --no-index --find-links="$PACKAGES_DIR" -q
echo "✓ PyTorch 安装完成"

echo -e "\n[4/4] 安装其他依赖..."
pip install -r "$SCRIPT_DIR/requirements.txt" \
    --no-index --find-links="$PACKAGES_DIR" -q
echo "✓ 所有依赖安装完成"

# 验证
echo -e "\n验证安装..."
python -c "
import torch
import monai
import streamlit
import SimpleITK
print(f'  ✓ torch: {torch.__version__}')
print(f'  ✓ CUDA: {torch.cuda.is_available()}')
print(f'  ✓ monai: {monai.__version__}')
print(f'  ✓ streamlit: {streamlit.__version__}')
"

echo -e "\n============================================================"
echo "✅ 离线安装完成！"
echo "============================================================"
echo ""
echo "激活环境:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "启动应用:"
echo "  cd $(dirname $SCRIPT_DIR)"
echo "  streamlit run streamlit_app.py"

