#!/bin/bash
# ============================================================
# 下载离线安装包 (在有网络的环境运行)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGES_DIR="$SCRIPT_DIR/packages"

echo "============================================================"
echo "📦 下载 NeuroScan AI 离线安装包"
echo "============================================================"

mkdir -p "$PACKAGES_DIR"

echo -e "\n[1/2] 下载 PyTorch (CUDA 12.1)..."
pip download torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 \
    -d "$PACKAGES_DIR" \
    --platform linux_x86_64 \
    --python-version 311 \
    --only-binary=:all:

echo -e "\n[2/2] 下载其他依赖..."
pip download -r "$SCRIPT_DIR/requirements.txt" -d "$PACKAGES_DIR"

echo -e "\n============================================================"
echo "✅ 下载完成！"
echo "============================================================"
echo ""
echo "离线包位置: $PACKAGES_DIR"
echo "离线包大小: $(du -sh $PACKAGES_DIR | cut -f1)"
echo ""
echo "离线安装命令:"
echo "  pip install --no-index --find-links=$PACKAGES_DIR -r requirements.txt"

