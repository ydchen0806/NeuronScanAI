#!/bin/bash
# ============================================================
# NeuroScan AI 环境部署脚本
# 适用于: Ubuntu 20.04+, Python 3.11+, CUDA 11.8+
# ============================================================

set -e

echo "============================================================"
echo "🚀 NeuroScan AI 环境部署"
echo "============================================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 检查Python版本
echo -e "\n${YELLOW}[1/6] 检查 Python 版本...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ $(echo "$PYTHON_VERSION >= 3.9" | bc -l) -eq 1 ]]; then
    echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ 需要 Python 3.9+，当前: $PYTHON_VERSION${NC}"
    exit 1
fi

# 创建虚拟环境
VENV_PATH="${1:-./venv}"
echo -e "\n${YELLOW}[2/6] 创建虚拟环境: $VENV_PATH${NC}"
if [ -d "$VENV_PATH" ]; then
    echo -e "${YELLOW}虚拟环境已存在，跳过创建${NC}"
else
    python3 -m venv "$VENV_PATH"
    echo -e "${GREEN}✓ 虚拟环境创建成功${NC}"
fi

# 激活虚拟环境
echo -e "\n${YELLOW}[3/6] 激活虚拟环境...${NC}"
source "$VENV_PATH/bin/activate"
echo -e "${GREEN}✓ 已激活: $(which python)${NC}"

# 升级pip
echo -e "\n${YELLOW}[4/6] 升级 pip...${NC}"
pip install --upgrade pip -q
echo -e "${GREEN}✓ pip 升级完成${NC}"

# 安装PyTorch (CUDA版本)
echo -e "\n${YELLOW}[5/6] 安装 PyTorch (CUDA 12.1)...${NC}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
echo -e "${GREEN}✓ PyTorch 安装完成${NC}"

# 检查CUDA
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.cuda.is_available()}')"

# 安装其他依赖
echo -e "\n${YELLOW}[6/6] 安装其他依赖...${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "$SCRIPT_DIR/requirements.txt" -q
echo -e "${GREEN}✓ 所有依赖安装完成${NC}"

# 验证安装
echo -e "\n${YELLOW}验证关键依赖...${NC}"
python -c "
import torch
import monai
import streamlit
import SimpleITK
import nibabel
print(f'  torch: {torch.__version__}')
print(f'  monai: {monai.__version__}')
print(f'  streamlit: {streamlit.__version__}')
print(f'  SimpleITK: {SimpleITK.__version__}')
print(f'  nibabel: {nibabel.__version__}')
"

echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}✅ 环境部署完成！${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "使用方法:"
echo "  source $VENV_PATH/bin/activate"
echo "  streamlit run streamlit_app.py"
echo ""

