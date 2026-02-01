#!/bin/bash
# ============================================================
# NeuroScan AI 一键部署脚本
# 在新服务器上运行此脚本即可完成部署
# ============================================================

set -e

echo "============================================================"
echo "🚀 NeuroScan AI 一键部署"
echo "============================================================"

# 项目目录
PROJECT_DIR="${1:-/root/NeuroScan}"

# 检查是否有GPU
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "✓ 检测到 GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
        return 0
    else
        echo "⚠ 未检测到 GPU，将使用 CPU 模式"
        return 1
    fi
}

# 安装系统依赖
install_system_deps() {
    echo -e "\n[1/5] 安装系统依赖..."
    apt-get update -qq
    apt-get install -y -qq python3.11 python3.11-venv python3-pip git curl
    echo "✓ 系统依赖安装完成"
}

# 克隆项目
clone_project() {
    echo -e "\n[2/5] 克隆项目..."
    if [ -d "$PROJECT_DIR" ]; then
        echo "项目目录已存在，执行 git pull..."
        cd "$PROJECT_DIR"
        git pull
    else
        git clone https://github.com/ydchen0806/NeuronScanAI.git "$PROJECT_DIR"
        cd "$PROJECT_DIR"
    fi
    echo "✓ 项目克隆完成"
}

# 安装Python环境
setup_python_env() {
    echo -e "\n[3/5] 配置 Python 环境..."
    cd "$PROJECT_DIR"
    
    # 创建虚拟环境
    python3.11 -m venv venv
    source venv/bin/activate
    
    # 升级pip
    pip install --upgrade pip -q
    
    # 安装PyTorch
    if check_gpu; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
    else
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
    fi
    
    # 安装其他依赖
    pip install -r env/requirements.txt -q
    
    echo "✓ Python 环境配置完成"
}

# 下载模型
download_models() {
    echo -e "\n[4/5] 下载模型权重..."
    source "$PROJECT_DIR/venv/bin/activate"
    
    # 从 Hugging Face 下载模型
    python -c "
from huggingface_hub import snapshot_download
print('  下载 MONAI 分割模型...')
snapshot_download(
    repo_id='cyd0806/neuroscan-ai-models',
    local_dir='$PROJECT_DIR/models',
    ignore_patterns=['*.md']
)
print('  ✓ 模型下载完成')
"
    
    echo "✓ 模型下载完成"
}

# 安装 Ollama (可选)
install_ollama() {
    echo -e "\n[5/5] 安装 Ollama LLM (可选)..."
    if command -v ollama &> /dev/null; then
        echo "Ollama 已安装"
    else
        curl -fsSL https://ollama.com/install.sh | sh
        # 下载模型
        ollama pull qwen2.5:7b
    fi
    echo "✓ Ollama 安装完成"
}

# 创建启动脚本
create_start_script() {
    cat > "$PROJECT_DIR/start.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
EOF
    chmod +x "$PROJECT_DIR/start.sh"
    echo "✓ 启动脚本创建完成: $PROJECT_DIR/start.sh"
}

# 主流程
main() {
    install_system_deps
    clone_project
    setup_python_env
    download_models
    # install_ollama  # 取消注释以安装 Ollama
    create_start_script
    
    echo ""
    echo "============================================================"
    echo "✅ 部署完成！"
    echo "============================================================"
    echo ""
    echo "启动服务:"
    echo "  cd $PROJECT_DIR"
    echo "  ./start.sh"
    echo ""
    echo "访问地址: http://<服务器IP>:8501"
}

main "$@"

