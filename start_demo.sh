#!/bin/bash
#
# NeuroScan AI - 智能医学影像纵向诊疗系统
# 启动脚本
#
# 使用方法:
#   ./start_demo.sh          # 启动 Streamlit 前端
#   ./start_demo.sh api      # 启动 FastAPI 后端
#   ./start_demo.sh debug    # 运行后端调试脚本
#   ./start_demo.sh download # 下载示例数据集
#
#
# NeuroScan AI - 完整服务启动脚本
# 启动所有服务：Ollama LLM、FastAPI 后端、Streamlit UI
#

set -e

echo "=============================================="
echo "🏥 NeuroScan AI - 智能医学影像纵向诊断系统"
echo "=============================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# 创建日志目录
mkdir -p logs

# 检查并杀死占用端口的进程
kill_port() {
    local port=$1
    local pid=$(lsof -ti:$port 2>/dev/null)
    if [ -n "$pid" ]; then
        echo "   ⚠️ Port $port in use, releasing..."
        kill -9 $pid 2>/dev/null || true
        sleep 1
        return 0
    fi
    return 0  # Always return success
}

# 检查 Python 环境
check_python() {
    echo -e "${BLUE}[1/6]${NC} 检查 Python 环境..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version)
        echo -e "   ✅ $PYTHON_VERSION"
    else
        echo "   ❌ Python 未安装"
        exit 1
    fi
}

# 检查依赖
check_dependencies() {
    echo -e "${BLUE}[2/6]${NC} 检查 Python 依赖..."
    
    # 检查关键依赖
    MISSING=""
    python3 -c "import streamlit" 2>/dev/null || MISSING="$MISSING streamlit"
    python3 -c "import fastapi" 2>/dev/null || MISSING="$MISSING fastapi"
    python3 -c "import uvicorn" 2>/dev/null || MISSING="$MISSING uvicorn"
    python3 -c "import nibabel" 2>/dev/null || MISSING="$MISSING nibabel"
    
    if [ -n "$MISSING" ]; then
        echo "   ⚠️ 缺少依赖:$MISSING"
        echo "   📦 正在安装..."
        pip install streamlit fastapi uvicorn nibabel pydantic-settings -q
    fi
    echo "   ✅ 依赖检查完成"
}

# 检查 Ollama
check_ollama() {
    echo -e "${BLUE}[3/6]${NC} 检查 Ollama LLM 服务..."
    
    # 设置模型目录为项目内的模型
    export OLLAMA_MODELS="$PROJECT_DIR/models/ollama"
    
    if command -v ollama &> /dev/null; then
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "   ✅ Ollama 服务已运行"
        else
            echo "   ⏳ 启动 Ollama 服务..."
            OLLAMA_MODELS="$OLLAMA_MODELS" nohup ollama serve > logs/ollama.log 2>&1 &
            sleep 3
            if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
                echo "   ✅ Ollama 已启动"
            else
                echo "   ⚠️ Ollama 启动失败，报告生成将使用模板模式"
            fi
        fi
        
        # 显示可用模型
        MODELS=$(ollama list 2>/dev/null | tail -n +2)
        if [ -n "$MODELS" ]; then
            echo "   📦 可用模型:"
            echo "$MODELS" | while read line; do
                echo "      - $line"
            done
        fi
    else
        echo "   ⚠️ Ollama 未安装，报告生成将使用模板模式"
    fi
}

# 启动 FastAPI 后端
start_api_server() {
    echo -e "${BLUE}[4/6]${NC} 启动 FastAPI 后端服务..."
    
    kill_port 8000
    
    cd "$PROJECT_DIR"
    nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &
    API_PID=$!
    
    # 等待启动
    sleep 3
    if curl -s http://localhost:8000/health > /dev/null 2>&1 || curl -s http://localhost:8000/docs > /dev/null 2>&1; then
        echo "   ✅ FastAPI 后端已启动 (PID: $API_PID)"
        echo "   📍 API 地址: http://localhost:8000"
        echo "   📖 API 文档: http://localhost:8000/docs"
    else
        echo "   ⚠️ FastAPI 启动中... (查看 logs/api.log)"
    fi
}

# 启动 Streamlit UI
start_streamlit() {
    echo -e "${BLUE}[5/6]${NC} Starting Streamlit UI..."
    
    # Kill existing Streamlit processes
    pkill -f "streamlit run" 2>/dev/null || true
    sleep 1
    
    STREAMLIT_PORT=8501
    kill_port 8501
    
    cd "$PROJECT_DIR"
    nohup streamlit run streamlit_app.py \
        --server.port $STREAMLIT_PORT \
        --server.address 0.0.0.0 \
        --server.headless true \
        --server.runOnSave false \
        --browser.gatherUsageStats false \
        > logs/streamlit.log 2>&1 &
    STREAMLIT_PID=$!
    
    # Wait for startup (max 10 seconds)
    echo "   ⏳ Waiting for Streamlit to start..."
    for i in {1..10}; do
        if curl -s --max-time 2 http://localhost:$STREAMLIT_PORT > /dev/null 2>&1; then
            echo "   ✅ Streamlit UI started (PID: $STREAMLIT_PID)"
            echo "   🌐 UI: http://localhost:$STREAMLIT_PORT"
            return 0
        fi
        sleep 1
    done
    echo "   ⏳ Streamlit starting... (check logs/streamlit.log)"
}

# 启动 Demo 展示页面
start_demo_server() {
    echo -e "${BLUE}[6/6]${NC} 启动 Demo 展示服务器..."
    
    kill_port 8080
    
    cd "$PROJECT_DIR/demo"
    nohup python3 -m http.server 8080 > "$PROJECT_DIR/logs/demo.log" 2>&1 &
    DEMO_PID=$!
    cd "$PROJECT_DIR"
    
    sleep 2
    echo "   ✅ Demo 服务器已启动 (PID: $DEMO_PID)"
    echo "   🎯 Demo 地址: http://localhost:8080"
}

# 显示访问信息
show_info() {
    echo ""
    echo "=============================================="
    echo -e "${GREEN}🎉 NeuroScan AI 所有服务启动成功！${NC}"
    echo "=============================================="
    echo ""
    echo "📌 访问地址："
    echo ""
    echo -e "   ${YELLOW}🖥️  主界面 (Streamlit):${NC}  http://localhost:8501"
    echo -e "   ${YELLOW}📡 API 文档 (FastAPI):${NC}  http://localhost:8000/docs"
    echo -e "   ${YELLOW}🎯 融资 Demo 页面:${NC}      http://localhost:8080"
    echo -e "   ${YELLOW}🤖 LLM 服务 (Ollama):${NC}   http://localhost:11434"
    echo ""
    echo "=============================================="
    echo ""
    echo "📁 日志文件："
    echo "   - logs/streamlit.log  (UI 日志)"
    echo "   - logs/api.log        (API 日志)"
    echo "   - logs/ollama.log     (LLM 日志)"
    echo ""
    echo "💡 使用说明："
    echo "   1. 打开浏览器访问 http://localhost:8501"
    echo "   2. 上传 DICOM 或 NIfTI 格式的 CT 扫描"
    echo "   3. 选择分析模式（单次/纵向对比）"
    echo "   4. 查看 AI 生成的诊断报告"
    echo ""
    echo "=============================================="
    echo ""
}

# 清理函数
cleanup() {
    echo ""
    echo "正在停止所有服务..."
    pkill -f "streamlit run" 2>/dev/null || true
    pkill -f "uvicorn app.main:app" 2>/dev/null || true
    pkill -f "http.server 8080" 2>/dev/null || true
    echo "✅ 所有服务已停止"
    exit 0
}

# 捕获 Ctrl+C
trap cleanup SIGINT SIGTERM

# 主流程
main() {
    check_python
    check_dependencies
    check_ollama
    start_api_server
    start_streamlit
    start_demo_server
    show_info
    
    # 保持脚本运行
    echo -e "${GREEN}服务运行中...${NC} (按 Ctrl+C 停止所有服务)"
    echo ""
    
    # 实时显示日志
    echo "📋 实时日志 (Streamlit):"
    echo "─────────────────────────────────────────────"
    tail -f logs/streamlit.log 2>/dev/null || while true; do sleep 1; done
}

# 处理命令行参数
case "${1:-}" in
    api)
        echo "仅启动 FastAPI 后端..."
        check_python
        check_dependencies
        cd "$PROJECT_DIR"
        python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
        ;;
    debug)
        echo "运行后端调试脚本..."
        check_python
        check_dependencies
        cd "$PROJECT_DIR"
        python3 scripts/debug_backend.py
        ;;
    download)
        echo "下载示例数据集..."
        check_python
        cd "$PROJECT_DIR"
        python3 scripts/download_datasets.py --dataset learn2reg
        ;;
    streamlit)
        echo "仅启动 Streamlit..."
        check_python
        check_dependencies
        cd "$PROJECT_DIR"
        streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
        ;;
    stop)
        cleanup
        ;;
    help|--help|-h)
        echo "使用方法: ./start_demo.sh [命令]"
        echo ""
        echo "命令:"
        echo "  (无)       启动所有服务（完整模式）"
        echo "  api        仅启动 FastAPI 后端"
        echo "  streamlit  仅启动 Streamlit 前端"
        echo "  debug      运行后端调试脚本"
        echo "  download   下载示例数据集"
        echo "  stop       停止所有服务"
        echo "  help       显示帮助信息"
        ;;
    *)
        # 默认：启动所有服务
        main
        ;;
esac
