#!/bin/bash
# ==========================================
# NeuroScan AI - 推送到 GitHub 脚本
# ==========================================

set -e

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo "=================================================="
echo "🚀 NeuroScan AI - 推送到 GitHub"
echo "=================================================="

# 1. 清理临时文件
echo ""
echo "🧹 清理临时文件..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
rm -rf logs/*.log 2>/dev/null || true
rm -rf output/* 2>/dev/null || true
rm -rf data/cache/* 2>/dev/null || true
rm -rf data/raw/Patient_* 2>/dev/null || true
rm -rf data/processed/Patient_* 2>/dev/null || true
echo "   ✅ 清理完成"

# 2. 初始化 Git（如果需要）
echo ""
echo "📦 初始化 Git 仓库..."
if [ ! -d ".git" ]; then
    git init
    echo "   ✅ Git 仓库已初始化"
else
    echo "   ℹ️  Git 仓库已存在"
fi

# 3. 配置 Git 用户
echo ""
echo "👤 配置 Git 用户..."
git config user.email "yindachen@mail.ustc.edu.cn"
git config user.name "ydchen0806"
echo "   ✅ 用户配置完成"

# 4. 添加远程仓库
echo ""
echo "🔗 配置远程仓库..."
# 使用环境变量 GITHUB_TOKEN，或手动输入
if [ -z "$GITHUB_TOKEN" ]; then
    echo "   ⚠️  请设置环境变量 GITHUB_TOKEN 或手动配置远程仓库"
    REMOTE_URL="https://github.com/ydchen0806/NeuronScanAI.git"
else
    REMOTE_URL="https://ydchen0806:${GITHUB_TOKEN}@github.com/ydchen0806/NeuronScanAI.git"
fi

if git remote | grep -q "origin"; then
    git remote set-url origin "$REMOTE_URL"
    echo "   ✅ 远程仓库 URL 已更新"
else
    git remote add origin "$REMOTE_URL"
    echo "   ✅ 远程仓库已添加"
fi

# 5. 添加文件
echo ""
echo "📁 添加文件到暂存区..."
git add .
echo "   ✅ 文件已添加"

# 6. 查看状态
echo ""
echo "📋 Git 状态:"
git status --short

# 7. 提交
echo ""
echo "💾 提交更改..."
git commit -m "Initial commit: NeuroScan AI - Medical Imaging Analysis Platform

Features:
- DICOM/NIfTI/NRRD multi-format support
- MONAI-based organ segmentation  
- Image registration (rigid + deformable)
- Longitudinal change detection
- LLM-powered report generation (Ollama)
- Streamlit web interface
- FastAPI backend"

echo "   ✅ 提交完成"

# 8. 推送
echo ""
echo "🚀 推送到 GitHub..."
git branch -M main
git push -u origin main --force

echo ""
echo "=================================================="
echo "✅ 推送完成!"
echo "=================================================="
echo ""
echo "🔗 仓库地址: https://github.com/ydchen0806/NeuronScanAI"
echo ""
