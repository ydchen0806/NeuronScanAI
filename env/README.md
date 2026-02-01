# NeuroScan AI 环境部署包

## 📦 目录结构

```
env/
├── README.md              # 本文档
├── requirements.txt       # 核心依赖列表
├── requirements_full.txt  # 完整依赖列表 (426个包)
├── setup.sh              # 在线安装脚本
├── download_packages.sh  # 下载离线包脚本
├── install_offline.sh    # 离线安装脚本
├── quick_deploy.sh       # 一键部署脚本
└── packages/             # 离线安装包 (~6.4 GB)
    ├── torch-*.whl
    ├── monai-*.whl
    └── ...
```

## 🚀 部署方式

### 方式一：在线安装 (推荐)

```bash
# 1. 克隆项目
git clone https://github.com/ydchen0806/NeuronScanAI.git
cd NeuronScanAI

# 2. 运行安装脚本
chmod +x env/setup.sh
./env/setup.sh

# 3. 激活环境并启动
source venv/bin/activate
streamlit run streamlit_app.py
```

### 方式二：离线安装

```bash
# 1. 将整个项目（含 env/packages/）拷贝到目标服务器
scp -r NeuroScan/ user@server:/path/to/

# 2. 运行离线安装
cd /path/to/NeuroScan
chmod +x env/install_offline.sh
./env/install_offline.sh

# 3. 激活环境并启动
source env/venv/bin/activate
streamlit run streamlit_app.py
```

### 方式三：一键部署 (新服务器)

```bash
# 在新服务器上执行
curl -fsSL https://raw.githubusercontent.com/ydchen0806/NeuronScanAI/main/env/quick_deploy.sh | bash
```

## 📋 依赖版本

| 包名 | 版本 | 说明 |
|------|------|------|
| Python | 3.11+ | 推荐 3.11 |
| PyTorch | 2.0+ | CUDA 12.1 |
| MONAI | 1.3+ | 医学影像 |
| Streamlit | 1.30+ | Web UI |
| SimpleITK | 2.2+ | 配准 |

## 💾 空间需求

| 组件 | 大小 |
|------|------|
| 离线包 (packages/) | ~6.4 GB |
| 安装后环境 | ~8 GB |
| 模型权重 | ~12 GB |
| **总计** | ~20 GB |

## ⚠️ 注意事项

1. **CUDA 版本**: 离线包基于 CUDA 12.1 编译，确保服务器 CUDA 兼容
2. **Python 版本**: 必须使用 Python 3.11.x
3. **系统**: 仅支持 Linux x86_64

## 🔧 常见问题

### Q: 安装失败 "No matching distribution"

A: 检查 Python 版本是否为 3.11，离线包是针对 3.11 编译的

### Q: CUDA 不可用

A: 检查 nvidia-smi 是否正常，CUDA 版本是否 >= 12.1

### Q: 启动后模型加载失败

A: 需要单独下载模型权重:
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('cyd0806/neuroscan-ai-models', local_dir='models')"
```

