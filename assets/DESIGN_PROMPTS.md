# NeuroScan AI 设计素材 Prompts

## 1. Logo 设计 Prompt

### Prompt (英文，适用于 Midjourney/DALL-E/Stable Diffusion)

```
A modern, minimalist medical AI logo for "NeuroScan AI". 

Design elements:
- Abstract brain or neural network pattern combined with CT scan cross-section
- Circular or hexagonal shape suggesting precision and technology
- Clean lines representing medical imaging scan lines
- Subtle gradient from deep blue (#0066CC) to cyan (#00A3E0)
- Optional: small pulse/heartbeat line integrated into design

Style: 
- Flat design, vector style
- Professional healthcare aesthetic
- Silicon Valley tech company feel
- Works on both light and dark backgrounds

Colors:
- Primary: Deep blue (#0066CC)
- Secondary: Cyan (#00A3E0)  
- Accent: Orange (#FF6B35) for highlights

No text in the logo. Icon only. Clean white background. 
High resolution, suitable for app icon and website header.
--ar 1:1 --v 6
```

### Prompt (中文版，可翻译使用)

```
为 "NeuroScan AI" 设计一个现代简约的医疗AI标志。

设计元素：
- 抽象的大脑或神经网络图案与CT扫描横截面结合
- 圆形或六边形轮廓，象征精准与科技
- 简洁的线条代表医学影像扫描线
- 从深蓝色(#0066CC)到青色(#00A3E0)的微妙渐变
- 可选：融入细微的脉搏/心跳线

风格：
- 扁平化设计，矢量风格
- 专业医疗保健美学
- 硅谷科技公司感觉
- 适用于浅色和深色背景

不包含文字，仅图标。干净的白色背景。
高分辨率，适合应用图标和网站头部。
```

---

## 2. 系统架构框图 Prompt

### Prompt (英文，适用于图表生成或设计软件)

```
Create a professional system architecture diagram for a medical imaging AI platform called "NeuroScan AI".

Layout: Top-to-bottom flow diagram with 4 main layers

Layer 1 - User Interface (Top):
- "Web Dashboard" (Streamlit icon)
- "REST API" (FastAPI icon)
- Connected by horizontal line

Layer 2 - AI Services (Middle):
4 boxes in a row:
- "DICOM Loader" with medical file icon
- "Organ Segmentation" with brain/organ icon
- "Image Registration" with alignment icon  
- "Report Generator" with document icon

Layer 3 - AI Models (Lower Middle):
3 boxes:
- "MONAI Segmentation Model" 
- "SimpleITK Registration"
- "Ollama LLM (Llama3)"

Layer 4 - Data Layer (Bottom):
- "Patient Data" database icon
- "Model Weights" folder icon
- "Generated Reports" document icon

Visual Style:
- Dark theme background (#0A1628)
- Cyan accent color (#00A3E0)
- Rounded rectangle boxes
- Glowing neon effect on connections
- Modern tech startup aesthetic
- Icons inside each box
- Arrows showing data flow direction

Size: 1920x1080px landscape
Format: PNG with transparency for boxes
```

### Prompt (中文版)

```
为医学影像AI平台 "NeuroScan AI" 创建专业的系统架构图。

布局：自上而下的流程图，共4个主要层次

第一层 - 用户界面（顶部）：
- "Web 界面"（Streamlit 图标）
- "REST API"（FastAPI 图标）
- 用水平线连接

第二层 - AI 服务（中部）：
4个方框横排：
- "DICOM 加载器" 配医疗文件图标
- "器官分割" 配大脑/器官图标
- "图像配准" 配对齐图标
- "报告生成" 配文档图标

第三层 - AI 模型（中下部）：
3个方框：
- "MONAI 分割模型"
- "SimpleITK 配准"
- "Ollama LLM (Llama3)"

第四层 - 数据层（底部）：
- "患者数据" 数据库图标
- "模型权重" 文件夹图标
- "生成报告" 文档图标

视觉风格：
- 深色主题背景 (#0A1628)
- 青色强调色 (#00A3E0)
- 圆角矩形方框
- 连接线发光霓虹效果
- 现代科技创业公司美学
- 每个方框内有图标
- 箭头显示数据流向

尺寸：1920x1080 横向
格式：PNG，方框透明
```

---

## 3. 产品宣传图 Prompt

### Prompt (适用于宣传材料)

```
Create a hero image for medical AI software "NeuroScan AI".

Scene composition:
- Left side: 3D rendered CT scan of human torso, semi-transparent showing organs
- Center: Floating holographic interface panels showing:
  - Brain scan with highlighted regions
  - Medical report with charts
  - Progress indicators
- Right side: Abstract neural network visualization
- Background: Dark gradient with subtle grid pattern

Visual effects:
- Blue and cyan color scheme
- Glowing edges on 3D elements
- Particle effects suggesting data flow
- Lens flare for futuristic feel

Text overlay area:
- Clean space at bottom for tagline
- "NeuroScan AI" watermark position

Style: Cinematic, futuristic medical technology
Mood: Professional, trustworthy, innovative
Resolution: 3840x2160 (4K)
--ar 16:9 --v 6 --style raw
```

---

## 4. 使用说明

### 推荐工具

| 用途 | 推荐工具 |
|------|----------|
| Logo 生成 | Midjourney, DALL-E 3, Ideogram |
| 架构图 | Figma, Excalidraw, Draw.io |
| 宣传图 | Midjourney, Stable Diffusion XL |
| 图标设计 | Figma, Adobe Illustrator |

### 配色方案

```css
/* NeuroScan AI 品牌色 */
--primary: #0066CC;      /* 深蓝 - 可信赖 */
--secondary: #00A3E0;    /* 青色 - 科技感 */
--accent: #FF6B35;       /* 橙色 - 活力 */
--dark-bg: #0A1628;      /* 深色背景 */
--card-bg: #1A2942;      /* 卡片背景 */
--text-primary: #FFFFFF; /* 主要文字 */
--text-secondary: #A0AEC0; /* 次要文字 */
```

### 导出规格

| 用途 | 尺寸 | 格式 |
|------|------|------|
| Logo (大) | 512x512 | PNG/SVG |
| Logo (中) | 256x256 | PNG |
| Logo (小) | 64x64 | PNG |
| Favicon | 32x32 | ICO/PNG |
| 架构图 | 1920x1080 | PNG |
| 宣传图 | 3840x2160 | PNG/JPG |

---

## 5. 品牌关键词

**英文关键词：**
- Medical AI, Healthcare Technology
- CT Scan Analysis, MRI Processing
- Longitudinal Analysis, Tumor Tracking
- Deep Learning, Neural Network
- HIPAA Compliant, Privacy-First
- Clinical Decision Support

**中文关键词：**
- 医学影像AI, 智能诊断
- CT分析, MRI处理
- 纵向追踪, 肿瘤监测
- 深度学习, 神经网络
- 数据安全, 本地部署
- 临床辅助决策
