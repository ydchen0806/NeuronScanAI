#!/usr/bin/env python3
"""
NeuroScan AI 完整工作流演示脚本

这个脚本展示了从输入到输出的完整流程：
1. 加载数据
2. 图像配准
3. 变化检测
4. 特征提取
5. RECIST 评估
6. LLM 报告生成
7. 可视化输出

使用方法:
    python scripts/workflow_demo.py
"""

import os
import sys

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from datetime import datetime
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def print_header(title):
    """打印格式化标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_step(step_num, title):
    """打印步骤标题"""
    print(f"\n{'─' * 70}")
    print(f"  📌 步骤 {step_num}: {title}")
    print(f"{'─' * 70}")


def load_data(baseline_path, followup_path):
    """
    阶段 1: 加载数据
    
    输入: NIfTI 文件路径
    输出: numpy 数组和元数据
    """
    print_step(1, "加载医学影像数据")
    
    print(f"\n  📂 基线扫描: {baseline_path}")
    print(f"  📂 随访扫描: {followup_path}")
    
    # 加载 NIfTI 文件
    baseline_nii = nib.load(baseline_path)
    followup_nii = nib.load(followup_path)
    
    # 获取数据
    baseline_data = baseline_nii.get_fdata().astype(np.float32)
    followup_data = followup_nii.get_fdata().astype(np.float32)
    
    # 获取空间信息
    spacing = baseline_nii.header.get_zooms()[:3]
    affine = baseline_nii.affine
    
    print(f"\n  ✅ 数据加载完成:")
    print(f"     - 基线尺寸: {baseline_data.shape}")
    print(f"     - 随访尺寸: {followup_data.shape}")
    print(f"     - 体素间距: {spacing} mm")
    print(f"     - 基线 HU 范围: [{baseline_data.min():.0f}, {baseline_data.max():.0f}]")
    print(f"     - 随访 HU 范围: [{followup_data.min():.0f}, {followup_data.max():.0f}]")
    
    return {
        'baseline': baseline_data,
        'followup': followup_data,
        'spacing': spacing,
        'affine': affine
    }


def perform_registration(baseline, followup, spacing):
    """
    阶段 2: 图像配准
    
    将随访扫描对齐到基线扫描
    """
    print_step(2, "图像配准 (Registration)")
    
    print("\n  🔧 配准策略:")
    print("     1. 刚性配准 (Rigid): 校正体位差异")
    print("     2. 非刚性配准 (Deformable): 校正呼吸运动")
    
    try:
        from app.services.registration import ImageRegistrator
        
        registrator = ImageRegistrator()
        
        # 刚性配准
        print("\n  ⏳ 执行刚性配准...")
        rigid_result = registrator.rigid_registration(
            fixed_image=baseline,
            moving_image=followup,
            spacing=spacing
        )
        print("     ✅ 刚性配准完成")
        
        # 非刚性配准
        print("  ⏳ 执行非刚性配准...")
        deformable_result = registrator.deformable_registration(
            fixed_image=baseline,
            moving_image=rigid_result['registered_image'],
            spacing=spacing
        )
        print("     ✅ 非刚性配准完成")
        
        registered = deformable_result['registered_image']
        
    except Exception as e:
        print(f"\n  ⚠️ 配准服务不可用: {e}")
        print("     使用原始图像继续...")
        registered = followup
    
    print(f"\n  ✅ 配准完成:")
    print(f"     - 输出尺寸: {registered.shape}")
    
    return registered


def detect_changes(baseline, registered, spacing):
    """
    阶段 3: 变化检测
    
    计算两次扫描之间的差异
    """
    print_step(3, "变化检测 (Change Detection)")
    
    print("\n  🔍 分析内容:")
    print("     - 体素级差异计算")
    print("     - 变化区域识别")
    print("     - 量化指标提取")
    
    # 计算差异图
    diff_map = registered - baseline
    abs_diff = np.abs(diff_map)
    
    # 设置阈值 (30 HU 为显著变化)
    threshold = 30
    significant_mask = abs_diff > threshold
    
    # 计算统计量
    voxel_volume = np.prod(spacing)  # mm³
    changed_voxels = significant_mask.sum()
    changed_volume = changed_voxels * voxel_volume
    
    # 计算变化统计
    if changed_voxels > 0:
        mean_change = diff_map[significant_mask].mean()
        max_increase = diff_map.max()
        max_decrease = diff_map.min()
    else:
        mean_change = 0
        max_increase = 0
        max_decrease = 0
    
    print(f"\n  ✅ 变化检测完成:")
    print(f"     - 显著变化阈值: {threshold} HU")
    print(f"     - 变化体素数: {changed_voxels:,}")
    print(f"     - 变化体积: {changed_volume/1000:.2f} cm³")
    print(f"     - 平均变化: {mean_change:+.1f} HU")
    print(f"     - 最大增加: {max_increase:+.1f} HU")
    print(f"     - 最大减少: {max_decrease:+.1f} HU")
    
    return {
        'diff_map': diff_map,
        'significant_mask': significant_mask,
        'changed_volume_mm3': changed_volume,
        'mean_change': mean_change,
        'max_increase': max_increase,
        'max_decrease': max_decrease
    }


def extract_features(baseline, registered, diff_map, spacing):
    """
    阶段 4: 特征提取
    
    提取病灶的量化特征
    """
    print_step(4, "特征提取 (Feature Extraction)")
    
    print("\n  📊 提取特征:")
    print("     - 体积测量")
    print("     - 密度分析")
    print("     - 形态学特征")
    
    # 找到变化最显著的区域作为 ROI
    abs_diff = np.abs(diff_map)
    threshold = np.percentile(abs_diff, 99)  # 取变化最大的 1%
    roi_mask = abs_diff > threshold
    
    if roi_mask.sum() == 0:
        roi_mask = abs_diff > 30
    
    voxel_volume = np.prod(spacing)
    
    # 基线特征
    baseline_roi = baseline[roi_mask] if roi_mask.sum() > 0 else baseline.flatten()
    baseline_features = {
        'volume_mm3': roi_mask.sum() * voxel_volume,
        'mean_hu': float(baseline_roi.mean()),
        'std_hu': float(baseline_roi.std()),
        'min_hu': float(baseline_roi.min()),
        'max_hu': float(baseline_roi.max())
    }
    
    # 随访特征
    followup_roi = registered[roi_mask] if roi_mask.sum() > 0 else registered.flatten()
    followup_features = {
        'volume_mm3': roi_mask.sum() * voxel_volume,
        'mean_hu': float(followup_roi.mean()),
        'std_hu': float(followup_roi.std()),
        'min_hu': float(followup_roi.min()),
        'max_hu': float(followup_roi.max())
    }
    
    # 计算变化
    density_change = followup_features['mean_hu'] - baseline_features['mean_hu']
    
    print(f"\n  ✅ 特征提取完成:")
    print(f"     基线特征:")
    print(f"       - ROI 体积: {baseline_features['volume_mm3']/1000:.2f} cm³")
    print(f"       - 平均密度: {baseline_features['mean_hu']:.1f} HU")
    print(f"     随访特征:")
    print(f"       - ROI 体积: {followup_features['volume_mm3']/1000:.2f} cm³")
    print(f"       - 平均密度: {followup_features['mean_hu']:.1f} HU")
    print(f"     变化:")
    print(f"       - 密度变化: {density_change:+.1f} HU")
    
    return {
        'baseline': baseline_features,
        'followup': followup_features,
        'density_change': density_change
    }


def evaluate_recist(baseline_diameter=10.0, followup_diameter=12.5):
    """
    阶段 5: RECIST 1.1 评估
    
    根据病灶直径变化评估疗效
    """
    print_step(5, "RECIST 1.1 评估")
    
    print("\n  📋 RECIST 1.1 标准:")
    print("     - CR (完全缓解): 所有靶病灶消失")
    print("     - PR (部分缓解): 直径总和减少 ≥30%")
    print("     - SD (疾病稳定): 介于 PR 和 PD 之间")
    print("     - PD (疾病进展): 直径总和增加 ≥20%")
    
    # 计算变化百分比
    change_percent = (followup_diameter - baseline_diameter) / baseline_diameter * 100
    
    # 评估
    if followup_diameter == 0:
        recist_code = "CR"
        recist_text = "完全缓解 (Complete Response)"
        recist_color = "green"
    elif change_percent <= -30:
        recist_code = "PR"
        recist_text = "部分缓解 (Partial Response)"
        recist_color = "blue"
    elif change_percent >= 20:
        recist_code = "PD"
        recist_text = "疾病进展 (Progressive Disease)"
        recist_color = "red"
    else:
        recist_code = "SD"
        recist_text = "疾病稳定 (Stable Disease)"
        recist_color = "orange"
    
    print(f"\n  ✅ RECIST 评估完成:")
    print(f"     - 基线最长径: {baseline_diameter:.1f} mm")
    print(f"     - 随访最长径: {followup_diameter:.1f} mm")
    print(f"     - 变化百分比: {change_percent:+.1f}%")
    print(f"     - 评估结果: {recist_code} - {recist_text}")
    
    return {
        'baseline_diameter': baseline_diameter,
        'followup_diameter': followup_diameter,
        'change_percent': change_percent,
        'recist_code': recist_code,
        'recist_text': recist_text,
        'recist_color': recist_color
    }


def generate_report(data, changes, features, recist, output_dir):
    """
    阶段 6: LLM 智能报告生成
    """
    print_step(6, "LLM 智能报告生成")
    
    print("\n  🤖 报告生成配置:")
    print("     - LLM 后端: Ollama (本地)")
    print("     - 模型: llama3.1:8b / meditron:7b")
    print("     - 报告格式: ACR 标准")
    
    # 尝试使用 LLM 生成报告
    try:
        from app.services.report import ReportGenerator
        
        generator = ReportGenerator(llm_backend="ollama")
        
        # 准备报告数据
        report_data = {
            'patient_id': 'WORKFLOW_DEMO',
            'baseline_date': '2025-10-01',
            'followup_date': datetime.now().strftime('%Y-%m-%d'),
            'baseline_findings': {
                'description': '右肺上叶后段见一结节灶，边界清晰',
                'size_mm': recist['baseline_diameter'],
                'density_hu': features['baseline']['mean_hu']
            },
            'followup_findings': {
                'description': '右肺上叶后段结节',
                'size_mm': recist['followup_diameter'],
                'density_hu': features['followup']['mean_hu']
            },
            'changes': {
                'size_change_percent': recist['change_percent'],
                'density_change': features['density_change']
            },
            'recist_evaluation': recist['recist_text']
        }
        
        print("\n  ⏳ 正在调用 LLM 生成报告...")
        report_content = generator.generate_longitudinal_report(**report_data)
        
        # 保存报告
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'ai_report.html')
        generator.save_report(report_content, report_path.replace('.html', ''), format='html')
        
        print(f"     ✅ LLM 报告已生成: {report_path}")
        llm_success = True
        
    except Exception as e:
        print(f"\n  ⚠️ LLM 报告生成失败: {e}")
        print("     使用模板生成报告...")
        llm_success = False
        report_content = None
    
    # 生成模板报告 (作为备份或补充)
    template_report = generate_template_report(features, recist, changes)
    
    os.makedirs(output_dir, exist_ok=True)
    template_path = os.path.join(output_dir, 'template_report.html')
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(template_report)
    
    print(f"     ✅ 模板报告已生成: {template_path}")
    
    return {
        'llm_success': llm_success,
        'report_content': report_content,
        'template_path': template_path
    }


def generate_template_report(features, recist, changes):
    """生成 HTML 模板报告"""
    
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>NeuroScan AI - 纵向对比诊断报告</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 40px;
        }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        .header {{
            text-align: center;
            padding: 30px;
            background: rgba(255,255,255,0.05);
            border-radius: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{ 
            font-size: 2.5em; 
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .section {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
        }}
        .section h2 {{
            color: #00d9ff;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(0,217,255,0.3);
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }}
        .info-item {{
            background: rgba(0,0,0,0.2);
            padding: 15px;
            border-radius: 10px;
        }}
        .info-label {{ color: #888; font-size: 0.9em; }}
        .info-value {{ font-size: 1.3em; color: #fff; margin-top: 5px; }}
        .recist-badge {{
            display: inline-block;
            padding: 10px 25px;
            border-radius: 25px;
            font-weight: bold;
            font-size: 1.2em;
            background: {'#ff4444' if recist['recist_code'] == 'PD' else '#44ff44' if recist['recist_code'] == 'CR' else '#4488ff' if recist['recist_code'] == 'PR' else '#ffaa44'};
            color: #000;
        }}
        .findings {{ line-height: 1.8; }}
        .highlight {{ color: #00ff88; font-weight: bold; }}
        .warning {{ color: #ff6b6b; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 NeuroScan AI</h1>
            <p style="margin-top: 10px; color: #888;">纵向对比诊断报告</p>
        </div>
        
        <div class="section">
            <h2>📋 患者信息</h2>
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">患者 ID</div>
                    <div class="info-value">WORKFLOW_DEMO</div>
                </div>
                <div class="info-item">
                    <div class="info-label">检查类型</div>
                    <div class="info-value">胸部 CT 纵向对比</div>
                </div>
                <div class="info-item">
                    <div class="info-label">基线日期</div>
                    <div class="info-value">2025-10-01</div>
                </div>
                <div class="info-item">
                    <div class="info-label">随访日期</div>
                    <div class="info-value">{datetime.now().strftime('%Y-%m-%d')}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 量化分析</h2>
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">基线最长径</div>
                    <div class="info-value">{recist['baseline_diameter']:.1f} mm</div>
                </div>
                <div class="info-item">
                    <div class="info-label">随访最长径</div>
                    <div class="info-value">{recist['followup_diameter']:.1f} mm</div>
                </div>
                <div class="info-item">
                    <div class="info-label">直径变化</div>
                    <div class="info-value {'warning' if recist['change_percent'] > 0 else 'highlight'}">{recist['change_percent']:+.1f}%</div>
                </div>
                <div class="info-item">
                    <div class="info-label">密度变化</div>
                    <div class="info-value">{features['density_change']:+.1f} HU</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>📋 RECIST 1.1 评估</h2>
            <div style="text-align: center; padding: 20px;">
                <span class="recist-badge">{recist['recist_code']}</span>
                <p style="margin-top: 15px; font-size: 1.2em;">{recist['recist_text']}</p>
            </div>
        </div>
        
        <div class="section">
            <h2>🔍 影像所见</h2>
            <div class="findings">
                <p>右肺上叶后段见一结节灶，与 <span class="highlight">2025-10-01</span> 基线对比：</p>
                <ul style="margin: 15px 0 15px 20px;">
                    <li>病灶由 <span class="highlight">{recist['baseline_diameter']:.1f}mm</span> 增大至 <span class="warning">{recist['followup_diameter']:.1f}mm</span></li>
                    <li>增大约 <span class="warning">{recist['change_percent']:+.1f}%</span></li>
                    <li>密度变化 <span class="highlight">{features['density_change']:+.1f} HU</span></li>
                </ul>
            </div>
        </div>
        
        <div class="section">
            <h2>💡 诊断建议</h2>
            <div class="findings">
                <p>根据 RECIST 1.1 标准评估为 <span class="warning">{recist['recist_text']}</span>，建议：</p>
                <ol style="margin: 15px 0 15px 20px;">
                    <li>立即安排胸部专家会诊</li>
                    <li>考虑 PET-CT 进一步评估代谢活性</li>
                    <li>建议进行穿刺活检明确病灶性质</li>
                    <li>3 个月后复查胸部 CT</li>
                </ol>
            </div>
        </div>
        
        <div style="text-align: center; padding: 20px; color: #666;">
            <p>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>本报告由 NeuroScan AI 自动生成，仅供参考，最终诊断请以医师意见为准</p>
        </div>
    </div>
</body>
</html>"""
    
    return html


def create_visualization(data, registered, changes, recist, output_dir):
    """
    阶段 7: 可视化输出
    """
    print_step(7, "可视化输出")
    
    print("\n  🎨 生成可视化:")
    print("     - 多平面对比图")
    print("     - 差异热力图")
    print("     - RECIST 评估图")
    
    os.makedirs(output_dir, exist_ok=True)
    
    baseline = data['baseline']
    followup = data['followup']
    diff_map = changes['diff_map']
    
    # 选择中间切片
    mid_slice = baseline.shape[2] // 2
    
    # 创建综合可视化
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#1a1a2e')
    
    # 标题
    fig.suptitle('NeuroScan AI - 纵向分析工作流演示', 
                 fontsize=20, color='white', fontweight='bold', y=0.98)
    
    # 1. 基线扫描
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(baseline[:, :, mid_slice].T, cmap='gray', origin='lower',
               vmin=-1000, vmax=400)
    ax1.set_title('Step 1: 基线扫描', color='white', fontsize=12)
    ax1.axis('off')
    
    # 2. 随访扫描
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(followup[:, :, mid_slice].T, cmap='gray', origin='lower',
               vmin=-1000, vmax=400)
    ax2.set_title('Step 2: 随访扫描', color='white', fontsize=12)
    ax2.axis('off')
    
    # 3. 配准结果
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(registered[:, :, mid_slice].T, cmap='gray', origin='lower',
               vmin=-1000, vmax=400)
    ax3.set_title('Step 3: 配准后', color='white', fontsize=12)
    ax3.axis('off')
    
    # 4. 差异图
    ax4 = fig.add_subplot(2, 3, 4)
    im4 = ax4.imshow(diff_map[:, :, mid_slice].T, cmap='RdBu_r', origin='lower',
                     vmin=-100, vmax=100)
    ax4.set_title('Step 4: 差异图', color='white', fontsize=12)
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, label='HU 变化', shrink=0.8)
    
    # 5. 热力图叠加
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(baseline[:, :, mid_slice].T, cmap='gray', origin='lower',
               vmin=-1000, vmax=400)
    mask = np.abs(diff_map[:, :, mid_slice].T) > 30
    overlay = np.ma.masked_where(~mask, np.abs(diff_map[:, :, mid_slice].T))
    ax5.imshow(overlay, cmap='hot', origin='lower', alpha=0.7, vmin=0, vmax=100)
    ax5.set_title('Step 5: 变化热力图', color='white', fontsize=12)
    ax5.axis('off')
    
    # 6. RECIST 评估结果
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_facecolor('#1a1a2e')
    
    # 绘制 RECIST 结果
    colors = {'CR': '#00ff00', 'PR': '#00aaff', 'SD': '#ffaa00', 'PD': '#ff4444'}
    color = colors.get(recist['recist_code'], '#ffffff')
    
    ax6.text(0.5, 0.7, 'RECIST 1.1 评估', ha='center', va='center',
             fontsize=16, color='white', transform=ax6.transAxes)
    ax6.text(0.5, 0.5, recist['recist_code'], ha='center', va='center',
             fontsize=48, color=color, fontweight='bold', transform=ax6.transAxes)
    ax6.text(0.5, 0.3, recist['recist_text'], ha='center', va='center',
             fontsize=12, color='white', transform=ax6.transAxes)
    ax6.text(0.5, 0.15, f"变化: {recist['change_percent']:+.1f}%", ha='center', va='center',
             fontsize=14, color=color, transform=ax6.transAxes)
    ax6.axis('off')
    ax6.set_title('Step 6: 评估结果', color='white', fontsize=12)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    
    # 保存
    viz_path = os.path.join(output_dir, 'workflow_visualization.png')
    plt.savefig(viz_path, dpi=150, facecolor='#1a1a2e', edgecolor='none',
                bbox_inches='tight')
    plt.close()
    
    print(f"\n  ✅ 可视化已保存: {viz_path}")
    
    return viz_path


def save_results(data, changes, features, recist, report, output_dir):
    """保存所有结果"""
    print_step(8, "保存结果")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存 JSON 结果
    results = {
        'timestamp': datetime.now().isoformat(),
        'patient_id': 'WORKFLOW_DEMO',
        'baseline_date': '2025-10-01',
        'followup_date': datetime.now().strftime('%Y-%m-%d'),
        'changes': {
            'changed_volume_mm3': float(changes['changed_volume_mm3']),
            'mean_change_hu': float(changes['mean_change']),
            'max_increase_hu': float(changes['max_increase']),
            'max_decrease_hu': float(changes['max_decrease'])
        },
        'features': {
            'baseline': features['baseline'],
            'followup': features['followup'],
            'density_change': float(features['density_change'])
        },
        'recist': {
            'baseline_diameter_mm': recist['baseline_diameter'],
            'followup_diameter_mm': recist['followup_diameter'],
            'change_percent': recist['change_percent'],
            'evaluation': recist['recist_code'],
            'description': recist['recist_text']
        }
    }
    
    json_path = os.path.join(output_dir, 'analysis_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n  ✅ 结果已保存:")
    print(f"     - JSON 数据: {json_path}")
    print(f"     - HTML 报告: {report['template_path']}")
    
    return json_path


def main():
    """主函数"""
    print_header("🏥 NeuroScan AI - 完整工作流演示")
    
    print("\n" + "─" * 70)
    print("  本演示展示从输入到输出的完整流程:")
    print("  输入 → 预处理 → 配准 → 变化检测 → 特征提取 → RECIST → 报告")
    print("─" * 70)
    
    # 设置路径
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    output_dir = os.path.join(PROJECT_ROOT, 'output', 'workflow_demo')
    
    # 查找可用数据
    baseline_path = None
    followup_path = None
    
    # 尝试 Learn2Reg 数据
    for folder in ['real_lung_001', 'real_lung_002', 'real_lung_003']:
        folder_path = os.path.join(data_dir, folder)
        if os.path.exists(folder_path):
            b = os.path.join(folder_path, 'baseline.nii.gz')
            f = os.path.join(folder_path, 'followup.nii.gz')
            if os.path.exists(b) and os.path.exists(f):
                baseline_path = b
                followup_path = f
                break
    
    if not baseline_path:
        print("\n  ❌ 未找到测试数据!")
        print("     请先运行: python scripts/download_real_data.py")
        return
    
    # 执行工作流
    try:
        # 阶段 1: 加载数据
        data = load_data(baseline_path, followup_path)
        
        # 阶段 2: 图像配准
        registered = perform_registration(
            data['baseline'], 
            data['followup'], 
            data['spacing']
        )
        
        # 阶段 3: 变化检测
        changes = detect_changes(
            data['baseline'], 
            registered, 
            data['spacing']
        )
        
        # 阶段 4: 特征提取
        features = extract_features(
            data['baseline'], 
            registered, 
            changes['diff_map'],
            data['spacing']
        )
        
        # 阶段 5: RECIST 评估
        recist = evaluate_recist()
        
        # 阶段 6: 报告生成
        report = generate_report(data, changes, features, recist, output_dir)
        
        # 阶段 7: 可视化
        viz_path = create_visualization(
            data, registered, changes, recist, output_dir
        )
        
        # 阶段 8: 保存结果
        json_path = save_results(data, changes, features, recist, report, output_dir)
        
        # 完成
        print_header("✅ 工作流演示完成!")
        
        print(f"""
  📁 输出目录: {output_dir}
  
  📄 生成的文件:
     1. analysis_results.json  - 量化分析数据
     2. template_report.html   - 诊断报告
     3. workflow_visualization.png - 可视化图
     {'4. ai_report.html         - LLM 智能报告' if report['llm_success'] else ''}
  
  🌐 查看报告:
     cd {output_dir} && python -m http.server 8899
     然后访问 http://localhost:8899/template_report.html
        """)
        
    except Exception as e:
        print(f"\n  ❌ 工作流执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

