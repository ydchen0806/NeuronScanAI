#!/usr/bin/env python3
"""
NeuroScan AI - Learn2Reg 真实数据可视化测试
使用真实的肺部 CT 数据 (吸气-呼气对) 测试配准和变化检测

数据来源: Learn2Reg Challenge Task 02 (Lung CT)
特点: 同一患者的吸气末和呼气末扫描，包含显著的解剖形变
"""

import os
import sys
import json
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 数据和输出目录
DATA_DIR = Path(__file__).parent.parent / "data" / "processed" / "real_lung_001"
OUTPUT_DIR = Path(__file__).parent / "visualizations_learn2reg"
OUTPUT_DIR.mkdir(exist_ok=True)

# 设置 matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_real_data():
    """加载真实的 Learn2Reg 数据"""
    print("📂 加载 Learn2Reg 真实数据...")
    
    baseline_path = DATA_DIR / "baseline.nii.gz"
    followup_path = DATA_DIR / "followup.nii.gz"
    baseline_mask_path = DATA_DIR / "baseline_mask.nii.gz"
    followup_mask_path = DATA_DIR / "followup_mask.nii.gz"
    
    if not baseline_path.exists():
        print(f"   ❌ 数据文件不存在: {baseline_path}")
        print("   请先运行: python scripts/download_real_data.py")
        return None
    
    # 加载数据
    baseline_img = nib.load(baseline_path)
    followup_img = nib.load(followup_path)
    
    baseline = baseline_img.get_fdata()
    followup = followup_img.get_fdata()
    
    # 加载掩码 (如果存在)
    baseline_mask = None
    followup_mask = None
    
    if baseline_mask_path.exists():
        baseline_mask = nib.load(baseline_mask_path).get_fdata()
    if followup_mask_path.exists():
        followup_mask = nib.load(followup_mask_path).get_fdata()
    
    data = {
        'baseline': baseline,
        'followup': followup,
        'baseline_mask': baseline_mask,
        'followup_mask': followup_mask,
        'affine': baseline_img.affine,
        'spacing': baseline_img.header.get_zooms()[:3],
        'shape': baseline.shape
    }
    
    print(f"   ✅ Baseline (吸气末): {baseline.shape}")
    print(f"   ✅ Followup (呼气末): {followup.shape}")
    print(f"   ✅ 体素间距: {data['spacing']}")
    print(f"   ✅ HU 范围: [{baseline.min():.0f}, {baseline.max():.0f}]")
    
    return data


def apply_lung_window(image, window_center=-600, window_width=1500):
    """应用肺窗"""
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    windowed = np.clip(image, min_val, max_val)
    windowed = (windowed - min_val) / (max_val - min_val) * 255
    return windowed.astype(np.uint8)


def visualize_real_ct_comparison(data):
    """可视化真实 CT 对比 (吸气 vs 呼气)"""
    print("📊 生成真实 CT 对比视图...")
    
    baseline = data['baseline']
    followup = data['followup']
    baseline_mask = data['baseline_mask']
    followup_mask = data['followup_mask']
    
    # 选择多个切片位置
    z_slices = [
        baseline.shape[2] // 4,      # 上部
        baseline.shape[2] // 2,      # 中部
        3 * baseline.shape[2] // 4,  # 下部
    ]
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Learn2Reg Real Lung CT - Inspiration vs Expiration', fontsize=16, fontweight='bold')
    
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.2)
    
    for col, z in enumerate(z_slices):
        # 第一行: Baseline (吸气末)
        ax1 = fig.add_subplot(gs[0, col])
        img1 = apply_lung_window(baseline[:, :, z])
        ax1.imshow(img1.T, cmap='gray', origin='lower')
        if baseline_mask is not None and baseline_mask[:, :, z].max() > 0:
            ax1.contour(baseline_mask[:, :, z].T, levels=[0.5], colors='cyan', linewidths=1)
        ax1.set_title(f'Inspiration (Slice {z})', fontsize=11)
        ax1.axis('off')
        if col == 0:
            ax1.set_ylabel('Baseline\n(Inspiration)', fontsize=10)
        
        # 第二行: Followup (呼气末)
        ax2 = fig.add_subplot(gs[1, col])
        img2 = apply_lung_window(followup[:, :, z])
        ax2.imshow(img2.T, cmap='gray', origin='lower')
        if followup_mask is not None and followup_mask[:, :, z].max() > 0:
            ax2.contour(followup_mask[:, :, z].T, levels=[0.5], colors='lime', linewidths=1)
        ax2.set_title(f'Expiration (Slice {z})', fontsize=11)
        ax2.axis('off')
        if col == 0:
            ax2.set_ylabel('Followup\n(Expiration)', fontsize=10)
        
        # 第三行: 差异图 (未配准)
        ax3 = fig.add_subplot(gs[2, col])
        diff = followup[:, :, z] - baseline[:, :, z]
        im3 = ax3.imshow(diff.T, cmap='RdBu_r', origin='lower', vmin=-500, vmax=500)
        ax3.set_title(f'Difference (Slice {z})', fontsize=11)
        ax3.axis('off')
        if col == 0:
            ax3.set_ylabel('Diff Map\n(Before Reg)', fontsize=10)
        
        # 第四行: 叠加视图
        ax4 = fig.add_subplot(gs[3, col])
        # 红绿叠加
        overlay = np.zeros((*img1.T.shape, 3), dtype=np.uint8)
        overlay[:, :, 0] = img1.T  # 红色通道 = 吸气
        overlay[:, :, 1] = img2.T  # 绿色通道 = 呼气
        ax4.imshow(overlay, origin='lower')
        ax4.set_title(f'Overlay (R=Insp, G=Exp)', fontsize=11)
        ax4.axis('off')
        if col == 0:
            ax4.set_ylabel('Color\nOverlay', fontsize=10)
    
    # 添加颜色条
    cbar_ax = fig.add_axes([0.92, 0.35, 0.02, 0.15])
    cbar = fig.colorbar(im3, cax=cbar_ax)
    cbar.set_label('Delta HU', fontsize=10)
    
    output_path = OUTPUT_DIR / "real_ct_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✅ Saved: {output_path}")
    return output_path


def visualize_registration_test(data):
    """测试配准效果"""
    print("📊 Testing registration on real data...")
    
    from app.services.registration.registrator import ImageRegistrator
    from app.services.analysis.change_detector import ChangeDetector
    
    baseline = data['baseline'].astype(np.float32)
    followup = data['followup'].astype(np.float32)
    spacing = tuple(float(s) for s in data['spacing'])
    
    # 为了加速测试，使用下采样
    downsample = 2
    baseline_ds = baseline[::downsample, ::downsample, ::downsample]
    followup_ds = followup[::downsample, ::downsample, ::downsample]
    spacing_ds = tuple(s * downsample for s in spacing)
    
    print(f"   Original shape: {baseline.shape}")
    print(f"   Downsampled shape: {baseline_ds.shape}")
    
    # 执行配准
    print("   Running rigid registration...")
    registrator = ImageRegistrator()
    
    try:
        registered_baseline, transform = registrator.rigid_registration(
            followup_ds, baseline_ds, spacing=spacing_ds
        )
        print("   ✅ Registration completed")
        
        # 变化检测
        print("   Computing difference maps...")
        detector = ChangeDetector()
        diff_before = followup_ds - baseline_ds
        diff_after = followup_ds - registered_baseline
        
        # 可视化
        center_z = baseline_ds.shape[2] // 2
        
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle('Registration Test on Real Lung CT Data', fontsize=16, fontweight='bold')
        
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.2)
        
        # 第一行: 原始图像
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(apply_lung_window(baseline_ds[:, :, center_z]).T, cmap='gray', origin='lower')
        ax1.set_title('Baseline (Inspiration)', fontsize=11)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(apply_lung_window(followup_ds[:, :, center_z]).T, cmap='gray', origin='lower')
        ax2.set_title('Followup (Expiration)', fontsize=11)
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(apply_lung_window(registered_baseline[:, :, center_z]).T, cmap='gray', origin='lower')
        ax3.set_title('Registered Baseline', fontsize=11)
        ax3.axis('off')
        
        # 棋盘格
        ax4 = fig.add_subplot(gs[0, 3])
        checkerboard = np.zeros_like(followup_ds[:, :, center_z])
        block_size = 16
        for i in range(0, checkerboard.shape[0], block_size):
            for j in range(0, checkerboard.shape[1], block_size):
                if ((i // block_size) + (j // block_size)) % 2 == 0:
                    checkerboard[i:i+block_size, j:j+block_size] = followup_ds[i:i+block_size, j:j+block_size, center_z]
                else:
                    checkerboard[i:i+block_size, j:j+block_size] = registered_baseline[i:i+block_size, j:j+block_size, center_z]
        ax4.imshow(apply_lung_window(checkerboard).T, cmap='gray', origin='lower')
        ax4.set_title('Checkerboard (Followup + Reg)', fontsize=11)
        ax4.axis('off')
        
        # 第二行: 差异图
        ax5 = fig.add_subplot(gs[1, 0])
        im5 = ax5.imshow(diff_before[:, :, center_z].T, cmap='RdBu_r', origin='lower', vmin=-500, vmax=500)
        ax5.set_title('Diff BEFORE Registration', fontsize=11)
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        
        ax6 = fig.add_subplot(gs[1, 1])
        im6 = ax6.imshow(diff_after[:, :, center_z].T, cmap='RdBu_r', origin='lower', vmin=-500, vmax=500)
        ax6.set_title('Diff AFTER Registration', fontsize=11)
        ax6.axis('off')
        plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
        
        # 绝对差异
        ax7 = fig.add_subplot(gs[1, 2])
        abs_diff_before = np.abs(diff_before[:, :, center_z])
        ax7.imshow(abs_diff_before.T, cmap='hot', origin='lower', vmin=0, vmax=500)
        ax7.set_title('|Diff| Before', fontsize=11)
        ax7.axis('off')
        
        ax8 = fig.add_subplot(gs[1, 3])
        abs_diff_after = np.abs(diff_after[:, :, center_z])
        ax8.imshow(abs_diff_after.T, cmap='hot', origin='lower', vmin=0, vmax=500)
        ax8.set_title('|Diff| After', fontsize=11)
        ax8.axis('off')
        
        # 第三行: 统计和评估
        ax9 = fig.add_subplot(gs[2, :2])
        
        # 计算配准质量指标
        mae_before = np.mean(np.abs(diff_before))
        mae_after = np.mean(np.abs(diff_after))
        improvement = (mae_before - mae_after) / mae_before * 100
        
        # 直方图
        ax9.hist(diff_before.flatten(), bins=100, alpha=0.5, label=f'Before (MAE={mae_before:.1f})', color='red')
        ax9.hist(diff_after.flatten(), bins=100, alpha=0.5, label=f'After (MAE={mae_after:.1f})', color='green')
        ax9.set_xlabel('Difference (HU)')
        ax9.set_ylabel('Frequency')
        ax9.set_title('Difference Distribution', fontsize=12)
        ax9.legend()
        ax9.set_xlim(-1000, 1000)
        
        # 统计信息
        ax10 = fig.add_subplot(gs[2, 2:])
        ax10.axis('off')
        
        stats_text = f"""
Registration Quality Assessment
{'='*40}

Input Data:
  - Dataset: Learn2Reg Task02 (Lung CT)
  - Baseline: Inspiration (end-inhale)
  - Followup: Expiration (end-exhale)
  - Shape: {baseline_ds.shape}
  - Spacing: {spacing_ds} mm

Registration Results:
  - MAE Before: {mae_before:.2f} HU
  - MAE After:  {mae_after:.2f} HU
  - Improvement: {improvement:.1f}%

Assessment:
  {'SUCCESS' if improvement > 0 else 'NEEDS IMPROVEMENT'}
  
Note: Large residual differences around
the diaphragm are expected due to the
significant breathing motion.
"""
        ax10.text(0.05, 0.95, stats_text, transform=ax10.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        output_path = OUTPUT_DIR / "registration_test.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Saved: {output_path}")
        print(f"   📊 MAE Before: {mae_before:.2f} HU")
        print(f"   📊 MAE After: {mae_after:.2f} HU")
        print(f"   📊 Improvement: {improvement:.1f}%")
        
        return output_path, {
            'mae_before': mae_before,
            'mae_after': mae_after,
            'improvement': improvement
        }
        
    except Exception as e:
        print(f"   ❌ Registration failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def visualize_multiplanar_view(data):
    """多平面视图 (轴位、矢状位、冠状位)"""
    print("📊 Generating multiplanar views...")
    
    baseline = data['baseline']
    followup = data['followup']
    
    # 中心切片
    cx, cy, cz = [s // 2 for s in baseline.shape]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Multiplanar Views - Inspiration vs Expiration', fontsize=16, fontweight='bold')
    
    # Baseline (吸气)
    axes[0, 0].imshow(apply_lung_window(baseline[:, :, cz]).T, cmap='gray', origin='lower')
    axes[0, 0].set_title('Baseline - Axial', fontsize=11)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(apply_lung_window(baseline[:, cy, :]).T, cmap='gray', origin='lower', aspect='auto')
    axes[0, 1].set_title('Baseline - Coronal', fontsize=11)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(apply_lung_window(baseline[cx, :, :]).T, cmap='gray', origin='lower', aspect='auto')
    axes[0, 2].set_title('Baseline - Sagittal', fontsize=11)
    axes[0, 2].axis('off')
    
    # Followup (呼气)
    axes[1, 0].imshow(apply_lung_window(followup[:, :, cz]).T, cmap='gray', origin='lower')
    axes[1, 0].set_title('Followup - Axial', fontsize=11)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(apply_lung_window(followup[:, cy, :]).T, cmap='gray', origin='lower', aspect='auto')
    axes[1, 1].set_title('Followup - Coronal', fontsize=11)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(apply_lung_window(followup[cx, :, :]).T, cmap='gray', origin='lower', aspect='auto')
    axes[1, 2].set_title('Followup - Sagittal', fontsize=11)
    axes[1, 2].axis('off')
    
    # 添加标签
    axes[0, 0].set_ylabel('Inspiration\n(Baseline)', fontsize=12)
    axes[1, 0].set_ylabel('Expiration\n(Followup)', fontsize=12)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "multiplanar_view.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✅ Saved: {output_path}")
    return output_path


def visualize_lung_volume_change(data):
    """可视化肺容量变化"""
    print("📊 Analyzing lung volume change...")
    
    baseline_mask = data['baseline_mask']
    followup_mask = data['followup_mask']
    spacing = data['spacing']
    
    if baseline_mask is None or followup_mask is None:
        print("   ⚠️  Lung masks not available")
        return None
    
    # 计算肺容量
    voxel_volume_ml = np.prod(spacing) / 1000  # mm³ -> ml
    
    baseline_volume = np.sum(baseline_mask > 0) * voxel_volume_ml
    followup_volume = np.sum(followup_mask > 0) * voxel_volume_ml
    volume_change = followup_volume - baseline_volume
    volume_change_pct = (volume_change / baseline_volume) * 100
    
    print(f"   Baseline lung volume: {baseline_volume:.0f} ml")
    print(f"   Followup lung volume: {followup_volume:.0f} ml")
    print(f"   Volume change: {volume_change:.0f} ml ({volume_change_pct:.1f}%)")
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Lung Volume Analysis - Breathing Cycle', fontsize=16, fontweight='bold')
    
    # 中心切片
    cz = baseline_mask.shape[2] // 2
    
    # 轴位视图
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(apply_lung_window(data['baseline'][:, :, cz]).T, cmap='gray', origin='lower')
    ax1.contour(baseline_mask[:, :, cz].T, levels=[0.5], colors='cyan', linewidths=2)
    ax1.set_title(f'Inspiration\nVolume: {baseline_volume:.0f} ml', fontsize=11)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(apply_lung_window(data['followup'][:, :, cz]).T, cmap='gray', origin='lower')
    ax2.contour(followup_mask[:, :, cz].T, levels=[0.5], colors='lime', linewidths=2)
    ax2.set_title(f'Expiration\nVolume: {followup_volume:.0f} ml', fontsize=11)
    ax2.axis('off')
    
    # 叠加对比
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(apply_lung_window(data['baseline'][:, :, cz]).T, cmap='gray', origin='lower', alpha=0.5)
    ax3.contour(baseline_mask[:, :, cz].T, levels=[0.5], colors='cyan', linewidths=2, linestyles='solid')
    ax3.contour(followup_mask[:, :, cz].T, levels=[0.5], colors='lime', linewidths=2, linestyles='dashed')
    ax3.set_title('Overlay\n(Cyan=Insp, Green=Exp)', fontsize=11)
    ax3.axis('off')
    
    # 体积条形图
    ax4 = fig.add_subplot(gs[1, 0])
    bars = ax4.bar(['Inspiration', 'Expiration'], [baseline_volume, followup_volume], 
                   color=['steelblue', 'coral'], edgecolor='black', linewidth=2)
    for bar, vol in zip(bars, [baseline_volume, followup_volume]):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                f'{vol:.0f} ml', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Volume (ml)', fontsize=11)
    ax4.set_title('Lung Volume Comparison', fontsize=12)
    
    # 体积变化饼图
    ax5 = fig.add_subplot(gs[1, 1])
    if volume_change < 0:
        sizes = [followup_volume, -volume_change]
        labels = ['Remaining', 'Exhaled']
        colors = ['steelblue', 'lightcoral']
    else:
        sizes = [baseline_volume, volume_change]
        labels = ['Baseline', 'Inhaled']
        colors = ['steelblue', 'lightgreen']
    ax5.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax5.set_title('Volume Distribution', fontsize=12)
    
    # 统计信息
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    stats_text = f"""
Lung Volume Analysis
{'='*35}

Measurements:
  - Inspiration: {baseline_volume:.0f} ml
  - Expiration:  {followup_volume:.0f} ml
  - Difference:  {volume_change:.0f} ml

Change:
  - Percentage: {volume_change_pct:.1f}%
  - Tidal Volume: ~{abs(volume_change):.0f} ml

Clinical Reference:
  - Normal tidal volume: 500-600 ml
  - Vital capacity: 3000-5000 ml

Note: These values reflect the
physiological lung volume change
during normal breathing.
"""
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    output_path = OUTPUT_DIR / "lung_volume_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✅ Saved: {output_path}")
    return output_path


def generate_html_report(visualizations: dict, reg_stats: dict, data: dict):
    """生成 HTML 报告"""
    print("📊 Generating HTML report...")
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuroScan AI - Learn2Reg Real Data Test Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0a192f 0%, #112240 50%, #1d3557 100%);
            color: #ccd6f6;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        header {{
            text-align: center;
            padding: 40px 20px;
            background: linear-gradient(135deg, #112240 0%, #1d3557 100%);
            border-radius: 20px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.4);
            border: 1px solid rgba(100, 255, 218, 0.1);
        }}
        header h1 {{
            font-size: 2.8em;
            background: linear-gradient(90deg, #64ffda, #00d9ff, #bd93f9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 15px;
        }}
        .subtitle {{ color: #8892b0; font-size: 1.2em; }}
        .badge {{
            display: inline-block;
            padding: 8px 20px;
            border-radius: 25px;
            font-weight: bold;
            font-size: 0.95em;
            margin: 5px;
        }}
        .badge.success {{ background: linear-gradient(90deg, #64ffda, #00d9ff); color: #0a192f; }}
        .badge.info {{ background: rgba(100, 255, 218, 0.1); color: #64ffda; border: 1px solid #64ffda; }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.03);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            border: 1px solid rgba(100, 255, 218, 0.1);
            transition: all 0.3s ease;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
            border-color: #64ffda;
        }}
        .stat-card h3 {{ font-size: 2.2em; color: #64ffda; margin-bottom: 10px; }}
        .stat-card p {{ color: #8892b0; }}
        .section {{
            background: rgba(255,255,255,0.02);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid rgba(100, 255, 218, 0.1);
        }}
        .section h2 {{
            color: #64ffda;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(100, 255, 218, 0.2);
        }}
        .visualization {{ text-align: center; margin: 20px 0; }}
        .visualization img {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 25px rgba(0,0,0,0.4);
            transition: transform 0.3s ease;
        }}
        .visualization img:hover {{ transform: scale(1.01); }}
        .description {{
            background: rgba(100, 255, 218, 0.05);
            padding: 15px 20px;
            border-radius: 10px;
            margin-top: 15px;
            border-left: 4px solid #64ffda;
        }}
        footer {{
            text-align: center;
            padding: 30px;
            color: #8892b0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🫁 NeuroScan AI</h1>
            <p class="subtitle">Learn2Reg Real Lung CT Data - Registration Test Report</p>
            <div style="margin-top: 20px;">
                <span class="badge success">REAL DATA</span>
                <span class="badge info">Learn2Reg Task02</span>
                <span class="badge info">Inspiration-Expiration Pair</span>
            </div>
        </header>
        
        <div class="stats">
            <div class="stat-card">
                <h3>{data['shape'][0]}x{data['shape'][1]}x{data['shape'][2]}</h3>
                <p>Volume Dimensions</p>
            </div>
            <div class="stat-card">
                <h3>{reg_stats['improvement']:.1f}%</h3>
                <p>Registration Improvement</p>
            </div>
            <div class="stat-card">
                <h3>{reg_stats['mae_after']:.1f}</h3>
                <p>MAE After Registration (HU)</p>
            </div>
            <div class="stat-card">
                <h3>REAL</h3>
                <p>Clinical CT Data</p>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 CT Comparison: Inspiration vs Expiration</h2>
            <div class="visualization">
                <img src="real_ct_comparison.png" alt="Real CT Comparison">
            </div>
            <div class="description">
                <p><strong>Description:</strong> Comparison of real lung CT scans from the Learn2Reg Challenge.
                The baseline shows the lungs at end-inspiration (maximum volume), while the followup shows
                end-expiration (minimum volume). The significant anatomical deformation is visible,
                especially around the diaphragm.</p>
            </div>
        </div>
        
        <div class="section">
            <h2>🔄 Registration Test Results</h2>
            <div class="visualization">
                <img src="registration_test.png" alt="Registration Test">
            </div>
            <div class="description">
                <p><strong>Description:</strong> Registration test using real breathing motion data.
                The algorithm attempts to align the inspiration and expiration scans.
                <br><br>
                <strong>Results:</strong>
                <br>• MAE Before Registration: {reg_stats['mae_before']:.2f} HU
                <br>• MAE After Registration: {reg_stats['mae_after']:.2f} HU
                <br>• Improvement: {reg_stats['improvement']:.1f}%
                <br><br>
                <strong>Note:</strong> Large residual differences around the diaphragm are expected
                due to the significant breathing motion (typically 2-4 cm displacement).</p>
            </div>
        </div>
        
        <div class="section">
            <h2>🖼️ Multiplanar Views</h2>
            <div class="visualization">
                <img src="multiplanar_view.png" alt="Multiplanar Views">
            </div>
            <div class="description">
                <p><strong>Description:</strong> Three orthogonal views (axial, coronal, sagittal)
                showing the lung anatomy during inspiration and expiration phases.</p>
            </div>
        </div>
        
        <div class="section">
            <h2>📏 Lung Volume Analysis</h2>
            <div class="visualization">
                <img src="lung_volume_analysis.png" alt="Lung Volume Analysis">
            </div>
            <div class="description">
                <p><strong>Description:</strong> Quantitative analysis of lung volume change during breathing.
                The cyan contour represents inspiration (larger volume), while the green contour represents
                expiration (smaller volume). This demonstrates the physiological lung volume change
                that our registration algorithm must handle.</p>
            </div>
        </div>
        
        <footer>
            <p>NeuroScan AI - Intelligent Medical Imaging Longitudinal Diagnosis System</p>
            <p style="margin-top: 10px;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p style="margin-top: 10px; font-size: 0.9em;">
                Data Source: Learn2Reg Challenge Task02 (Zenodo) | Powered by MONAI | SimpleITK
            </p>
        </footer>
    </div>
</body>
</html>
"""
    
    output_path = OUTPUT_DIR / "learn2reg_report.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"   ✅ Saved: {output_path}")
    return output_path


def run_all_visualizations():
    """运行所有可视化"""
    print("\n" + "="*60)
    print("NeuroScan AI - Learn2Reg Real Data Visualization Test")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data: {DATA_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*60 + "\n")
    
    # 加载数据
    data = load_real_data()
    if data is None:
        return False
    
    visualizations = {}
    reg_stats = {'mae_before': 0, 'mae_after': 0, 'improvement': 0}
    
    try:
        # 生成可视化
        visualizations['comparison'] = visualize_real_ct_comparison(data)
        visualizations['multiplanar'] = visualize_multiplanar_view(data)
        visualizations['volume'] = visualize_lung_volume_change(data)
        
        # 配准测试
        reg_path, stats = visualize_registration_test(data)
        if reg_path:
            visualizations['registration'] = reg_path
            reg_stats = stats
        
        # 生成 HTML 报告
        visualizations['html'] = generate_html_report(visualizations, reg_stats, data)
        
        print("\n" + "="*60)
        print("✅ Visualization Complete!")
        print("="*60)
        print(f"Generated {len(visualizations)} visualization files")
        print(f"\n📁 Output: {OUTPUT_DIR}")
        print(f"🌐 Report: {visualizations.get('html', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_visualizations()
    sys.exit(0 if success else 1)

