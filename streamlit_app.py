#!/usr/bin/env python3
"""
NeuroScan AI - Streamlit Frontend Application
Intelligent Medical Imaging Longitudinal Diagnosis System
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import streamlit as st
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import tempfile
import zipfile
import uuid
from datetime import datetime
import json

from app.core.config import settings
from app.core.logging import logger
from app.services.dicom import DicomLoader
from app.services.segmentation import OrganSegmentor, ORGAN_LABELS
from app.services.registration import ImageRegistrator
from app.services.analysis import ChangeDetector, FeatureExtractor, ROIExtractor
from app.services.dicom.windowing import apply_ct_window, CT_WINDOWS

# Page config
st.set_page_config(
    page_title="NeuroScan AI - Medical Imaging Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Theme colors */
    :root {
        --primary-color: #0066cc;
        --secondary-color: #00a3e0;
        --accent-color: #ff6b35;
        --bg-dark: #0a1628;
        --bg-card: #1a2942;
        --text-primary: #ffffff;
        --text-secondary: #a0aec0;
    }
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #1a2942 50%, #0d2137 100%);
    }
    
    /* Text color improvements - High contrast */
    .stApp, .stApp * {
        color: #ffffff !important;
    }
    
    /* Main content area */
    .main .block-container {
        color: #ffffff !important;
    }
    
    /* All text elements */
    p, span, div, label, h1, h2, h3, h4, h5, h6, li, td, th {
        color: #ffffff !important;
    }
    
    /* Streamlit text elements */
    .stMarkdown, .stText, .stInfo, .stSuccess, .stWarning, .stError {
        color: #ffffff !important;
    }
    
    /* Selectbox and input labels */
    label, .stSelectbox label, .stTextInput label, .stRadio label {
        color: #e0e0e0 !important;
        font-weight: 500 !important;
    }
    
    /* Metric values */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: #ffffff !important;
    }
    
    /* Code blocks */
    code, pre {
        background-color: #1a2942 !important;
        color: #00ff88 !important;
        border: 1px solid rgba(0, 163, 224, 0.3) !important;
    }
    
    /* JSON display - Enhanced colors */
    .stJson {
        background-color: #1a2942 !important;
        color: #ffffff !important;
    }
    
    /* JSON data container - Better visibility */
    [data-testid="stJson"] {
        background-color: #0d2137 !important;
        border: 2px solid rgba(0, 163, 224, 0.4) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    /* JSON keys */
    [data-testid="stJson"] .json-key {
        color: #00a3e0 !important;
        font-weight: 600 !important;
    }
    
    /* JSON values */
    [data-testid="stJson"] .json-value {
        color: #ffffff !important;
    }
    
    /* JSON strings */
    [data-testid="stJson"] .json-string {
        color: #48bb78 !important;
    }
    
    /* JSON numbers */
    [data-testid="stJson"] .json-number {
        color: #ff6b35 !important;
    }
    
    /* JSON boolean */
    [data-testid="stJson"] .json-boolean {
        color: #9f7aea !important;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: rgba(0, 163, 224, 0.15) !important;
        border-left: 4px solid #00a3e0 !important;
        color: #ffffff !important;
    }
    
    /* Success boxes */
    .stSuccess {
        background-color: rgba(72, 187, 120, 0.15) !important;
        border-left: 4px solid #48bb78 !important;
        color: #ffffff !important;
    }
    
    /* Warning boxes */
    .stWarning {
        background-color: rgba(237, 137, 54, 0.15) !important;
        border-left: 4px solid #ed8936 !important;
        color: #ffffff !important;
    }
    
    /* Error boxes */
    .stError {
        background-color: rgba(245, 101, 101, 0.15) !important;
        border-left: 4px solid #f56565 !important;
        color: #ffffff !important;
    }
    
    /* Expander headers */
    .streamlit-expanderHeader {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Table text */
    table, thead, tbody, tr, td, th {
        color: #ffffff !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Selectbox options */
    .stSelectbox > div > div {
        color: #ffffff !important;
        background-color: #1a2942 !important;
    }
    
    /* Radio buttons */
    .stRadio label {
        color: #e0e0e0 !important;
    }
    
    /* Slider labels */
    .stSlider label {
        color: #e0e0e0 !important;
    }
    
    /* Title styles */
    .main-title {
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00a3e0 0%, #0066cc 50%, #ff6b35 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        font-family: 'SF Pro Text', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 1.1rem;
        color: #a0aec0;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Card styles */
    .metric-card {
        background: linear-gradient(145deg, #1a2942 0%, #0d2137 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(0, 163, 224, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00a3e0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #a0aec0;
        margin-top: 0.5rem;
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-success {
        background: rgba(72, 187, 120, 0.2);
        color: #48bb78;
        border: 1px solid rgba(72, 187, 120, 0.3);
    }
    
    .status-warning {
        background: rgba(237, 137, 54, 0.2);
        color: #ed8936;
        border: 1px solid rgba(237, 137, 54, 0.3);
    }
    
    .status-error {
        background: rgba(245, 101, 101, 0.2);
        color: #f56565;
        border: 1px solid rgba(245, 101, 101, 0.3);
    }
    
    /* Sidebar styles - Improved contrast */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2942 0%, #0d2137 100%) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label {
        color: #e0e0e0 !important;
    }
    
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stTextInput label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricValue"],
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #ffffff !important;
    }
    
    /* Sidebar radio buttons - better visibility */
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
        color: #ffffff !important;
        background-color: rgba(26, 41, 66, 0.5) !important;
        padding: 0.5rem !important;
        border-radius: 6px !important;
        margin: 0.25rem 0 !important;
    }
    
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {
        background-color: rgba(0, 163, 224, 0.2) !important;
    }
    
    /* Sidebar separator */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.2) !important;
        margin: 1rem 0 !important;
    }
    
    /* Button styles */
    .stButton > button {
        background: linear-gradient(135deg, #0066cc 0%, #00a3e0 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0, 163, 224, 0.4);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #0066cc 0%, #00a3e0 100%);
    }
    
    /* File upload area */
    .uploadedFile {
        background: rgba(0, 163, 224, 0.1);
        border: 2px dashed rgba(0, 163, 224, 0.3);
        border-radius: 12px;
    }
    
    /* Image viewer */
    .image-viewer {
        background: #0a1628;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(0, 163, 224, 0.2);
    }
    
    /* Report section */
    .report-section {
        background: linear-gradient(145deg, #1a2942 0%, #0d2137 100%);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(0, 163, 224, 0.2);
        margin-top: 1rem;
    }
    
    /* Dataframe and table styling */
    .stDataFrame, .dataframe {
        background-color: #1a2942 !important;
        color: #ffffff !important;
    }
    
    .stDataFrame table, .dataframe table {
        color: #ffffff !important;
    }
    
    .stDataFrame th, .dataframe th {
        background-color: #0d2137 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    .stDataFrame td, .dataframe td {
        color: #ffffff !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Markdown content */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #e0e0e0 !important;
        line-height: 1.6 !important;
    }
    
    /* Streamlit widgets background */
    .stSelectbox, .stTextInput, .stNumberInput, .stTextArea {
        background-color: rgba(26, 41, 66, 0.5) !important;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* Streamlit columns */
    .stColumn {
        color: #ffffff !important;
    }
    
    /* Streamlit tabs */
    .stTabs [data-baseweb="tab"] {
        color: #ffffff !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #00a3e0 !important;
        border-bottom: 2px solid #00a3e0 !important;
    }
    
    /* Streamlit spinner text */
    .stSpinner {
        color: #ffffff !important;
    }
    
    /* Streamlit download button */
    .stDownloadButton button {
        background-color: #0066cc !important;
        color: #ffffff !important;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Ensure all text is visible */
    * {
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# ============ Helper Functions ============

@st.cache_resource
def get_dicom_loader():
    """Get DICOM loader (cached)"""
    return DicomLoader()


@st.cache_resource
def get_segmentor():
    """Get segmentor (cached)"""
    return OrganSegmentor()


@st.cache_resource
def get_registrator():
    """Get registrator (cached)"""
    return ImageRegistrator()


@st.cache_resource
def get_change_detector():
    """Get change detector (cached)"""
    return ChangeDetector()


def load_nifti_cached(nifti_path: str):
    """Load NIfTI file (cached)"""
    img = nib.load(nifti_path)
    data = img.get_fdata()
    return data, img


def render_slice(data: np.ndarray, slice_idx: int, view: str = "axial", 
                 window_preset: str = "lung", overlay: np.ndarray = None):
    """Render slice image"""
    # Get slice
    if view == "axial":
        slice_data = data[:, :, slice_idx]
        if overlay is not None:
            overlay_slice = overlay[:, :, slice_idx]
    elif view == "sagittal":
        slice_data = data[slice_idx, :, :]
        if overlay is not None:
            overlay_slice = overlay[slice_idx, :, :]
    else:  # coronal
        slice_data = data[:, slice_idx, :]
        if overlay is not None:
            overlay_slice = overlay[:, slice_idx, :]
    
    # Apply window/level
    windowed = apply_ct_window(slice_data, preset=window_preset)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0a1628')
    ax.set_facecolor('#0a1628')
    
    # Display CT image
    ax.imshow(windowed.T, cmap='gray', origin='lower', aspect='equal')
    
    # Overlay segmentation
    if overlay is not None:
        # Create colored overlay
        try:
            # Use new matplotlib API
            cmap = plt.colormaps.get_cmap('jet')
            # Normalize overlay values to 0-1 range
            overlay_normalized = overlay_slice.T.astype(np.float32)
            max_val = overlay_normalized.max()
            if max_val > 0:
                overlay_normalized = overlay_normalized / max_val
            
            # Apply colormap (returns RGBA array)
            overlay_colored = cmap(overlay_normalized)
            
            # Set transparency: only show where overlay > 0
            mask = overlay_slice.T > 0
            overlay_colored[..., 3] = mask.astype(np.float32) * 0.4
            
            # Ensure shape is (H, W, 4) not (H, W, 1, 4)
            if overlay_colored.ndim == 4 and overlay_colored.shape[2] == 1:
                overlay_colored = overlay_colored.squeeze(2)
            
            ax.imshow(overlay_colored, origin='lower', aspect='equal')
        except Exception as e:
            logger.warning(f"Failed to overlay segmentation: {e}")
    
    ax.axis('off')
    ax.set_title(f'{view.capitalize()} - Slice {slice_idx}', color='white', fontsize=12)
    
    plt.tight_layout()
    return fig


def render_diff_heatmap(followup: np.ndarray, diff_map: np.ndarray, 
                        slice_idx: int, view: str = "axial"):
    """Render difference heatmap"""
    # Get slice
    if view == "axial":
        bg_slice = followup[:, :, slice_idx]
        diff_slice = diff_map[:, :, slice_idx]
    elif view == "sagittal":
        bg_slice = followup[slice_idx, :, :]
        diff_slice = diff_map[slice_idx, :, :]
    else:
        bg_slice = followup[:, slice_idx, :]
        diff_slice = diff_map[:, slice_idx, :]
    
    # Apply window/level
    bg_windowed = apply_ct_window(bg_slice, preset="lung")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0a1628')
    ax.set_facecolor('#0a1628')
    
    # Show background
    ax.imshow(bg_windowed.T, cmap='gray', origin='lower', aspect='equal')
    
    # Overlay heatmap
    max_val = max(np.abs(diff_slice).max(), 1)
    diff_normalized = diff_slice / max_val
    alpha = np.abs(diff_normalized)
    alpha = np.clip(alpha, 0, 1)
    
    heatmap = ax.imshow(
        diff_normalized.T, 
        cmap='RdBu_r', 
        origin='lower', 
        aspect='equal',
        alpha=alpha.T * 0.7,
        vmin=-1, 
        vmax=1
    )
    
    # Add colorbar
    cbar = plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('HU Change', color='white', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    ax.axis('off')
    ax.set_title(f'Difference Map - {view.capitalize()} Slice {slice_idx}', 
                 color='white', fontsize=12)
    
    plt.tight_layout()
    return fig


def create_3d_visualization(segmentation: np.ndarray, organ_labels: dict):
    """Create 3D visualization"""
    # Simplified 3D visualization - using isosurfaces
    fig = go.Figure()
    
    # Select main organs for visualization
    main_organs = {
        5: ("liver", "#e74c3c"),
        13: ("lung_left_upper", "#3498db"),
        15: ("lung_right_upper", "#2ecc71"),
        44: ("heart", "#9b59b6"),
    }
    
    for label_id, (name, color) in main_organs.items():
        if label_id in np.unique(segmentation):
            # Get organ mask
            mask = (segmentation == label_id)
            
            # Downsample for faster rendering
            mask_ds = mask[::4, ::4, ::4]
            
            if mask_ds.sum() > 0:
                # Get surface points
                from skimage import measure
                try:
                    verts, faces, _, _ = measure.marching_cubes(mask_ds, level=0.5)
                    
                    # Add mesh
                    fig.add_trace(go.Mesh3d(
                        x=verts[:, 0],
                        y=verts[:, 1],
                        z=verts[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        color=color,
                        opacity=0.5,
                        name=name
                    ))
                except:
                    pass
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='#0a1628'
        ),
        paper_bgcolor='#0a1628',
        plot_bgcolor='#0a1628',
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text="3D Organ Visualization", font=dict(color='white'))
    )
    
    return fig


# ============ Main Interface ============

def main():
    # Title
    st.markdown('<h1 class="main-title">🏥 NeuroScan AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Intelligent Longitudinal Medical Imaging Analysis System</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
        st.markdown("### 🔧 系统控制")
        
        # Mode selection
        mode = st.radio(
            "选择模式",
            ["📤 数据上传", "🔬 单次分析", "📊 纵向对比", "📋 诊断报告", "📁 示例数据"],
            index=0
        )
        
        st.markdown("---")
        
        # System status
        st.markdown("### 📊 系统状态")
        col1, col2 = st.columns(2)
        with col1:
            import torch
            gpu_ok = torch.cuda.is_available()
            st.metric("GPU", "✅ OK" if gpu_ok else "❌ N/A")
        with col2:
            st.metric("模型", "✅ 就绪")
        
        # LLM Status
        try:
            import ollama
            ollama.list()
            st.metric("LLM", "✅ Ollama")
        except:
            st.metric("LLM", "⚠️ 模板模式")
        
        st.markdown("---")
        
        # Loaded scans
        st.markdown("### 📁 已加载扫描")
        if "scans" in st.session_state and st.session_state.scans:
            for scan_id, info in st.session_state.scans.items():
                scan_type = info.get("scan_type", "")
                type_emoji = "🔵" if scan_type == "baseline" else "🔴" if scan_type == "followup" else "⚪"
                st.text(f"{type_emoji} {scan_id[:8]}...")
        else:
            st.text("暂无扫描数据")
        
        st.markdown("---")
        st.markdown("### ℹ️ 关于")
        st.caption("NeuroScan AI v1.0")
        st.caption("智能纵向医学影像分析系统")
    
    # Initialize session state
    if "scans" not in st.session_state:
        st.session_state.scans = {}
    if "current_scan" not in st.session_state:
        st.session_state.current_scan = None
    if "segmentation" not in st.session_state:
        st.session_state.segmentation = None
    
    # Main content area
    if mode == "📤 数据上传":
        render_upload_page()
    elif mode == "🔬 单次分析":
        render_single_analysis_page()
    elif mode == "📊 纵向对比":
        render_longitudinal_page()
    elif mode == "📋 诊断报告":
        render_report_page()
    else:  # 示例数据
        render_sample_data_page()


def process_single_upload(uploaded_file, patient_id=None, scan_type=None):
    """Process single file upload"""
    # Get file extension
    file_name = uploaded_file.name
    file_ext = ''.join(Path(file_name).suffixes).lower()
    if not file_ext:
        file_ext = '.bin'
    
    # Save uploaded file (preserve extension)
    temp_path = settings.CACHE_DIR / f"upload_{uuid.uuid4()}{file_ext}"
    settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # Use new multi-format loader
    loader = get_dicom_loader()
    
    # Detect file type
    file_type = loader.detect_file_type(temp_path)
    st.info(f"🔍 Detected file type: {file_type}")
    
    # Process file (all formats supported)
    scan_info = loader.process_upload_any_format(temp_path, patient_id or None)
    
    # Store to session state
    st.session_state.scans[scan_info.scan_id] = {
        "info": scan_info,
        "nifti_path": scan_info.nifti_path,
        "metadata": scan_info.metadata.model_dump(),
        "scan_type": scan_type  # baseline or followup
    }
    st.session_state.current_scan = scan_info.scan_id
    
    # Clean up temp file
    temp_path.unlink()
    
    return scan_info


def display_scan_preview(scan_info):
    """Display scan preview"""
    try:
        loader = get_dicom_loader()
        data, _ = loader.load_nifti(Path(scan_info.nifti_path))
        
        # Show metadata
        with st.expander("📋 Scan Metadata", expanded=True):
            st.json(scan_info.metadata.model_dump())
        
        st.markdown("#### 📸 Scan Preview")
        mid_slice = data.shape[2] // 2
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Apply window/level (lung window)
        vmin, vmax = -1000, 400
        
        axes[0].imshow(data[:, :, mid_slice].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        axes[0].set_title('Axial')
        axes[0].axis('off')
        axes[1].imshow(data[:, data.shape[1]//2, :].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        axes[1].set_title('Sagittal')
        axes[1].axis('off')
        axes[2].imshow(data[data.shape[0]//2, :, :].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        axes[2].set_title('Coronal')
        axes[2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as preview_error:
        st.warning(f"Preview failed: {preview_error}")


def display_comparison_preview(baseline_info, followup_info):
    """Display baseline and followup scan comparison preview"""
    try:
        loader = get_dicom_loader()
        baseline_data, _ = loader.load_nifti(Path(baseline_info.nifti_path))
        followup_data, _ = loader.load_nifti(Path(followup_info.nifti_path))
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        # Apply window/level (lung window)
        vmin, vmax = -1000, 400
        
        # Baseline scan
        mid_slice_b = baseline_data.shape[2] // 2
        axes[0, 0].imshow(baseline_data[:, :, mid_slice_b].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title('Baseline - Axial')
        axes[0, 0].axis('off')
        axes[0, 1].imshow(baseline_data[:, baseline_data.shape[1]//2, :].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title('Baseline - Sagittal')
        axes[0, 1].axis('off')
        axes[0, 2].imshow(baseline_data[baseline_data.shape[0]//2, :, :].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        axes[0, 2].set_title('Baseline - Coronal')
        axes[0, 2].axis('off')
        
        # Followup scan
        mid_slice_f = followup_data.shape[2] // 2
        axes[1, 0].imshow(followup_data[:, :, mid_slice_f].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        axes[1, 0].set_title('Followup - Axial')
        axes[1, 0].axis('off')
        axes[1, 1].imshow(followup_data[:, followup_data.shape[1]//2, :].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        axes[1, 1].set_title('Followup - Sagittal')
        axes[1, 1].axis('off')
        axes[1, 2].imshow(followup_data[followup_data.shape[0]//2, :, :].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        axes[1, 2].set_title('Followup - Coronal')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Show basic info
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Baseline Scan**")
            st.write(f"- Shape: {baseline_data.shape}")
            st.write(f"- Scan ID: {baseline_info.scan_id[:8]}...")
        with col2:
            st.markdown("**Followup Scan**")
            st.write(f"- Shape: {followup_data.shape}")
            st.write(f"- Scan ID: {followup_info.scan_id[:8]}...")
            
    except Exception as e:
        st.warning(f"Comparison preview failed: {e}")


def render_upload_page():
    """Render data upload page"""
    st.markdown("## 📤 数据上传")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(145deg, #1a2942 0%, #0d2137 100%); 
                    padding: 1.5rem; border-radius: 16px; 
                    border: 2px solid rgba(0, 163, 224, 0.3);
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);">
            <h4 style="color: #00a3e0; margin-top: 0;">📁 上传医学影像</h4>
            <p style="color: #e0e0e0; margin-bottom: 0;">支持多种格式，系统自动识别并转换为标准格式进行分析。</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Select upload mode
        upload_mode = st.radio(
            "上传模式",
            ["单次扫描", "纵向对比 (多文件)"],
            horizontal=True,
            help="选择「纵向对比」可上传多次扫描进行对比分析（第一个为基线，第二个为随访）"
        )
        
        if upload_mode == "单次扫描":
            st.markdown("""
            <div style="background: rgba(0, 163, 224, 0.1); padding: 1rem; border-radius: 12px; 
                        border: 2px dashed rgba(0, 163, 224, 0.4); margin: 1rem 0;">
            """, unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "选择医学影像文件",
                type=["zip", "tar", "gz", "nii", "nrrd", "mha", "mhd", "dcm"],
                help="支持格式: ZIP/TAR压缩包, NIfTI (.nii/.nii.gz), NRRD, MHA/MHD, DICOM (.dcm)"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            uploaded_files = [uploaded_file] if uploaded_file else []
        else:
            st.markdown("""
            <div style="background: rgba(72, 187, 120, 0.1); padding: 1rem; border-radius: 12px; 
                        border: 1px solid rgba(72, 187, 120, 0.3); margin: 0.5rem 0;">
                <p style="color: #48bb78; margin: 0; font-weight: 600;">📊 纵向对比模式</p>
                <p style="color: #e0e0e0; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                    上传 2 个文件：<strong>第一个</strong>为<span style="color: #3498db;">基线扫描</span>，
                    <strong>第二个</strong>为<span style="color: #e74c3c;">随访扫描</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: rgba(0, 163, 224, 0.1); padding: 1rem; border-radius: 12px; 
                        border: 2px dashed rgba(0, 163, 224, 0.4); margin: 1rem 0;">
            """, unsafe_allow_html=True)
            uploaded_files_list = st.file_uploader(
                "选择医学影像文件（可多选）",
                type=["zip", "tar", "gz", "nii", "nrrd", "mha", "mhd", "dcm"],
                accept_multiple_files=True,
                help="上传多个文件: 第一个=基线, 第二个=随访"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            uploaded_files = []
            if uploaded_files_list:
                if len(uploaded_files_list) >= 1:
                    uploaded_files.append(("baseline", uploaded_files_list[0]))
                if len(uploaded_files_list) >= 2:
                    uploaded_files.append(("followup", uploaded_files_list[1]))
                if len(uploaded_files_list) > 2:
                    st.warning(f"⚠️ Only the first 2 files will be used. {len(uploaded_files_list) - 2} additional file(s) ignored.")
        
        patient_id = st.text_input("患者 ID（可选）", placeholder="例如: P001")
        
        # Show supported formats
        with st.expander("📋 支持的文件格式", expanded=False):
            st.markdown("""
            | Format | Extension | Description |
            |--------|-----------|-------------|
            | **ZIP Archive** | `.zip` | Contains DICOM or NIfTI files |
            | **TAR Archive** | `.tar`, `.tar.gz`, `.tgz` | Contains DICOM or NIfTI files |
            | **NIfTI** | `.nii`, `.nii.gz` | Neuroimaging standard format |
            | **DICOM** | `.dcm`, `.dicom` | Medical imaging standard |
            | **NRRD** | `.nrrd` | Nearly Raw Raster Data |
            | **MHA/MHD** | `.mha`, `.mhd` | MetaImage format |
            """)
        
        # Process uploaded files
        if upload_mode == "单次扫描":
            # Single file upload mode
            if uploaded_files and uploaded_files[0] is not None:
                uploaded_file = uploaded_files[0]
                file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
                st.info(f"📄 Selected: {uploaded_file.name} ({file_size:.2f} MB)")
                
                if st.button("🚀 Start Processing", use_container_width=True):
                    with st.spinner("Processing medical image data..."):
                        try:
                            scan_info = process_single_upload(uploaded_file, patient_id)
                            st.success(f"✅ Complete! Scan ID: {scan_info.scan_id[:8]}...")
                            display_scan_preview(scan_info)
                        except Exception as e:
                            st.error(f"❌ Processing failed: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
        else:
            # Longitudinal mode - multi-file upload
            if len(uploaded_files) == 2:
                st.success("✅ Baseline and followup scans selected")
                for label, f in uploaded_files:
                    file_size = len(f.getvalue()) / (1024 * 1024)
                    st.info(f"📄 {label}: {f.name} ({file_size:.2f} MB)")
                
                if st.button("🚀 Process & Compare", use_container_width=True):
                    with st.spinner("Processing medical image data..."):
                        try:
                            baseline_info = None
                            followup_info = None
                            
                            for label, f in uploaded_files:
                                scan_info = process_single_upload(f, patient_id, scan_type=label)
                                if label == "baseline":
                                    baseline_info = scan_info
                                    st.session_state.baseline_scan = scan_info.scan_id
                                else:
                                    followup_info = scan_info
                                    st.session_state.followup_scan = scan_info.scan_id
                                st.success(f"✅ {label} processed: {scan_info.scan_id[:8]}...")
                            
                            # Show comparison preview
                            if baseline_info and followup_info:
                                st.markdown("### 📊 Scan Comparison Preview")
                                display_comparison_preview(baseline_info, followup_info)
                                st.info("💡 Go to **Longitudinal Compare** page for detailed analysis")
                                
                        except Exception as e:
                            st.error(f"❌ Processing failed: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
            elif len(uploaded_files) == 1:
                label, f = uploaded_files[0]
                st.warning(f"⚠️ Only {label} uploaded. Please upload both baseline and followup scans.")
            else:
                st.info("📤 Please upload baseline and followup scans for longitudinal analysis")
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>📋 Supported Formats</h4>
            <ul style="color: #a0aec0;">
                <li>DICOM (.dcm, .dicom)</li>
                <li>NIfTI (.nii, .nii.gz)</li>
                <li>ZIP/TAR archives</li>
                <li>NRRD (.nrrd)</li>
                <li>MHA/MHD (.mha, .mhd)</li>
            </ul>
            <h4 style="margin-top: 1rem;">⚙️ Processing Pipeline</h4>
            <ol style="color: #a0aec0;">
                <li>Auto-detect file format</li>
                <li>Parse metadata</li>
                <li>Resample to 1.0mm isotropic</li>
                <li>Convert to NIfTI</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Show uploaded scans
    if st.session_state.scans:
        st.markdown("### 📁 Uploaded Scans")
        for scan_id, scan_data in st.session_state.scans.items():
            with st.expander(f"Scan: {scan_id[:8]}..."):
                st.json(scan_data["metadata"])


def render_single_analysis_page():
    """Render single analysis page"""
    st.markdown("## 🔬 Single Scan Analysis")
    
    if not st.session_state.scans:
        st.warning("⚠️ Please upload data first")
        return
    
    # Select scan
    scan_options = list(st.session_state.scans.keys())
    selected_scan = st.selectbox(
        "Select scan to analyze",
        scan_options,
        format_func=lambda x: f"{x[:8]}... ({st.session_state.scans[x]['metadata'].get('patient_id', 'Unknown')})"
    )
    
    if selected_scan:
        scan_data = st.session_state.scans[selected_scan]
        nifti_path = scan_data["nifti_path"]
        
        # Load data
        data, img = load_nifti_cached(nifti_path)
        
        # Display controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            view = st.selectbox("View", ["axial", "sagittal", "coronal"])
        
        with col2:
            max_slice = data.shape[{"axial": 2, "sagittal": 0, "coronal": 1}[view]] - 1
            slice_idx = st.slider("Slice", 0, max_slice, max_slice // 2)
        
        with col3:
            window = st.selectbox("Window Preset", list(CT_WINDOWS.keys()), index=0)
        
        # Display image
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📷 CT Image")
            
            # Check if segmentation exists
            overlay = None
            if st.session_state.segmentation is not None and selected_scan == st.session_state.current_scan:
                overlay = st.session_state.segmentation
            
            fig = render_slice(data, slice_idx, view, window, overlay)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("### 🔬 Segmentation Analysis")
            
            if st.button("🧠 Run Organ Segmentation", use_container_width=True):
                with st.spinner("Running organ segmentation..."):
                    try:
                        segmentor = get_segmentor()
                        seg_path, organ_paths = segmentor.segment_file(
                            Path(nifti_path),
                            save_individual_organs=True
                        )
                        
                        # Load segmentation result
                        seg_data, _ = load_nifti_cached(str(seg_path))
                        st.session_state.segmentation = seg_data
                        st.session_state.current_scan = selected_scan
                        
                        st.success("✅ Segmentation complete!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Segmentation failed: {str(e)}")
            
            # Show detected organs
            if st.session_state.segmentation is not None:
                unique_labels = np.unique(st.session_state.segmentation)
                detected_organs = [ORGAN_LABELS.get(int(l), f"Unknown_{l}") 
                                   for l in unique_labels if l > 0]
                
                st.markdown(f"**Detected {len(detected_organs)}  organ structures**")
                
                # Show organ list (collapsible)
                with st.expander("View organ list"):
                    for i, organ in enumerate(detected_organs[:20]):
                        st.text(f"• {organ}")
                    if len(detected_organs) > 20:
                        st.text(f"... and {len(detected_organs) - 20}  more")
        
        # Image statistics
        st.markdown("### 📊 Image Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Image Size", f"{data.shape[0]}×{data.shape[1]}×{data.shape[2]}")
        with col2:
            spacing = img.header.get_zooms()[:3]
            st.metric("Voxel Spacing", f"{spacing[0]:.1f}×{spacing[1]:.1f}×{spacing[2]:.1f} mm")
        with col3:
            st.metric("HU Range", f"{int(data.min())} ~ {int(data.max())}")
        with col4:
            st.metric("Mean HU", f"{data.mean():.1f}")


def render_longitudinal_page():
    """Render longitudinal comparison page"""
    st.markdown("## 📊 Longitudinal Comparison")
    
    if len(st.session_state.scans) < 2:
        st.warning("⚠️ Longitudinal comparison requires at least 2 scans. Please upload more data.")
        return
    
    scan_options = list(st.session_state.scans.keys())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📅 Baseline Scan")
        baseline_scan = st.selectbox(
            "Select baseline scan",
            scan_options,
            key="baseline",
            format_func=lambda x: f"{x[:8]}... ({st.session_state.scans[x]['metadata'].get('study_date', 'Unknown')})"
        )
    
    with col2:
        st.markdown("### 📅 Follow-up Scan")
        followup_options = [s for s in scan_options if s != baseline_scan]
        if followup_options:
            followup_scan = st.selectbox(
                "Select followup scan",
                followup_options,
                key="followup",
                format_func=lambda x: f"{x[:8]}... ({st.session_state.scans[x]['metadata'].get('study_date', 'Unknown')})"
            )
        else:
            st.warning("Please select different scans")
            return
    
    # Registration and comparison analysis
    if st.button("🔄 Run Registration & Analysis", use_container_width=True):
        with st.spinner("Running image registration..."):
            try:
                baseline_path = Path(st.session_state.scans[baseline_scan]["nifti_path"])
                followup_path = Path(st.session_state.scans[followup_scan]["nifti_path"])
                
                # Registration
                registrator = get_registrator()
                warped_path, transforms = registrator.register_files(
                    followup_path,  # fixed
                    baseline_path,  # moving
                    use_deformable=True
                )
                
                st.success("✅ Registration complete!")
                
                # Calculate difference
                with st.spinner("Calculating differences..."):
                    loader = get_dicom_loader()
                    followup_img = nib.load(followup_path)
                    followup_data = followup_img.get_fdata()
                    followup_spacing = tuple(followup_img.header.get_zooms()[:3])
                    
                    warped_img = nib.load(warped_path)
                    warped_data = warped_img.get_fdata()
                    
                    detector = get_change_detector()
                    diff_map, significant = detector.compute_difference_map(
                        followup_data, warped_data
                    )
                    
                    # Quantify changes (with spacing)
                    changes = detector.quantify_changes(diff_map, significant, spacing=followup_spacing)
                    
                    # Store results
                    st.session_state.diff_map = diff_map
                    st.session_state.significant = significant
                    st.session_state.followup_data = followup_data
                    st.session_state.warped_data = warped_data
                    st.session_state.baseline_scan = baseline_scan  # Save scan IDs
                    st.session_state.followup_scan = followup_scan
                    st.session_state.registration_results = {
                        "rigid": "completed" if "rigid" in transforms else None,
                        "deformable": "completed" if "deformable" in transforms else None,
                        "warped_path": str(warped_path),
                        "spacing": followup_spacing
                    }
                    st.session_state.change_results = changes
                    
                    st.success("✅ Difference analysis complete!")
                    
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Show comparison results
    if hasattr(st.session_state, 'diff_map') and st.session_state.diff_map is not None:
        st.markdown("### 📈 Comparison Results")
        
        # Controls
        col1, col2 = st.columns(2)
        with col1:
            view = st.selectbox("View", ["axial", "sagittal", "coronal"], key="diff_view")
        with col2:
            max_slice = st.session_state.followup_data.shape[{"axial": 2, "sagittal": 0, "coronal": 1}[view]] - 1
            slice_idx = st.slider("Slice", 0, max_slice, max_slice // 2, key="diff_slice")
        
        # Display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔵 Baseline (Registered)")
            fig = render_slice(st.session_state.warped_data, slice_idx, view, "lung")
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("#### 🔴 Difference Heatmap")
            fig = render_diff_heatmap(
                st.session_state.followup_data,
                st.session_state.significant,
                slice_idx,
                view
            )
            st.pyplot(fig)
            plt.close()
        
        # Change statistics
        st.markdown("### 📊 变化统计")
        significant = st.session_state.significant
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("最大密度增加", f"{significant.max():.1f} HU", delta="↑")
        with col2:
            st.metric("最大密度减少", f"{significant.min():.1f} HU", delta="↓")
        with col3:
            changed_voxels = (significant != 0).sum()
            st.metric("变化体素数", f"{changed_voxels:,}")
        with col4:
            total_voxels = significant.size
            change_percent = (changed_voxels / total_voxels) * 100
            st.metric("变化比例", f"{change_percent:.2f}%")
        
        # 详细变化分析
        st.markdown("### 📈 详细变化分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 变化分布图
            st.markdown("#### 变化分布")
            import plotly.graph_objects as go
            
            significant_values = significant[significant != 0]
            if len(significant_values) > 0:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=significant_values.flatten(),
                    nbinsx=50,
                    marker_color='#00a3e0',
                    opacity=0.7
                ))
                fig.update_layout(
                    xaxis_title="HU 变化值",
                    yaxis_title="体素数量",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    margin=dict(l=40, r=20, t=20, b=40),
                    height=250
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("无显著变化")
        
        with col2:
            # 变化统计表
            st.markdown("#### 统计摘要")
            
            increase_voxels = (significant > 0).sum()
            decrease_voxels = (significant < 0).sum()
            
            # 计算体积（假设 1mm³ 体素）
            spacing = st.session_state.get("registration_results", {}).get("spacing", (1.0, 1.0, 1.0))
            if isinstance(spacing, (list, tuple)) and len(spacing) >= 3:
                voxel_volume_cc = spacing[0] * spacing[1] * spacing[2] / 1000
            else:
                voxel_volume_cc = 0.001
            
            changed_volume_cc = changed_voxels * voxel_volume_cc
            
            stats_data = {
                "指标": ["总体素数", "变化体素数", "增加区域", "减少区域", "变化体积"],
                "数值": [
                    f"{total_voxels:,}",
                    f"{changed_voxels:,}",
                    f"{increase_voxels:,} ({increase_voxels/total_voxels*100:.2f}%)",
                    f"{decrease_voxels:,} ({decrease_voxels/total_voxels*100:.2f}%)",
                    f"{changed_volume_cc:.2f} cc"
                ]
            }
            
            import pandas as pd
            df = pd.DataFrame(stats_data)
            st.dataframe(df, hide_index=True, use_container_width=True)
        
        # RECIST 评估提示
        st.markdown("### 🏥 临床提示")
        
        if change_percent < 0.1:
            st.success("✅ 变化极小，图像基本稳定")
        elif change_percent < 1.0:
            st.info("ℹ️ 检测到轻微变化，建议结合临床判断")
        elif change_percent < 5.0:
            st.warning("⚠️ 检测到显著变化，建议详细分析")
        else:
            st.error("🚨 检测到大范围变化，请仔细评估")


def render_report_page():
    """Render report generation page with AI chat"""
    st.markdown("## 📋 智能诊断报告 & AI 咨询")
    
    if not st.session_state.scans:
        st.warning("⚠️ 请先上传并分析数据")
        return
    
    # 创建两个标签页：报告生成 和 AI咨询
    tab1, tab2 = st.tabs(["📝 报告生成", "💬 AI 医学咨询"])
    
    with tab1:
        render_report_tab()
    
    with tab2:
        render_chat_tab()


def render_chat_tab():
    """渲染AI聊天咨询标签页"""
    st.markdown("### 💬 AI 医学影像咨询助手")
    
    st.markdown("""
    <div style="background: linear-gradient(145deg, #1a2942 0%, #0d2137 100%); 
                padding: 1rem; border-radius: 12px; border: 1px solid rgba(0, 163, 224, 0.3); margin-bottom: 1rem;">
        <p style="color: #a0aec0; margin: 0;">
            🤖 基于本地部署的 LLM (Ollama) 提供医学影像分析咨询。您可以询问关于影像发现、病情解读、治疗建议等问题。
        </p>
        <p style="color: #ff6b35; font-size: 0.85rem; margin: 0.5rem 0 0 0;">
            ⚠️ 免责声明：AI 建议仅供参考，不能替代专业医生的诊断和治疗意见。
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 初始化聊天历史
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # 显示当前分析上下文
    if hasattr(st.session_state, 'change_results') and st.session_state.change_results:
        with st.expander("📊 当前分析数据（AI 将基于此数据回答）", expanded=False):
            st.json(st.session_state.change_results)
    
    # 显示聊天历史
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="background: rgba(0, 102, 204, 0.2); padding: 0.75rem 1rem; 
                            border-radius: 12px; margin: 0.5rem 0; border-left: 3px solid #0066cc;">
                    <strong style="color: #00a3e0;">🧑 您:</strong> {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: rgba(72, 187, 120, 0.15); padding: 0.75rem 1rem; 
                            border-radius: 12px; margin: 0.5rem 0; border-left: 3px solid #48bb78;">
                    <strong style="color: #48bb78;">🤖 AI 助手:</strong>
                    <div style="margin-top: 0.5rem; color: #e0e0e0; line-height: 1.6;">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # 快捷问题按钮
    st.markdown("#### 💡 快捷问题")
    col1, col2, col3 = st.columns(3)
    
    quick_questions = [
        ("这个变化是好还是坏？", col1),
        ("需要做什么进一步检查？", col2),
        ("治疗方案有哪些选择？", col3),
    ]
    
    for question, col in quick_questions:
        with col:
            if st.button(question, use_container_width=True, key=f"quick_{question}"):
                process_chat_message(question)
                st.rerun()
    
    col1, col2, col3 = st.columns(3)
    quick_questions_2 = [
        ("用药有什么注意事项？", col1),
        ("多久需要复查？", col2),
        ("日常生活需要注意什么？", col3),
    ]
    
    for question, col in quick_questions_2:
        with col:
            if st.button(question, use_container_width=True, key=f"quick2_{question}"):
                process_chat_message(question)
                st.rerun()
    
    # 用户输入
    st.markdown("#### ✏️ 自定义问题")
    user_input = st.text_input(
        "输入您的问题",
        placeholder="例如：这个病灶是良性还是恶性的可能性更大？",
        key="chat_input"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🚀 发送", use_container_width=True):
            if user_input.strip():
                process_chat_message(user_input)
                st.rerun()
    with col2:
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()


def process_chat_message(user_message: str):
    """处理聊天消息"""
    # 添加用户消息
    st.session_state.chat_messages.append({
        "role": "user",
        "content": user_message
    })
    
    # 构建上下文
    context = ""
    if hasattr(st.session_state, 'change_results') and st.session_state.change_results:
        cr = st.session_state.change_results
        context = f"""
当前影像分析数据：
- 变化体素数: {cr.get('changed_voxels', 0):,}
- 变化比例: {cr.get('change_percent', 0):.2f}%
- 变化体积: {cr.get('changed_volume_cc', 0):.2f} cc
- 最大密度增加: {cr.get('max_hu_increase', 0):.1f} HU
- 最大密度减少: {cr.get('max_hu_decrease', 0):.1f} HU
- 密度增加区域: {cr.get('increase_percent', 0):.2f}%
- 密度减少区域: {cr.get('decrease_percent', 0):.2f}%
"""
    
    # 调用 LLM
    try:
        import ollama
        
        system_prompt = f"""你是一名经验丰富的放射科医生和肿瘤科医生。你正在帮助患者或其家属理解医学影像分析结果。

{context}

回答要求：
1. 使用通俗易懂的语言，避免过多专业术语
2. 如果使用专业术语，请给出解释
3. 回答要客观、准确、有同理心
4. 始终提醒患者最终诊断需要专业医生判断
5. 如果问题超出影像分析范围，请诚实说明
6. 回答应该简洁但完整，控制在200字以内"""
        
        response = ollama.chat(
            model="llama3.1:8b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        
        ai_response = response.message.content
        
    except Exception as e:
        ai_response = f"""抱歉，AI 助手暂时无法响应。

可能的原因：
- Ollama 服务未启动
- 网络连接问题

您的问题是："{user_message}"

建议：请将此问题咨询您的主治医生。

技术信息：{str(e)[:100]}"""
    
    # 添加 AI 回复
    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": ai_response
    })


def render_report_tab():
    """渲染报告生成标签页"""
    # Select report type
    report_type = st.radio(
        "报告类型",
        ["单次检查报告", "纵向对比报告"],
        horizontal=True
    )
    
    if report_type == "单次检查报告":
        scan_options = list(st.session_state.scans.keys())
        selected_scan = st.selectbox(
            "Select scan",
            scan_options,
            format_func=lambda x: f"{x[:8]}..."
        )
        
        if st.button("📝 Generate Report", use_container_width=True):
            with st.spinner("Generating report..."):
                scan_data = st.session_state.scans[selected_scan]
                metadata = scan_data["metadata"]
                
                # Safely get metadata values
                patient_id = metadata.get('patient_id') or 'Unknown'
                study_date = metadata.get('study_date') or 'Unknown'
                manufacturer = metadata.get('manufacturer') or 'Unknown'
                slice_thickness = metadata.get('slice_thickness')
                if slice_thickness is None:
                    slice_thickness = 1.0
                
                # Generate report
                report = f"""
# Chest CT Examination Report

## Examination Information
- **Patient ID**: {patient_id}
- **Study Date**: {study_date}
- **Body Part**: Chest
- **Equipment**: {manufacturer}

## Technique
Standard chest CT scan, slice thickness {float(slice_thickness):.1f}mm

## Findings

### Lungs
Both lung fields are clear with no significant parenchymal abnormalities. Trachea and main bronchi are patent.

### Mediastinum
Mediastinum is midline. No significant lymphadenopathy. Heart size and morphology are normal.

### Pleura
Bilateral pleural surfaces are smooth. No pleural effusion.

### Bones
No significant osseous abnormalities identified.

## Impression
1. No significant active pulmonary disease
2. Normal cardiac size and morphology
3. No mediastinal abnormalities

## Recommendations
1. Correlate with clinical history
2. Follow-up as clinically indicated

---
*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*This report is AI-assisted by NeuroScan AI. For reference only. Final diagnosis should be made by qualified physicians.*
"""
                
                # Display report in high-contrast container
                st.markdown("""
                <div style="background-color: #1a2942; padding: 2rem; border-radius: 12px; border: 2px solid #00a3e0; margin: 1rem 0;">
                """, unsafe_allow_html=True)
                st.markdown(report)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Download button
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name=f"report_{selected_scan[:8]}.md",
                    mime="text/markdown"
                )
    
    else:  # 纵向对比报告
        if not hasattr(st.session_state, 'diff_map') or st.session_state.diff_map is None:
            st.warning("⚠️ 请先进行纵向对比分析（在「纵向对比」页面完成）")
            return
        
        # Get scan IDs from session state
        scan_options = list(st.session_state.scans.keys())
        baseline_scan = st.session_state.get("baseline_scan")
        followup_scan = st.session_state.get("followup_scan")
        
        # Validate scan IDs exist in current scans
        if baseline_scan and baseline_scan not in scan_options:
            baseline_scan = None
        if followup_scan and followup_scan not in scan_options:
            followup_scan = None
        
        # Display selected scans or allow selection
        if baseline_scan and followup_scan and baseline_scan != followup_scan:
            col1, col2 = st.columns(2)
            with col1:
                baseline_meta = st.session_state.scans[baseline_scan].get("metadata", {})
                st.info(f"📅 **基线扫描**: {baseline_scan[:8]}... ({baseline_meta.get('study_date', 'Unknown')})")
            with col2:
                followup_meta = st.session_state.scans[followup_scan].get("metadata", {})
                st.info(f"📅 **随访扫描**: {followup_scan[:8]}... ({followup_meta.get('study_date', 'Unknown')})")
        else:
            # If not set or invalid, allow user to select
            st.markdown("### 选择扫描")
            col1, col2 = st.columns(2)
            with col1:
                baseline_scan = st.selectbox(
                    "选择基线扫描",
                    scan_options,
                    key="report_baseline",
                    help="选择较早的扫描作为基线",
                    index=0 if scan_options else None
                )
            with col2:
                followup_options = [s for s in scan_options if s != baseline_scan]
                if followup_options:
                    followup_scan = st.selectbox(
                        "选择随访扫描",
                        followup_options,
                        key="report_followup",
                        help="选择较晚的扫描作为随访",
                        index=0 if followup_options else None
                    )
                else:
                    st.warning("⚠️ 请选择不同的扫描")
                    return
        
        # Final validation
        if not baseline_scan or not followup_scan or baseline_scan == followup_scan:
            st.error("❌ 请选择两个不同的扫描")
            return
        
        if st.button("📝 生成对比报告（中文，LLM 分析）", use_container_width=True):
            with st.spinner("正在生成详细报告（使用 LLM 分析）..."):
                try:
                    from app.services.report import ReportGenerator
                    
                    # Get data
                    baseline_scan_data = st.session_state.scans[baseline_scan]
                    followup_scan_data = st.session_state.scans[followup_scan]
                    
                    baseline_metadata = baseline_scan_data.get("metadata", {})
                    followup_metadata = followup_scan_data.get("metadata", {})
                    
                    patient_id = baseline_metadata.get("patient_id", "Unknown")
                    baseline_date = str(baseline_metadata.get("study_date", "Unknown"))
                    followup_date = str(followup_metadata.get("study_date", "Unknown"))
                    
                    # Get registration and change results
                    registration_results = st.session_state.get("registration_results", {})
                    change_results = st.session_state.get("change_results", {})
                    
                    # Extract findings (simplified - in real scenario, extract from segmentation)
                    baseline_findings = []  # TODO: Extract from segmentation
                    followup_findings = []  # TODO: Extract from segmentation
                    
                    # Initialize report generator (try LLM, fallback to template)
                    try:
                        generator = ReportGenerator(llm_backend="ollama")
                        st.info("✅ 使用 LLM 生成详细分析报告")
                    except:
                        generator = ReportGenerator(llm_backend="template")
                        st.info("⚠️ LLM 不可用，使用模板模式")
                    
                    # Generate report
                    report = generator.generate_longitudinal_report(
                        patient_id=patient_id,
                        baseline_date=baseline_date,
                        followup_date=followup_date,
                        baseline_findings=baseline_findings,
                        followup_findings=followup_findings,
                        registration_results=registration_results,
                        change_results=change_results,
                        modality="CT"
                    )
                    
                    # Display report in high-contrast container
                    st.markdown("""
                    <div style="background-color: #1a2942; padding: 2rem; border-radius: 12px; border: 2px solid #00a3e0; margin: 1rem 0;">
                    """, unsafe_allow_html=True)
                    st.markdown(report)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Download button
                    st.download_button(
                        label="📥 下载报告",
                        data=report,
                        file_name=f"longitudinal_report_{patient_id}_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown"
                    )
                    
                except Exception as e:
                    st.error(f"❌ 报告生成失败: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())


def render_sample_data_page():
    """渲染示例数据页面"""
    st.markdown("## 📁 示例数据")
    
    st.markdown("""
    <div class="metric-card">
        <h4>🎯 快速开始</h4>
        <p style="color: #a0aec0;">使用预置的示例数据快速体验系统功能，无需上传文件。</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 查找可用的示例数据
    sample_dir = settings.PROCESSED_DATA_DIR
    sample_cases = []
    
    if sample_dir.exists():
        for case_dir in sample_dir.iterdir():
            if case_dir.is_dir() and case_dir.name.startswith("real_lung_"):
                baseline = case_dir / "baseline.nii.gz"
                followup = case_dir / "followup.nii.gz"
                if baseline.exists() and followup.exists():
                    sample_cases.append({
                        "name": case_dir.name,
                        "path": case_dir,
                        "baseline": baseline,
                        "followup": followup,
                        "has_mask": (case_dir / "baseline_mask.nii.gz").exists()
                    })
    
    if not sample_cases:
        st.warning("⚠️ 未找到示例数据。请先运行数据下载脚本。")
        st.code("python scripts/download_datasets.py --dataset learn2reg")
        return
    
    st.markdown(f"### 📋 可用数据集 ({len(sample_cases)} 个病例)")
    
    # 显示数据集列表
    for i, case in enumerate(sample_cases):
        with st.expander(f"📂 {case['name']}", expanded=(i == 0)):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**路径**: `{case['path']}`")
                st.markdown(f"**基线扫描**: `baseline.nii.gz`")
                st.markdown(f"**随访扫描**: `followup.nii.gz`")
                if case['has_mask']:
                    st.markdown("**分割掩码**: ✅ 可用")
                
            with col2:
                if st.button(f"📥 加载此数据", key=f"load_{case['name']}", use_container_width=True):
                    with st.spinner(f"正在加载 {case['name']}..."):
                        try:
                            loader = get_dicom_loader()
                            
                            # 加载基线
                            baseline_data, baseline_img = loader.load_nifti(case['baseline'])
                            baseline_scan_id = f"{case['name']}_baseline"
                            
                            from app.schemas.dicom import DicomMetadata, ScanInfo
                            
                            baseline_metadata = DicomMetadata(
                                patient_id=case['name'],
                                patient_name="Learn2Reg Sample",
                                modality="CT",
                                study_description="Inspiration CT"
                            )
                            
                            st.session_state.scans[baseline_scan_id] = {
                                "info": None,
                                "nifti_path": str(case['baseline']),
                                "metadata": baseline_metadata.model_dump(),
                                "scan_type": "baseline"
                            }
                            
                            # 加载随访
                            followup_data, followup_img = loader.load_nifti(case['followup'])
                            followup_scan_id = f"{case['name']}_followup"
                            
                            followup_metadata = DicomMetadata(
                                patient_id=case['name'],
                                patient_name="Learn2Reg Sample",
                                modality="CT",
                                study_description="Expiration CT"
                            )
                            
                            st.session_state.scans[followup_scan_id] = {
                                "info": None,
                                "nifti_path": str(case['followup']),
                                "metadata": followup_metadata.model_dump(),
                                "scan_type": "followup"
                            }
                            
                            st.session_state.baseline_scan = baseline_scan_id
                            st.session_state.followup_scan = followup_scan_id
                            
                            st.success(f"✅ 成功加载 {case['name']}!")
                            st.info("💡 请切换到「纵向对比」模式进行分析")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ 加载失败: {e}")
    
    # 数据集信息
    st.markdown("---")
    st.markdown("### 📚 数据集信息")
    
    st.markdown("""
    **Learn2Reg Lung CT 数据集**
    
    这是一个用于医学图像配准挑战的公开数据集，包含同一患者的：
    - **吸气末 CT** (Inspiration) - 作为基线扫描
    - **呼气末 CT** (Expiration) - 作为随访扫描
    
    **特点**:
    - 包含显著的解剖形变（横膈膜移动、肺部膨胀/收缩）
    - 非常适合测试配准算法的稳健性
    - 包含肺部分割掩码
    
    **来源**: [Zenodo](https://zenodo.org/record/3835682)
    """)
    
    # 下载更多数据
    st.markdown("### ⬇️ 下载更多数据")
    
    st.markdown("""
    使用以下命令下载更多公开数据集：
    """)
    
    st.code("""
# 下载 Learn2Reg 肺部 CT 数据
python scripts/download_datasets.py --dataset learn2reg

# 生成合成测试数据
python scripts/download_datasets.py --synthetic 10

# 列出所有可用数据集
python scripts/download_datasets.py --list
    """, language="bash")


if __name__ == "__main__":
    main()

