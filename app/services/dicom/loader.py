"""
DICOM 加载器 - 基于 MONAI 和 pydicom
实现 DICOM 转 NIfTI 功能

支持的输入格式:
- DICOM ZIP 压缩包 (.zip)
- NIfTI 文件 (.nii, .nii.gz)
- DICOM 文件夹 (包含 .dcm 文件)
- 单个 DICOM 文件 (.dcm)
- NRRD 文件 (.nrrd)
- MHA/MHD 文件 (.mha, .mhd)
"""
import os
import zipfile
import tarfile
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union
from datetime import datetime
import uuid
import gzip

import numpy as np
import pydicom
import nibabel as nib
from scipy import ndimage

from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import DicomLoadError
from app.schemas.dicom import DicomMetadata, ScanInfo


# 支持的文件格式
SUPPORTED_FORMATS = {
    'zip': ['.zip'],
    'tar': ['.tar', '.tar.gz', '.tgz'],
    'nifti': ['.nii', '.nii.gz'],
    'dicom': ['.dcm', '.dicom', '.ima'],
    'nrrd': ['.nrrd'],
    'mha': ['.mha', '.mhd'],
}


class DicomLoader:
    """
    多格式医学影像加载器
    
    支持格式:
    - ZIP/TAR 压缩包 (包含 DICOM 或 NIfTI)
    - NIfTI (.nii, .nii.gz)
    - DICOM 文件夹或单文件 (.dcm)
    - NRRD (.nrrd)
    - MHA/MHD (.mha, .mhd)
    """
    
    def __init__(self):
        """初始化加载器"""
        self._monai_available = False
        try:
            from monai.transforms import (
                Compose,
                LoadImage,
                EnsureChannelFirst,
                Orientation,
                Spacing,
            )
            # 尝试使用 NibabelReader 而不是 ITKReader
            from monai.data import NibabelReader
            self.reader = NibabelReader()
            self.preprocess_transforms = Compose([
                LoadImage(reader=self.reader, image_only=True),
                EnsureChannelFirst(),
                Orientation(axcodes=settings.ORIENTATION),
                Spacing(
                    pixdim=settings.TARGET_SPACING,
                    mode="bilinear"
                ),
            ])
            self._monai_available = True
        except Exception as e:
            logger.warning(f"MONAI 预处理不可用，使用 pydicom 后备方案: {e}")
            self.preprocess_transforms = None
        
    def extract_zip(self, zip_path: Path, extract_to: Path) -> Path:
        """解压 DICOM ZIP 包"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            
            # 找到 DICOM 文件所在目录
            for root, dirs, files in os.walk(extract_to):
                dcm_files = [f for f in files if f.endswith('.dcm') or not '.' in f]
                if dcm_files:
                    return Path(root)
            
            raise DicomLoadError("ZIP 包中未找到 DICOM 文件")
            
        except zipfile.BadZipFile:
            raise DicomLoadError("无效的 ZIP 文件")
        except Exception as e:
            raise DicomLoadError(f"解压失败: {str(e)}")
    
    def parse_dicom_metadata(self, dicom_dir: Path) -> DicomMetadata:
        """解析 DICOM 元数据"""
        dcm_files = list(dicom_dir.glob("**/*.dcm"))
        if not dcm_files:
            # 尝试没有扩展名的文件
            dcm_files = [f for f in dicom_dir.iterdir() if f.is_file() and '.' not in f.name]
        
        if not dcm_files:
            raise DicomLoadError("目录中未找到 DICOM 文件")
        
        # 读取第一个文件获取元数据
        ds = pydicom.dcmread(dcm_files[0], force=True)
        
        # 解析日期
        study_date = None
        if hasattr(ds, 'StudyDate') and ds.StudyDate:
            try:
                study_date = datetime.strptime(ds.StudyDate, '%Y%m%d')
            except ValueError:
                pass
        
        # 解析像素间距
        pixel_spacing = None
        if hasattr(ds, 'PixelSpacing'):
            pixel_spacing = [float(x) for x in ds.PixelSpacing]
        
        return DicomMetadata(
            patient_id=getattr(ds, 'PatientID', 'Unknown'),
            patient_name=str(getattr(ds, 'PatientName', '')),
            study_date=study_date,
            study_description=getattr(ds, 'StudyDescription', ''),
            series_description=getattr(ds, 'SeriesDescription', ''),
            modality=getattr(ds, 'Modality', 'CT'),
            manufacturer=getattr(ds, 'Manufacturer', ''),
            slice_thickness=float(getattr(ds, 'SliceThickness', 0)) if hasattr(ds, 'SliceThickness') else None,
            pixel_spacing=pixel_spacing,
            image_count=len(dcm_files)
        )
    
    def _load_dicom_series_pydicom(self, dicom_dir: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        使用 pydicom 加载 DICOM 序列
        
        Args:
            dicom_dir: DICOM 文件目录
            
        Returns:
            (图像数组, 元数据字典)
        """
        # 找到所有 DICOM 文件
        dcm_files = list(dicom_dir.glob("**/*.dcm"))
        if not dcm_files:
            # 尝试没有扩展名的文件
            dcm_files = [f for f in dicom_dir.rglob("*") if f.is_file() and '.' not in f.name]
        
        if not dcm_files:
            raise DicomLoadError("目录中未找到 DICOM 文件")
        
        # 读取所有切片
        slices = []
        for dcm_file in dcm_files:
            try:
                ds = pydicom.dcmread(dcm_file, force=True)
                if hasattr(ds, 'pixel_array'):
                    slices.append(ds)
            except Exception:
                continue
        
        if not slices:
            raise DicomLoadError("无法读取任何 DICOM 切片")
        
        # 按位置排序
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]) if hasattr(x, 'ImagePositionPatient') else float(x.InstanceNumber))
        except Exception:
            slices.sort(key=lambda x: float(getattr(x, 'InstanceNumber', 0)))
        
        # 获取像素间距
        ds = slices[0]
        pixel_spacing = [1.0, 1.0]
        if hasattr(ds, 'PixelSpacing'):
            pixel_spacing = [float(x) for x in ds.PixelSpacing]
        
        slice_thickness = 1.0
        if hasattr(ds, 'SliceThickness'):
            slice_thickness = float(ds.SliceThickness)
        elif len(slices) > 1:
            # 从位置计算
            try:
                z1 = float(slices[0].ImagePositionPatient[2])
                z2 = float(slices[1].ImagePositionPatient[2])
                slice_thickness = abs(z2 - z1)
            except Exception:
                pass
        
        # 堆叠切片
        pixel_arrays = []
        for s in slices:
            arr = s.pixel_array.astype(np.float32)
            # 应用 rescale
            if hasattr(s, 'RescaleSlope') and hasattr(s, 'RescaleIntercept'):
                arr = arr * float(s.RescaleSlope) + float(s.RescaleIntercept)
            pixel_arrays.append(arr)
        
        volume = np.stack(pixel_arrays, axis=-1)
        
        metadata = {
            'spacing': (pixel_spacing[0], pixel_spacing[1], slice_thickness),
            'shape': volume.shape,
            'dtype': str(volume.dtype)
        }
        
        return volume, metadata
    
    def _resample_volume(
        self, 
        volume: np.ndarray, 
        original_spacing: Tuple[float, ...],
        target_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)
    ) -> np.ndarray:
        """
        重采样体积数据
        
        Args:
            volume: 输入体积
            original_spacing: 原始间距
            target_spacing: 目标间距
            
        Returns:
            重采样后的体积
        """
        # 计算缩放因子
        resize_factor = np.array(original_spacing) / np.array(target_spacing)
        new_shape = np.round(volume.shape * resize_factor).astype(int)
        
        # 使用 scipy 进行重采样
        resampled = ndimage.zoom(volume, resize_factor, order=1)
        
        return resampled
    
    def convert_to_nifti(
        self, 
        dicom_dir: Path, 
        output_path: Path,
        apply_preprocessing: bool = True
    ) -> Tuple[Path, np.ndarray]:
        """
        将 DICOM 目录转换为 NIfTI 格式
        
        Args:
            dicom_dir: DICOM 文件目录
            output_path: 输出 NIfTI 文件路径
            apply_preprocessing: 是否应用预处理
            
        Returns:
            (NIfTI 文件路径, 图像数组)
        """
        try:
            logger.info(f"开始转换 DICOM: {dicom_dir}")
            
            # 尝试使用 MONAI
            if self._monai_available and apply_preprocessing:
                try:
                    from monai.transforms import LoadImage
                    image_data = self.preprocess_transforms(str(dicom_dir))
                    
                    # 转换为 numpy 数组
                    if hasattr(image_data, 'numpy'):
                        arr = image_data.numpy()
                    else:
                        arr = np.array(image_data)
                    
                    # 移除 channel 维度 (如果存在)
                    if arr.ndim == 4 and arr.shape[0] == 1:
                        arr = arr[0]
                except Exception as e:
                    logger.warning(f"MONAI 加载失败，使用 pydicom: {e}")
                    arr, metadata = self._load_dicom_series_pydicom(dicom_dir)
                    if apply_preprocessing:
                        arr = self._resample_volume(arr, metadata['spacing'], settings.TARGET_SPACING)
            else:
                # 使用 pydicom 后备方案
                arr, metadata = self._load_dicom_series_pydicom(dicom_dir)
                if apply_preprocessing:
                    arr = self._resample_volume(arr, metadata['spacing'], settings.TARGET_SPACING)
            
            # 创建 NIfTI 图像
            nifti_img = nib.Nifti1Image(arr, affine=np.eye(4))
            
            # 设置体素大小
            nifti_img.header.set_zooms(settings.TARGET_SPACING)
            
            # 保存
            output_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nifti_img, output_path)
            
            logger.info(f"NIfTI 保存成功: {output_path}, shape: {arr.shape}")
            return output_path, arr
            
        except Exception as e:
            logger.error(f"DICOM 转换失败: {str(e)}")
            raise DicomLoadError(f"DICOM 转 NIfTI 失败: {str(e)}")
    
    def process_upload(
        self, 
        zip_file_path: Path,
        patient_id: Optional[str] = None
    ) -> ScanInfo:
        """
        处理上传的 DICOM ZIP 包
        
        Args:
            zip_file_path: ZIP 文件路径
            patient_id: 可选的患者ID
            
        Returns:
            ScanInfo 对象
        """
        scan_id = str(uuid.uuid4())
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 解压
            dicom_dir = self.extract_zip(zip_file_path, temp_path)
            
            # 解析元数据
            metadata = self.parse_dicom_metadata(dicom_dir)
            if patient_id:
                metadata.patient_id = patient_id
            
            # 创建存储目录
            raw_dir = settings.RAW_DATA_DIR / metadata.patient_id / scan_id
            raw_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制原始 DICOM
            shutil.copytree(dicom_dir, raw_dir / "dicom", dirs_exist_ok=True)
            
            # 转换为 NIfTI
            nifti_path = settings.PROCESSED_DATA_DIR / metadata.patient_id / f"{scan_id}.nii.gz"
            self.convert_to_nifti(dicom_dir, nifti_path)
            
        return ScanInfo(
            scan_id=scan_id,
            metadata=metadata,
            nifti_path=str(nifti_path),
            raw_path=str(raw_dir),
            status="completed"
        )
    
    def load_nifti(self, nifti_path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
        """加载 NIfTI 文件"""
        img = nib.load(nifti_path)
        data = img.get_fdata()
        return data, img
    
    def detect_file_type(self, file_path: Path) -> str:
        """
        检测文件类型
        
        Returns:
            'zip', 'tar', 'nifti', 'dicom', 'nrrd', 'mha', 'folder', 'unknown'
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        name = file_path.name.lower()
        
        # 检查 .nii.gz 特殊情况
        if name.endswith('.nii.gz'):
            return 'nifti'
        
        # 检查 .tar.gz 特殊情况
        if name.endswith('.tar.gz') or name.endswith('.tgz'):
            return 'tar'
        
        # 按后缀判断
        for fmt, extensions in SUPPORTED_FORMATS.items():
            if suffix in extensions:
                return fmt
        
        # 检查是否是文件夹
        if file_path.is_dir():
            return 'folder'
        
        # 尝试读取文件头判断
        try:
            with open(file_path, 'rb') as f:
                header = f.read(132)
                # DICOM 文件通常在 128 字节后有 'DICM' 标识
                if len(header) >= 132 and header[128:132] == b'DICM':
                    return 'dicom'
                # ZIP 文件以 PK 开头
                if header[:2] == b'PK':
                    return 'zip'
                # GZIP 文件
                if header[:2] == b'\x1f\x8b':
                    return 'gzip'
        except Exception:
            pass
        
        return 'unknown'
    
    def extract_archive(self, archive_path: Path, extract_to: Path) -> Path:
        """
        解压压缩包 (支持 ZIP 和 TAR)
        
        Returns:
            解压后的目录路径
        """
        file_type = self.detect_file_type(archive_path)
        
        if file_type == 'zip':
            return self.extract_zip(archive_path, extract_to)
        elif file_type == 'tar':
            return self._extract_tar(archive_path, extract_to)
        else:
            raise DicomLoadError(f"不支持的压缩格式: {archive_path.suffix}")
    
    def _extract_tar(self, tar_path: Path, extract_to: Path) -> Path:
        """解压 TAR 包"""
        try:
            with tarfile.open(tar_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
            
            # 找到数据文件所在目录
            for root, dirs, files in os.walk(extract_to):
                # 检查是否有 DICOM 或 NIfTI 文件
                dcm_files = [f for f in files if f.endswith('.dcm') or not '.' in f]
                nii_files = [f for f in files if f.endswith('.nii') or f.endswith('.nii.gz')]
                if dcm_files or nii_files:
                    return Path(root)
            
            return extract_to
            
        except Exception as e:
            raise DicomLoadError(f"TAR 解压失败: {str(e)}")
    
    def load_nrrd(self, nrrd_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        加载 NRRD 文件
        
        Returns:
            (图像数组, 元数据字典)
        """
        try:
            import nrrd
            data, header = nrrd.read(str(nrrd_path))
            
            # 提取间距信息
            spacing = [1.0, 1.0, 1.0]
            if 'space directions' in header:
                dirs = header['space directions']
                spacing = [np.linalg.norm(d) for d in dirs if d is not None]
            elif 'spacings' in header:
                spacing = list(header['spacings'])
            
            metadata = {
                'spacing': tuple(spacing[:3]),
                'shape': data.shape,
                'dtype': str(data.dtype)
            }
            
            return data.astype(np.float32), metadata
            
        except ImportError:
            raise DicomLoadError("需要安装 pynrrd: pip install pynrrd")
        except Exception as e:
            raise DicomLoadError(f"NRRD 加载失败: {str(e)}")
    
    def load_mha(self, mha_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        加载 MHA/MHD 文件 (使用 SimpleITK)
        
        Returns:
            (图像数组, 元数据字典)
        """
        try:
            import SimpleITK as sitk
            img = sitk.ReadImage(str(mha_path))
            data = sitk.GetArrayFromImage(img)
            
            # SimpleITK 返回 ZYX 顺序，转换为 XYZ
            data = np.transpose(data, (2, 1, 0))
            
            spacing = img.GetSpacing()
            
            metadata = {
                'spacing': spacing,
                'shape': data.shape,
                'dtype': str(data.dtype),
                'origin': img.GetOrigin(),
                'direction': img.GetDirection()
            }
            
            return data.astype(np.float32), metadata
            
        except ImportError:
            raise DicomLoadError("需要安装 SimpleITK: pip install SimpleITK")
        except Exception as e:
            raise DicomLoadError(f"MHA 加载失败: {str(e)}")
    
    def load_any_format(
        self, 
        file_path: Path,
        apply_preprocessing: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        自动检测并加载任意支持的格式
        
        Args:
            file_path: 文件或目录路径
            apply_preprocessing: 是否应用预处理 (重采样)
            
        Returns:
            (图像数组, 元数据字典)
        """
        file_path = Path(file_path)
        file_type = self.detect_file_type(file_path)
        
        logger.info(f"检测到文件类型: {file_type}, 路径: {file_path}")
        
        if file_type == 'nifti':
            # 直接加载 NIfTI
            img = nib.load(file_path)
            data = img.get_fdata().astype(np.float32)
            spacing = img.header.get_zooms()[:3]
            metadata = {
                'spacing': spacing,
                'shape': data.shape,
                'dtype': str(data.dtype)
            }
            
        elif file_type == 'nrrd':
            data, metadata = self.load_nrrd(file_path)
            
        elif file_type == 'mha':
            data, metadata = self.load_mha(file_path)
            
        elif file_type in ['zip', 'tar']:
            # 解压并加载
            with tempfile.TemporaryDirectory() as temp_dir:
                extract_path = self.extract_archive(file_path, Path(temp_dir))
                # 递归检测解压后的内容
                return self._load_from_directory(extract_path, apply_preprocessing)
            
        elif file_type == 'dicom' or file_type == 'folder':
            # 加载 DICOM
            if file_path.is_file():
                dicom_dir = file_path.parent
            else:
                dicom_dir = file_path
            data, metadata = self._load_dicom_series_pydicom(dicom_dir)
            
        else:
            raise DicomLoadError(f"不支持的文件格式: {file_path}")
        
        # 应用预处理 (重采样)
        if apply_preprocessing and 'spacing' in metadata:
            original_spacing = metadata['spacing']
            if original_spacing != settings.TARGET_SPACING:
                data = self._resample_volume(data, original_spacing, settings.TARGET_SPACING)
                metadata['original_spacing'] = original_spacing
                metadata['spacing'] = settings.TARGET_SPACING
                metadata['shape'] = data.shape
        
        return data, metadata
    
    def _load_from_directory(
        self, 
        directory: Path,
        apply_preprocessing: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """从目录加载数据 (自动检测格式)"""
        directory = Path(directory)
        
        # 查找 NIfTI 文件
        nii_files = list(directory.glob("**/*.nii.gz")) + list(directory.glob("**/*.nii"))
        if nii_files:
            return self.load_any_format(nii_files[0], apply_preprocessing)
        
        # 查找 NRRD 文件
        nrrd_files = list(directory.glob("**/*.nrrd"))
        if nrrd_files:
            return self.load_any_format(nrrd_files[0], apply_preprocessing)
        
        # 查找 MHA 文件
        mha_files = list(directory.glob("**/*.mha")) + list(directory.glob("**/*.mhd"))
        if mha_files:
            return self.load_any_format(mha_files[0], apply_preprocessing)
        
        # 查找 DICOM 文件
        dcm_files = list(directory.glob("**/*.dcm"))
        if not dcm_files:
            # 尝试没有扩展名的文件
            dcm_files = [f for f in directory.rglob("*") if f.is_file() and '.' not in f.name]
        
        if dcm_files:
            data, metadata = self._load_dicom_series_pydicom(dcm_files[0].parent)
            if apply_preprocessing:
                data = self._resample_volume(data, metadata['spacing'], settings.TARGET_SPACING)
                metadata['original_spacing'] = metadata['spacing']
                metadata['spacing'] = settings.TARGET_SPACING
            return data, metadata
        
        raise DicomLoadError(f"目录中未找到支持的医学影像文件: {directory}")
    
    def process_upload_any_format(
        self, 
        file_path: Path,
        patient_id: Optional[str] = None
    ) -> ScanInfo:
        """
        处理任意格式的上传文件
        
        支持:
        - ZIP/TAR 压缩包
        - NIfTI 文件
        - DICOM 文件夹
        - NRRD/MHA 文件
        
        Args:
            file_path: 上传的文件路径
            patient_id: 可选的患者ID
            
        Returns:
            ScanInfo 对象
        """
        scan_id = str(uuid.uuid4())
        file_path = Path(file_path)
        file_type = self.detect_file_type(file_path)
        
        logger.info(f"处理上传文件: {file_path}, 类型: {file_type}")
        
        # 生成患者ID
        if not patient_id:
            patient_id = f"Patient_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 创建存储目录
        raw_dir = settings.RAW_DATA_DIR / patient_id / scan_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        nifti_path = settings.PROCESSED_DATA_DIR / patient_id / f"{scan_id}.nii.gz"
        nifti_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 根据文件类型处理
        if file_type in ['zip', 'tar']:
            # 解压并处理
            with tempfile.TemporaryDirectory() as temp_dir:
                extract_path = self.extract_archive(file_path, Path(temp_dir))
                
                # 尝试解析 DICOM 元数据
                try:
                    metadata = self.parse_dicom_metadata(extract_path)
                except Exception:
                    metadata = DicomMetadata(
                        patient_id=patient_id,
                        patient_name="",
                        modality="CT"
                    )
                
                # 加载并转换
                data, meta = self._load_from_directory(extract_path, apply_preprocessing=True)
                
                # 保存原始文件
                shutil.copy(file_path, raw_dir / file_path.name)
                
        elif file_type == 'nifti':
            # 直接加载 NIfTI
            data, meta = self.load_any_format(file_path, apply_preprocessing=True)
            metadata = DicomMetadata(
                patient_id=patient_id,
                patient_name="",
                modality="CT"
            )
            # 复制原始文件
            shutil.copy(file_path, raw_dir / file_path.name)
            
        elif file_type in ['nrrd', 'mha']:
            # 加载 NRRD/MHA
            data, meta = self.load_any_format(file_path, apply_preprocessing=True)
            metadata = DicomMetadata(
                patient_id=patient_id,
                patient_name="",
                modality="CT"
            )
            shutil.copy(file_path, raw_dir / file_path.name)
            
        elif file_type == 'dicom' or file_type == 'folder':
            # DICOM 处理
            if file_path.is_file():
                dicom_dir = file_path.parent
            else:
                dicom_dir = file_path
            
            metadata = self.parse_dicom_metadata(dicom_dir)
            data, meta = self._load_dicom_series_pydicom(dicom_dir)
            data = self._resample_volume(data, meta['spacing'], settings.TARGET_SPACING)
            
            # 复制原始 DICOM
            shutil.copytree(dicom_dir, raw_dir / "dicom", dirs_exist_ok=True)
            
        else:
            raise DicomLoadError(f"不支持的文件格式: {file_path}")
        
        # 保存为 NIfTI
        nifti_img = nib.Nifti1Image(data, affine=np.eye(4))
        nifti_img.header.set_zooms(settings.TARGET_SPACING)
        nib.save(nifti_img, nifti_path)
        
        logger.info(f"文件处理完成: {nifti_path}, shape: {data.shape}")
        
        if patient_id:
            metadata.patient_id = patient_id
        
        return ScanInfo(
            scan_id=scan_id,
            metadata=metadata,
            nifti_path=str(nifti_path),
            raw_path=str(raw_dir),
            status="completed"
        )


