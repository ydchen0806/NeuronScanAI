"""
图像配准服务 - 基于 SimpleITK/Elastix
实现两级配准策略:
  L1: 刚性变换 (Rigid) - 修正体位差异
  L2: 非刚性变换 (Deformable) - 修正呼吸导致的软组织形变
"""
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
import SimpleITK as sitk
import nibabel as nib

from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import RegistrationError


class ImageRegistrator:
    """图像配准器"""
    
    def __init__(self):
        """初始化配准器"""
        self.iterations = settings.REGISTRATION_ITERATIONS
        self.sampling_rate = settings.REGISTRATION_SAMPLING_RATE
        
    def _numpy_to_sitk(self, image: np.ndarray, spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)) -> sitk.Image:
        """将 numpy 数组转换为 SimpleITK 图像"""
        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
        # 确保 spacing 是 Python float 列表 (SimpleITK 要求)
        spacing_list = [float(s) for s in spacing]
        # SimpleITK 使用 (x, y, z) 顺序，而 numpy 是 (z, y, x)
        # 所以需要反转 spacing
        if len(spacing_list) == 3:
            spacing_list = list(reversed(spacing_list))
        sitk_image.SetSpacing(spacing_list)
        return sitk_image
    
    def _sitk_to_numpy(self, sitk_image: sitk.Image) -> np.ndarray:
        """将 SimpleITK 图像转换为 numpy 数组"""
        return sitk.GetArrayFromImage(sitk_image)
    
    def rigid_registration(
        self,
        fixed_image: np.ndarray,
        moving_image: np.ndarray,
        spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
        fast_mode: bool = True
    ) -> Tuple[np.ndarray, sitk.Transform]:
        """
        Rigid registration (L1)
        
        Args:
            fixed_image: Fixed image (Followup)
            moving_image: Moving image (Baseline)
            spacing: Voxel spacing
            fast_mode: Use faster settings
            
        Returns:
            (Registered image, Transform parameters)
        """
        logger.info("Starting rigid registration (L1)...")
        
        try:
            # Convert to SimpleITK format
            fixed_sitk = self._numpy_to_sitk(fixed_image, spacing)
            moving_sitk = self._numpy_to_sitk(moving_image, spacing)
            
            # Initialize transform (center alignment)
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_sitk,
                moving_sitk,
                sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
            
            # Configure registration method
            registration = sitk.ImageRegistrationMethod()
            
            # Similarity metric: Mutual Information
            registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32 if fast_mode else 50)
            registration.SetMetricSamplingStrategy(registration.RANDOM)
            sampling_rate = 0.05 if fast_mode else self.sampling_rate
            registration.SetMetricSamplingPercentage(sampling_rate)
            
            # Interpolation method
            registration.SetInterpolator(sitk.sitkLinear)
            
            # Optimizer: Gradient descent
            iterations = 100 if fast_mode else self.iterations
            registration.SetOptimizerAsGradientDescent(
                learningRate=1.0,
                numberOfIterations=iterations,
                convergenceMinimumValue=1e-5,
                convergenceWindowSize=5
            )
            registration.SetOptimizerScalesFromPhysicalShift()
            
            # Multi-resolution strategy (2 levels for fast mode)
            if fast_mode:
                registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2])
                registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1])
            else:
                registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
                registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
            registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            
            # Set initial transform
            registration.SetInitialTransform(initial_transform, inPlace=False)
            
            # Execute registration
            final_transform = registration.Execute(fixed_sitk, moving_sitk)
            
            # Apply transform
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed_sitk)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(float(np.min(moving_image)))
            resampler.SetTransform(final_transform)
            
            warped_sitk = resampler.Execute(moving_sitk)
            warped_image = self._sitk_to_numpy(warped_sitk)
            
            logger.info(f"Rigid registration complete, final metric: {registration.GetMetricValue():.6f}")
            
            return warped_image, final_transform
            
        except Exception as e:
            logger.error(f"Rigid registration failed: {str(e)}")
            raise RegistrationError(f"Rigid registration failed: {str(e)}")
    
    def deformable_registration(
        self,
        fixed_image: np.ndarray,
        moving_image: np.ndarray,
        spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
        grid_spacing: Tuple[int, ...] = (80, 80, 80),  # Increased for faster computation
        fast_mode: bool = True  # Enable fast mode by default
    ) -> Tuple[np.ndarray, sitk.Transform]:
        """
        Non-rigid registration (L2) - B-Spline deformation
        
        Args:
            fixed_image: Fixed image (Followup)
            moving_image: Moving image (Baseline, already rigid-registered)
            spacing: Voxel spacing
            grid_spacing: B-Spline control point grid spacing (larger = faster)
            fast_mode: Use faster but less accurate settings
            
        Returns:
            (Registered image, Transform parameters)
        """
        logger.info("Starting deformable registration (L2)...")
        
        try:
            # Convert to SimpleITK format
            fixed_sitk = self._numpy_to_sitk(fixed_image, spacing)
            moving_sitk = self._numpy_to_sitk(moving_image, spacing)
            
            # For fast mode, downsample images first
            if fast_mode:
                # Downsample by factor of 2 for speed
                fixed_sitk = sitk.Shrink(fixed_sitk, [2, 2, 2])
                moving_sitk = sitk.Shrink(moving_sitk, [2, 2, 2])
                logger.info("Fast mode: downsampled images for registration")
            
            # Initialize B-Spline transform with coarser grid
            transform_domain_mesh_size = [
                max(2, int(np.ceil(fixed_sitk.GetSize()[i] * fixed_sitk.GetSpacing()[i] / grid_spacing[i])))
                for i in range(3)
            ]
            
            initial_transform = sitk.BSplineTransformInitializer(
                fixed_sitk,
                transform_domain_mesh_size
            )
            
            # Configure registration method
            registration = sitk.ImageRegistrationMethod()
            
            # Similarity metric: Mutual Information
            registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)  # Reduced bins
            registration.SetMetricSamplingStrategy(registration.RANDOM)
            # Use lower sampling rate for speed
            sampling_rate = 0.05 if fast_mode else self.sampling_rate
            registration.SetMetricSamplingPercentage(sampling_rate)
            
            # Interpolation method
            registration.SetInterpolator(sitk.sitkLinear)
            
            # Optimizer: Gradient descent (faster than LBFGSB)
            if fast_mode:
                registration.SetOptimizerAsGradientDescent(
                    learningRate=1.0,
                    numberOfIterations=50,  # Reduced iterations
                    convergenceMinimumValue=1e-4,
                    convergenceWindowSize=5
                )
                registration.SetOptimizerScalesFromPhysicalShift()
                
                # Simpler multi-resolution (2 levels instead of 3)
                registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2])
                registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1])
            else:
                registration.SetOptimizerAsLBFGSB(
                    gradientConvergenceTolerance=1e-5,
                    numberOfIterations=100,
                    maximumNumberOfCorrections=5,
                    maximumNumberOfFunctionEvaluations=500,
                    costFunctionConvergenceFactor=1e7
                )
                registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
                registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
            
            registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            
            # Set initial transform
            registration.SetInitialTransform(initial_transform, inPlace=False)
            
            # Add progress callback
            def progress_callback(method):
                logger.info(f"  Iteration {method.GetOptimizerIteration()}: metric = {method.GetMetricValue():.6f}")
            
            # Only log every 10 iterations to reduce overhead
            iteration_count = [0]
            def sparse_callback(method):
                iteration_count[0] += 1
                if iteration_count[0] % 10 == 0:
                    logger.info(f"  Progress: iteration {method.GetOptimizerIteration()}, metric = {method.GetMetricValue():.6f}")
            
            registration.AddCommand(sitk.sitkIterationEvent, lambda: sparse_callback(registration))
            
            # Execute registration
            logger.info("Running optimization...")
            final_transform = registration.Execute(fixed_sitk, moving_sitk)
            
            # Apply transform to original (non-downsampled) images
            if fast_mode:
                # Re-load original images for final resampling
                fixed_sitk_orig = self._numpy_to_sitk(fixed_image, spacing)
                moving_sitk_orig = self._numpy_to_sitk(moving_image, spacing)
            else:
                fixed_sitk_orig = fixed_sitk
                moving_sitk_orig = moving_sitk
            
            # Apply transform
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed_sitk_orig)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(float(np.min(moving_image)))
            resampler.SetTransform(final_transform)
            
            warped_sitk = resampler.Execute(moving_sitk_orig)
            warped_image = self._sitk_to_numpy(warped_sitk)
            
            logger.info(f"Deformable registration complete, final metric: {registration.GetMetricValue():.6f}")
            
            return warped_image, final_transform
            
        except Exception as e:
            logger.error(f"Deformable registration failed: {str(e)}")
            raise RegistrationError(f"Deformable registration failed: {str(e)}")
    
    def register(
        self,
        fixed_image: np.ndarray,
        moving_image: np.ndarray,
        spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
        use_deformable: bool = True,
        fast_mode: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Execute two-level registration
        
        Args:
            fixed_image: Fixed image (Followup)
            moving_image: Moving image (Baseline)
            spacing: Voxel spacing
            use_deformable: Whether to use deformable registration
            fast_mode: Use faster but less accurate settings (default: True)
            
        Returns:
            (Registered image, Transform parameters dict)
        """
        import time
        start_time = time.time()
        
        # L1: Rigid registration
        logger.info("=" * 50)
        logger.info("Step 1/2: Rigid Registration")
        warped_rigid, rigid_transform = self.rigid_registration(
            fixed_image, moving_image, spacing, fast_mode=fast_mode
        )
        rigid_time = time.time() - start_time
        logger.info(f"Rigid registration took {rigid_time:.1f}s")
        
        transforms = {"rigid": rigid_transform}
        
        if use_deformable:
            # L2: Deformable registration
            logger.info("=" * 50)
            logger.info("Step 2/2: Deformable Registration")
            deform_start = time.time()
            warped_final, deform_transform = self.deformable_registration(
                fixed_image, warped_rigid, spacing, fast_mode=fast_mode
            )
            deform_time = time.time() - deform_start
            logger.info(f"Deformable registration took {deform_time:.1f}s")
            transforms["deformable"] = deform_transform
        else:
            warped_final = warped_rigid
        
        total_time = time.time() - start_time
        logger.info("=" * 50)
        logger.info(f"Total registration time: {total_time:.1f}s")
        
        return warped_final, transforms
    
    def register_files(
        self,
        fixed_path: Path,
        moving_path: Path,
        output_path: Optional[Path] = None,
        use_deformable: bool = True
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        配准两个 NIfTI 文件
        
        Args:
            fixed_path: 固定图像路径 (Followup)
            moving_path: 移动图像路径 (Baseline)
            output_path: 输出路径
            use_deformable: 是否使用非刚性配准
            
        Returns:
            (配准结果路径, 配准参数)
        """
        # 加载图像
        fixed_nii = nib.load(fixed_path)
        moving_nii = nib.load(moving_path)
        
        fixed_data = fixed_nii.get_fdata()
        moving_data = moving_nii.get_fdata()
        
        # 获取体素间距
        spacing = tuple(fixed_nii.header.get_zooms()[:3])
        
        # 执行配准
        warped_data, transforms = self.register(
            fixed_data, moving_data, spacing, use_deformable
        )
        
        # 确定输出路径
        if output_path is None:
            output_path = moving_path.parent / f"{moving_path.stem.replace('.nii', '')}_warped.nii.gz"
        
        # 保存结果
        warped_nii = nib.Nifti1Image(warped_data, fixed_nii.affine, fixed_nii.header)
        nib.save(warped_nii, output_path)
        
        logger.info(f"配准完成: {output_path}")
        
        return output_path, transforms


