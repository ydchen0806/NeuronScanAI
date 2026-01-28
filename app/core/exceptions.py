"""
自定义异常类
"""


class NeuroScanException(Exception):
    """NeuroScan 基础异常"""
    def __init__(self, message: str, code: str = "UNKNOWN_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)


class DicomLoadError(NeuroScanException):
    """DICOM 加载错误"""
    def __init__(self, message: str):
        super().__init__(message, code="DICOM_LOAD_ERROR")


class RegistrationError(NeuroScanException):
    """图像配准错误"""
    def __init__(self, message: str):
        super().__init__(message, code="REGISTRATION_ERROR")


class SegmentationError(NeuroScanException):
    """分割错误"""
    def __init__(self, message: str):
        super().__init__(message, code="SEGMENTATION_ERROR")


class AnalysisError(NeuroScanException):
    """分析错误"""
    def __init__(self, message: str):
        super().__init__(message, code="ANALYSIS_ERROR")


class ModelNotFoundError(NeuroScanException):
    """模型未找到错误"""
    def __init__(self, message: str):
        super().__init__(message, code="MODEL_NOT_FOUND")


class InvalidInputError(NeuroScanException):
    """无效输入错误"""
    def __init__(self, message: str):
        super().__init__(message, code="INVALID_INPUT")


