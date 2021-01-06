import enum

from .platform import Platform


class PlatformName(enum.Enum):
    ONNX = "onnxruntime_onnx"
    TRT = "tensorrt_plan"
    DYNAMIC = None


platforms = {}
try:
    from .torch import TorchOnnxPlatform
except ImportError:
    pass
else:
    platforms[PlatformName.ONNX] = TorchOnnxPlatform
