import enum

from .platform import Platform
from .torch import TorchOnnxPlatform


class PlatformName(enum.Enum):
    ONNX = "onnxruntime_onnx"
    TRT = "tensorrt_plan"
    DYNAMIC = None


platforms = {PlatformName.ONNX: TorchOnnxPlatform}
