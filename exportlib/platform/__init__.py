from .ensemble import EnsemblePlatform
from .platform import Platform, PlatformName

platforms = {PlatformName.ENSEMBLE: EnsemblePlatform}

try:
    from .torch import TorchOnnxPlatform
except ImportError:
    pass
else:
    platforms[PlatformName.ONNX] = TorchOnnxPlatform


try:
    from .tensorrt import TensorRTTorchPlatform
except ImportError:
    pass
else:
    platforms[PlatformName.TRT] = TensorRTTorchPlatform
