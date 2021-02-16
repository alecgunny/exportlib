import tensorflow as tf

from exportlib.platform import Platform


class TensorFlowSavedModelPlatform:  # (Platform):
    # not inheriting for now to avoid having
    # to implement abstract methods
    def __init__(self, model):
        pass
