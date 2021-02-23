import os

from google.protobuf import text_format
from tritonclient.grpc import model_config_pb2 as model_config


def soft_makedirs(path):
    """
    directory making utility that returns
    `True` if we had to make the directory,
    otherwise `False` if it already existed
    """
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False


def read_config(path):
    config = model_config.ModelConfig()
    with open(path, "r") as f:
        text_format.Merge(f.read(), config)
    return config


def write_config(config, path):
    with open(path, "w") as f:
        f.write(str(config))
