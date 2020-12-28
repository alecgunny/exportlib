import os

from google.protobuf import text_format
from tritonclient.grpc import model_config_pb2 as model_config


def soft_makedirs(path):
    # basically just a reminder to myself to
    # get rid of this function and replace it
    # with the exist_ok syntax when I'm confident
    # I have the right version (os doesn't have
    # a __version__ attribute unfortunately)
    try:
        os.makedirs(path, exist_ok=True)
    except TypeError:
        if not os.path.exists(path):
            os.makedirs(path)


def read_config(path):
    config = model_config.ModelConfig()
    with open(path, "r") as f:
        text_format.Merge(f.read(), config)
    return config


def write_config(config, path):
    with open(path, "w") as f:
        f.write(str(config))
