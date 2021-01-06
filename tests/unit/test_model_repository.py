import os

import pytest
import torch

from exportlib.model_repository import ModelRepository
from exportlib.platform import PlatformName


def test_model_repository(input_dim=64):
    nn = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 128), torch.nn.Linear(128, 1)
    )

    repo = ModelRepository("/tmp/repo")
    model = repo.create_model("my_nn", platform=PlatformName.ONNX)
    model.max_batch_size = 8
    model.config.add_instance_group(kind="gpu", count=4)

    export_path = model.export_version(
        nn, input_shapes={"input": (None, input_dim)}
    )
    assert export_path == "/tmp/repo/my_nn/1/model.onnx"
    assert os.path.exists(export_path)

    with pytest.raises(ValueError):
        duplicate_model = repo.create_model("my_nn", platform=PlatformName.ONNX)
    duplicate_model = repo.create_model(
        "my_nn", platform=PlatformName.ONNX, force=True
    )
    assert duplicate_model.name == "my_nn_0"

    export_path = duplicate_model.export_version(nn, {"input": (None, input_dim)})
    assert export_path == "/tmp/repo/my_nn_0/1/model.onnx"

    new_repo = ModelRepository("/tmp/repo")
    for model_name in ["my_nn", "my_nn_0"]:
        assert model_name in new_repo.models
    assert new_repo.models["my_nn"].config.instance_groups[0].count == 4
