import os

import pytest
import torch

from exportlib.model_repository import ModelRepository
from exportlib.platform import PlatformName


def test_model_repository(input_dim=64):
    nn = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 128), torch.nn.Linear(128, 1)
    )

    # instantiate a repository and add a model to it
    repo = ModelRepository("/tmp/repo")
    model = repo.create_model("my_nn", platform=PlatformName.ONNX)
    model.max_batch_size = 8
    model.config.add_instance_group(kind="gpu", count=4)

    # export the "trained" network
    export_path = model.export_version(
        nn, input_shapes={"input": (None, input_dim)}
    )

    # make sure the path matches what we expect
    # and that it exists
    assert export_path == "/tmp/repo/my_nn/1/model.onnx"
    assert os.path.exists(export_path)

    # this should create an error since `force` is
    # False by default, so it will crash with the
    # existing model
    with pytest.raises(ValueError):
        duplicate_model = repo.create_model("my_nn", platform=PlatformName.ONNX)

    # now try it with `force=True` and make sure
    # the appropriate postfix has been added
    duplicate_model = repo.create_model(
        "my_nn", platform=PlatformName.ONNX, force=True
    )
    assert duplicate_model.name == "my_nn_0"

    # export this model and make sure the path is right
    export_path = duplicate_model.export_version(
        nn, input_shapes={"input": (None, input_dim)}
    )
    assert export_path == "/tmp/repo/my_nn_0/1/model.onnx"

    # create a new repository at the same location and
    # make sure that it loads in all the existing models
    # and populates the configs
    new_repo = ModelRepository("/tmp/repo")
    for model_name in ["my_nn", "my_nn_0"]:
        assert model_name in new_repo.models
    assert new_repo.models["my_nn"].config.instance_group[0].count == 4

    # export a second version of the nn to
    # the same place
    nn2 = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 128), torch.nn.Linear(128, 1)
    )
    new_repo.models["my_nn"].export_version(nn2)

    # make sure we have a new version
    versions = next(os.walk("/tmp/repo/my_nn"))[1]
    versions = list(map(int, versions))
    assert set(versions) == {1, 2}
