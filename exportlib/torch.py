import os
import typing
from inspect import signature

import torch
from exportlib.io import write_config
from exportlib.model_repository import ModelRepository
from tritongrpcclient import model_config_pb2 as model_config


def export(
    repository: ModelRepository,
    model_name: str,
    model_fn: typing.Callable,
    input_shapes: dict[str, tuple],
    model_version: int = 1,
    formats: typing.List[str] = "onnx",
    output_names: typing.List[str] = None,
    max_batch_size: int = 8,
    concurrent_models=1,
) -> str:
    model = repository.create(model_name, platform=formats, force=True)
    if all([shape[0] is None for shape in input_shapes.values()]):
        model.config.max_batch_size = max_batch_size

    inputs, dynamic_axes = {}, {}
    for input_name, shape in input_shapes.items():
        # only allow variable sized shape for batch
        # for right now, TODO: how to handle generally?
        if not all(shape[1:]):
            raise ValueError("Cannot use shape {}".format(shape))

        model.config.add_input(
            name=input_name, shape=(x or -1 for x in shape), dtype="float32"
        )
        inputs[input_name] = torch.randn(*(x or 1 for x in shape))
        if shape[0] is None:
            dynamic_axes[input_name] = {0: "batch"}

    parameters = signature(model_fn).parameters
    assert len(parameters) == len(input_shapes)
    if len(parameters) == 1:
        # if we have simple 1 -> 1 mapping, don't overcomplicate it
        input_names = [input_name]
    else:
        input_names = [parameter.name for parameter in parameters]

    try:
        inputs = [inputs[name] for name in input_names]
    except KeyError:
        # input names don't match model_fn parameter names,
        # so give our best guess on ordering and sort the
        # inputs alphabetically
        input_names = sorted(inputs.keys())
        inputs = [inputs[name] for name in input_names]
    outputs = model_fn(*inputs)

    if isinstance(outputs, torch.Tensor):
        outputs = [outputs]
    if output_names is None:
        output_names = [x.name for x in outputs]
    else:
        assert len(output_names) == len(outputs)
    outputs = dict(zip(output_names, outputs))

    for output_name, output in outputs.items():
        shape = output.shape
        if any([shape[0] is None for shape in input_shapes.values()]):
            shape[0] = -1
            dynamic_axes[output_name] = {0: "batch"}

        model.add_output(name=output_name, shape=shape, dtype="float32")

    export_path = os.path.join(model.path, str(model_version), "model.onnx")
    torch.onnx.export(
        model_fn,
        inputs,
        export_path,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes or None,
    )

    model.config.add_instance_group(
        kind="gpu", gpu=0, num_models=concurrent_models
    )
    return export_path
