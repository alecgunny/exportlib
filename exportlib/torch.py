import os
import typing
from inspect import signature

import torch

from exportlib import io
from exportlib.model_repository import ModelRepository, Platform


def export(
    repository: ModelRepository,
    model_name: str,
    module: torch.nn.Module,
    input_shapes: typing.Dict[str, tuple],
    model_version: int = 1,
    output_names: typing.List[str] = None,
    max_batch_size: int = 8,
    concurrent_models: int = 1,
    verbose: bool = False,
) -> str:
    """
    Export an instantiated PyTorch `Module` as an ONNX runtime to
    a Triton model `repository` under the name `model_name`.
    """
    model = repository.create(model_name, platform=Platform.ONNX, force=True)
    if all([shape[0] is None for shape in input_shapes.values()]):
        model.config.max_batch_size = max_batch_size

    inputs, dynamic_axes = {}, {}
    for input_name, shape in input_shapes.items():
        # only allow variable sized shape for batch
        # for right now, TODO: how to handle generally?
        # TODO: make error more specific
        if not all(shape[1:]):
            raise ValueError("Cannot use shape {}".format(shape))

        model.config.add_input(name=input_name, shape=shape, dtype="float32")
        inputs[input_name] = torch.randn(*(x or 1 for x in shape))
        if shape[0] is None:
            dynamic_axes[input_name] = {0: "batch"}

    # use function signature from module.forward
    # method to infer the order in which to pass inputs
    parameters = dict(signature(module.forward).parameters)

    # get rid of any **kwargs
    try:
        parameters.pop("kwargs")
    except KeyError:
        pass

    assert len(parameters) == len(input_shapes)
    if len(parameters) == 1:
        # if we have simple 1 -> 1 mapping, don't overcomplicate it
        input_names = [input_name]
    else:
        input_names = [parameter.name for parameter in parameters]

    try:
        inputs = [inputs[name] for name in input_names]
    except KeyError:
        # input names don't match module.forward parameter names,
        # so give our best guess on ordering and sort the
        # inputs alphabetically
        # TODO: at this point, does it make sense to just raise
        # an error here? Alternatively, since parameter names like
        # "x" are sort of standard in Module.forward definitions,
        # does it make sense to provide an `input_name_map` kwarg
        # that maps to more descriptive input names for Triton?
        input_names = sorted(inputs.keys())
        inputs = [inputs[name] for name in input_names]
    outputs = module(*inputs)

    if isinstance(outputs, torch.Tensor):
        outputs = [outputs]
        if output_names is None:
            output_names = ["output"]

    if output_names is None:
        output_names = [f"output_{i}" for i in range(len(outputs))]
    elif isinstance(output_names, str):
        output_names = [output_names]
    assert len(output_names) == len(outputs)
    outputs = dict(zip(output_names, outputs))

    for output_name, output in outputs.items():
        shape = tuple(output.shape)
        if any([shape[0] is None for shape in input_shapes.values()]):
            shape = (None,) + shape[1:]
            dynamic_axes[output_name] = {0: "batch"}

        # TODO: map dtype from tensor dtype directly
        model.config.add_output(name=output_name, shape=shape, dtype="float32")

    version_dir = os.path.join(model.path, str(model_version))
    io.soft_makedirs(version_dir)
    export_path = os.path.join(version_dir, "model.onnx")

    if len(inputs) == 1:
        inputs = inputs[0]

    torch.onnx.export(
        module,
        inputs,
        export_path,
        verbose=verbose,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes or None,
    )

    model.config.add_instance_group(kind="gpu", gpus=[0], count=concurrent_models)
    model.config.write()
    return export_path
