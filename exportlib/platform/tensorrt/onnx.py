import contextlib
import typing
from functools import wraps

import tensorrt as trt

if typing.TYPE_CHECKING:
    from tritonclient.grpc import model_config_pb2 as model_config


def convert_network(
    model_binary: bytes,
    config: "model_config.ModelConfig",
    use_fp16: bool = False
) -> bytes:
    with contextlib.ExitStack() as stack:
        return _convert_network(stack, model_binary, config, use_fp16)


def _convert_network(
    stack: contextlib.ExitStack,
    model_binary: bytes,
    model_config: "model_config.ModelConfig",
    use_fp16: bool,
):
    """
    using a cheap wrapper to save myself some tabs
    """
    logger = trt.Logger()
    builder = stack.enter_context(trt.Builder(logger))
    config = stack.enter_context(builder.create_builder_config())
    builder.max_workspace_size = 1 << 28
    builder.max_batch_size = max(model_config.max_batch_size, 1)
    if use_fp16:
        builder.fp16_mode = True
        builder.strict_type_constraints = True

    for input in model_config.input:
        if input.dims[0] != -1:
            continue
        profile = builder.create_optimization_profile()
        min_shape = tuple([1] + input.dims[1:])
        max_shape = tuple([builder.max_batch_size] + input.dims[1:])
        optimal_shape = max_shape

        profile.set_shape(input.name, min_shape, optimal_shape, max_shape)
        config.add_optimization_profile(profile)

    network = stack.enter_context(
        builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
    )
    parser = stack.enter_context(trt.OnnxParser(network, logger))
    parser.parse(model_binary)

    if len(model_config.output) == 1 and network.num_outputs == 0:
        last_layer = network.get_layer(network.num_layers - 1)
        network_output = last_layer.get_output(0)
        network.mark_output(network_output)
    elif len(model_config.output) != network.num_outputs:
        raise ValueError(
            "Number of config outputs {} doesn't "
            "match number of outputs {} in network.".format(
                len(model_config.output), network.num_outputs
            )
        )

    for n, output in enumerate(model_config.output):
        network_output = network.get_output(n)
        network_output.name = output.name

        # rather than do a full shape check, only do
        # it on the dimensions we have at build time
        if len(network_output.shape) != len(output.dims):
            raise ValueError(
                "Number of dimensions {} specified for "
                "output {} not equal to number {} found "
                "in TensorRT network".format(
                    len(output.dims),
                    output.name,
                    len(network_output.shape),
                )
            )
        for ndim, cdim in zip(network_output.shape, output.dims):
            if ndim != -1 and ndim != cdim:
                raise ValueError(
                    "Shape mismatch for output {} between "
                    "config shape {} and network shape {}".format(
                        output.name, output.dims, network_output.shape
                    )
                )
    return builder.build_cuda_engine(network)
