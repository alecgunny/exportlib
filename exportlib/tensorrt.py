import contextlib
import os

import tensorrt as trt
from exportlib import io


def export(
    model_name,
    model_repository,
    base_config,
    onnx_model,
    model_version=1,
    onnx_model_version=None,
    use_fp16=False,
):
    model = model_repository.create(model_name, platform="tensorrt_plan")
    model.config.MergeFrom(base_config)

    version_path = os.path.join(model.path, str(model_version))
    io.soft_makedirs(version_path)

    # set up a plan builder
    TRT_LOGGER = trt.Logger()
    with contextlib.ExitStack() as stack:
        builder = stack.enter_context(trt.Builder(TRT_LOGGER))
        builder.max_workspace_size = 1 << 28  # 256 MiB
        builder.max_batch_size = 1  # flags['batch_size']
        if use_fp16:
            builder.fp16_mode = True
            builder.strict_type_constraints = True

        #   config = builder.create_builder_config()
        #   profile = builder.create_optimization_profile()
        #   min_shape = tuple([1] + onnx_config.input[0].dims[1:])
        #   max_shape = tuple([8] + onnx_config.input[0].dims[1:])

        #   optimal_shape = max_shape
        #   profile.set_shape('input', min_shape, optimal_shape, max_shape)
        #   config.add_optimization_profile(profile)

        # initialize a parser with a network and fill in that
        # network with the onnx file we just saved
        network = stack.enter_context(
            builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
        )

        parser = stack.enter_context(trt.OnnxParser(network, TRT_LOGGER))
        onnx_model_version = onnx_model_version or model_version
        onnx_path = os.path.join(
            onnx_model.path, onnx_model_version, "model.onnx"
        )
        with open(onnx_path, "rb") as f:
            parser.parse(f.read())

        last_layer = network.get_layer(network.num_layers - 1)
        if not last_layer.get_output(0):
            # logger.info("Marking output layer")
            network.mark_output(last_layer.get_output(0))

        # build an engine from that network
        engine = builder.build_cuda_engine(network)
        engine_path = os.path.join(version_path, "model.plan")
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())

        # export config
        io.write_config(model.config_path)
