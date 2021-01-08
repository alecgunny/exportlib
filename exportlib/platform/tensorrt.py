from __future__ import annotations

import contextlib
import io
import os
import typing
from copy import deepcopy

import tensorrt as trt

from exportlib.io import soft_makedirs
from exportlib.platform import Platform, PlatformName, TorchOnnxPlatform
from exportlib.platform.platform import _SHAPE_TYPE

if typing.TYPE_CHECKING:
    from exportlib.model_repository import Model


class TensorRTPlatform(Platform):
    def _make_export_path(self, version):
        return os.path.join(self.model.path, str(version), "model.plan")

    def export(
        self,
        model_fn: typing.Union[typing.Callable, Model, str],
        version: int,
        input_shapes: _SHAPE_TYPE = None,
        output_names: typing.Optional[typing.List[str]] = None,
        verbose: int = 0,
        use_fp16: bool = False,
    ):
        version_dir = os.path.join(self.model.path, str(version))
        if isinstance(model_fn, typing.Callable):
            model_fn = super().export(
                model_fn, version, input_shapes, output_names, verbose
            )
            model_fn = model_fn.getvalue()

        else:
            if not isinstance(model_fn, str):
                temp_config = deepcopy(self.model.config._config)
                temp_config.MergeFrom(model_fn.config._config)
                temp_config.MergeFrom(self.model.config._config)
                self.model.config._config = temp_config

                model_fn = model_fn.platform._make_export_path(version)

            with open(model_fn, "rb") as f:
                model_fn = f.read()
            soft_makedirs(version_dir)

            self._check_exposed_tensors("input", input_shapes)

            # we don't have any function to check output shapes,
            # so just make sure we have valid outputs
            if len(self.model.config.output) == 0:
                raise ValueError("No model outputs")

        TRT_LOGGER = trt.Logger()
        with contextlib.ExitStack() as stack:
            builder = stack.enter_context(trt.Builder(TRT_LOGGER))
            config = stack.enter_context(builder.create_builder_config())

            builder.max_workspace_size = 1 << 28  # 256 MiB
            builder.max_batch_size = max(self.model.config.max_batch_size, 1)
            if use_fp16:
                builder.fp16_mode = True
                builder.strict_type_constraints = True

            for input in self.model.config.input:
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

            parser = stack.enter_context(trt.OnnxParser(network, TRT_LOGGER))
            parser.parse(model_fn)

            output_shapes = {}
            for n, output in enumerate(self.model.config.output):
                network_output = network.get_output(n)
                if network_output is None:
                    # if we only have one output, default to marking
                    # the last layer as the output layer
                    if len(self.model.config.output) == 1:
                        last_layer = network.get_layer(network.num_layers - 1)
                        network_output = last_layer.get_output(0)
                        network.mark_output(network_output)

                    else:
                        raise IndexError(
                            "Number of config outputs {} doesn't "
                            "match number of outputs {} in network.".format(
                                len(self.model.config.output), n
                            )
                        )

                network_output.name = output.name
                output_shapes[output.name] = network_output.shape
            # self._check_exposed_tensors("output", output_shapes)

            engine = builder.build_cuda_engine(network)
            if engine is None:
                raise RuntimeError("TensorRT conversion failed")

            engine_path = os.path.join(version_dir, "model.plan")
            with open(engine_path, "wb") as f:
                f.write(engine.serialize())

            # write and clean up
            self.model.config.write()
            return engine_path


class TensorRTTorchPlatform(TensorRTPlatform, TorchOnnxPlatform):
    def _do_export(self, model_fn, export_obj, verbose=0):
        export_obj = io.BytesIO()
        return super()._do_export(model_fn, export_obj, verbose)
