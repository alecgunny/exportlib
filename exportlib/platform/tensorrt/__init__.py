import os
import pickle
import requests
from copy import deepcopy

from exportlib.platform import Platform, TorchOnnxPlatform

from .onnx import convert_network


class TensorRTPlatform(Platform):
    def _make_export_path(self, version):
        return os.path.join(self.model.path, str(version), "model.plan")

    def export(
        self,
        model_fn: typing.Union[typing.Callable, "Model", str],
        version: int,
        input_shapes: _SHAPE_TYPE = None,
        output_names: typing.Optional[typing.List[str]] = None,
        verbose: int = 0,
        use_fp16: bool = False,
        url: typing.Optional[str] = None
    ):
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

            self._check_exposed_tensors("input", input_shapes)

            # we don't have any function to check output shapes,
            # so just make sure we have valid outputs
            if len(self.model.config.output) == 0:
                raise ValueError("No model outputs")

        config = self.model.config._config
        if url is None:
            engine = convert_network(model_fn, config, use_fp16)
            if engine is None:
                raise RuntimeError("Model conversion failed")
            engine = engine.serialize()
        else:
            # TODO: error catching here
            data = {
                "config": config.SerializeToString(),
                "model": model_fn
            }
            response = requests.post(
                url=url,
                data=pickle.dumps(data),
                headers={'Content-Type': 'application/octet-stream'}
            )
            engine = response.content

        engine_path = self._make_export_path(version)
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())

        # write and clean up
        self.model.config.write()
        return engine_path


class TensorRTTorchPlatform(TensorRTPlatform, TorchOnnxPlatform):
    def _do_export(self, model_fn, export_obj, verbose=0):
        export_obj = io.BytesIO()
        return super()._do_export(model_fn, export_obj, verbose)
