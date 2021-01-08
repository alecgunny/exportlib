import os
import typing
from inspect import signature

import torch

from exportlib.platform import Platform


class TorchOnnxPlatform(Platform):
    @property
    def _tensor_type(self):
        return torch.Tensor

    def _make_tensor(self, shape):
        return torch.randn(*(x or 1 for x in shape))

    def _make_export_path(self, version):
        return os.path.join(self.model.path, str(version), "model.onnx")

    def _parse_model_fn_parameters(self, model_fn):
        if isinstance(model_fn, torch.nn.Module):
            model_fn = model_fn.forward
        parameters = signature(model_fn).parameters
        return parameters

    def _do_export(self, model_fn, export_obj, verbose=0):
        inputs, dynamic_axes = [], {}
        for input in self.model.config.input:
            shape = list(input.dims)
            if shape[0] == -1:
                dynamic_axes[input.name] = {0: "batch"}
                shape[0] = self.model.config.max_batch_size
            inputs.append(self._make_tensor(shape))

        if len(dynamic_axes) > 0:
            for output in self.model.config.output:
                dynamic_axes[output.name] = {0: "batch"}

        if len(inputs) == 1:
            inputs = inputs[0]

        torch.onnx.export(
            model_fn,
            inputs,
            export_obj,
            input_names=[x.name for x in self.model.config.input],
            output_names=[x.name for x in self.model.config.output],
            dynamic_axes=dynamic_axes or None,
        )
        return export_obj
