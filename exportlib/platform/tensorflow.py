import tensorflow as tf

from exportlib.platform import Platform


class TensorFlowSavedModelPlatform(Platform):
    @property
    def _tensor_type(self):
        pass

    def _make_tensor(self, shape):
        pass

    def _do_export(self, model_fn, export_dir, verbose=0):
        pass

    def _parse_model_fn_parameters(self, model_fn):
        pass

    def _make_export_path(version):
        pass

