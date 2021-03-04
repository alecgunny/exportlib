import os

from exportlib.platform import Platform


class EnsemblePlatform(Platform):
    @property
    def _tensor_type(self):
        raise TypeError

    def _make_tensor(self, shape):
        raise TypeError

    def _parse_model_fn_parameters(self, model_fn):
        raise TypeError

    def _do_export(self, model_fn, export_obj, verbose=0):
        raise TypeError

    def _make_export_path(self, version):
        return os.path.join(self.model.path, str(version), "model.empty")

    def export(self, model_fn, version, *args, **kwargs):
        self.model.config.write(os.path.join(self.model.path, "config.pbtxt"))
        with open(self._make_export_path(version), "w") as f:
            f.write("")
        return None
