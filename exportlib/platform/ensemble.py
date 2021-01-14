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
        raise TypeError

    def export(self, *args, **kwargs):
        self.model.config.write()
        return None
