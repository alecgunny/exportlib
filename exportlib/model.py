import contextlib
import os
import shutil
import typing

import attr

from exportlib import ModelConfig, io
from exportlib.platform import Platform, PlatformName, platforms
from exportlib.platform.platform import _SHAPE_TYPE

if typing.TYPE_CHECKING:
    from tritonclient.grpc.model_config_pb2 import ModelEnsembling

    from exportlib import ModelRepository


@contextlib.contextmanager
def _create_subdir(path: str, dirname: str):
    path = os.path.join(path, dirname)
    do_remove = io.soft_makedirs(path)
    try:
        yield path
    except Exception:
        if do_remove:
            shutil.rmtree(path)
        raise


@attr.s(auto_attribs=True)
class ExposedTensor:
    model: "Model"
    name: str
    shape: _SHAPE_TYPE


def _infer_platform(name, repo, platform=None):
    config_path = os.path.join(repo.path, name, "config.pbtxt")
    if platform is not None:
        try:
            platform = PlatformName(platform)
        except ValueError:
            raise ValueError(f"Unrecognized platform {platform}")
    else:
        platform = PlatformName.DYNAMIC

    try:
        config = ModelConfig.read(config_path)
    except FileNotFoundError:
        if platform is PlatformName.DYNAMIC:
            raise ValueError("Must specify platform for new model")
        config = ModelConfig(name, platform)
    else:
        if config.platform != "" and platform is not PlatformName.DYNAMIC:
            # if the config specifies a platform and the
            # initialization did too, make sure they match
            if platform.value != config.platform:
                raise ValueError(
                    f"Existing config for model {name} "
                    f"specifies platform {config.platform}, which "
                    f"doesn't match specified platform {platform.value}"
                )
        elif platform is PlatformName.DYNAMIC:
            # otherwise if the initialization didn't specify
            # anything, try to grab the platform from the config
            try:
                platform = PlatformName(config.platform)
            except ValueError:
                raise ValueError(
                    f"Existing config for model {config.name} "
                    f"specifies unknown platform {config.platform}"
                )
        else:
            # otherwise we don't have a platform from anywhere so
            # raise an error
            raise ValueError(f"Model {config.name} config missing platform")
    return platform, config


@attr.s(auto_attribs=True)
class Model:
    name: str
    repository: "ModelRepository"
    platform: typing.Optional[str]

    def __new__(
        cls,
        name: str,
        repository: "ModelRepository",
        platform: typing.Optional[str],
    ):
        platform, _ = _infer_platform(name, repository, platform)

        if platform is PlatformName.ENSEMBLE:
            cls = EnsembleModel
        return super().__new__(cls)

    def __attrs_post_init__(self):
        with _create_subdir(self.repository.path, self.name):
            platform, config = _infer_platform(
                self.name, self.repository, self.platform
            )

            try:
                self.platform = platforms[platform](self)
            except KeyError:
                raise ValueError(
                    "No exporter associated with platform {}".format(platform)
                )
            self.config = config

    @property
    def path(self):
        return os.path.join(self.repository.path, self.name)

    @property
    def versions(self):
        # TODO: what if there are other dirs? Is Triton cool
        # with that? Do we filter as ints?
        return next(os.walk(self.path))[1]

    @property
    def inputs(self):
        inputs = {}
        for input in self.config.input:
            shape = tuple(x if x != -1 else None for x in input.dims)
            inputs[input.name] = ExposedTensor(self, input.name, shape)
        return inputs

    @property
    def outputs(self):
        outputs = {}
        for output in self.config.output:
            shape = tuple(x if x != -1 else None for x in output.dims)
            outputs[output.name] = ExposedTensor(self, output.name, shape)
        return outputs

    def export_version(
        self,
        model_fn: typing.Optional[typing.Union[typing.Callable, "Model"]] = None,
        version: typing.Optional[int] = None,
        input_shapes: _SHAPE_TYPE = None,
        output_names: typing.Optional[typing.List[str]] = None,
        verbose: int = 0,
        **kwargs,
    ) -> str:
        if model_fn is None:
            ensemble = platforms[PlatformName.ENSEMBLE]
            if not isinstance(self.platform, ensemble):
                raise ValueError(
                    "Must specify model function for non " "ensemble model"
                )

        version = version or len(self.versions) + 1
        with _create_subdir(self.path, str(version)):
            return self.platform.export(
                model_fn,
                version,
                input_shapes=input_shapes,
                output_names=output_names,
                verbose=verbose,
                **kwargs,
            )


_tensor_type = typing.Union[str, ExposedTensor]


class EnsembleModel(Model):
    @property
    def models(self):
        return [
            self.repository.models[step.model_name]
            for step in self._config.ensemble_scheduling.step
        ]

    def _find_tensor(
        self,
        tensor: _tensor_type,
        exposed_type: str,
        version: typing.Optional[int] = None,
    ) -> typing.Tuple[ExposedTensor, "ModelEnsembling.Step"]:
        assert exposed_type in ["input", "output"]
        repo_models = list(self.repository.models.values())

        if isinstance(tensor, str):
            for model in repo_models:
                tensors = getattr(model, exposed_type + "s")
                try:
                    tensor = tensors[tensor]
                except KeyError:
                    continue
                break
            else:
                raise ValueError(
                    f"Coludn't find model with input {input} "
                    "in model repository."
                )

        for step in self.config.ensemble_scheduling.step:
            if step.model_name == tensor.model.name:
                break
        else:
            if tensor.model not in repo_models:
                raise ValueError(
                    f"Trying to add model {tensor.model.name} to "
                    "ensemble that doesn't exist in repo."
                )
            self.config.add_step(tensor.model, version=version)
        return tensor

    def _update_step_map(self, model_name, key, value, map_type):
        for step in self.config.ensemble_scheduling.step:
            if step.model_name == model_name:
                step_map = getattr(step, map_type + "_map")
                step_map[key] = value

    def _update_input_map(self, model_name, key, value):
        self._update_step_map(model_name, key, value, "input")

    def _update_output_map(self, model_name, key, value):
        self._update_step_map(model_name, key, value, "output")

    def add_input(
        self,
        input: _tensor_type,
        version: typing.Optional[int] = None,
        name: typing.Optional[str] = None,
    ) -> ExposedTensor:
        input = self._find_tensor(input, "input", version)
        name = name or input.name
        if input.name not in self.inputs:
            self.config.add_input(
                name,
                input.shape,
                dtype="float32",  # TODO: dynamic dtype mapping
            )
        self._update_input_map(input.model.name, input.name, name)
        return self.inputs[name]

    def add_output(
        self,
        output: _tensor_type,
        version: typing.Optional[int] = None,
        name: typing.Optional[str] = None,
    ) -> ExposedTensor:
        output = self._find_tensor(output, "output", version)
        name = name or output.name
        if output.name not in self.outputs:
            self.config.add_output(
                name,
                output.shape,
                dtype="float32",  # TODO: dynamic dtype mapping
            )
        self._update_output_map(output.model.name, output.name, name)
        return self.outputs[name]

    def add_streaming_inputs(
        self,
        inputs: typing.Union[typing.List[_tensor_type]],
        stream_size: int,
        name: typing.Optional[str] = None,
    ):
        tensors = []
        for input in inputs:
            tensor = self._find_tensor(input, "input")
            tensors.append(tensor)

        try:
            from exportlib.stream import make_streaming_input_model
        except ImportError as e:
            if "tensorflow" in str(e):
                raise RuntimeError(
                    "Unable to leverage streaming input, "
                    "must install TensorFlow first"
                )
        streaming_model = make_streaming_input_model(
            self.repository, tensors, stream_size, name
        )

        self.add_input(streaming_model.inputs["stream"])

        metadata = []
        for tensor, output in zip(tensors, streaming_model.config.output):
            self.pipe(streaming_model.outputs[output.name], tensor)
            metadata.append("{}/{}".format(tensor.model.name, tensor.name))

        self.config.parameters["stream_channels"].string_value = ",".join(
            metadata
        )

    def pipe(
        self,
        input: typing.Union[str, "ExposedTensor"],
        output: typing.Union[str, "ExposedTensor"],
        name: typing.Optional[str] = None,
        version: typing.Optional[int] = None,
    ) -> None:
        input = self._find_tensor(input, "output")
        output = self._find_tensor(output, "input", version)

        try:
            for step in self.config.ensemble_scheduling.step:
                if step.model_name == input.model.name:
                    break
            current_key = step.output_map[input.name]
            if current_key == "":
                raise KeyError
        except KeyError:
            name = name or input.name
            self._update_output_map(input.model.name, input.name, name)
        else:
            if name is not None and current_key != name:
                raise ValueError(
                    f"Output {input.name} from {input.model.name} "
                    f"already using key {current_key}, couldn't "
                    f"use provided key {name}"
                )
            name = current_key

        try:
            for step in self.config.ensemble_scheduling.step:
                if step.model_name == output.model.name:
                    break
            current_key = step.input_map[output.name]
            if current_key == "":
                raise KeyError
        except KeyError:
            self._update_input_map(output.model.name, output.name, name)
        else:
            if current_key != name:
                raise ValueError(
                    f"Input {output.name} to {output.model.name} "
                    f"already receiving input from {current_key}, "
                    f"can't pipe input {name}"
                )
