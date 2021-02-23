import contextlib
import os
import shutil
import typing

import attr

from exportlib import ModelConfig, io
from exportlib.platform import Platform, PlatformName, platforms
from exportlib.platform.platform import _SHAPE_TYPE

if typing.TYPE_CHEKCKING:
    from tritonclient.grpc.model_config_pb2 import ModelEnsembling

    from exportlib import ModelRepository


@contextlib.context_manager
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


@attr.s(auto_attribs=True)
class Model:
    repository: "ModelRepository"
    config: ModelConfig

    def __new__(
        cls,
        name: str,
        repository: "ModelRepository",
        platform: typing.Optional[str],
    ):
        if platform is not None:
            try:
                platform = Platform(platform)
            except ValueError:
                raise ValueError(f"Unrecognized platform {platform}")

        with _create_subdir(repository.path, name) as path:
            try:
                config = ModelConfig.read(os.path.join(path, "config.pbtxt"))
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
                    raise ValueError(
                        f"Model {config.name} config missing platform"
                    )

            if platform is PlatformName.ENSEMBLE:
                cls = EnsembleModel
            obj = super().__new__(cls)
            obj.__init__(repository, config)
            return obj

    @property
    def name(self):
        return self.config.name

    @property
    def platform(self):
        try:
            platform = PlatformName(self.config.platform)
            return platforms[platform](self)
        except KeyError:
            raise ValueError(
                "No exporter associated with platform {}".format(
                    self.config.platform
                )
            )

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
            self.repo.models[step.model_name]
            for step in self._config.ensemble_scheduling.step
        ]

    def _find_tensor(
        self,
        tensor: _tensor_type,
        exposed_type: str,
        version: typing.Optional[int] = None,
    ) -> typing.Tuple[ExposedTensor, ModelEnsembling.Step]:
        assert exposed_type in ["input", "output"]
        repo_models = list(self.repo.models.values())

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
            step = self.config.add_model(tensor.model, version=version)
        return tensor, step

    def add_input(
        self,
        input: typing.Union[str, ExposedTensor],
        version: typing.Optional[int] = None,
    ) -> ExposedTensor:
        input, step = self._find_tensor(input, "input", version)
        if input.name not in self.inputs:
            self.config.add_input(
                input.name,
                input.shape,
                dtype="float32",  # TODO: dynamic dtype mapping
            )
        step.input_map[input.name] = input.name
        return self.inputs[input.name]

    def add_output(
        self,
        output: typing.Union[str, ExposedTensor],
        version: typing.Optional[int] = None,
    ) -> ExposedTensor:
        output, step = self._find_tensor(output, "output", version)
        if output.name not in self.outputs:
            self.config.add_output(
                output.name,
                output.shape,
                dtype="float32",  # TODO: dynamic dtype mapping
            )
        step.output_map[output.name] = output.name
        return self.outputs[output.name]

    def pipe(
        self,
        input: typing.Union[str, "ExposedTensor"],
        output: typing.Union[str, "ExposedTensor"],
        name: typing.Optional[str] = None,
        version: typing.Optional[int] = None,
    ) -> None:
        input, input_step = self._find_tensor(input, "output")
        output, output_step = self._find_tensor(output, "input", version)

        try:
            current_key = input_step.output_map[input.name]
        except KeyError:
            name = name or input.name
            input_step.output_map[input.name] = name
        else:
            if name is not None and current_key != name:
                raise ValueError(
                    f"Output {input.name} from {input.model.name} "
                    f"already using key {current_key}, couldn't "
                    f"use provided key {name}"
                )
            name = current_key

        try:
            current_key = output_step.input_map[output.name]
        except KeyError:
            output_step.input_map[output.name] = name
        else:
            if current_key != name:
                raise ValueError(
                    f"Input {output.name} to {output.model.name} "
                    f"already receiving input from {current_key}, "
                    f"can't pipe input {name}"
                )
