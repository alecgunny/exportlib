import os
import shutil
import typing

import attr

from exportlib import ModelConfig, io
from exportlib.platform import PlatformName, platforms
from exportlib.platform.platform import _SHAPE_TYPE

if typing.TYPE_CHEKCKING:
    from exportlib import ModelRepository


@attr.s(auto_attribs=True)
class ExposedTensor:
    model: "Model"
    name: str
    shape: _SHAPE_TYPE


@attr.s(auto_attribs=True)
class Model:
    name: str
    repository: "ModelRepository"
    platform: str = attr.ib(default=None, converter=PlatformName)

    def __attrs_post_init__(self):
        io.soft_makedirs(self.path)

        # try to load an existing config, if it exists
        try:
            self.config = ModelConfig.read(self)
        except FileNotFoundError:
            # if it doesn't, we have to specify what platform
            # our new model uses
            if self.platform == PlatformName.DYNAMIC:
                raise ValueError("Must specify platform for new model")

            self.config = ModelConfig(self, platform=self.platform)

        else:
            if (
                self.config.platform is not None
                and self.platform != PlatformName.DYNAMIC
            ):
                # if the config specifies a platform and the
                # initialization did too, make sure they match
                # TODO: just warn instead and prefer the specified
                # one?
                if self.platform.value != self.config.platform:
                    raise ValueError(
                        f"Existing config for model {self.name} "
                        f"specifies platform {self.config.platform}, which "
                        f"doesn't match specified platform {self.platform.value}"
                    )
            elif self.platform == PlatformName.DYNAMIC:
                # otherwise if the initialization didn't specify
                # anything, try to grab the platform from the config
                try:
                    self.platform = PlatformName(self.config.platform)
                except ValueError:
                    raise ValueError(
                        f"Existing config for model {self.name} "
                        f"specifies unknown platform {self.config.platform}"
                    )
            else:
                # otherwise we don't have a platform from anywhere so
                # raise an error
                raise ValueError(f"Model {self.name} config missing platform")

        finally:
            try:
                platform = platforms[self.platform]
            except KeyError:
                raise ValueError(
                    "No exporter found for platform {}".format(
                        self.platform.value
                    )
                )
        self.platform = platform(self)

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
        for input in self.config.inputs:
            shape = tuple(x if x != -1 else None for x in input.dims)
            inputs[input.name] = ExposedTensor(self, input.name, shape)
        return inputs

    @property
    def outputs(self):
        outputs = {}
        for output in self.config.outputs:
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
        version_dir = os.path.join(self.path, str(version))
        io.soft_makedirs(version_dir)
        try:
            return self.platform.export(
                model_fn,
                version,
                input_shapes=input_shapes,
                output_names=output_names,
                verbose=verbose,
                **kwargs,
            )
        except Exception:
            shutil.rmtree(version_dir)
            raise
