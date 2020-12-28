import enum
import os
import re
import types
import typing
import warnings
from itertools import count

import attr
from exportlib import io


class ModelConfig:
    def __init__(
        self, model: "Model", platform: "Platform", max_batch_size: int = None
    ):
        self._config = io.model_config.ModelConfig(
            name=model.name, platform=platform.value
        )
        if max_batch_size is not None:
            self._config.max_batch_size = max_batch_size
        self.path = os.path.join(model.path, "config.pbtxt")

        # dynamically add methods for creating inputs and outputs
        # this will make things like updating datatype mappings
        # and shape checks etc. simpler
        def _add_exposed_tensor(exposed_type):
            output_type = getattr(
                io.model_config, "Model{}".format(exposed_type.title())
            )

            def f(
                obj,
                name: str,
                shape: typing.Tuple[typing.Optional[int], ...],
                dtype: str = "float32",  # TODO: include as some sort of enum?
            ) -> output_type:
                assert dtype in ("float32", "int64"), f"Unknown dtype {dtype}"

                shape = (x or -1 for x in shape)
                # TODO: add data_type mapping
                exposed_obj = output_type(
                    name=name,
                    dims=shape,
                    data_type=io.model_config.DataType.TYPE_FP32,
                )

                current_exposed = getattr(obj._config, exposed_type)
                current_exposed.append(exposed_obj)
                return exposed_obj

            setattr(self, f"add_{exposed_type}", types.MethodType(f, self))

        for exposed in ["input", "output"]:
            _add_exposed_tensor(exposed)

    def __getattr__(self, name):
        try:
            return self._config.__getattribute__(name)
        except AttributeError as e:
            raise AttributeError from e

    @classmethod
    def read(cls, model: "Model"):
        obj = cls(model, Platform.DYNAMIC)
        config = io.read_config(obj.path)
        obj._config = config
        return obj

    def write(self):
        io.write_config(self._config, self.path)

    def add_instance_group(
        self,
        kind: str = "gpu",  # TODO: add choices
        gpus: typing.Optional[typing.Union[int, typing.List[int]]] = None,
        count: int = 1,
    ) -> io.model_config.ModelInstanceGroup:
        assert kind in ("cpu", "gpu")
        if isinstance(gpus, int):
            if gpus == 0 and kind == "gpu":
                gpus = [gpus]
            elif kind == "gpu":
                # TODO: is this the behavior we want?
                gpus = [i for i in range(gpus)]
        if kind == "gpu" and gpus is None:
            gpus = [0]

        instance_group = io.model_config.ModelInstanceGroup(
            kind=kind, count=count
        )
        if gpus is not None:
            instance_group.gpus.extend(gpus)
        self._config.instance_group

        return instance_group

    def __repr__(self):
        return str(self._config)


class Platform(enum.Enum):
    ONNX = "onnx"
    TRT = "tensorrt_plan"
    DYNAMIC = None


@attr.s(auto_attribs=True)
class Model:
    name: str
    repository: "ModelRepository"
    platform: str = attr.ib(default=None, converter=Platform)

    def __attrs_post_init__(self):
        io.soft_makedirs(self.path)

        # try to load an existing config, if it exists
        try:
            self.config = ModelConfig.read(self)
        except FileNotFoundError:
            # if it doesn't, we have to specify what platform
            # our new model uses
            if self.platform == Platform.DYNAMIC:
                raise ValueError("Must specify platform for new model")
            self.config = ModelConfig(name=self.name, platform=self.platform)
            return

        if self.config.platform is not None and self.platform != Platform.DYNAMIC:
            # if the config specifies a platform and the
            # initialization did too, make sure they match
            # TODO: just warn instead and prefer the specified
            # one?
            if self.platform.value != self.config.platform:
                raise ValueError(
                    f"Existing config for model {self.name} "
                    f"specifies platform {self.config.platform}, which doesn't match "
                    f"specified platform {self.platform.value}"
                )
        elif self.platform == Platform.DYNAMIC:
            # otherwise if the initialization didn't specify
            # anything, try to grab the platform from the config
            try:
                self.platform = Platform(self.config.platform)
            except ValueError:
                raise ValueError(
                    f"Existing config for model {self.name} "
                    f"specifies unknown platform {self.config.platform}"
                )
        else:
            # otherwise we don't have a platform from anywhere so
            # raise an error
            raise ValueError(f"Model {self.name} config missing platform")

    @property
    def path(self):
        return os.path.join(self.repository.path, self.name)


@attr.s(auto_attribs=True)
class ModelRepository:
    path: str
    default_platform: str = attr.ib(default="tensorrt_plan")

    def __attrs_post_init__(self):
        io.soft_makedirs(self.path)

        self.models = []
        model_names = next(os.walk(self.path))[1]
        for model_name in model_names:
            try:
                self.create(model_name)
            except ValueError:
                self.create(model_name, self.default_platform)

    def create(self, name, platform=None, force=False):
        if any([model.name == name for model in self.models]) and not force:
            raise ValueError("Model {} already exists".format(name))
        elif force:
            # append an index to the name of the model starting at 0
            pattern = re.compile(f"{name}_[0-9]+")
            matches = [
                model.name
                for model in self.models
                if pattern.fullmatch(model.name) is not None
            ]

            if len(matches) == 0:
                # no postfixed models have been made yet, start at 0
                index = 0
            else:
                # search for the first available index
                pattern = re.compile(f"(?<={name}_)[0-9]+")
                postfixes = [int(pattern.search(x).group(0)) for x in matches]
                for index, postfix in zip(count(0), sorted(postfixes)):
                    if index != postfix:
                        break
                else:
                    # indices up to len(matches) are taken,
                    # increment to the next available
                    index += 1
            name += f"_{index}"

        try:
            model = Model(name=name, repository=self, platform=platform)
        except ValueError:
            # TODO: this catch needs to be more specific since
            # there's a bunch of ValueErrors in the post_init now
            raise ValueError("Unknown platform {}".format(platform))

        self.models.append(model)
        return model
