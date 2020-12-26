import enum
import os
import re
import typing
import warnings
from itertools import count

import attr

from exportlib import io


class model_config:
    @attr.s(auto_attribs=True)
    class ModelConfig:
        name: str
        platform: str


class Platform(enum.Enum):
    ONNX = "onnx"
    TRT = "tensorrt_plan"
    DYNAMIC = None


@attr.s(auto_attribs=True)
class Model:
    name: int
    repository: "ModelRepository"
    platform: str = attr.ib(default=None, converter=Platform)

    def __attrs_post_init__(self):
        io.soft_makedirs(self.path)

        config = None
        if os.path.exists(self.config_path):
            config = io.read_config(self.config_path)
            if config is None:
                pass
            elif (
                config.platform is not None and self.platform != Platform.DYNAMIC
            ):
                assert self.platform.value == config.platform
            elif self.platform == Platform.DYNAMIC:
                self.platform = Platform(config.platform)
            else:
                raise ValueError(f"Model {self.name} config missing platform")

        if config is not None:
            self.config = config
        elif self.platform == Platform.DYNAMIC:
            raise ValueError("Must specify platform for new model")
        else:
            self.config = io.model_config.ModelConfig(
                name=self.name, platform=self.platform.value
            )

    @property
    def path(self):
        return os.path.join(self.repository.path, self.name)

    @property
    def config_path(self):
        return os.path.join(self.path, "config.pbtxt")


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
            raise ValueError("Unknown platform {}".format(platform))

        self.models.append(model)
        return model
