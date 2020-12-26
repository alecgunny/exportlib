import enum
import os
import re
import typing
import warnings
from itertools import count

import attr

from exportlib import io


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

        # try to load an existing config, if it exists
        try:
            config = io.read_config(self.config_path)
        except FileNotFoundError:
            # if it doesn't, we have to specify what platform
            # our new model uses
            if self.platform == Platform.DYNAMIC:
                raise ValueError("Must specify platform for new model")
            self.config = io.model_config.ModelConfig(
                name=self.name, platform=self.platform.value
            )
            return

        if config.platform is not None and self.platform != Platform.DYNAMIC:
            # if the config specifies a platform and the
            # initialization did too, make sure they match
            # TODO: just warn instead and prefer the specified
            # one?
            if self.platform.value != config.platform:
                raise ValueError(
                    f"Existing config for model {self.name} "
                    f"specifies platform {config.platform}, which doesn't match "
                    f"specified platform {self.platform.value}"
                )
        elif self.platform == Platform.DYNAMIC:
            # otherwise if the initialization didn't specify
            # anything, try to grab the platform from the config
            try:
                self.platform = Platform(config.platform)
            except ValueError:
                raise ValueError(
                    f"Existing config for model {self.name} "
                    f"specifies unknown platform {config.platform}"
                )
        else:
            # otherwise we don't have a platform from anywhere so
            # raise an error
            raise ValueError(f"Model {self.name} config missing platform")

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
            # TODO: this catch needs to be more specific since
            # there's a bunch of ValueErrors in the post_init now
            raise ValueError("Unknown platform {}".format(platform))

        self.models.append(model)
        return model
