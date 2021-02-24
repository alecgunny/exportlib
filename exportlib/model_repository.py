import os
import re
import shutil
import typing
from itertools import count

import attr

from exportlib import Model, io
from exportlib.platform import PlatformName


@attr.s(auto_attribs=True)
class ModelRepository:
    path: str
    default_platform: str = attr.ib(default=PlatformName.TRT)

    def __attrs_post_init__(self):
        io.soft_makedirs(self.path)

        self.models = {}
        model_names = next(os.walk(self.path))[1]
        for model_name in model_names:
            try:
                self.create_model(model_name)
            except ValueError:
                self.create_model(model_name, self.default_platform)

    def create_model(
        self,
        name: str,
        platform: typing.Optional[str] = None,
        force: bool = False,
    ) -> "Model":
        if name in self.models and not force:
            raise ValueError("Model {} already exists".format(name))
        elif name in self.models:
            # append an index to the name of the model starting at 0
            pattern = re.compile(f"{name}_[0-9]+")
            matches = list(filter(pattern.fullmatch, self.models))

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
            raise

        self.models[name] = model
        return model

    def remove_model(self, name: str):
        try:
            model = self.models.pop(name)
        except KeyError:
            raise ValueError(f"Unrecognized model {name}")

        shutil.rmtree(model.path)

    def delete(self):
        model_names = self.models.keys()
        for model_name in model_names:
            self.remove_model(model_name)
        shutil.rmtree(self.path)
