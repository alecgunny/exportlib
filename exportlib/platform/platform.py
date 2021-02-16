# from __future__ import annotations

import abc
import enum
import os
import typing
from collections import OrderedDict

import attr

if typing.TYPE_CHECKING:
    from exportlib.model_repository import Model


_SHAPE_TYPE = typing.Optional[
    typing.Dict[str, typing.Tuple[typing.Optional[int], ...]]
]


class PlatformName(enum.Enum):
    ONNX = "onnxruntime_onnx"
    TF = "tensorflow_savedmodel"  # TODO: support pbtxt?
    TRT = "tensorrt_plan"
    ENSEMBLE = "ensemble"
    DYNAMIC = None


@attr.s(auto_attribs=True)
class Platform(metaclass=abc.ABCMeta):
    model: "Model"

    def _check_exposed_tensors(self, exposed_type, provided=None):
        exposed = getattr(self.model.config, exposed_type)
        if len(exposed) == 0 and provided is None:
            # our config doesn't have any exposed tensors
            # already, and we haven't provided any
            raise ValueError("Must specify {} shapes".format(exposed_type))
        elif len(exposed) == 0:
            # our config doesn't have any exposed tensors,
            # but we've provided some, so add them to the
            # config
            if not isinstance(provided, dict):
                provided = {
                    f"{exposed_type}_{i}": shape
                    for i, shape in enumerate(provided)
                }

            for name, shape in provided.items():
                # TODO: support variable length axes beyond
                # just the batch dimension
                if any([i is None for i in shape[1:]]):
                    raise ValueError(
                        "Shape {} has variable length axes outside "
                        "of the first dimension. This isn't allowed "
                        "at the moment".format(shape)
                    )
                add_fn = getattr(self.model.config, "add_" + exposed_type)

                # TODO: don't hardcode dtype
                add_fn(name=name, shape=shape, dtype="float32")
        elif provided is not None:
            # our config has some exposed tensors already, and
            # we've provided some, so make sure everything matches
            if not isinstance(provided, dict):
                provided = {x.name: shape for x, shape in zip(exposed, provided)}

            if len(provided) != len(exposed) or set(provided) != set(
                [x.name for x in exposed]
            ):
                raise ValueError(
                    "Provided {exposed_type}s {provided} "
                    "don't match config {exposed_type}s {config}".format(
                        exposed_type=exposed_type,
                        provided=list(provided.keys()),
                        config=[x.name for x in exposed],
                    )
                )

            # next check that the shapes match
            for ex in exposed:
                config_shape = list(ex.dims)
                provided_shape = [i or -1 for i in provided[ex.name]]
                if (
                    len(config_shape) != len(provided_shape)
                    or config_shape != provided_shape
                ):
                    raise ValueError(
                        "Shapes {}, {} don't match".format(
                            tuple(config_shape), tuple(provided_shape)
                        )
                    )

    def export(
        self,
        model_fn: typing.Callable,
        version: int,
        input_shapes: _SHAPE_TYPE = None,
        output_names: typing.Optional[typing.List[str]] = None,
        verbose: int = 0,
    ) -> str:
        self._check_exposed_tensors("input", input_shapes)

        # now that we know we have inputs added to our
        # model config, use that config to generate
        # framework tensors that we'll feed through
        # the network to inspect the output
        input_tensors = {}
        for input in self.model.config.input:
            shape = (
                i if i != -1 else self.model.config.max_batch_size
                for i in input.dims
            )
            input_tensors[input.name] = self._make_tensor(shape)

        # use function signature from module.forward
        # method to infer the order in which to pass inputs
        # TODO: what will this do for *args
        parameters = OrderedDict(self._parse_model_fn_parameters(model_fn))

        # get rid of any **kwargs
        try:
            parameters.pop("kwargs")
        except KeyError:
            pass

        assert len(parameters) == len(input_tensors)

        if len(parameters) == 1:
            # if we have simple 1 -> 1 mapping, don't overcomplicate it
            input_names = list(input_tensors)
        else:
            input_names = list(parameters)

        try:
            input_tensors = [input_tensors[name] for name in input_names]
        except KeyError:
            # input names don't match module.forward parameter names,
            # so give our best guess on ordering and sort the
            # inputs alphabetically
            # TODO: at this point, does it make sense to just raise
            # an error here? Alternatively, since parameter names like
            # "x" are sort of standard in Module.forward definitions,
            # does it make sense to provide an `input_name_map` kwarg
            # that maps to more descriptive input names for Triton?
            input_names = sorted(input_tensors.keys())
            input_tensors = [input_tensors[name] for name in input_names]

        # TODO: in Keras, we'll be able to get most of this
        # information by static inspection of the model.
        # How will that fit in to this section?
        outputs = model_fn(*input_tensors)
        if isinstance(outputs, self._tensor_type):
            outputs = [outputs]
        shapes = [tuple(x.shape) for x in outputs]
        if any([x.dims[0] == -1 for x in self.model.config.input]):
            shapes = [(None,) + s[1:] for s in shapes]

        if output_names is not None:
            shapes = {name: shape for name, shape in zip(output_names, shapes)}
        self._check_exposed_tensors("output", shapes)

        export_path = self._do_export(
            model_fn, self._make_export_path(version), verbose=verbose
        )

        # write out the config for good measure
        self.model.config.write()
        return export_path

    @property
    @abc.abstractmethod
    def _tensor_type(self):
        pass

    @abc.abstractmethod
    def _make_tensor(self, shape):
        pass

    @abc.abstractmethod
    def _do_export(self, model_fn, export_dir, verbose=0):
        pass

    @abc.abstractmethod
    def _parse_model_fn_parameters(self, model_fn):
        pass

    @abc.abstractmethod
    def _make_export_path(version):
        pass
