import os
import typing

from exportlib import io
from exportlib.platform import PlatformName

if typing.TYPE_CHECKING:
    from exportlib.model import ExposedTensor, Model


def _add_exposed_tensor(f):
    """
    Decorator for adding input/output adding methods
    to the config class. Doing it this way in order to simplify
    things like syntax updates and building the data type map
    """
    exposed_type = f.__name__.split("_")[1]
    output_type = getattr(io.model_config, "Model{}".format(exposed_type.title()))

    def wrapper(
        obj: "ModelConfig",
        name: str,
        shape: typing.Tuple[typing.Optional[int], ...],
        dtype: str = "float32",
        # TODO: should dtype be some sort of enum? How to
        # handle more general data types?
        **kwargs,  # including kwargs for reshaping later or something
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
        f(exposed_obj, **kwargs)
        return exposed_obj

    wrapper.__name__ = f.__name__

    docstring = f"""
    :param name: name for the {exposed_type}
    :param shape: tuple of tensor dimensions.
        Use `None` to indicate a variable size.
    :param dtype: Data type of the {exposed_type}
    """
    wrapper.__doc__ = f.__doc__ + "\n" + docstring
    return wrapper


class ModelConfig:
    """
    Wrapper around the `tritonclient.grpc.model_config_pb2.ModelConfig`
    protobuf Message to simplify and standardize syntax around things
    like path naming, reading/writing, and adding inputs/outputs.
    :param model: the Triton `Model` object that this config describes
    :param platform: the desired inference software `Platform` for the config
    :param max_batch_size: the maximimum batch size this model will see
        during inference. If left as `None`, the model will have a fixed batch
        size and all inputs and outputs must have explicitly described dims
    """

    def __new__(cls, name: str, platform: PlatformName, **kwargs) -> None:
        if platform is PlatformName.ENSEMBLE:
            cls = EnsembleConfig

        obj = super().__new__(cls)
        obj.__init__(name, platform, **kwargs)
        return obj

    def __init__(self, name: str, platform: PlatformName, **kwargs) -> None:
        self._config = io.model_config.ModelConfig(
            name=name, platform=platform.value, **kwargs
        )

    def __getattr__(self, name):
        try:
            return self._config.__getattribute__(name)
        except AttributeError as e:
            raise AttributeError from e

    @classmethod
    def read(cls, path: str):
        config = io.read_config(path)
        # TODO: check for valid platform names here?
        obj = cls(config.name, PlatformName.DYNAMIC)
        obj._config = config
        return obj

    def write(self, path):
        """
        Write out the protobuf config to the model's
        folder in the model repository
        """
        # TODO: add some validation checks here
        # For example:
        # ------------
        #    - if max_batch_size is not set, ensure that
        #        all inputs and outputs have dims > 0
        io.write_config(self._config, path)

    @_add_exposed_tensor
    def add_input(input: io.model_config.ModelInput, **kwargs):
        """
        add an input
        """
        return

    @_add_exposed_tensor
    def add_output(output: io.model_config.ModelOutput, **kwargs):
        """
        add an output
        """
        return

    def add_instance_group(
        self,
        kind: str = "gpu",
        gpus: typing.Optional[typing.Union[int, typing.List[int]]] = None,
        count: int = 1,
    ) -> io.model_config.ModelInstanceGroup:
        # first figure out which GPUs to use
        # passing a single integer will be interpreted as a
        # range of GPU indices, except for 0, which will
        # default to using GPU 0. The default of None is
        # interpreted as using GPU 0 as well.
        # TODO: is this range behavior really what we want?
        # I think so, if you want to specify a single non-zero
        # GPU index, you can always put it in a one element list
        if isinstance(gpus, int):
            if gpus == 0 and kind == "gpu":
                gpus = [gpus]
            elif kind == "gpu":
                gpus = [i for i in range(gpus)]
        if kind == "gpu" and gpus is None:
            gpus = [0]

        # next deal with the instance group, mapping from
        # our lowercase value to the protobuf enum expected
        # by the config
        try:
            kind = io.model_config.ModelInstanceGroup.Kind.Value(
                "KIND_{}".format(kind.upper())
            )
        except ValueError:
            options = ", ".join(typing.get_args(self._INSTANCE_GROUP_KINDS))
            raise ValueError(
                f"Could not understand instance group kind {kind}, "
                f"must be one of {options}"
            )

        # create the instance group, only assign GPUs
        # if they've been specified
        instance_group = io.model_config.ModelInstanceGroup(
            kind=kind, count=count
        )
        if gpus is not None:
            instance_group.gpus.extend(gpus)
        self._config.instance_group.append(instance_group)

        return instance_group

    def __repr__(self):
        return str(self._config)


class EnsembleConfig(ModelConfig):
    def add_model(self, model: "Model", version: typing.Optional[int] = None):
        version = version or -1
        step = io.model_config.ModelEnsembling.Step(
            model_name=model.name, model_version=version
        )
        self._config.ensemble_scheduling.step.append(step)
        self.models[model.name] = model
        return step

    def pipe(
        self,
        input_tensor: "ExposedTensor",
        output_tensor: "ExposedTensor",
        name: typing.Optional[str] = None,
    ):
        input_tensor_model, input_tensor_name = input_tensor.split(".")
        output_tensor_model, output_tensor_name = output_tensor.split(".")

        # TODO: combine this into one function that
        # gets applied for both inputs and outputs
        for step in self._config.ensemble_scheduling.step:
            if step.model_name == input_tensor_model:
                model = self.models[step.model_name]

                if input_tensor_name not in model.outputs:
                    raise ValueError(
                        "Unrecognized output tensor {} from "
                        "model {}".format(input_tensor_name, model.name)
                    )
                if input_tensor_name not in step.output_map:
                    # we haven't mapped this tensor to anything
                    # before, so add it with either its own key
                    # or the provided one
                    name = name or input_tensor_name
                    step.output_map[input_tensor_name] = name
                elif step.output_map[input_tensor_name] != name:
                    # we have seen it before, but didn't pass
                    # the name we saw before
                    if name is None:
                        # if this is because we didn't pass a name
                        # at all, then just use the existing one
                        name = step.output_map[input_tensor_name]
                    else:
                        # otherwise throw an error
                        raise ValueError(
                            "Output tensor {} for model {} "
                            "already maps to name {}. Name "
                            "{} was provided".format(
                                input_tensor_name,
                                step.model_name,
                                step.output_map[input_tensor_name],
                                name,
                            )
                        )
                break
        else:
            if name is not None:
                raise ValueError("Can't specify key for input name")
            name = input_tensor_name

            if input_tensor_model == "INPUT" and name not in [
                x.name for x in self._config.input
            ]:
                raise ValueError("Unrecognized input tensor {}".format(name))
            elif input_tensor_model != "INPUT":
                raise ValueError(
                    "Model {} not in ensemble!".format(input_tensor_model)
                )

        for step in self._config.ensemble_scheduling.step:
            if step.model_name == output_tensor_model:
                model = self.models[step.model_name]

                if output_tensor_name not in model.inputs:
                    raise ValueError(
                        "Unrecognized input tensor {} from "
                        "model {}".format(output_tensor_name, model.name)
                    )
                if output_tensor_name in step.input_map:
                    raise ValueError(
                        "Input {} for model {} is already "
                        "receiving output from tensor {}".format(
                            output_tensor_name,
                            model.name,
                            step.input_map[output_tensor_name],
                        )
                    )
                step.input_map[output_tensor_name] = name
                break

        else:
            if output_tensor_model == "OUTPUT" and name not in [
                x.name for x in self._config.output
            ]:
                raise ValueError("Unrecognized output tensor {}".format(name))
            elif output_tensor_model != "OUTPUT":
                raise ValueError(
                    "Model {} not in ensemble!".format(output_tensor_model)
                )
