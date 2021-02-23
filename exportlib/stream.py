import os
import typing

import tensorflow as tf
from tritonclient.grpc import model_config_pb2 as model_config

from exportlib.platform import PlatformName

if typing.TYPE_CHECKING:
    from exportlib import ModelRepository
    from exportlib.model import ExposedTensor, Model


@tf.keras.utils.register_keras_serializable(name="Snapshotter")
class Snapshotter(tf.keras.layers.Layer):
    def __init__(
        self,
        snapshot_size: int,
        channels: typing.Union[int, typing.List[int]],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.snapshot_size = snapshot_size

        if isinstance(channels, int):
            channels = [channels]
        self.channels = channels

    def build(self, input_shape):
        if input_shape[0] is None:
            raise ValueError("Must specify batch dimension")
        if input_shape[0] != 1:
            # TODO: support batching
            raise ValueError("Batching not currently supported")
        if sum(self.channels) != input_shape[1]:
            raise ValueError(
                "Number of channels specified {} doesn't "
                "match number of channels found {}".format(
                    sum(self.channels), input_shape[1]
                )
            )

        self.snapshot = self.add_weight(
            name="snapshot",
            shape=(input_shape[0], input_shape[1], self.size),
            dtype=tf.float32,
            initializer="zeros",
            trainable=False,
        )
        self.update_size = input_shape[2]

    def call(self, stream):
        update = tf.concat(
            [self.snapshot[:, :, self.update_size :], stream], axis=2
        )
        self.snapshot.assign(update)
        return tf.split(update, self.channels, axis=1)

    def compute_output_shape(self, input_shape):
        return [
            tf.TensorShape([input_shape[0], i, self.snapshot_size])
            for i in self.channels
        ]

    def get_config(self):
        config = {"snapshot_size": self.snapshot_size, "channels": self.channels}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def make_streaming_input_model(
    repository: "ModelRepository",
    inputs: "ExposedTensor",
    stream_size: int,
    name: typing.Optional[str] = None,
) -> "Model":
    channels = [x.shape[1] for x in inputs]
    input = tf.keras.Input(
        name="stream",
        shape=(sum(channels), stream_size),
        batch_size=1,  # TODO: other batch sizes
        dtype=tf.float32,
    )
    output = Snapshotter(inputs[0].shape[-1], channels)
    model = tf.keras.Model(inputs=input, outputs=output)

    tf_model = repository.create_model(
        name=name or "snapshotter", platform=PlatformName.TF, force=True
    )
    tf_model.config.sequence_batching = model_config.ModelSequenceBatching(
        max_sequence_idle_microseconds=5000000,
        direct=model_config.ModelSequenceBatching.StrategyDirect,
    )
    tf_model.config.model_warmup.append(
        model_config.ModelWarmup(
            inputs={
                "stream": model_config.ModelWarmup.Input(
                    dims=[1, sum(channels), stream_size],
                    data_type=model_config.TYPE_FP32,
                    zero_data=True,
                )
            },
            name="zeros_warmup",
        )
    )
    tf_model.config.add_input(
        name="stream", shape=(1, sum(channels), stream_size), dtype="float32"
    )
    for n, x in enumerate(inputs):
        postfix = "" if n == 0 else f"_{n}"
        tf_model.config.add_output(
            name="snapshotter" + postfix, shape=x.shape, dtype="float32"
        )

    # TODO: add actual platform save
    model.save(os.path.join(tf_model.path, "1", "model.savedmodel"))
    tf_model.config.write(os.path.join(tf_model.path, "config.pbtxt"))
    return tf_model
