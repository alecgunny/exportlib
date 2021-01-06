# Exporting Deep Learning Models for Accelerated Inference on Triton Inference Server

Tools for facilitating the export of trained neural networks to a Triton Inference Server model repository. Right now, this involves a fair amount of boilerplate for keeping track of standardized paths and filenames, as well as general protobuf boilerplate. This is an attempt to consolidate this to more standard, intuitive API calls that make managing such a repository simpler.

Since this is very much a work in progress, there's not much to document for now. The main question is whether to package this all up as a command line tool, or just as libraries to be imported and used in a training script after training has completed. Right now I'm learning towards this usage, with something like:

```python
import torch

from exportlib.model_repository import ModelRepository
from exportlib.platform import PlatformName


# define your torch model
class NN(torch.nn.Module):
    ...

# do whatever your usual training on it is
nn = NN(...)
do_some_training(nn, ...)

# instantiate a model repository and add your model to it
repo = ModelRepository("/tmp/repo")
model = repo.create_model("my_nn", platform=PlatformName.ONNX)

# configure some stuff about your model, might
# consider how to fit these into `create_model`
model.config.max_batch_size = 64
model.add_instance_group("gpu", count=4)

# export the onnx binary and the protobuf config
export_path = model.export(nn, input_shapes={"input": (None, 256)})

# <repo_dir>/<model.name>/<version: defaulted to 1>/<standard onnx filename>
# "/tmp/repo/my_nn/1/model.onnx"
print(export_path)
```

Since the TensorRT components need to be utilized on the same hardware that will be used at inference time (which will presumably be different than your training hardware), it may end up making sense to package the TensorRT portion of this as a Flask application as a sort of conversion service to be deployed on nodes with the relevant hardware, possibly with an nginx server in front of them to route calls to the appropriate hardware.
