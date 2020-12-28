# Exporting Deep Learning Models for Accelerated Inference on Triton Inference Server

Tools for facilitating the export of trained neural networks to a Triton Inference Server model repository. Right now, this involves a fair amount of boilerplate for keeping track of standardized paths and filenames, as well as general protobuf boilerplate. This is an attempt to consolidate this to more standard, intuitive API calls that make managing such a repository simpler.

Since this is very much a work in progress, there's not much to document for now. The main question is whether to package this all up as a command line tool, or just as libraries to be imported and used in a training script after training has completed. Since the TensorRT components need to be utilized on the same hardware that will be used at inference time (which will presumably be different than your training hardware), it may end up making sense to package this part up as a Flask application as a sort of conversion service to be deployed on nodes with the relevant hardware, possibly with an nginx server in front of them to route calls to the appropriate hardware.