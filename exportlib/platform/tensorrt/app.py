import pickle

from flask import Flask, Response, request
from tritonclient.grpc.model_config_pb2 import ModelConfig

from exportlib.platform.tensorrt.onnx import convert_network

app = Flask(__name__)


@app.errorhandler(Exception)
def handle_500(e):
    app.logger.error(str(e))
    return str(e), 500


@app.route("/onnx", methods=["POST", "GET"])
def index():
    # TODO: this is mega unsafe
    data = pickle.loads(request.data)

    # TODO: add in checking and error returns
    # for bad configs, models, etc.
    config = ModelConfig()
    config.MergeFromString(data["config"])

    try:
        use_fp16 = data["use_fp16"]
    except KeyError:
        use_fp16 = False

    engine = convert_network(data["network"], config, use_fp16)

    if engine is None:
        app.logger.error("Model conversion failed")
        return "Model conversion failed", 500
    engine = bytes(engine.serialize())
    return Response(engine, content_type="application/octet-stream")


if __name__ == "__main__":
    app.run()
