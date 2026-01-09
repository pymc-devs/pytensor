"""ONNX dispatch system for converting PyTensor operations to ONNX."""

# isort: off
from pytensor.link.onnx.dispatch.basic import onnx_funcify, onnx_typify

# Load dispatch specializations
import pytensor.link.onnx.dispatch.elemwise
import pytensor.link.onnx.dispatch.shape
import pytensor.link.onnx.dispatch.math
import pytensor.link.onnx.dispatch.tensor_basic
import pytensor.link.onnx.dispatch.subtensor
import pytensor.link.onnx.dispatch.nlinalg
import pytensor.link.onnx.dispatch.nnet

# isort: on
