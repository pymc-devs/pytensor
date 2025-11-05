"""ONNX dispatch system for converting PyTensor operations to ONNX."""

# isort: off
from pytensor.link.onnx.dispatch.basic import onnx_funcify, onnx_typify

# Load dispatch specializations
import pytensor.link.onnx.dispatch.elemwise  # noqa: F401
import pytensor.link.onnx.dispatch.shape  # noqa: F401

# isort: on
