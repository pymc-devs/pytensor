"""ONNX backend for PyTensor.

This module provides functionality to export PyTensor graphs to ONNX format
and execute them using ONNX Runtime.
"""

from pytensor.link.onnx.dispatch import onnx_funcify, onnx_typify
from pytensor.link.onnx.export import compile_onnx, export_function_onnx, export_onnx
from pytensor.link.onnx.linker import ONNXLinker


# ONNX opset version used by default
ONNX_OPSET_VERSION = 18

__all__ = [
    "ONNX_OPSET_VERSION",
    "ONNXLinker",
    "compile_onnx",
    "export_function_onnx",
    "export_onnx",
    "onnx_funcify",
    "onnx_typify",
]
