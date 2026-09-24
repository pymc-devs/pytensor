"""JavaScript code generation for PyTensor graphs."""

# isort: off
from pytensor.link.js.dispatch.basic import JSProgram, js_funcify, lower

# Register the supported Ops with js_funcify.
import pytensor.link.js.dispatch.elemwise

# isort: on


__all__ = ["JSProgram", "js_funcify", "lower"]
