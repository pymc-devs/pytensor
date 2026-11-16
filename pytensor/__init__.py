import sys


__docformat__ = "restructuredtext en"


from pytensor import _version


__version__: str = _version.get_versions()["version"]

del _version


from pytensor.configdefaults import config


# isort: off
from pytensor import tensor
from pytensor.compile import (
    In,
    Mode,
    Out,
    shared,
    wrap_py,
)
from pytensor.compile.maker import function
from pytensor.compile.mode import get_mode
from pytensor.gradient import Lop, Rop, grad, pullback, pushforward
from pytensor.printing import debugprint as dprint

# Preload the deprecated module before binding the helper with the same name.
import pytensor.ifelse
from pytensor.conditional import ifelse
from pytensor.scan.basic import scan
from pytensor.scan.views import map
from pytensor.compile.builders import OpFromGraph
import pytensor.compile.rewriting  # register OpFromGraph rewrites in optdb
from pytensor.link.jax.ops import wrap_jax
from pytensor import _sparse_lazy
# isort: on


def __getattr__(name):
    if name == "sparse":
        # During pytensor.sparse's own import, submodules may do
        # `import pytensor.sparse.X as Y` which probes pytensor.sparse via
        # getattr before the parent attribute has been set. Return the
        # partially-loaded module from sys.modules to avoid re-entry.
        if "pytensor.sparse" in sys.modules:
            return sys.modules["pytensor.sparse"]
        import pytensor.sparse as sparse

        return sparse
    raise AttributeError(f"module 'pytensor' has no attribute {name!r}")


# Some config variables are registered by submodules. Only after all those
# imports were executed, we can warn about remaining flags provided by the user
# through PYTENSOR_FLAGS.
config.warn_unused_flags()

from pytensor.compile.rewriting import register_canonicalize
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.tensor.basic import Alloc, MakeVector, constant
from pytensor.tensor.elemwise import Cast
from pytensor.tensor.shape import Reshape
from pytensor.tensor.variable import TensorConstant

def _is_int(dtype):
    return dtype.startswith(("int", "uint"))

def _strip_cast(var):
    if var.owner is not None and isinstance(var.owner.op, Cast):
        inp = var.owner.inputs[0]
        if _is_int(var.dtype) and _is_int(inp.dtype):
            return _strip_cast(inp)
    return var

def _strip_shape(var):
    if var.owner is None:
        return var
    if isinstance(var.owner.op, Cast):
        inp = var.owner.inputs[0]
        if _is_int(var.dtype) and _is_int(inp.dtype):
            return _strip_shape(inp)
    if isinstance(var.owner.op, MakeVector):
        new = [_strip_cast(i) for i in var.owner.inputs]
        if all(n is o for n, o in zip(new, var.owner.inputs)):
            return var
        dtypes = {i.dtype for i in new}
        if len(dtypes) == 1:
            return MakeVector(dtypes.pop())(*new)
        target = next((i.dtype for i in new if not isinstance(i, TensorConstant)), new[0].dtype)
        out = []
        for i in new:
            if i.dtype == target:
                out.append(i)
            elif isinstance(i, TensorConstant):
                try:
                    out.append(constant(i.data, dtype=target))
                except (OverflowError, ValueError):
                    return var
            else:
                return var
        return MakeVector(target)(*out)
    return var

@register_canonicalize
@node_rewriter([Alloc, Reshape])
def local_useless_int_shape_cast(fgraph, node):
    if isinstance(node.op, Alloc):
        new = [node.inputs[0]] + [_strip_cast(i) for i in node.inputs[1:]]
        if all(n is o for n, o in zip(new, node.inputs)):
            return None
        return [node.op(*new)]
    new = _strip_shape(node.inputs[1])
    if new is node.inputs[1]:
        return None
    return [node.op(node.inputs[0], new)]
