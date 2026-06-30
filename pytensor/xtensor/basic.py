from collections.abc import Sequence

from pytensor.compile.ops import TypeCastingOp
from pytensor.gradient import DisconnectedType, disconnected_type, grad_undefined
from pytensor.graph import Apply, Op
from pytensor.graph.basic import Variable
from pytensor.tensor.type import TensorType, continuous_dtypes
from pytensor.xtensor.type import XTensorType, as_xtensor, xtensor


def grad_connected(var: Variable) -> bool:
    """Whether an XOp input can carry a cotangent (a continuous-dtype xtensor)."""
    return isinstance(var.type, XTensorType) and var.type.dtype in continuous_dtypes


class XOp(Op):
    """A base class for XOps that shouldn't be materialized"""

    def perform(self, node, inputs, outputs):
        raise NotImplementedError(
            f"xtensor operation {self} must be lowered to equivalent tensor operations"
        )

    def do_constant_folding(self, fgraph, node):
        return False

    def pullback(self, inputs, outputs, cotangents):
        # XOps carry no gradient of their own. Defer to LazyGrad, which the
        # expand_lazy_grad rewrite differentiates by lowering core_op to tensor ops and
        # taking their pullback, so no XOp runs lowering inside its own pullback. Discrete
        # xtensor inputs (e.g. integer indices) have an undefined gradient; structural
        # inputs (slices, rngs) are disconnected.
        from pytensor.xtensor.shape import zeros_like

        # A disconnected cotangent (no contribution from that output) becomes a zero,
        # so LazyGrad never takes a DisconnectedType as an input.
        cotangents = [
            zeros_like(out) if isinstance(cot.type, DisconnectedType) else cot
            for cot, out in zip(cotangents, outputs)
        ]
        grads = iter(
            LazyGrad(self, len(outputs))(*inputs, *cotangents, return_list=True)
        )
        return [
            next(grads)
            if grad_connected(inp)
            else grad_undefined(self, i, inp)
            if isinstance(inp.type, XTensorType)
            else disconnected_type()
            for i, inp in enumerate(inputs)
        ]

    def vectorize_node(
        self, node, *new_inputs, new_dim: str | None
    ) -> Sequence[Variable]:
        raise NotImplementedError(f"Vectorized node not implemented for {self}")


class LazyGrad(XOp):
    """Deferred vector-Jacobian product of another XOp.

    Wraps the differentiated ``core_op`` with its inputs and the output cotangents. The
    ``expand_lazy_grad`` rewrite differentiates it by lowering ``core_op`` to tensor ops
    and taking their pullback, so no XOp ever runs lowering inside its own pullback.
    There is one output per differentiable (continuous-dtype) input.
    """

    __props__ = ("core_op", "n_cotangents")

    def __init__(self, core_op: Op, n_cotangents: int):
        self.core_op = core_op
        self.n_cotangents = n_cotangents

    def make_node(self, *inputs):
        forward_inputs = inputs[: -self.n_cotangents]
        outputs = [inp.type() for inp in forward_inputs if grad_connected(inp)]
        return Apply(self, list(inputs), outputs)


class XTypeCastOp(TypeCastingOp):
    """Base class for Ops that type cast between TensorType and XTensorType.

    This is like a `ViewOp` but without the expectation the input and output have identical types.
    """

    def infer_shape(self, node, input_shapes):
        return input_shapes

    def vectorize_node(
        self, node, *new_inputs, new_dim: str | None
    ) -> Sequence[Variable]:
        raise NotImplementedError(f"Vectorized node not implemented for {self}")


class TensorFromXTensor(XTypeCastOp):
    __props__ = ()

    def make_node(self, x):
        if not isinstance(x.type, XTensorType):
            raise TypeError(f"x must be have an XTensorType, got {type(x.type)}")
        output = TensorType(x.type.dtype, shape=x.type.shape)()
        return Apply(self, [x], [output])

    def pullback(self, inputs, outs, g_outs):
        [x] = inputs
        [g_out] = g_outs
        return [xtensor_from_tensor(g_out, dims=x.type.dims)]

    def vectorize_node(self, node, new_x, new_dim):
        [old_x] = node.inputs
        if (new_x.ndim - old_x.ndim) > 1:
            raise NotImplementedError(
                f"Vectorization of {self} cannot guarantee correct placement of multiple batch dimensions. "
                "You can call vectorize_graph one batch dimension at a time, "
                "or pytensor.xtensor.vectorization.vectorize_graph instead."
            )
        new_x = new_x.transpose(..., *old_x.dims)
        return [self(new_x)]


tensor_from_xtensor = TensorFromXTensor()


class XTensorFromTensor(XTypeCastOp):
    __props__ = ("dims",)

    def __init__(self, dims: Sequence[str]):
        super().__init__()

        if isinstance(dims, str):
            dims = (dims,)
        self.dims = tuple(dims)

    def make_node(self, x):
        if not isinstance(x.type, TensorType):
            raise TypeError(f"x must be an TensorType type, got {type(x.type)}")
        output = xtensor(dtype=x.type.dtype, dims=self.dims, shape=x.type.shape)
        return Apply(self, [x], [output])

    def pullback(self, inputs, outs, g_outs):
        [g_out] = g_outs
        return [tensor_from_xtensor(g_out)]

    def vectorize_node(self, node, new_x, new_dim):
        [old_x] = node.inputs
        if new_x.ndim != old_x.ndim:
            if new_dim is None:
                raise NotImplementedError(
                    f"Vectorization of {self} cannot infer the new dimension labels. "
                    "Use pytensor.xtensor.vectorization.vectorize_graph instead."
                )
            return [type(self)(dims=(new_dim, *self.dims))(new_x)]
        else:
            return [self(new_x)]


def xtensor_from_tensor(x, dims, name=None):
    return XTensorFromTensor(dims=dims)(x, name=name)


class Rename(XTypeCastOp):
    __props__ = ("new_dims",)

    def __init__(self, new_dims: tuple[str, ...]):
        super().__init__()
        self.new_dims = new_dims

    def make_node(self, x):
        x = as_xtensor(x)
        output = x.type.clone(dims=self.new_dims)()
        return Apply(self, [x], [output])

    def pullback(self, inputs, outs, g_outs):
        [x] = inputs
        [g_out] = g_outs
        return [type(self)(x.type.dims)(g_out)]

    def vectorize_node(self, node, new_x, new_dim):
        [old_x] = node.inputs
        old_dim_mapping = dict(zip(old_x.dims, self.new_dims, strict=True))

        # new_dims may include a mix of old dims (possibly re-ordered), and new dims which won't be renamed
        new_dims = tuple(
            old_dim_mapping.get(new_dim, new_dim) for new_dim in new_x.dims
        )
        return [type(self)(new_dims)(new_x)]


def rename(x, name_dict: dict[str, str] | None = None, **names: str):
    if name_dict is not None:
        if names:
            raise ValueError("Cannot use both positional and keyword names in rename")
        names = name_dict

    x = as_xtensor(x)
    old_names = x.type.dims
    new_names = list(old_names)
    for old_name, new_name in names.items():
        try:
            new_names[old_names.index(old_name)] = new_name
        except ValueError:
            raise ValueError(
                f"Cannot rename {old_name} to {new_name}: {old_name} not in {old_names}"
            )

    return Rename(tuple(new_names))(x)
