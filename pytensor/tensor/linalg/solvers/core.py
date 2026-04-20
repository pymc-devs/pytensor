import logging

import pytensor.tensor.math as ptm
from pytensor import tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor.basic import as_tensor_variable
from pytensor.tensor.linalg.dtype_utils import linalg_output_dtype
from pytensor.tensor.type import tensor


logger = logging.getLogger(__name__)


class SolveBase(Op):
    """Base class for `scipy.linalg` matrix equation solvers."""

    __props__: tuple[str, ...] = (
        "lower",
        "b_ndim",
        "overwrite_a",
        "overwrite_b",
    )

    def __init__(
        self,
        *,
        lower=False,
        b_ndim,
        overwrite_a=False,
        overwrite_b=False,
    ):
        self.lower = lower

        assert b_ndim in (1, 2)
        self.b_ndim = b_ndim
        if b_ndim == 1:
            self.gufunc_signature = "(m,m),(m)->(m)"
        else:
            self.gufunc_signature = "(m,m),(m,n)->(m,n)"
        self.overwrite_a = overwrite_a
        self.overwrite_b = overwrite_b
        destroy_map = {}
        if self.overwrite_a and self.overwrite_b:
            # An output destroying two inputs is not yet supported
            # destroy_map[0] = [0, 1]
            raise NotImplementedError(
                "It's not yet possible to overwrite_a and overwrite_b simultaneously"
            )
        elif self.overwrite_a:
            destroy_map[0] = [0]
        elif self.overwrite_b:
            destroy_map[0] = [1]
        self.destroy_map = destroy_map

    def perform(self, node, inputs, outputs):
        raise NotImplementedError(
            "SolveBase should be subclassed with an perform method"
        )

    def make_node(self, A, b):
        A = as_tensor_variable(A)
        b = as_tensor_variable(b)

        if A.ndim != 2:
            raise ValueError(f"`A` must be a matrix; got {A.type} instead.")
        if b.ndim != self.b_ndim:
            raise ValueError(f"`b` must have {self.b_ndim} dims; got {b.type} instead.")

        o_dtype = linalg_output_dtype(A.dtype, b.dtype)
        x = tensor(dtype=o_dtype, shape=b.type.shape)
        return Apply(self, [A, b], [x])

    def infer_shape(self, fgraph, node, shapes):
        Ashape, Bshape = shapes
        rows = Ashape[1]
        if len(Bshape) == 1:
            return [(rows,)]
        else:
            cols = Bshape[1]
            return [(rows, cols)]

    def pullback(self, inputs, outputs, output_gradients):
        r"""Reverse-mode gradient updates for matrix solve operation :math:`c = A^{-1} b`.

        Symbolic expression for updates taken from [#]_.

        References
        ----------
        .. [#] M. B. Giles, "An extended collection of matrix derivative results
          for forward and reverse mode automatic differentiation",
          http://eprints.maths.ox.ac.uk/1079/

        """
        A, _b = inputs

        c = outputs[0]
        # C is a scalar representing the entire graph
        # `output_gradients` is (dC/dc,)
        # We need to return (dC/d[inv(A)], dC/db)
        c_bar = output_gradients[0]

        props_dict = self._props_dict()
        props_dict["lower"] = not self.lower

        solve_op = type(self)(**props_dict)

        b_bar = solve_op(A.mT, c_bar)
        # force outer product if vector second input
        A_bar = -ptm.outer(b_bar, c) if c.ndim == 1 else -b_bar.dot(c.T)

        if props_dict.get("unit_diagonal", False):
            n = A_bar.shape[-1]
            A_bar = A_bar[pt.arange(n), pt.arange(n)].set(pt.zeros(n))

        return [A_bar, b_bar]


def _default_b_ndim(b, b_ndim):
    if b_ndim is not None:
        assert b_ndim in (1, 2)
        return b_ndim

    b = as_tensor_variable(b)
    if b_ndim is None:
        return min(b.ndim, 2)  # By default, assume the core case is a matrix
