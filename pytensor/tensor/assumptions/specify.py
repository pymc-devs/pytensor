from pytensor.compile.ops import TypeCastingOp
from pytensor.graph.basic import Apply, Variable
from pytensor.tensor.basic import as_tensor_variable


class SpecifyAssumptions(TypeCastingOp):
    """No-op that declares structural assumptions on a tensor for use by graph rewrites."""

    __props__ = ("assumptions",)

    def __init__(self, assumptions: frozenset[str]):
        super().__init__()
        self.assumptions = assumptions

    def make_node(self, x):
        x = as_tensor_variable(x)
        out = x.type()
        return Apply(self, [x], [out])

    def infer_shape(self, fgraph, node, input_shapes):
        return input_shapes

    def infer_assumption(self, key, feature, fgraph, node, input_states):
        from pytensor.tensor.assumptions.core import FactState

        if key.name in self.assumptions:
            return [FactState.TRUE]
        return [input_states[0]]

    def grad(self, inputs, output_grads):
        return output_grads


def specify_assumptions(
    x: Variable,
    diagonal: bool | None = None,
    lower_triangular: bool | None = None,
    upper_triangular: bool | None = None,
    symmetric: bool | None = False,
    positive_definite: bool | None = False,
    orthogonal: bool | None = False,
):
    """Attach structural assumptions to a symbolic tensor.

    Returns a tensor identical to *x* at runtime but carrying the declared assumptions so that graph rewrites can exploit them.

    Parameters
    ----------
    x : Variable
        The symbolic variable to annotate.
    diagonal : bool, optional
        Assert that *x* is a diagonal matrix.
    lower_triangular : bool, optional
        Assert that *x* is lower-triangular.
    upper_triangular : bool, optional
        Assert that *x* is upper-triangular.
    symmetric : bool, optional
        Assert that *x* is symmetric.
    positive_definite : bool, optional
        Assert that *x* is positive-definite.
    orthogonal : bool, optional
        Assert that *x* is orthogonal.

    Returns
    -------
    out : TensorVariable
        A view of *x* with the assumptions attached.

    Examples
    --------
    >>> import pytensor.tensor as pt
    >>> x = pt.dmatrix("x")
    >>> x_diag = pt.specify_assumptions(x, diagonal=True)
    """
    x = as_tensor_variable(x)

    names: set[str] = set()
    updates = {
        "diagonal": diagonal,
        "lower_triangular": lower_triangular,
        "upper_triangular": upper_triangular,
        "symmetric": symmetric,
        "positive_definite": positive_definite,
        "orthogonal": orthogonal,
    }
    for kwarg, value in updates.items():
        if value:
            names.add(kwarg)

    if not names:
        return x

    return SpecifyAssumptions(frozenset(names))(x)
