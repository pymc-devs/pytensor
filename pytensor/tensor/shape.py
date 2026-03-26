import warnings
from numbers import Number
from textwrap import dedent
from types import EllipsisType
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple

import pytensor
from pytensor.gradient import disconnected_type
from pytensor.graph import Op
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.replace import _vectorize_node
from pytensor.graph.type import HasShape
from pytensor.link.c.op import COp
from pytensor.link.c.params_type import ParamsType
from pytensor.tensor import _get_vector_length, as_tensor_variable
from pytensor.tensor import basic as ptb
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.type import DenseTensorType, TensorType
from pytensor.tensor.type_other import NoneConst, NoneTypeT
from pytensor.tensor.variable import TensorVariable


if TYPE_CHECKING:
    pass

ShapeValueType = None | EllipsisType | np.integer | int | Variable


def register_shape_c_code(type, code, version=()):
    """
    Tell Shape Op how to generate C code for an PyTensor Type.

    Parameters
    ----------
    typ : PyTensor type
        It must be the PyTensor class itself and not an instance of the class.
    code : C code
        Returns a vector representing the shape for the PyTensor type 'typ'.
        Use %(iname)s and %(oname)s for the input and output C variable names
        respectively.
    version
        A number indicating the version of the code, for cache.

    """
    Shape.c_code_and_version[type] = (code, version)


class Shape(COp):
    """
    L{Op} to return the shape of a matrix.

    Notes
    -----
    Non-differentiable.

    """

    _f16_ok = True

    # Mapping from Type to C code (and version) to use.
    # In the C code, the name of the input variable is %(iname)s,
    # the output variable is %(oname)s.
    c_code_and_version: dict = {}

    check_input = False
    __props__ = ()

    def make_node(self, x):
        x = ptb.as_tensor_variable(x)
        out_var = tensor(dtype="int64", shape=(x.type.ndim,))

        return Apply(self, [x], [out_var])

    def perform(self, node, inp, out_):
        (x,) = inp
        (out,) = out_
        out[0] = np.asarray(np.shape(x), dtype="int64")

    def infer_shape(self, fgraph, node, in_shapes):
        return [[len(in_shapes[0])]]

    def connection_pattern(self, node):
        # the grad returns the gradient with respect to the
        # elements of a tensor variable
        # the elements of the tensor variable do not participate
        # in the computation of the shape, so they are not really
        # part of the graph
        return [[False]]

    def grad(self, inp, grads):
        # the grad returns the gradient with respect to the
        # elements of a tensor variable
        # the elements of the tensor variable do not participate
        # in the computation of the shape, so they are not really
        # part of the graph
        return [disconnected_type()]

    def R_op(self, inputs, eval_points):
        return [None]

    def c_code(self, node, name, inames, onames, sub):
        (iname,) = inames
        (oname,) = onames
        fail = sub["fail"]

        itype = node.inputs[0].type.__class__
        if itype in self.c_code_and_version:
            code, version = self.c_code_and_version[itype]
            return code % locals()

        # Else, no C code
        raise NotImplementedError()

    def c_code_cache_version(self):
        version = []
        # If any of the c code is unversioned, we have to return ()
        # Else, we will return a list of (type name, version) pairs.
        for t, (c, v) in sorted(
            self.c_code_and_version.items(), key=lambda pair: str(pair[0])
        ):
            if not v:
                warnings.warn(
                    f"Type {t} has C code for Shape, but it has no "
                    "version. You should add a 'version' keyword "
                    "arg when calling register_shape_c_code.",
                    stacklevel=2,
                )
                return ()
            version.append((str(t), v))

        if version:
            version.append(1)

        return tuple(version)


_shape = Shape()


def shape(x: np.ndarray | Number | Variable) -> TensorVariable:
    """Return the shape of `x`."""
    if not isinstance(x, Variable):
        # The following is a type error in Python 3.9 but not 3.12.
        # Thus we need to ignore unused-ignore on 3.12.
        x = ptb.as_tensor_variable(x)  # type: ignore[arg-type,unused-ignore]

    return cast(TensorVariable, _shape(x))


@_get_vector_length.register(Shape)  # type: ignore
def _get_vector_length_Shape(op: Op, var: TensorVariable) -> int:
    return cast(int, var.owner.inputs[0].type.ndim)


@_vectorize_node.register(Shape)
def vectorize_shape(op, node, batched_x):
    from pytensor.tensor.extra_ops import broadcast_to

    [old_x] = node.inputs
    core_ndims = old_x.type.ndim
    batch_ndims = batched_x.type.ndim - core_ndims
    batched_x_shape = shape(batched_x)
    if not batch_ndims:
        return batched_x_shape.owner
    else:
        batch_shape = batched_x_shape[:batch_ndims]
        core_shape = batched_x_shape[batch_ndims:]
        return broadcast_to(core_shape, (*batch_shape, core_ndims)).owner


def shape_tuple(x: TensorVariable) -> tuple[Variable, ...]:
    r"""Get a tuple of symbolic shape values.

    This will return `ScalarConstant`\s for static shape values.

    """
    if not isinstance(x.type, HasShape):
        # We assume/call it a scalar
        return ()

    res: tuple[Variable, ...] = ()
    symbolic_shape = shape(x)
    static_shape = x.type.shape
    for i in range(x.type.ndim):
        shape_val = static_shape[i]

        if shape_val is not None:
            # TODO: Why not use uint64?
            res += (pytensor.scalar.ScalarConstant(pytensor.scalar.int64, shape_val),)
        else:
            res += (symbolic_shape[i],)

    return res


class Shape_i(COp):
    """
    L{Op} to return the shape of a matrix.

    Notes
    -----
    Non-differentiable.

    """

    _f16_ok = True

    # Mapping from Type to C code (and version) to use.
    # In the C code, the name of the input variable is %(iname)s,
    # the output variable is %(oname)s.
    c_code_and_version: dict = {}

    check_input = False

    __props__ = ("i",)

    def __init__(self, i):
        # As i will be used in the hash and that ndarray are not hashable,
        # we need to convert it to an int as it is hashable.
        if isinstance(i, np.ndarray):
            assert i.dtype in pytensor.tensor.type.integer_dtypes
        assert i == int(i)
        i = int(i)
        self.i = i

    # NB:
    # 1) params_type is defined as a property to avoid
    #    loop in Python import caused by importing pytensor.scalar below
    #    when params_type is defined directly in class code.
    # 2) We wrap scalar into ParamsType (instead of directly using scalar as op param)
    #    to avoid PyTensor converting scalar param to constant that would be later
    #    hardcoded as literal in C code, making us loose all the advantages of
    #    using params.
    @property
    def params_type(self):
        return ParamsType(i=pytensor.scalar.basic.int64)

    def __str__(self):
        return f"{self.__class__.__name__}{{{self.i}}}"

    def make_node(self, x):
        if not (isinstance(x, Variable) and hasattr(x.type, "ndim")):
            raise TypeError(
                f"{x} must be `Variable` with a `Type` having an ndim attribute"
            )
        if x.type.ndim <= self.i:
            raise TypeError(f"{x} has too few dimensions for Shape_i")
        return Apply(self, [x], [pytensor.tensor.type.lscalar()])

    def perform(self, node, inp, out_):
        (x,) = inp
        (out,) = out_
        if out[0] is None:
            out[0] = np.asarray(np.shape(x)[self.i], dtype="int64")
        else:
            out[0][...] = np.shape(x)[self.i]

    def c_code_cache_version(self):
        version = []
        # If any of the c code is unversioned, we have to return ()
        # Else, we will return a list of (type name, version) pairs.
        for t, (c, ci, v) in sorted(
            self.c_code_and_version.items(), key=lambda pair: str(pair[0])
        ):
            if not v:
                warnings.warn(
                    f"Type {t} has C code for Shape_i, but it has "
                    "no version. You should add a 'version' keyword "
                    "arg when calling register_shape_i_c_code.",
                    stacklevel=2,
                )
                return ()
            version.append((str(t), v))

        if version:
            version.append(2)

        return tuple(version)

    def c_code(self, node, name, inames, onames, sub):
        (iname,) = inames
        (oname,) = onames
        fail = sub["fail"]
        # i is then 'params->i', not just 'params'.
        i = sub["params"] + "->i"

        itype = node.inputs[0].type.__class__
        if itype in self.c_code_and_version:
            code, check_input, version = self.c_code_and_version[itype]
            return (check_input + code) % locals()

        # Else, no C code
        raise NotImplementedError()

    def infer_shape(self, fgraph, node, input_shapes):
        return [()]

    def connection_pattern(self, node):
        # the grad returns the gradient with respect to the
        # elements of a tensor variable
        # the elements of the tensor variable do not participate
        # in the computation of the shape, so they are not really
        # part of the graph
        return [[False]]

    def grad(self, inp, grads):
        return [
            pytensor.gradient.grad_not_implemented(
                op=self,
                x_pos=0,
                x=inp[0],
                comment="No gradient for the shape of a matrix is implemented.",
            )
        ]


def shape_i(var, i, fgraph=None):
    """
    Equivalent of var.shape[i], but apply if possible the shape feature
    optimization.

    This is useful in optimization that need to get the shape. This
    remove the need of the following shape_feature optimization that
    convert it. So this speed up optimization and remove Equilibrium
    max iteration problems.

    Parameters
    ----------
    var : Variable
        The variable we want to take the shape of.
    i : int
        The shape dimensions we want
    fgraph : FunctionGraph (optional)

    """
    if fgraph and hasattr(fgraph, "shape_feature"):
        shape_feature = fgraph.shape_feature
        shape_of = shape_feature.shape_of

        def recur(node):
            if node.outputs[0] not in shape_of:
                for inp in node.inputs:
                    if inp.owner:
                        recur(inp.owner)
                # If the output var isn't marked as being in the graph,
                # we need to add it in the ShapeFeature.
                shape_feature.on_import(fgraph, node, "graph.ops.shape_i")

        if var not in shape_of:
            recur(var.owner)
        return shape_of[var][i]

    # If we are not able to use the shape feature, we should not put
    # Shape_i in the graph. Otherwise, the shape feature optimization
    # won't get applied.
    return shape(var)[i]


def register_shape_i_c_code(typ, code, check_input, version=()):
    """
    Tell Shape_i how to generate C code for an PyTensor Type.

    Parameters
    ----------
    typ : PyTensor type
        It must be the PyTensor class itself and not an instance of the class.
    code : C code
        Gets the shape of dimensions %(i)s for the PyTensor type 'typ'.
        Use %(iname)s and %(oname)s for the input and output C variable names
        respectively.
    version
        A number indicating the version of the code, for cache.

    """
    Shape_i.c_code_and_version[typ] = (code, check_input, version)


class SpecifyShape(COp):
    """
    L{Op} that puts into the graph the user-provided shape.

    In the case where this `Op` stays in the final graph, we assert the shape.
    For this the output of this op must be used in the graph. This is not
    the case most of the time if we only take the shape of the output.
    Maybe there are other optimizations that will mess with this.

    Notes
    -----
    Maybe in the future we will never do the assert!
    """

    view_map = {0: [0]}
    __props__ = ()
    _f16_ok = True
    _output_type_depends_on_input_value = True

    def make_node(self, x, *shape):
        x = ptb.as_tensor_variable(x)

        shape = tuple(
            NoneConst
            if (
                s is None or (isinstance(s, Variable) and isinstance(s.type, NoneTypeT))
            )
            else ptb.as_tensor_variable(s, ndim=0)
            for s in shape
        )

        if any(
            s.dtype not in pytensor.tensor.type.integer_dtypes
            for s in shape
            if hasattr(s, "dtype")
        ):
            raise TypeError("Shape values must be integer types")

        if len(shape) != x.type.ndim:
            raise ValueError(
                f"Input `x` is {x.type.ndim}-dimensional and will never match a shape of length {len(shape)}."
            )

        type_shape = [None] * x.ndim
        for i, (xts, s) in enumerate(zip(x.type.shape, shape, strict=True)):
            if xts is not None:
                type_shape[i] = xts
            elif not isinstance(s.type, NoneTypeT):
                try:
                    type_shape[i] = int(ptb.get_scalar_constant_value(s))
                except NotScalarConstantError:
                    pass

        out_var = x.type.clone(shape=type_shape)()

        return Apply(self, [x, *shape], [out_var])

    def perform(self, node, inp, out_):
        x, *shape = inp
        (out,) = out_
        ndim = len(shape)
        if x.ndim != ndim:
            raise AssertionError(
                f"SpecifyShape: Got {x.ndim} dimensions (shape {x.shape}), expected {ndim} dimensions with shape {tuple(shape)}."
            )
        # zip strict not specified because we are in a hot loop
        if not all(xs == s for xs, s in zip(x.shape, shape) if s is not None):
            raise AssertionError(
                f"SpecifyShape: Got shape {x.shape}, expected {tuple(int(s) if s is not None else None for s in shape)}."
            )
        out[0] = x

    def infer_shape(self, fgraph, node, shapes):
        xshape, *_ = shapes
        shape = node.inputs[1:]
        # Use x shape if specified dim is None, otherwise the specified shape
        return [
            [
                xshape[i] if isinstance(dim.type, NoneTypeT) else dim
                for i, dim in enumerate(shape)
            ]
        ]

    def connection_pattern(self, node):
        return [[True], *[[False]] * len(node.inputs[1:])]

    def grad(self, inp, grads):
        _x, *shape = inp
        (gz,) = grads
        return [
            specify_shape(gz, shape),
            *(disconnected_type() for _ in range(len(shape))),
        ]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            # It means that this op sits on top of a non-differentiable path
            return [None]
        return self.make_node(eval_points[0], *inputs[1:]).outputs

    def c_code(self, node, name, i_names, o_names, sub):
        if not isinstance(node.inputs[0].type, DenseTensorType):
            raise NotImplementedError(
                f"Specify_shape c_code not implemented for input type {node.inputs[0].type}"
            )

        x_name, *shape_names = i_names
        (o_name,) = o_names
        fail = sub["fail"]

        code = dedent(
            f"""
            if (PyArray_NDIM({x_name}) != {len(shape_names)}) {{
                PyErr_Format(PyExc_AssertionError,
                    "SpecifyShape: Got %d dimensions, expected %d dimensions.",
                    PyArray_NDIM({x_name}), {len(shape_names)}
                );
                {fail};
            }}
            """
        )

        for i, (shp_name, shp) in enumerate(
            zip(shape_names, node.inputs[1:], strict=True)
        ):
            if isinstance(shp.type, NoneTypeT):
                continue
            code += dedent(
                f"""
                if (py_{shp_name} != Py_None){{
                    dtype_{shp_name} shp = ((dtype_{shp_name}*)PyArray_GETPTR1({shp_name}, 0))[0];
                    if (PyArray_DIMS({x_name})[{i}] != shp) {{
                        PyErr_Format(PyExc_AssertionError,
                            "SpecifyShape: dim %d of input has shape %d, expected %d.",
                            {i}, PyArray_DIMS({x_name})[{i}], shp
                        );
                        {fail};
                    }}
                }}
                """
            )

        code += dedent(
            f"""
            Py_XDECREF({o_name});
            {o_name} = {x_name};
            Py_XINCREF({o_name});
            """
        )
        return code

    def c_code_cache_version(self):
        return (2,)


_specify_shape = SpecifyShape()


def specify_shape(
    x: np.ndarray | Number | Variable,
    shape: ShapeValueType | list[ShapeValueType] | tuple[ShapeValueType, ...],
):
    """Specify a fixed shape for a `Variable`.

    If a dimension's shape value is ``None``, the size of that dimension is not
    considered fixed/static at runtime.

    A single ``Ellipsis`` can be used to imply multiple ``None`` specified dimensions
    """
    x = as_tensor_variable(x)  # type: ignore[arg-type]

    if not isinstance(shape, tuple | list):
        shape = (shape,)

    # If shape is a symbolic 1d vector of fixed length, we separate the items into a
    # tuple with one entry per shape dimension
    if len(shape) == 1 and shape[0] not in (None, Ellipsis):
        shape_vector = ptb.as_tensor_variable(shape[0])  # type: ignore[arg-type]
        if shape_vector.ndim == 1:
            try:
                shape = tuple(shape_vector)
            except ValueError:
                raise ValueError("Shape vector must have fixed dimensions")

    if Ellipsis in shape:
        ellipsis_pos = shape.index(Ellipsis)
        implied_none = x.type.ndim - (len(shape) - 1)
        shape = (
            *shape[:ellipsis_pos],
            *((None,) * implied_none),
            *shape[ellipsis_pos + 1 :],
        )
        if Ellipsis in shape[ellipsis_pos + 1 :]:
            raise ValueError("Multiple Ellipsis in specify_shape")

    # If the specified shape is already encoded in the input static shape, do nothing
    # This ignores PyTensor constants in shape
    new_shape_info = any(
        s != xts for (s, xts) in zip(shape, x.type.shape, strict=False) if s is not None
    )

    # If shape does not match x.ndim, we rely on the `Op` to raise a ValueError
    if not new_shape_info and len(shape) == x.type.ndim:
        return x

    return _specify_shape(x, *shape)


@_get_vector_length.register(SpecifyShape)  # type: ignore
def _get_vector_length_SpecifyShape(op: Op, var: TensorVariable) -> int:
    try:
        return int(ptb.get_scalar_constant_value(var.owner.inputs[1]).item())
    except NotScalarConstantError:
        raise ValueError(f"Length of {var} cannot be determined")


@_vectorize_node.register(SpecifyShape)
def _vectorize_specify_shape(op, node, x, *shape):
    old_x, *old_shape = node.inputs
    batched_ndims = x.type.ndim - old_x.type.ndim

    if any(
        as_tensor_variable(dim).type.ndim != 0
        for dim in shape
        if not (
            (isinstance(dim, Variable) and isinstance(dim.type, NoneTypeT))
            or dim is None
        )
    ):
        raise NotImplementedError(
            "It is not possible to vectorize the shape argument of SpecifyShape"
        )

    if len(shape) == len(old_shape):
        new_shape = tuple([None] * batched_ndims) + shape
    elif len(shape) == (len(old_shape) + batched_ndims):
        new_shape = shape
    else:
        raise ValueError(
            "Invalid number of shape arguments passed into vectorize node of SpecifyShape"
        )

    return specify_shape(x, new_shape).owner


def shape_padleft(t, n_ones=1):
    """Reshape `t` by left-padding the shape with `n_ones` 1s.

    See Also
    --------
    shape_padaxis
    shape_padright
    Dimshuffle

    """
    _t = ptb.as_tensor_variable(t)
    if n_ones == 0:
        return _t
    pattern = ["x"] * n_ones + list(range(_t.type.ndim))
    return _t.dimshuffle(pattern)


def shape_padright(t, n_ones=1):
    """Reshape `t` by right-padding the shape with `n_ones` 1s.

    See Also
    --------
    shape_padaxis
    shape_padleft
    Dimshuffle

    """
    _t = ptb.as_tensor_variable(t)
    if n_ones == 0:
        return _t
    pattern = list(range(_t.type.ndim)) + ["x"] * n_ones
    return _t.dimshuffle(pattern)


def shape_padaxis(t, axis):
    """Reshape `t` by inserting 1 at the dimension `axis`.

    Examples
    --------
    >>> tensor = pytensor.tensor.type.tensor3()
    >>> pytensor.tensor.shape_padaxis(tensor, axis=0)
    ExpandDims{axis=0}.0
    >>> pytensor.tensor.shape_padaxis(tensor, axis=1)
    ExpandDims{axis=1}.0
    >>> pytensor.tensor.shape_padaxis(tensor, axis=3)
    ExpandDims{axis=3}.0
    >>> pytensor.tensor.shape_padaxis(tensor, axis=-1)
    ExpandDims{axis=3}.0

    See Also
    --------
    shape_padleft
    shape_padright
    Dimshuffle

    """
    _t = ptb.as_tensor_variable(t)

    ndim = _t.ndim + 1
    if not -ndim <= axis < ndim:
        msg = f"axis {axis} is out of bounds [-{ndim}, {ndim})"
        raise IndexError(msg)
    if axis < 0:
        axis += ndim

    pattern = list(range(_t.type.ndim))
    pattern.insert(axis, "x")
    return _t.dimshuffle(pattern)


register_shape_c_code(
    TensorType,
    """
    npy_intp shape[] = {PyArray_NDIM(%(iname)s)};
    if(%(oname)s == NULL || (PyArray_DIMS(%(oname)s)[0] != shape[0]))
    {
        Py_XDECREF(%(oname)s);
        %(oname)s = (PyArrayObject*) PyArray_SimpleNew(1, shape, NPY_INT64);
    }
    for(int i=0;i<shape[0];i++)
    {
        ((npy_int64*)PyArray_GETPTR1(%(oname)s, i))[0] = PyArray_DIMS(%(iname)s)[i];
    }
    """,
    version=1,
)


register_shape_i_c_code(
    TensorType,
    """
    if(!%(oname)s)
        %(oname)s=(PyArrayObject*)PyArray_EMPTY(0, NULL, NPY_INT64, 0);
    ((npy_int64*)PyArray_DATA(%(oname)s))[0]=PyArray_DIMS(%(iname)s)[%(i)s];
    """,
    """
    if (%(i)s>=PyArray_NDIM(%(iname)s)){
        PyErr_SetString(PyExc_TypeError,
            "Number of dimensions lower than expected");
        %(fail)s
    }
    """,
    version=3,
)


def specify_broadcastable(x, *axes):
    """Specify the input as being broadcastable in the specified axes.

    For example, specify_broadcastable(x, 0) will make the first dimension of
    x broadcastable. When performing the function, if the length of
    x along that dimension is not 1, a ValueError will be raised.

    Parameters
    ----------
    x : tensor_like
        Input pytensor tensor.
    axis : an int or an iterable object such as list or tuple of int values
        The dimension along which the tensor x should be broadcastable.
        If the length of x along these dimensions is not 1, a ValueError will
        be raised.

    Returns
    -------
    tensor
        A pytensor tensor, which is broadcastable along the specified dimensions.

    """
    x = as_tensor_variable(x)

    if not axes:
        return x

    axes = normalize_axis_tuple(axes, x.type.ndim)
    shape_info = [1 if i in axes else s for i, s in enumerate(x.type.shape)]
    return specify_shape(x, shape_info)
