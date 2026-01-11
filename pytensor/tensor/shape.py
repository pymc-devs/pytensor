import warnings
from collections.abc import Sequence
from numbers import Number
from textwrap import dedent
from types import EllipsisType
from typing import TYPE_CHECKING, Union, cast
from typing import cast as typing_cast

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
from pytensor.tensor import _get_vector_length, as_tensor_variable, get_vector_length
from pytensor.tensor import basic as ptb
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.type import DenseTensorType, TensorType, int_dtypes, tensor
from pytensor.tensor.type_other import NoneConst, NoneTypeT
from pytensor.tensor.variable import TensorConstant, TensorVariable


if TYPE_CHECKING:
    from pytensor.tensor import TensorLike

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
        if not isinstance(x, Variable):
            x = ptb.as_tensor_variable(x)

        if isinstance(x.type, TensorType):
            out_var = TensorType("int64", (x.type.ndim,))()
        else:
            out_var = pytensor.tensor.type.lvector()

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


class Reshape(COp):
    """Perform a reshape operation of the input x to the new shape shp.
    The number of dimensions to which to reshape to (ndim) must be
    known at graph build time.
    """

    view_map = {0: [0]}  # output 0 is potentially aliased to inputs [0]
    _f16_ok = True
    _output_type_depends_on_input_value = True

    check_input = False
    __props__ = ("ndim",)

    def __init__(self, ndim):
        self.ndim = int(ndim)
        if ndim < 0:
            raise ValueError("The output dimensions after reshape must be 0 or greater")

    def __str__(self):
        return f"{self.__class__.__name__}{{{self.ndim}}}"

    def make_node(self, x, shp):
        x = ptb.as_tensor_variable(x)
        shp_orig = shp
        shp = ptb.as_tensor_variable(shp, ndim=1)
        if shp.type.shape == (None,):
            shp = specify_shape(shp, self.ndim)
        if not (
            shp.dtype in int_dtypes
            or (isinstance(shp, TensorConstant) and shp.data.size == 0)
        ):
            # It raises an error if shp is not of integer type,
            # except when shp is constant and empty
            # (in this case, shp.dtype does not matter anymore).
            raise TypeError(f"Shape must be integers; got {shp.dtype}")

        assert shp.ndim == 1

        if isinstance(shp, TensorConstant):
            out_shape = [int(s) if s >= 0 else None for s in shp.data]
        else:
            out_shape = [None] * self.ndim
            shp_list = shp_orig
            if hasattr(shp_orig, "ndim") and shp_orig.ndim == 0:
                shp_list = [shp_orig]
            for index in range(self.ndim):
                y = shp_list[index]
                y = ptb.as_tensor_variable(y)
                try:
                    s_val = ptb.get_scalar_constant_value(y).item()
                    if s_val >= 0:
                        out_shape[index] = s_val
                except NotScalarConstantError:
                    pass

        # If we only don't know the size of one output dimension,
        # but we know all the input dimensions we can deduce it
        # This happens often when there is -1 as an input of Reshape
        if None not in x.type.shape and out_shape.count(None) == 1:
            full_size = np.prod(x.type.shape)
            known_size = np.prod([s for s in out_shape if s is not None])
            out_shape[out_shape.index(None)] = int(full_size // known_size)

        out_shape = tuple(out_shape)

        # Run some eager error checks
        if len(out_shape) != self.ndim:
            raise ValueError(
                "Shape argument to Reshape has incorrect length:"
                f" {len(out_shape)}, should be {self.ndim}"
            )

        if None not in x.type.shape and None not in out_shape:
            if np.prod(x.type.shape) != np.prod(out_shape):
                raise ValueError(
                    f"Reshape: Input shape {x.type.shape} is incompatible with new shape {out_shape}"
                )

        return Apply(self, [x, shp], [tensor(dtype=x.type.dtype, shape=out_shape)])

    def perform(self, node, inp, out_):
        x, shp = inp
        (out,) = out_
        if len(shp) != self.ndim:
            raise ValueError(
                "Shape argument to Reshape has incorrect"
                f" length: {len(shp)}, should be {self.ndim}"
            )
        out[0] = np.reshape(x, shp)

    def connection_pattern(self, node):
        return [[True], [False]]

    def grad(self, inp, grads):
        x, _shp = inp
        (g_out,) = grads
        return [reshape(g_out, shape(x), ndim=x.ndim), disconnected_type()]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return [None]
        return self(eval_points[0], *inputs[1:], return_list=True)

    def infer_shape(self, fgraph, node, ishapes):
        from pytensor.tensor.math import eq, maximum, mul

        # inputs[1] can contain at most one value of '-1', meaning the actual
        # shape of the output will be automatically computed by reshape, so
        # that the total number of elements stays the same.
        # TODO: Maybe put that formula here?
        # It's not trivial, because we would have to check if the product of
        # all the non-minus-one shapes is a divisor of the product of the
        # original shapes.
        # The following expression leads to cycles in feature_shape,
        # because it tries to replace the Shape_i node by the switch
        # statement, which depends on Shape_i.
        # return [tuple([switch(eq(node.inputs[1][i], -1),
        #                      Shape_i(i)(node.outputs[0]),
        #                      node.inputs[1][i])
        #                    for i in range(self.ndim)]
        #    )]
        # Here, we only simplify if the shape (node.inputs[1]) is a constant,
        # ideally it would suffice to check that it is always non-negative.
        # If current variable is a scalar and its dimensionality should
        # change to self.ndim, then use size 1 for all new dimensions.
        if len(ishapes[0]) == 0:
            return [(1,) * self.ndim]

        requ = node.inputs[1]
        input_size = mul(*ishapes[0])
        if isinstance(requ, TensorConstant):
            requ = list(requ.data)
            requ_part = [ele for ele in requ if ele != -1]
            crit = len(requ) - len(requ_part)
            if crit == 1 and len(requ_part) > 0:
                # If there are both 0 and -1 in requ_size, it is impossible
                # to determine a right output, but we can at least prevent
                # a division by 0. We do not want to keep a negative
                # size here as it could lead to further weird errors
                # after other optimizations.
                requ_size = mul(*requ_part)
                missing = input_size // (1 if requ_size == 0 else requ_size)
                for i, ele in enumerate(requ):
                    if ele == -1:
                        requ[i] = missing
            elif crit == 1:  # we reshape to -1
                requ = [input_size] if ishapes[0] else [1]
            elif crit > 1:
                raise ValueError(
                    "shape argument to Reshape.perform"
                    " must have at most one entry equal to -1"
                )
            return [requ]
        else:
            requ = [requ[i] for i in range(self.ndim)]
            # since new_dims can have negative value (-1), the
            # multiplication of all values should be negated
            # to give a positive value.
            # To avoid optimization complexity, we avoid checking
            # for the case when there are two or more '-1' values.
            if self.ndim:
                requ_size = -mul(*requ)
                # If there are both 0 and -1 in requ_size, it is impossible
                # to determine a right output, but we can at least prevent
                # a division by 0. We do not want to keep a negative
                # size here as it could lead to further weird errors
                # after other optimizations.
                rest_size = input_size // maximum(requ_size, 1)
            return [
                tuple(
                    ptb.switch(eq(requ[i], -1), rest_size, requ[i])
                    for i in range(self.ndim)
                )
            ]

    def c_code_cache_version(self):
        return (10,)

    def c_code(self, node, name, inputs, outputs, sub):
        x, shp = inputs
        shp_dtype = node.inputs[1].type.dtype_specs()[1]
        (z,) = outputs
        fail = sub["fail"]
        ndim = self.ndim

        return f"""
        assert (PyArray_NDIM({shp}) == 1);

        // Unpack shape into new_dims
        npy_intp new_dims[{ndim}];
        for (int ii = 0; ii < {ndim}; ++ii)
        {{
            new_dims[ii] = (({shp_dtype}*)(PyArray_BYTES({shp}) + ii * PyArray_STRIDES({shp})[0]))[0];
        }}

        PyArray_Dims newshape;
        newshape.len = {ndim};
        newshape.ptr = new_dims;

        Py_XDECREF({z});
        {z} = (PyArrayObject *) PyArray_Newshape({x}, &newshape, NPY_CORDER);

        if (!{z}) {{
            //The error message should have been set by PyArray_Newshape
            {fail};
        }}
        """


@_vectorize_node.register(Reshape)
def _vectorize_reshape(op, node, x, shape):
    from pytensor.tensor.blockwise import vectorize_node_fallback

    old_x, old_shape = node.inputs
    batched_ndims = x.type.ndim - old_x.type.ndim

    if as_tensor_variable(shape).type.ndim != 1:
        return vectorize_node_fallback(op, node, x, shape)

    if len(tuple(old_shape)) == len(tuple(shape)):
        new_shape = [*x.shape[:batched_ndims], *shape]
    elif len(tuple(old_shape)) == (len(tuple(shape)) - batched_ndims):
        new_shape = shape
    else:
        raise ValueError("Invalid shape length passed into vectorize node of Reshape")

    return reshape(x, new_shape, ndim=len(new_shape)).owner


def reshape(
    x: "TensorLike",
    newshape: Union["TensorLike", Sequence["TensorLike"]],
    *,
    ndim: int | None = None,
) -> TensorVariable:
    if ndim is None:
        newshape = ptb.as_tensor_variable(newshape)  # type: ignore
        if newshape.type.ndim != 1:
            raise TypeError(
                "New shape in reshape must be a vector or a list/tuple of"
                f" scalar. Got {newshape} after conversion to a vector."
            )
        try:
            ndim = get_vector_length(newshape)
        except ValueError:
            raise ValueError(
                f"The length of the provided shape ({newshape}) cannot "
                "be automatically determined, so PyTensor is not able "
                "to know what the number of dimensions of the reshaped "
                "variable will be. You can provide the 'ndim' keyword "
                "argument to 'reshape' to avoid this problem."
            )
    op = Reshape(ndim)
    rval = op(x, newshape)
    return typing_cast(TensorVariable, rval)


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
