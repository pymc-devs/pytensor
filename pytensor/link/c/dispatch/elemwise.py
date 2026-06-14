from collections.abc import Hashable
from textwrap import dedent
from typing import cast

import numpy as np

import pytensor.tensor.basic
from pytensor.configdefaults import config
from pytensor.graph.basic import Apply
from pytensor.graph.utils import MethodNotDefined
from pytensor.link.c.basic import failure_code
from pytensor.link.c.dispatch.basic import CImpl, c_funcify
from pytensor.link.c.op import openmp_supported
from pytensor.scalar import get_scalar_type
from pytensor.scalar.basic import identity as scalar_identity
from pytensor.tensor import elemwise_cgen as cgen
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.type import TensorType
from pytensor.utils import uniq, unzip


def _elemwise_c_all(op, node, nodename, inames, onames, sub, use_openmp):
    # Some `Op`s directly call this generator on a fresh Elemwise.
    # To not request all of them to call prepare_node(), do it here.
    # There is no harm if it get called multiple times.
    if not hasattr(node.tag, "fake_node"):
        op.prepare_node(node, None, None, "c")
    _inames = inames
    _onames = onames

    inames = uniq(inames)
    inputs = uniq(node.inputs)
    # assert that inames and inputs order stay consistent.
    # This is to protect again futur change of uniq.
    assert len(inames) == len(inputs)
    ii, iii = unzip(
        uniq(list(zip(_inames, node.inputs, strict=True))), n=2, strict=True
    )
    assert all(x == y for x, y in zip(ii, inames, strict=True))
    assert all(x == y for x, y in zip(iii, inputs, strict=True))

    defines = ""
    undefs = ""

    # The destroy map is a map of output indices to input indices
    # that overwrite them.  We just convert them to the actual
    # Variables.
    dmap = {node.outputs[o]: [node.inputs[i]] for o, i in op.inplace_pattern.items()}

    # dtypes of the inputs
    idtypes = [input.type.dtype_specs()[1] for input in inputs]

    # These are the outputs that we will need to allocate
    # (output, name, name of the c type), transposed
    real = list(
        zip(
            *[
                (r, s, r.type.dtype_specs()[1])
                for r, s in zip(node.outputs, onames, strict=True)
                if r not in dmap
            ],
            strict=True,
        )
    )
    if real:
        real_outputs, real_onames, real_odtypes = real
    else:
        real_outputs, real_onames, real_odtypes = [], [], []

    # Outputs that are aliased with an input (inplace)
    # (output, name), transposed (c type name not needed since we don't
    # need to allocate.
    aliased = list(
        zip(
            *[(r, s) for (r, s) in zip(node.outputs, onames, strict=True) if r in dmap],
            strict=True,
        )
    )
    if aliased:
        aliased_outputs, aliased_onames = aliased
    else:
        aliased_outputs, aliased_onames = [], []

    # for each input:
    # same as range(ndim), but with 'x' at all broadcastable positions
    orders = [
        [(s == 1 and "x") or i for i, s in enumerate(input.type.shape)]
        for input in inputs
    ]

    # number of nested loops we will need (all inputs have same
    # dimensionality)
    nnested = len(orders[0])
    sub = dict(sub)
    for i, (input, iname) in enumerate(zip(inputs, inames, strict=True)):
        # the c generators will substitute the input names for
        # references to loop variables lv0, lv1, ...
        sub[f"lv{i}"] = iname

    decl = cgen.make_declare(orders, idtypes, sub)
    checks = cgen.make_checks(orders, idtypes, sub)

    # Check if all inputs (except broadcasted scalar) are fortran.
    # In that case, create a fortran output ndarray.
    z = list(zip(inames, inputs, strict=True))
    alloc_fortran = " && ".join(
        f"PyArray_ISFORTRAN({arr})"
        for arr, var in z
        if not all(s == 1 for s in var.type.shape)
    )
    # If it is a scalar, make it c contig to prevent problem with
    # NumPy C and F contig not always set as both of them.
    if len(alloc_fortran) == 0:
        alloc_fortran = "0"

    alloc = ""
    # We loop over the "real" outputs, i.e., those that are not
    # inplace (must be allocated) and we declare/allocate/check
    # them
    for output, oname, odtype in zip(
        real_outputs, real_onames, real_odtypes, strict=True
    ):
        i += 1  # before this loop, i = number of inputs
        sub[f"lv{i}"] = oname
        sub["olv"] = oname
        alloc += cgen.make_declare(
            [list(range(nnested))], [odtype], dict(sub, lv0=oname)
        )
        alloc += cgen.make_alloc(orders, odtype, sub, fortran=alloc_fortran)
        alloc += cgen.make_checks(
            [list(range(nnested))], [odtype], dict(sub, lv0=oname)
        )
    olv_index = i  # index of the last output

    # We loop over the "aliased" outputs, i.e., those that are
    # inplace (overwrite the contents of one of the inputs) and
    # make the output pointers point to their corresponding input
    # pointers.
    for output, oname in zip(aliased_outputs, aliased_onames, strict=True):
        olv_index = inputs.index(dmap[output][0])
        iname = inames[olv_index]
        # We make the output point to the corresponding input and
        # decrease the reference of whatever the output contained
        # prior to this
        alloc += f"""
            if ({oname}) {{
                Py_XDECREF({oname});
            }}
            {oname} = {iname};
            Py_XINCREF({oname});
            """
        # We alias the scalar variables
        defines += f"#define {oname}_i {iname}_i\n"
        undefs += f"#undef {oname}_i\n"

    # Note: here, olv_index is either the index of the last output
    # which is allocated, OR, if there are any aliased outputs,
    # the index of the last of these aliased outputs.

    # We generate the C code of the inner loop using the scalar op
    if use_openmp:
        # If we are using openmp, we need to get rid of the "goto"
        # statement in sub['fail']. For now we recreate it here.
        fail = failure_code(sub, use_goto=False)
    else:
        fail = sub["fail"]
    task_code = op.scalar_op.c_code(
        node.tag.fake_node,
        nodename + "_scalar_",
        [f"{s}_i" for s in _inames],
        [f"{s}_i" for s in onames],
        dict(sub, fail=fail),
    )
    code = f"""
        {{
            {defines}
            {task_code}
            {undefs}
        }}
        """

    loop_orders = orders + [list(range(nnested))] * len(real_onames)
    dtypes = idtypes + list(real_odtypes)
    if all(
        [o.ndim <= 1 for o in node.outputs]
        or
        # Use simpler code when output ndim == 0 or 1
        # or for broadcated scalar.
        all(s == 1 for s in node.outputs[0].type.shape)
    ):
        if nnested:
            all_code = [("", "")] * (nnested - 1) + [("", code)] + [""]
        else:
            all_code = [code]
        if len(all_code) == 1:
            # No loops
            task_decl = "".join(
                f"{dtype}& {name}_i = *{name}_iter;\n"
                for name, dtype in zip(
                    inames + list(real_onames),
                    idtypes + list(real_odtypes),
                    strict=True,
                )
            )

            preloops = {}
            for i, (loop_order, dtype) in enumerate(
                zip(loop_orders, dtypes, strict=True)
            ):
                for j, index in enumerate(loop_order):
                    if index != "x":
                        preloops.setdefault(j, "")
                        preloops[j] += (
                            f"%(lv{i})s_iter = ({dtype}*)(PyArray_DATA(%(lv{i})s));\n"
                        ) % sub
                        break
                else:  # all broadcastable
                    preloops.setdefault(0, "")
                    preloops[0] += (
                        f"%(lv{i})s_iter = ({dtype}*)(PyArray_DATA(%(lv{i})s));\n"
                    ) % sub

            init_array = preloops.get(0, " ")
            loop = f"""
                {{
                  {defines}
                  {init_array}
                  {task_decl}
                  {task_code}
                  {undefs}
                }}
                """
        else:
            loop = cgen.make_loop(
                loop_orders=loop_orders,
                dtypes=dtypes,
                loop_tasks=all_code,
                sub=sub,
                openmp=use_openmp,
            )
    else:
        loop = cgen.make_reordered_loop(
            init_loop_orders=loop_orders,
            olv_index=olv_index,
            dtypes=dtypes,
            inner_task=code,
            sub=sub,
            openmp=use_openmp,
        )

    # If all inputs and outputs are contiguous
    # and the scalar op define optimized code for that case
    # use it! The scalar_op needs to check the type-level shapes itself.
    if (
        all(o.ndim >= 1 for o in node.outputs)
        and
        # Don't use the contig code for broadcasted scalar.
        not all(s == 1 for s in node.outputs[0].type.shape)
    ):
        contig = None
        try:
            contig = op.scalar_op.c_code_contiguous(
                node, nodename + "_scalar_contig_", _inames, onames, sub
            )
        except MethodNotDefined:
            # Try to make one generic version, this will help the
            # compiler to vectorize the code as their won't be as
            # many ptr and the stride will be hard coded.
            if all(
                # io.type.shape == node.outputs[1].type.shape
                # Elemwise does not specify non-broadcastable static/type-levelshape
                # information for its outputs yet
                node.outputs[0].type.is_super(io.type)
                for io in node.inputs + node.outputs
            ) and (
                len(node.inputs) <= 1
                # If either one of the inputs has a `None` shape, we cannot
                # assume they will have the same size
                or all(
                    len(set(inp_shape)) == 1 and None not in inp_shape
                    for inp_shape in zip(
                        *(inp.type.shape for inp in node.inputs), strict=True
                    )
                )
            ):
                z = onames[0]
                contig = f"""
                    // All output have the same size
                    npy_intp n = PyArray_SIZE({z});
                    """
                index = ""
                for x, var in zip(inames + onames, inputs + node.outputs, strict=True):
                    if not all(s == 1 for s in var.type.shape):
                        contig += f"""
            dtype_{x} * {x}_ptr = (dtype_{x}*) PyArray_DATA({x});
                            """
                        index += f"""
            dtype_{x}& {x}_i = {x}_ptr[i];
                            """
                    else:
                        contig += f"""
            dtype_{x}& {x}_i = ((dtype_{x}*) PyArray_DATA({x}))[0];
                            """
                if use_openmp:
                    contig += f"""#pragma omp parallel for if(n>={int(config.openmp_elemwise_minsize)})
                        """
                contig += f"""
                    for(int i=0; i<n; i++){{
                        {index}
                        {task_code};
                    }}
                    """
        if contig is not None:
            z = list(zip(inames + onames, inputs + node.outputs, strict=True))
            all_broadcastable = all(s == 1 for s in var.type.shape)
            cond1 = " && ".join(
                f"PyArray_ISCONTIGUOUS({arr})"
                for arr, var in z
                if not all_broadcastable
            )
            cond2 = " && ".join(
                f"PyArray_ISFORTRAN({arr})" for arr, var in z if not all_broadcastable
            )
            loop = f"""
            if(({cond1}) || ({cond2})){{
                {contig}
            }}else{{
                {loop}
            }}
            """
    return decl, checks, alloc, loop, ""


def _careduce_c_all(op, node, name, input_names, output_names, sub):
    [inp] = node.inputs
    [out] = node.outputs
    ndim = inp.type.ndim

    [inp_name] = input_names
    [out_name] = output_names

    inp_dtype = inp.type.dtype_specs()[1]
    out_dtype = out.type.dtype_specs()[1]

    acc_dtype = getattr(op, "acc_dtype", None)

    if acc_dtype is not None:
        if acc_dtype == "float16":
            raise MethodNotDefined("no c_code for float16")
        acc_type = TensorType(shape=node.outputs[0].type.shape, dtype=acc_dtype)
        acc_dtype = acc_type.dtype_specs()[1]
    else:
        acc_dtype = out_dtype

    axis = op.axis
    if axis is None:
        axis = list(range(inp.type.ndim))

    if len(axis) == 0:
        # This is just an Elemwise cast operation
        # The acc_dtype is never a downcast compared to the input dtype
        # So we just need a cast to the output dtype.
        var = pytensor.tensor.basic.cast(inp, node.outputs[0].dtype)
        if var is inp:
            var = Elemwise(scalar_identity)(inp)
        assert var.dtype == node.outputs[0].dtype
        inner_op = var.owner.op
        return _elemwise_c_all(
            inner_op,
            var.owner,
            name,
            input_names,
            output_names,
            sub,
            use_openmp=inner_op.openmp and openmp_supported(),
        )

    inp_dims = list(range(ndim))
    non_reduced_dims = [i for i in inp_dims if i not in axis]
    counter = iter(range(ndim))
    acc_dims = ["x" if i in axis else next(counter) for i in range(ndim)]

    sub = sub.copy()
    sub["lv0"] = inp_name
    sub["lv1"] = out_name
    sub["olv"] = out_name

    if acc_dtype != out_dtype:
        # Create an accumulator variable different from the output
        acc_name = "acc"
        setup = acc_type.c_declare(acc_name, sub) + acc_type.c_init(acc_name, sub)
    else:
        # the output is the accumulator variable
        acc_name = out_name
        setup = ""

    # Define strides of input array
    setup += cgen.make_declare(
        [inp_dims], [inp_dtype], sub, compute_stride_jump=False
    ) + cgen.make_checks([inp_dims], [inp_dtype], sub, compute_stride_jump=False)

    # Define strides of output array and allocate it
    out_sub = sub | {"lv0": out_name}
    alloc = (
        cgen.make_declare([acc_dims], [out_dtype], out_sub, compute_stride_jump=False)
        + cgen.make_alloc([non_reduced_dims], out_dtype, sub)
        + cgen.make_checks([acc_dims], [out_dtype], out_sub, compute_stride_jump=False)
    )

    if acc_dtype != out_dtype:
        # Define strides of accumulation buffer and allocate it
        sub["lv1"] = acc_name
        sub["olv"] = acc_name

        acc_sub = sub | {"lv0": acc_name}
        alloc += (
            cgen.make_declare(
                [acc_dims], [acc_dtype], acc_sub, compute_stride_jump=False
            )
            + cgen.make_alloc([non_reduced_dims], acc_dtype, sub)
            + cgen.make_checks(
                [acc_dims], [acc_dtype], acc_sub, compute_stride_jump=False
            )
        )

    identity = op.scalar_op.identity
    if np.isposinf(identity):
        if inp.type.dtype in ("float32", "float64"):
            identity = "__builtin_inf()"
        elif inp.type.dtype.startswith("uint") or inp.type.dtype == "bool":
            identity = "1"
        else:
            identity = "NPY_MAX_" + str(inp.type.dtype).upper()
    elif np.isneginf(identity):
        if inp.type.dtype in ("float32", "float64"):
            identity = "-__builtin_inf()"
        elif inp.type.dtype.startswith("uint") or inp.type.dtype == "bool":
            identity = "0"
        else:
            identity = "NPY_MIN_" + str(inp.type.dtype).upper()
    elif identity is None:
        raise TypeError(f"The {op.scalar_op} does not define an identity.")

    initial_value = f"{acc_name}_i = {identity};"

    inner_task = op.scalar_op.c_code(
        Apply(
            op.scalar_op,
            [
                get_scalar_type(dtype=iv.type.dtype).make_variable()
                for iv in (node.inputs * 2)
            ],
            [
                get_scalar_type(dtype=ov.type.dtype).make_variable()
                for ov in node.outputs
            ],
        ),
        None,
        [f"{acc_name}_i", f"{inp_name}_i"],
        [f"{acc_name}_i"],
        sub,
    )

    if out.type.ndim == 0:
        # Simple case where everything is reduced, no need for loop ordering
        loop = cgen.make_complete_loop_careduce(
            inp_var=inp_name,
            acc_var=acc_name,
            inp_dtype=inp_dtype,
            acc_dtype=acc_dtype,
            initial_value=initial_value,
            inner_task=inner_task,
            fail_code=sub["fail"],
        )
    else:
        loop = cgen.make_reordered_loop_careduce(
            inp_var=inp_name,
            acc_var=acc_name,
            inp_dtype=inp_dtype,
            acc_dtype=acc_dtype,
            inp_ndim=ndim,
            reduction_axes=axis,
            initial_value=initial_value,
            inner_task=inner_task,
        )

    if acc_dtype != out_dtype:
        cast = dedent(
            f"""
                PyArray_CopyInto({out_name}, {acc_name});
                {acc_type.c_cleanup(acc_name, sub)}
                """
        )
    else:
        cast = ""

    if op.error_on_empty_reduce_axis:
        # Ops with no identity element (e.g. Max/Min) are undefined on an empty
        # reduction, so guard against a zero-sized reduced axis at runtime.
        [iname] = input_names
        reduce_axis = op.axis
        if reduce_axis is None:
            reduce_axis = list(range(len(node.inputs[0].type.broadcastable)))
        pattern = [0] * len(node.inputs[0].broadcastable)
        for i in reduce_axis:
            pattern[i] = 1
        pattern_ = str(pattern)[1:-1]
        setup = f"int tosum[]={{{pattern_}}};" + setup
        alloc += dedent(
            f"""
            for(int i=0;i<PyArray_NDIM({iname});i++){{
                if(PyArray_DIMS({iname})[i]==0 && tosum[i]){{
                    PyErr_Format(PyExc_ValueError,
                        "Input of CAReduce{{{node.op.scalar_op}}} has zero-size on axis %%d",i);
                    {sub["fail"]};
                }}
            }}
            """
        )

    return setup, alloc, loop, cast


class DimShuffleImpl(CImpl):
    """Specialized C implementation of `DimShuffle`.

    Reads the op's static permutation and the input's static shape to emit a
    straight-line view construction; the only runtime guards are squeeze checks
    on dropped axes whose length is statically unknown.
    """

    op: DimShuffle

    def c_code_cache_version(self) -> tuple[Hashable, ...]:
        # `new_order`/`input_ndim` ride `DimShuffle.__props__` and the input's
        # static shape rides its type signature, so both are keyed automatically;
        # bump this only when the emitted C below changes.
        return (1,)

    def c_code(
        self,
        node: Apply,
        name: str,
        inputs: list[str],
        outputs: list[str],
        sub: dict[str, str],
    ) -> str:
        op = self.op
        (inp,) = inputs
        (out,) = outputs
        fail = sub["fail"]
        new_order = op._new_order
        nd_out = len(new_order)
        in_shape = node.inputs[0].type.shape

        # A dropped axis is guaranteed by `make_node` to be length 1 or unknown.
        # Only the unknown case needs a runtime check.
        guards = "\n".join(
            f"""
            if (PyArray_DIMS({inp})[{d}] != 1) {{
                PyErr_SetString(PyExc_ValueError,
                    "DimShuffle: cannot drop axis {d} with length not equal to one.");
                {fail}
            }}"""
            for d in op.drop
            if in_shape[d] is None
        )

        assigns = []
        for i, j in enumerate(new_order):
            if j == -1:
                # An augmented (broadcast) axis. The length-1 stride is set to the
                # itemsize rather than zero: the value is never dereferenced, but
                # some BLAS implementations mishandle a zero stride.
                assigns.append(f"dimensions[{i}] = 1;")
                assigns.append(f"strides[{i}] = itemsize;")
            else:
                assigns.append(f"dimensions[{i}] = PyArray_DIMS({inp})[{j}];")
                static_len = in_shape[j]
                if static_len == 1:
                    assigns.append(f"strides[{i}] = itemsize;")
                elif static_len is not None:
                    assigns.append(f"strides[{i}] = PyArray_STRIDES({inp})[{j}];")
                else:
                    assigns.append(
                        f"strides[{i}] = PyArray_DIMS({inp})[{j}] == 1 ? "
                        f"itemsize : PyArray_STRIDES({inp})[{j}];"
                    )

        if nd_out:
            shape_block = (
                f"npy_intp dimensions[{nd_out}];\n"
                f"npy_intp strides[{nd_out}];\n" + "\n".join(assigns)
            )
            dims_ptr = "dimensions"
            strides_ptr = "strides"
        else:
            shape_block = ""
            dims_ptr = "NULL"
            strides_ptr = "NULL"

        return f"""
        {{
            npy_intp itemsize = PyArray_ITEMSIZE({inp});
            {guards}
            {shape_block}

            Py_XDECREF({out});
            // Borrow only the writable flag from the input; NPY_OWNDATA stays 0.
            {out} = (PyArrayObject*)PyArray_New(
                &PyArray_Type, {nd_out}, {dims_ptr},
                PyArray_TYPE({inp}), {strides_ptr},
                PyArray_DATA({inp}), itemsize,
                (NPY_ARRAY_WRITEABLE * PyArray_ISWRITEABLE({inp})),
                NULL);
            if ({out} == NULL) {{
                {fail}
            }}

            // Declare the result a view of the input and recompute its flags.
            Py_INCREF((PyObject*){inp});
            PyArray_SetBaseObject({out}, (PyObject*){inp});
            PyArray_UpdateFlags({out}, NPY_ARRAY_UPDATE_ALL);
        }}
        """


@c_funcify.register(DimShuffle)
def c_funcify_dimshuffle(op, node=None, **kwargs) -> DimShuffleImpl:
    return DimShuffleImpl(op)


class ElemwiseImpl(CImpl):
    """C implementation of `Elemwise`.

    Emits the broadcasting loop over the per-element scalar code via
    `_elemwise_c_all`, delegating support code and headers to the scalar op.
    """

    op: Elemwise

    def _use_openmp(self) -> bool:
        """Return whether to emit OpenMP: requested by the op and compiler-supported."""
        return self.op.openmp and openmp_supported()

    def c_support_code(self, **kwargs) -> str:
        return cast(str, self.op.scalar_op.c_support_code(**kwargs))

    def c_support_code_apply(self, node: Apply, name: str) -> str:
        return cast(
            str, self.op.scalar_op.c_support_code_apply(node, name + "_scalar_")
        )

    def c_headers(self, **kwargs) -> list[str]:
        return ["<vector>", "<algorithm>"]

    def c_header_dirs(self, **kwargs) -> list[str]:
        return cast(list[str], self.op.scalar_op.c_header_dirs(**kwargs))

    def c_compile_args(self, **kwargs) -> list[str]:
        return ["-fopenmp"] if self._use_openmp() else []

    def c_code(
        self,
        node: Apply,
        name: str,
        inputs: list[str],
        outputs: list[str],
        sub: dict[str, str],
    ) -> str:
        scalar_op = self.op.scalar_op
        if (
            any(i.type.dtype == "float16" for i in node.inputs)
            or any(o.type.dtype == "float16" for o in node.outputs)
            # This is for Composite
            or getattr(scalar_op, "inner_float16", False)
        ):
            # No float16 C support; fall back to perform.
            raise NotImplementedError()
        return "\n".join(
            _elemwise_c_all(
                self.op, node, name, inputs, outputs, sub, self._use_openmp()
            )
        )

    def c_code_cache_version_apply(self, node: Apply) -> tuple[Hashable, ...]:
        scalar_op = self.op.scalar_op
        version: list[Hashable] = [17]  # bump when the emitted C changes
        scalar_node = Apply(
            scalar_op,
            [get_scalar_type(dtype=i.type.dtype).make_variable() for i in node.inputs],
            [get_scalar_type(dtype=o.type.dtype).make_variable() for o in node.outputs],
        )
        version.append(scalar_op.c_code_cache_version_apply(scalar_node))
        version.extend(
            get_scalar_type(dtype=i.type.dtype).c_code_cache_version()
            for i in node.inputs + node.outputs
        )
        version.append(("openmp", self._use_openmp()))
        version.append(("openmp_elemwise_minsize", config.openmp_elemwise_minsize))
        if all(version):
            return tuple(version)
        return ()


@c_funcify.register(Elemwise)
def c_funcify_elemwise(op, node=None, **kwargs) -> ElemwiseImpl:
    return ElemwiseImpl(op)


class CAReduceImpl(CImpl):
    """C implementation of `CAReduce` (and its subclasses).

    Emits the reduction loop over the per-element scalar code via
    `_careduce_c_all`; the loop inlines the scalar op's `c_code` directly, so no
    support code is delegated.
    """

    op: CAReduce

    def c_headers(self, **kwargs) -> list[str]:
        return ["<vector>", "<algorithm>"]

    def c_code(
        self,
        node: Apply,
        name: str,
        inputs: list[str],
        outputs: list[str],
        sub: dict[str, str],
    ) -> str:
        return "\n".join(_careduce_c_all(self.op, node, name, inputs, outputs, sub))

    def c_code_cache_version_apply(self, node: Apply) -> tuple[Hashable, ...]:
        scalar_op = self.op.scalar_op
        version = [11]  # bump when the emitted C changes
        scalar_node = Apply(
            scalar_op,
            [get_scalar_type(dtype=i.type.dtype).make_variable() for i in node.inputs],
            [get_scalar_type(dtype=o.type.dtype).make_variable() for o in node.outputs],
        )
        version.append(scalar_op.c_code_cache_version_apply(scalar_node))
        version.extend(
            get_scalar_type(dtype=i.type.dtype).c_code_cache_version()
            for i in node.inputs + node.outputs
        )
        if all(version):
            return tuple(version)
        return ()


@c_funcify.register(CAReduce)
def c_funcify_careduce(op, node=None, **kwargs) -> CAReduceImpl:
    return CAReduceImpl(op)
