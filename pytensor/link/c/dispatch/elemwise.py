from collections.abc import Hashable

from pytensor.configdefaults import config
from pytensor.graph.basic import Apply
from pytensor.link.c.dispatch.basic import CImpl, c_funcify
from pytensor.link.c.op import openmp_supported
from pytensor.scalar import get_scalar_type
from pytensor.tensor.elemwise import DimShuffle, Elemwise, _elemwise_c_all


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
        return self.op.scalar_op.c_support_code(**kwargs)

    def c_support_code_apply(self, node: Apply, name: str) -> str:
        return self.op.scalar_op.c_support_code_apply(node, name + "_scalar_")

    def c_headers(self, **kwargs) -> list[str]:
        return ["<vector>", "<algorithm>"]

    def c_header_dirs(self, **kwargs) -> list[str]:
        return self.op.scalar_op.c_header_dirs(**kwargs)

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
            any(i.dtype == "float16" for i in node.inputs)
            or any(o.dtype == "float16" for o in node.outputs)
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
        version = [17]  # bump when the emitted C changes
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
