import warnings
from collections.abc import Callable, Collection
from functools import cache
from typing import TYPE_CHECKING, cast

from pytensor.configdefaults import config
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.op import ComputeMapType, Op, StorageMapType, ThunkType
from pytensor.graph.utils import MethodNotDefined
from pytensor.link.c.interface import CLinkerOp


if TYPE_CHECKING:
    from pytensor.link.c.basic import _CThunk


class CThunkWrapperType(ThunkType):
    thunk: "_CThunk"
    cthunk: ThunkType


def is_cthunk_wrapper_type(thunk: Callable[[], None]) -> CThunkWrapperType:
    res = cast(CThunkWrapperType, thunk)
    return res


class COp(Op, CLinkerOp):
    """An `Op` with a C implementation."""

    def make_c_thunk(
        self,
        node: Apply,
        storage_map: StorageMapType,
        compute_map: ComputeMapType | None,
        no_recycling: Collection[Variable],
    ) -> CThunkWrapperType:
        """Create a thunk for a C implementation.

        Like :meth:`Op.make_thunk`, but will only try to make a C thunk.

        """
        # FIXME: Putting the following import on the module level causes an import cycle.
        #        The conclusion should be that the antire "make_c_thunk" method should be defined
        #        in pytensor.link.c and dispatched onto the Op!
        import pytensor.link.c.basic
        from pytensor.graph.fg import FunctionGraph

        node_input_storage = [storage_map[r] for r in node.inputs]
        node_output_storage = [storage_map[r] for r in node.outputs]

        e = FunctionGraph(node.inputs, node.outputs)
        e_no_recycling = [
            new_o
            for (new_o, old_o) in zip(e.outputs, node.outputs, strict=True)
            if old_o in no_recycling
        ]
        cl = pytensor.link.c.basic.CLinker().accept(e, no_recycling=e_no_recycling)
        # float16 gets special treatment since running
        # unprepared C code will get bad results.
        if not getattr(self, "_f16_ok", False):

            def is_f16(t):
                return getattr(t, "dtype", "") == "float16"

            if any(is_f16(i.type) for i in node.inputs) or any(
                is_f16(o.type) for o in node.outputs
            ):
                # get_dynamic_module is a subset of make_thunk that is reused.
                # This just try to build the c code
                # It will raise an error for ops
                # that don't implement c code. In those cases, we
                # don't want to print a warning.
                cl.get_dynamic_module()
                warnings.warn(f"Disabling C code for {self} due to unsupported float16")
                raise NotImplementedError("float16")
        outputs = cl.make_thunk(
            input_storage=node_input_storage, output_storage=node_output_storage
        )
        thunk, _node_input_filters, _node_output_filters = outputs

        if compute_map is None:
            rval = is_cthunk_wrapper_type(thunk)

        else:
            cm_entries = [compute_map[o] for o in node.outputs]

            @is_cthunk_wrapper_type
            def rval(thunk=thunk, cm_entries=cm_entries):
                thunk()
                for entry in cm_entries:
                    entry[0] = True

        rval.thunk = thunk
        rval.cthunk = thunk.cthunk
        rval.inputs = node_input_storage
        rval.outputs = node_output_storage
        rval.lazy = False
        return rval

    def make_thunk(self, node, storage_map, compute_map, no_recycling, impl=None):
        """Create a thunk.

        See :meth:`Op.make_thunk`.

        Parameters
        ----------
        impl :
            Currently, ``None``, ``'c'`` or ``'py'``. If ``'c'`` or ``'py'`` we
            will only try that version of the code.

        """
        if (impl is None and config.cxx) or impl == "c":
            self.prepare_node(
                node, storage_map=storage_map, compute_map=compute_map, impl="c"
            )
            try:
                return self.make_c_thunk(node, storage_map, compute_map, no_recycling)
            except (NotImplementedError, MethodNotDefined):
                # We requested the c code, so don't catch the error.
                if impl == "c":
                    raise

        return super().make_thunk(
            node, storage_map, compute_map, no_recycling, impl=impl
        )


@cache
def openmp_supported() -> bool:
    """Return whether the active C compiler can build OpenMP code.

    Memoized; the probe runs at most once per process. It is pure — it never
    mutates ``config`` or op state, so the result reflects only the compiler,
    not call order. Return ``False`` when there is no C compiler.
    """
    if not config.cxx:
        return False

    from pytensor.link.c.cmodule import GCC_compiler

    code = """
    #include <omp.h>
int main( int argc, const char* argv[] )
{
    int res[10];
    for(int i=0; i < 10; i++){
        res[i] = i;
    }
}
    """
    supported = bool(
        GCC_compiler.try_compile_tmp(
            src_code=code, tmp_prefix="test_omp_", flags=["-fopenmp"], try_run=False
        )
    )
    if not supported:
        warnings.warn(
            "Your C compiler fails to compile OpenMP code; PyTensor will run "
            "Elemwise operations single-threaded. Set the `openmp` flag to False "
            "to silence this warning.",
            stacklevel=2,
        )
    return supported


class _NoPythonCOp(COp):
    """A class used to indicate that a `COp` does not provide a Python implementation.

    XXX: Do not use this class; it's only for tracking bad implementations internally.

    """

    def perform(self, node, inputs, output_storage):
        raise NotImplementedError("No Python implementation is provided by this COp.")
