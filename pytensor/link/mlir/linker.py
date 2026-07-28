import numpy as np

from pytensor.link.basic import JITLinker


class MLIRLinker(JITLinker):
    """A linker that lowers a FunctionGraph to an IREE CPU module."""

    required_rewrites = ("minimum_compile",)
    incompatible_rewrites = (
        "cxx_only",
        "BlasOpt",
        "inplace",
        "fusion",
        "scan_reduce_trace_prealloc",
    )

    def fgraph_convert(self, fgraph, **kwargs):
        from pytensor.link.mlir.dispatch import mlir_funcify

        return mlir_funcify(fgraph)

    def jit_compile(self, module):
        try:
            import iree.compiler.tools
            import iree.runtime
        except ModuleNotFoundError as error:
            raise ImportError(
                "MLIR mode requires iree-base-compiler and iree-base-runtime"
            ) from error

        from pytensor.link.mlir.dispatch import mlir_typify

        vmfb = iree.compiler.tools.compile_str(
            module.text,
            target_backends=["llvm-cpu"],
            extra_args=["--iree-input-demote-f64-to-f32=false"],
        )
        entrypoint = iree.runtime.load_vm_flatbuffer(vmfb, backend="llvm-cpu")[
            module.entrypoint
        ]

        def run(*inputs):
            runtime_inputs = tuple(mlir_typify(input) for input in inputs)
            if module.runtime_validator is not None:
                module.runtime_validator(*runtime_inputs)
            result = entrypoint(*runtime_inputs)
            if module.output_count == 0:
                return None
            if module.output_count == 1:
                result = (result,)
            return tuple(np.asarray(value).copy() for value in result)

        return run

    def create_thunk_inputs(self, storage_map):
        return [storage_map[variable] for variable in self.fgraph.inputs]
