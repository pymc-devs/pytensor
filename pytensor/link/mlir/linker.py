import numpy as np

from pytensor.link.basic import JITLinker


class MLIRLinker(JITLinker):
    """A linker that lowers a FunctionGraph to an IREE target."""

    required_rewrites = ("minimum_compile",)
    incompatible_rewrites = (
        "cxx_only",
        "BlasOpt",
        "inplace",
        "fusion",
        "scan_reduce_trace_prealloc",
    )

    _target_options = {
        "llvm-cpu": (
            "local-task",
            ("--iree-input-demote-f64-to-f32=false",),
        ),
        "metal-spirv": (
            "metal",
            (
                "--iree-metal-target-platform=macos",
                "--iree-input-demote-f64-to-f32=false",
            ),
        ),
    }

    def __init__(self, target_backend="llvm-cpu", *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.runtime_driver, self.compile_args = self._target_options[target_backend]
        except KeyError:
            raise ValueError(f"Unsupported MLIR target backend: {target_backend}") from None
        self.target_backend = target_backend

    def accept(self, fgraph, no_recycling=None, profile=None):
        if self.fgraph is not None and self.fgraph is not fgraph:
            return type(self)(
                target_backend=self.target_backend, allow_gc=self.allow_gc
            ).accept(fgraph, no_recycling, profile)
        return super().accept(fgraph, no_recycling, profile)

    def fgraph_convert(self, fgraph, **kwargs):
        if self.target_backend == "metal-spirv" and any(
            getattr(variable.type, "dtype", None) == "float64"
            for variable in fgraph.variables
        ):
            raise TypeError("MLIR Metal only supports float32 graphs")

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

        if self.target_backend == "metal-spirv" and "f64" in module.text:
            raise TypeError("MLIR Metal only supports float32 graphs")

        vmfb = iree.compiler.tools.compile_str(
            module.text,
            target_backends=[self.target_backend],
            extra_args=self.compile_args,
        )
        entrypoint = iree.runtime.load_vm_flatbuffer(
            vmfb, driver=self.runtime_driver
        )[module.entrypoint]

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
