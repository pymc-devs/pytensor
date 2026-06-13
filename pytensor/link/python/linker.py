from pytensor.link.vm import VMLinker


class PythonLinker(VMLinker):
    """A pure-Python `VMLinker` that runs each node through the `python_funcify` registry.

    Per node, a registered `python_funcify` implementation (a fast numpy/scipy
    callable) is wrapped into a thunk; unregistered ops fall back to their
    ``perform`` method via ``Op.make_thunk(impl="py")``. Lazy ops such as
    ``IfElse`` fall through to their own thunks, so the VM still short-circuits
    them. Fusion is excluded because fused ``Composite`` loops run slower than
    vectorized numpy on this backend.
    """

    def __init__(
        self,
        allow_gc=None,
        use_cloop=False,
        callback=None,
        callback_input=None,
        lazy=None,
        schedule=None,
        c_thunks=None,
        allow_partial_eval=None,
    ):
        # The Python backend never emits C: per-node Python thunks, Python VM.
        super().__init__(
            allow_gc=allow_gc,
            use_cloop=False,
            callback=callback,
            callback_input=callback_input,
            lazy=lazy,
            schedule=schedule,
            c_thunks=False,
            allow_partial_eval=allow_partial_eval,
        )
        # ``c_thunks=False`` already gives ("minimum_compile", "py_only") /
        # ("cxx_only",); add fusion for the numpy backend.
        self.incompatible_rewrites = ("cxx_only", "fusion")

    def _make_node_thunk(self, node, storage_map, compute_map, impl):
        from pytensor.link.python.dispatch.basic import (
            make_node_thunk_with_python_dispatch,
        )

        return make_node_thunk_with_python_dispatch(
            node,
            storage_map,
            compute_map,
            fallback=super()._make_node_thunk,
            impl=impl,
        )
