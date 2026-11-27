from pytensor.link.basic import JITLinker


class PythonLinker(JITLinker):
    """Compose a `FunctionGraph` into a single pure-Python function.

    The whole graph is turned into one straight-line Python function by
    `fgraph_to_python`, dispatching each `Op` through the `python_funcify`
    registry (falling back to ``perform`` for unregistered ops). There is no
    compilation step, so `jit_compile` is the identity.
    """

    required_rewrites = ("minimum_compile", "py_only")
    incompatible_rewrites = ("cxx_only",)

    def fgraph_convert(self, fgraph, **kwargs):
        from pytensor.link.python.dispatch.basic import python_funcify

        return python_funcify(fgraph, **kwargs)

    def jit_compile(self, fn):
        return fn

    def create_thunk_inputs(self, storage_map):
        return [storage_map[n] for n in self.fgraph.inputs]
