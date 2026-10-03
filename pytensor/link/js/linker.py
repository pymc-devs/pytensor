"""Lower rewritten PyTensor graphs to JavaScript evaluated by V8."""

from pytensor.link.basic import JITLinker


def lower(fgraph):
    # Import dispatch after PyTensor's scalar and tensor modules initialize.
    from pytensor.link.js.dispatch import lower as lower_fgraph

    return lower_fgraph(fgraph)


class JSLinker(JITLinker):
    """Experimental V8 linker with dynamic axis lengths."""

    required_rewrites = ("minimum_compile", "js")
    incompatible_rewrites = (
        "cxx_only",
        "BlasOpt",
        "inplace",
        "scan_reduce_trace_prealloc",
    )

    def fgraph_convert(self, fgraph, **kwargs):
        return lower(fgraph)

    def jit_compile(self, program):
        from pytensor.link.js.node_runtime import NodeJSFunction

        return NodeJSFunction(program)

    def create_thunk_inputs(self, storage_map):
        return [storage_map[node] for node in self.fgraph.inputs]
