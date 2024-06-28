from typing import Any

from pytensor.graph.basic import Variable
from pytensor.link.basic import JITLinker


class PytorchLinker(JITLinker):
    """A `Linker` that compiles NumPy-based operations using torch.compile."""

    def input_filter(self, inp: Any) -> Any:
        from pytensor.link.pytorch.dispatch import pytorch_typify

        return pytorch_typify(inp)

    def output_filter(self, var: Variable, out: Any) -> Any:
        return out.cpu()

    def fgraph_convert(self, fgraph, input_storage, storage_map, **kwargs):
        from pytensor.link.pytorch.dispatch import pytorch_funcify

        return pytorch_funcify(
            fgraph, input_storage=input_storage, storage_map=storage_map, **kwargs
        )

    def jit_compile(self, fn):
        import torch

        return torch.compile(fn)

    def create_thunk_inputs(self, storage_map):
        thunk_inputs = []
        for n in self.fgraph.inputs:
            sinput = storage_map[n]
            thunk_inputs.append(sinput)

        return thunk_inputs
