import warnings
import torch

from pytensor.compile.sharedvalue import SharedVariable, shared
from pytensor.graph.basic import Constant
from pytensor.link.basic import JITLinker


class PyTorchLinker(JITLinker):
    """A `Linker` that JIT-compiles NumPy-based operations using PyTorch."""

    def fgraph_convert(self, fgraph, input_storage, storage_map, **kwargs):
        from pytensor.link.pytorch.dispatch import pytorch_funcify
        from pytensor.tensor.random.type import RandomType

        shared_rng_inputs = [
            inp
            for inp in fgraph.inputs
            if (isinstance(inp, SharedVariable) and isinstance(inp.type, RandomType))
        ]

        if shared_rng_inputs:
            warnings.warn(
                f"The RandomType SharedVariables {shared_rng_inputs} will not be used "
                f"in the compiled PyTorch graph. Instead a copy will be used.",
                UserWarning,
            )
            new_shared_rng_inputs = [
                shared(inp.get_value(borrow=False)) for inp in shared_rng_inputs
            ]

            fgraph.replace_all(
                zip(shared_rng_inputs, new_shared_rng_inputs),
                import_missing=True,
                reason="PyTorchLinker.fgraph_convert",
            )

            for old_inp, new_inp in zip(shared_rng_inputs, new_shared_rng_inputs):
                new_inp_storage = [new_inp.get_value(borrow=True)]
                storage_map[new_inp] = new_inp_storage
                old_inp_storage = storage_map.pop(old_inp)
                for input_storage_idx, input_storage_item in enumerate(input_storage):
                    if input_storage_item is old_inp_storage:
                        break
                else:  # no break
                    raise ValueError()
                input_storage[input_storage_idx] = new_inp_storage
                fgraph.remove_input(
                    fgraph.inputs.index(old_inp), reason="PyTorchLinker.fgraph_convert"
                )

        return pytorch_funcify(
            fgraph, input_storage=input_storage, storage_map=storage_map, **kwargs
        )

    def jit_compile(self, fn):
        # For PyTorch, the script mode allows for JIT compilation
        scripted_fn = torch.jit.script(fn)
        return scripted_fn

    def create_thunk_inputs(self, storage_map):
        from pytensor.link.pytorch.dispatch import pytorch_typify

        thunk_inputs = []
        for n in self.fgraph.inputs:
            sinput = storage_map[n]
            if isinstance(sinput[0], torch.Generator):
                new_value = pytorch_typify(
                    sinput[0], dtype=getattr(sinput[0], "dtype", None)
                )
                sinput[0] = new_value
            thunk_inputs.append(sinput)

        return thunk_inputs
