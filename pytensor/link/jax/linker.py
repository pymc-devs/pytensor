import warnings

from numpy.random import Generator, RandomState

from pytensor.compile.sharedvalue import SharedVariable, shared
from pytensor.link.basic import JITLinker


class JAXLinker(JITLinker):
    """A `Linker` that JIT-compiles NumPy-based operations using JAX."""

    def fgraph_convert(self, fgraph, input_storage, storage_map, **kwargs):
        from pytensor.link.jax.dispatch import jax_funcify
        from pytensor.tensor.random.type import RandomType

        shared_rng_inputs = [
            inp
            for inp in fgraph.inputs
            if (isinstance(inp, SharedVariable) and isinstance(inp.type, RandomType))
        ]

        # Replace any shared RNG inputs so that their values can be updated in place
        # without affecting the original RNG container. This is necessary because
        # JAX does not accept RandomState/Generators as inputs, and they will have to
        # be tipyfied
        if shared_rng_inputs:
            warnings.warn(
                f"The RandomType SharedVariables {shared_rng_inputs} will not be used "
                f"in the compiled JAX graph. Instead a copy will be used.",
                UserWarning,
            )
            new_shared_rng_inputs = [
                shared(inp.get_value(borrow=False)) for inp in shared_rng_inputs
            ]

            fgraph.replace_all(
                zip(shared_rng_inputs, new_shared_rng_inputs, strict=True),
                import_missing=True,
                reason="JAXLinker.fgraph_convert",
            )

            for old_inp, new_inp in zip(
                shared_rng_inputs, new_shared_rng_inputs, strict=True
            ):
                new_inp_storage = [new_inp.get_value(borrow=True)]
                storage_map[new_inp] = new_inp_storage
                old_inp_storage = storage_map.pop(old_inp)
                # Find index of old_inp_storage in input_storage
                for input_storage_idx, input_storage_item in enumerate(input_storage):
                    # We have to establish equality based on identity because input_storage may contain numpy arrays
                    if input_storage_item is old_inp_storage:
                        break
                else:  # no break
                    raise ValueError()
                input_storage[input_storage_idx] = new_inp_storage
                # We need to change the order of the inputs of the FunctionGraph
                # so that the new input is in the same position as to old one,
                # to align with the storage_map. We hope this is safe!
                old_inp_fgrap_index = fgraph.inputs.index(old_inp)
                fgraph.remove_input(
                    old_inp_fgrap_index,
                    reason="JAXLinker.fgraph_convert",
                )
                fgraph.inputs.remove(new_inp)
                fgraph.inputs.insert(old_inp_fgrap_index, new_inp)

        return jax_funcify(
            fgraph, input_storage=input_storage, storage_map=storage_map, **kwargs
        )

    def jit_compile(self, fn):
        import jax

        return jax.jit(fn)

    def create_thunk_inputs(self, storage_map):
        from pytensor.link.jax.dispatch import jax_typify

        thunk_inputs = []
        for n in self.fgraph.inputs:
            sinput = storage_map[n]
            if isinstance(sinput[0], RandomState | Generator):
                new_value = jax_typify(
                    sinput[0], dtype=getattr(sinput[0], "dtype", None)
                )
                sinput[0] = new_value
            thunk_inputs.append(sinput)

        return thunk_inputs
