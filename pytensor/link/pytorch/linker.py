from pytensor.link.basic import JITLinker
from pytensor.link.utils import unique_name_generator


class PytorchLinker(JITLinker):
    """A `Linker` that compiles NumPy-based operations using torch.compile."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen_functors = []

    def fgraph_convert(self, fgraph, input_storage, storage_map, **kwargs):
        from pytensor.link.pytorch.dispatch import pytorch_funcify

        # We want to have globally unique names
        # across the entire pytensor graph, not
        # just the subgraph
        generator = unique_name_generator(["torch_linker"])

        # Ensure that torch is aware of the generated
        # code so we can compile without graph breaks
        def conversion_func_register(*args, **kwargs):
            functor = pytorch_funcify(*args, **kwargs)
            name = kwargs["unique_name"](functor)
            self.gen_functors.append((f"_{name}", functor))
            return functor

        built_kwargs = {
            "unique_name": generator,
            "conversion_func": conversion_func_register,
            **kwargs,
        }
        return pytorch_funcify(
            fgraph, input_storage=input_storage, storage_map=storage_map, **built_kwargs
        )

    def jit_compile(self, fn):
        import torch

        from pytensor.link.pytorch.dispatch import pytorch_typify

        class wrapper:
            """
            Pytorch would fail compiling our method when trying
            to resolve some of the methods returned from dispatch
            calls. We want to be careful to not leak the methods,
            so this class just holds them and provisions the expected
            location accordingly

            https://discuss.pytorch.org/t/closures-are-being-gcd-and-causing-failures-to-compile/213319
            """

            def __init__(self, fn, gen_functors):
                self.fn = torch.compile(fn)
                self.gen_functors = gen_functors.copy()

            def __call__(self, *args, **kwargs):
                import pytensor.link.utils

                # set attrs
                for n, fn in self.gen_functors:
                    setattr(pytensor.link.utils, n[1:], fn)

                res = self.fn(*args, **kwargs)

                # unset attrs
                for n, _ in self.gen_functors:
                    if getattr(pytensor.link.utils, n[1:], False):
                        delattr(pytensor.link.utils, n[1:])

                return res

            def __del__(self):
                del self.gen_functors

        inner_fn = wrapper(fn, self.gen_functors)
        self.gen_functors = []

        # Torch does not accept numpy inputs and may return GPU objects
        def fn(*inputs, inner_fn=inner_fn):
            outs = inner_fn(*(pytorch_typify(inp) for inp in inputs))
            return tuple(out.cpu().numpy() for out in outs)

        return fn

    def create_thunk_inputs(self, storage_map):
        thunk_inputs = []
        for n in self.fgraph.inputs:
            sinput = storage_map[n]
            thunk_inputs.append(sinput)

        return thunk_inputs
