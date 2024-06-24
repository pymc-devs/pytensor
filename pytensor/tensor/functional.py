from collections.abc import Callable

from pytensor.graph import vectorize_graph
from pytensor.tensor.utils import _parse_gufunc_signature
from pytensor.tensor.variable import TensorVariable


def vectorize(func: Callable, signature: str | None = None) -> Callable:
    """Create a vectorized version of a python function that takes TensorVariables as inputs and outputs.

    Similar to numpy.vectorize. See respective docstrings for more details.

    Parameters
    ----------
    func: Callable
        Function that creates the desired outputs from TensorVariable inputs with the core dimensions.
    signature: str, optional
        Generalized universal function signature, e.g., (m,n),(n)->(m) for vectorized matrix-vector multiplication.
        If not provided, it is assumed all inputs have scalar core dimensions. Unlike numpy, the outputs
        can have arbitrary shapes when the signature is not provided.

    Returns
    -------
    vectorized_func: Callable
        Callable that takes TensorVariables with arbitrarily batched dimensions on the left
        and returns variables whose graphs correspond to the vectorized expressions of func.

    Notes
    -----
    Unlike numpy.vectorize, the equality of core dimensions implied by the signature is not explicitly asserted.

    To vectorize an existing graph, use `pytensor.graph.replace.vectorize_graph` instead.


    Examples
    --------
    .. code-block:: python

        import pytensor
        import pytensor.tensor as pt


        def func(x):
            return pt.exp(x) / pt.sum(pt.exp(x))


        vec_func = pt.vectorize(func, signature="(a)->(a)")

        x = pt.matrix("x")
        y = vec_func(x)

        fn = pytensor.function([x], y)
        fn([[0, 1, 2], [2, 1, 0]])
        # array([[0.09003057, 0.24472847, 0.66524096],
        #        [0.66524096, 0.24472847, 0.09003057]])


    .. code-block:: python

        import pytensor
        import pytensor.tensor as pt


        def func(x):
            return x[0], x[-1]


        vec_func = pt.vectorize(func, signature="(a)->(),()")

        x = pt.matrix("x")
        y1, y2 = vec_func(x)

        fn = pytensor.function([x], [y1, y2])
        fn([[-10, 0, 10], [-11, 0, 11]])
        # [array([-10., -11.]), array([10., 11.])]

    """

    def inner(*inputs):
        if signature is None:
            # Assume all inputs are scalar
            inputs_sig = [()] * len(inputs)
        else:
            inputs_sig, outputs_sig = _parse_gufunc_signature(signature)
            if len(inputs) != len(inputs_sig):
                raise ValueError(
                    f"Number of inputs does not match signature: {signature}"
                )

        # Create dummy core inputs by stripping the batched dimensions of inputs
        core_inputs = []
        for input, input_sig in zip(inputs, inputs_sig, strict=True):
            if not isinstance(input, TensorVariable):
                raise TypeError(
                    f"Inputs to vectorize function must be TensorVariable, got {type(input)}"
                )

            if input.ndim < len(input_sig):
                raise ValueError(
                    f"Input {input} has less dimensions than signature {input_sig}"
                )
            if len(input_sig):
                core_shape = input.type.shape[-len(input_sig) :]
            else:
                core_shape = ()

            core_input = input.type.clone(shape=core_shape)(name=input.name)
            core_inputs.append(core_input)

        # Call function on dummy core inputs
        core_outputs = func(*core_inputs)
        if core_outputs is None:
            raise ValueError("vectorize function returned no outputs")

        if signature is not None:
            if isinstance(core_outputs, list | tuple):
                n_core_outputs = len(core_outputs)
            else:
                n_core_outputs = 1
            if n_core_outputs != len(outputs_sig):
                raise ValueError(
                    f"Number of outputs does not match signature: {signature}"
                )

        # Vectorize graph by replacing dummy core inputs by original inputs
        outputs = vectorize_graph(
            core_outputs, replace=dict(zip(core_inputs, inputs, strict=True))
        )
        return outputs

    return inner
