"""Convert a jax function to a pytensor compatible function."""

from collections.abc import Sequence
from functools import wraps

import numpy as np

from pytensor.compile.function import function
from pytensor.compile.mode import Mode
from pytensor.gradient import DisconnectedType
from pytensor.graph import Apply, Op, Variable
from pytensor.tensor.basic import as_tensor, infer_static_shape
from pytensor.tensor.type import TensorType


class JAXOp(Op):
    """
    JAXOp is a PyTensor Op that wraps a JAX function, providing both forward
    computation and reverse-mode differentiation (via VJP).

    Parameters
    ----------
    input_types : list
        A list of PyTensor types for each input variable.
    output_types : list
        A list of PyTensor types for each output variable.
    jax_function : callable
        The JAX function that computes outputs from inputs. It should
        always return a tuple of outputs, even if there is only one output.
    name : str, optional
        A custom name for the Op instance. If provided, the class name will be
        updated accordingly.

    Example
    -------
    This example defines a simple function that sums the input array with a dynamic shape.

    >>> import numpy as np
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from pytensor.tensor import TensorType
    >>>
    >>> # Create the jax function that sums the input array.
    >>> def sum_function(x, y):
    ...     return (jnp.sum(x + y),)
    >>>
    >>> # Create the input and output types, input has a dynamic shape.
    >>> input_type = TensorType("float32", shape=(None,))
    >>> output_type = TensorType("float32", shape=())
    >>>
    >>> # Instantiate a JAXOp
    >>> op = JAXOp(
    ...     [input_type, input_type], [output_type], sum_function, name="DummyJAXOp"
    ... )
    >>> # Define symbolic input variables.
    >>> x = pt.tensor("x", dtype="float32", shape=(2,))
    >>> y = pt.tensor("y", dtype="float32", shape=(2,))
    >>> # Compile a PyTensor function.
    >>> result = op(x, y)
    >>> f = pytensor.function([x, y], [result])
    >>> print(
    ...     f(
    ...         np.array([2.0, 3.0], dtype=np.float32),
    ...         np.array([4.0, 5.0], dtype=np.float32),
    ...     )
    ... )
    [array(14., dtype=float32)]
    >>>
    >>> # Compute the gradient of op(x, y) with respect to x.
    >>> g = pt.grad(result, x)
    >>> grad_f = pytensor.function([x, y], [g])
    >>> print(
    ...     grad_f(
    ...         np.array([2.0, 3.0], dtype=np.float32),
    ...         np.array([4.0, 5.0], dtype=np.float32),
    ...     )
    ... )
    [array([1., 1.], dtype=float32)]
    """

    __props__ = ("input_types", "output_types", "jax_func")

    def __init__(self, input_types, output_types, jax_function, name=None):
        import jax

        self.input_types = tuple(input_types)
        self.output_types = tuple(output_types)
        self.jax_func = jax_function
        self.jitted_func = jax.jit(jax_function)
        self.name = name
        super().__init__()

    def __repr__(self):
        base = self.__class__.__name__
        props = list(self.__props__)
        if self.name is not None:
            props.insert(0, "name")
        props = ", ".join(f"{prop}={getattr(self, prop)}" for prop in props)
        return f"{base}({props})"

    def make_node(self, *inputs: Variable) -> Apply:
        """Create an Apply node with the given inputs and inferred outputs."""
        if len(inputs) != len(self.input_types):
            raise ValueError(
                f"Op {self} expected {len(self.input_types)} inputs, got {len(inputs)}"
            )
        filtered_inputs = [
            inp_type.filter_variable(inp)
            for inp, inp_type in zip(inputs, self.input_types)
        ]
        outputs = [output_type() for output_type in self.output_types]
        return Apply(self, filtered_inputs, outputs)

    def perform(self, node, inputs, outputs):
        """Execute the JAX function and store results in output storage."""
        results = self.jitted_func(*inputs)
        if not isinstance(results, tuple):
            raise TypeError("JAX function must return a tuple of outputs.")
        if len(results) != len(outputs):
            raise ValueError(
                f"JAX function returned {len(results)} outputs, but "
                f"{len(outputs)} were expected."
            )
        for output_container, result, out_type in zip(
            outputs, results, self.output_types
        ):
            output_container[0] = np.array(result, dtype=out_type.dtype)

    def perform_jax(self, *inputs):
        """Execute the JAX function directly, returning JAX arrays."""
        outputs = self.jitted_func(*inputs)
        if not isinstance(outputs, tuple):
            raise TypeError("JAX function must return a tuple of outputs.")
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def grad(self, inputs, output_gradients):
        """Compute gradients using JAX's vector-Jacobian product (VJP)."""
        import jax

        # Find indices of outputs that need gradients
        connected_output_indices = [
            i
            for i, output_grad in enumerate(output_gradients)
            if not isinstance(output_grad.type, DisconnectedType)
        ]

        num_inputs = len(inputs)

        def vjp_operation(*args):
            """VJP operation that computes gradients w.r.t. inputs."""
            input_values = args[:num_inputs]
            cotangent_vectors = args[num_inputs:]
            assert len(cotangent_vectors) == len(connected_output_indices)

            def restricted_function(*input_values):
                """Restricted function that only returns connected outputs."""
                outputs = self.jax_func(*input_values)
                return [
                    outputs[i].astype(self.output_types[i].dtype)
                    for i in connected_output_indices
                ]

            _primals, vjp_function = jax.vjp(restricted_function, *input_values)
            output_dtypes = [
                self.output_types[i].dtype for i in connected_output_indices
            ]
            return vjp_function(
                [
                    cotangent.astype(dtype)
                    for cotangent, dtype in zip(
                        cotangent_vectors, output_dtypes, strict=True
                    )
                ]
            )

        if self.name is not None:
            name = "vjp_" + self.name
        else:
            name = "vjp_jax_op"

        # Create VJP operation
        vjp_op = JAXOp(
            self.input_types
            + tuple(self.output_types[i] for i in connected_output_indices),
            [self.input_types[i] for i in range(num_inputs)],
            vjp_operation,
            name=name,
        )

        return vjp_op(
            *[*inputs, *[output_gradients[i] for i in connected_output_indices]],
            return_list=True,
        )


def wrap_jax(jax_function=None, *, allow_eval=True):
    """Return a PyTensor-compatible function from a JAX jittable function.

    This decorator wraps a JAX function so that it accepts and returns
    `pytensor.Variable` objects. The JAX-jittable function can accept any
    nested Python structure (a `Pytree
    <https://jax.readthedocs.io/en/latest/pytrees.html>`_) as input, and might
    return any nested Python structure.

    Parameters
    ----------
    jax_function : Callable, optional
        A JAX function to be wrapped. If None, returns a decorator function.
    allow_eval : bool, default=True
        Whether to allow evaluation of symbolic shapes when input shapes are
        not fully determined.

    Returns
    -------
    Callable
        A function that wraps the given JAX function so that it can be called with
        pytensor.Variable inputs and returns pytensor.Variable outputs.

    Examples
    --------

    >>> import jax.numpy as jnp
    >>> import pytensor.tensor as pt
    >>> from pytensor import wrap_jax
    >>> @wrap_jax
    ... def add(x, y):
    ...     return jnp.add(x, y)
    >>> x = pt.scalar("x")
    >>> y = pt.scalar("y")
    >>> result = add(x, y)
    >>> f = pytensor.function([x, y], [result])
    >>> print(f(1, 2))
    [array(3.)]

    We can also pass arbitrary jax pytree structures as inputs and outputs:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> import pytensor.tensor as pt
    >>> from pytensor import wrap_jax
    >>> @wrap_jax
    ... def complex_function(x, y, scale=1.0):
    ...     return {
    ...         "sum": jnp.add(x, y) * scale,
    ...     }
    >>> x = pt.vector("x", shape=(3,))
    >>> y = pt.vector("y", shape=(3,))
    >>> result = complex_function(x, y, scale=2.0)
    >>> f = pytensor.function([x, y], [result["sum"]])

    Or Equinox modules:

    >>> x = pt.tensor("x", shape=(3,))  # doctest +SKIP
    >>> y = pt.tensor("y", shape=(3,))  # doctest +SKIP
    >>> import equinox as eqx  # doctest +SKIP
    >>> mlp = eqx.nn.MLP(
    ...     3, 3, 3, depth=2, activation=jnp.tanh, key=jax.random.key(0)
    ... )  # doctest +SKIP
    >>> mlp = eqx.tree_at(lambda m: m.layers[0].bias, mlp, y)  # doctest +SKIP
    >>> @wrap_jax  # doctest +SKIP
    ... def neural_network(x, mlp):  # doctest +SKIP
    ...     return mlp(x)  # doctest +SKIP
    >>> out = neural_network(x, mlp)  # doctest +SKIP

    If the input shapes are not fully determined, and valid
    input shapes cannot be inferred by evaluating the inputs either,
    an error will be raised:

    >>> import jax.numpy as jnp
    >>> import pytensor.tensor as pt
    >>> @wrap_jax
    ... def add(x, y):
    ...     return jnp.add(x, y)
    >>> x = pt.vector("x")  # shape is not fully determined
    >>> y = pt.vector("y")  # shape is not fully determined
    >>> result = add(x, y)
    ValueError: Could not compile a function to infer example shapes. Please provide inputs with fully determined shapes by calling pt.specify_shape.
    ...
    """

    def decorator(func):
        name = func.__name__

        try:
            import jax
        except ImportError as e:
            raise ImportError(
                "The wrap_jax decorator requires jax to be installed."
            ) from e

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Partition inputs into dynamic PyTensor variables and static variables.
            # Static variables don't participate in the computational graph.
            pytensor_variables, static_values = _eqx_partition(
                (args, kwargs), lambda x: isinstance(x, Variable)
            )

            # Flatten the PyTensor variables for processing
            variables_flat, variables_treedef = jax.tree.flatten(pytensor_variables)
            input_types = [var.type for var in variables_flat]

            # Determine output types by calling the function through jax.eval_shape
            output_types, output_treedef, output_static = _find_output_types(
                func,
                variables_flat,
                variables_treedef,
                static_values,
                allow_eval=allow_eval,
            )

            def flattened_function(*flat_variables):
                """Execute the original function with flattened inputs."""
                variables = jax.tree.unflatten(variables_treedef, flat_variables)
                reconstructed_args, reconstructed_kwargs = _eqx_combine(
                    variables, static_values
                )
                function_outputs = func(*reconstructed_args, **reconstructed_kwargs)
                array_outputs, _ = _eqx_partition(function_outputs, _is_array)
                flattened_outputs, _ = jax.tree.flatten(array_outputs)
                return tuple(flattened_outputs)

            # Create the JAX operation
            jax_op_instance = JAXOp(
                input_types,
                output_types,
                flattened_function,
                name=name,
            )

            # Execute the operation and reconstruct the output structure
            flattened_results = jax_op_instance(*variables_flat)
            if not isinstance(flattened_results, Sequence):
                flattened_results = [flattened_results]

            output_variables = jax.tree.unflatten(output_treedef, flattened_results)
            final_outputs = _eqx_combine(output_variables, output_static)

            return final_outputs

        return wrapper

    if jax_function is None:
        return decorator
    else:
        return decorator(jax_function)


def _find_output_types(
    jax_function, inputs_flat, input_treedef, static_input, *, allow_eval=True
):
    """Determine output types with jax.eval_shape on dummy inputs."""
    import jax
    import jax.numpy as jnp

    resolved_input_shapes = []
    requires_shape_evaluation = False

    for variable in inputs_flat:
        # If shape is already fully determined, use it directly
        if not any(dimension is None for dimension in variable.type.shape):
            resolved_input_shapes.append(variable.type.shape)
            continue

        # Try to infer static shape
        _, inferred_shape = infer_static_shape(variable.shape)
        if not any(dimension is None for dimension in inferred_shape):
            resolved_input_shapes.append(inferred_shape)
            continue

        # Shape still has undetermined dimensions
        if not allow_eval:
            raise ValueError(
                f"Input variable {variable} has undetermined shape dimensions. "
                "Please provide inputs with fully determined shapes by calling "
                "pt.specify_shape."
            )
        requires_shape_evaluation = True
        resolved_input_shapes.append(variable.shape)

    if requires_shape_evaluation:
        try:
            shape_evaluation_function = function(
                [],
                [as_tensor(s, dtype="int64") for s in resolved_input_shapes],
                on_unused_input="ignore",
                mode=Mode(linker="py", optimizer="fast_compile"),
            )
        except Exception as e:
            raise ValueError(
                "Could not compile a function to infer example shapes. "
                "Please provide inputs with fully determined shapes by "
                "calling pt.specify_shape."
            ) from e
        resolved_input_shapes = [tuple(s) for s in shape_evaluation_function()]

    # Determine output types using jax.eval_shape with dummy inputs
    output_metadata_storage = {}

    dummy_input_arrays = [
        jnp.ones(shape, dtype=variable.type.dtype)
        for variable, shape in zip(inputs_flat, resolved_input_shapes, strict=True)
    ]

    def wrapped_jax_function(input_arrays):
        """Wrapper to extract output metadata during shape evaluation."""
        variables = jax.tree.unflatten(input_treedef, input_arrays)
        reconstructed_args, reconstructed_kwargs = _eqx_combine(variables, static_input)
        function_outputs = jax_function(*reconstructed_args, **reconstructed_kwargs)
        array_outputs, static_outputs = _eqx_partition(function_outputs, _is_array)

        # Store metadata for later use
        output_metadata_storage["output_static"] = static_outputs
        flattened_outputs, output_structure = jax.tree.flatten(array_outputs)
        output_metadata_storage["output_treedef"] = output_structure
        return flattened_outputs

    output_shapes_flat = jax.eval_shape(wrapped_jax_function, dummy_input_arrays)
    output_treedef = output_metadata_storage["output_treedef"]
    output_static = output_metadata_storage["output_static"]

    # If we used shape evaluation, set all output shapes to unknown
    # TODO: This is throwing away potential static shape information.
    if requires_shape_evaluation:
        output_types = [
            TensorType(
                dtype=output_shape.dtype, shape=tuple(None for _ in output_shape.shape)
            )
            for output_shape in output_shapes_flat
        ]
    else:
        output_types = [
            TensorType(dtype=output_shape.dtype, shape=output_shape.shape)
            for output_shape in output_shapes_flat
        ]

    return output_types, output_treedef, output_static


# From the equinox library, licensed under Apache 2.0
# https://github.com/patrick-kidger/equinox
#
# Copied here to avoid a dependency on equinox just these functions.
def _eqx_combine(*pytrees, is_leaf=None):
    """Combines multiple PyTrees into one PyTree, by replacing `None` leaves.

    !!! example

        ```python
        pytree1 = [None, 1, 2]
        pytree2 = [0, None, None]
        equinox.combine(pytree1, pytree2)  # [0, 1, 2]
        ```

    !!! tip

        The idea is that `equinox.combine` should be used to undo a call to
        [`equinox.filter`][] or [`equinox.partition`][].

    **Arguments:**

    - `*pytrees`: a sequence of PyTrees all with the same structure.
    - `is_leaf`: As [`equinox.partition`][].

    **Returns:**

    A PyTree with the same structure as its inputs. Each leaf will be the first
    non-`None` leaf found in the corresponding leaves of `pytrees` as they are
    iterated over.
    """
    import jax

    if is_leaf is None:
        _is_leaf = _is_none
    else:
        _is_leaf = lambda x: _is_none(x) or is_leaf(x)  # noqa: E731

    return jax.tree.map(_combine, *pytrees, is_leaf=_is_leaf)


def _eqx_partition(
    pytree,
    filter_spec,
    replace=None,
    is_leaf=None,
):
    """Splits a PyTree into two pieces. Equivalent to
    `filter(...), filter(..., inverse=True)`, but slightly more efficient.

    !!! info

        See also [`equinox.combine`][] to reconstitute the PyTree again.
    """
    import jax

    filter_tree = jax.tree.map(_make_filter_tree(is_leaf), filter_spec, pytree)
    left = jax.tree.map(lambda mask, x: x if mask else replace, filter_tree, pytree)
    right = jax.tree.map(lambda mask, x: replace if mask else x, filter_tree, pytree)
    return left, right


def _make_filter_tree(is_leaf):
    import jax
    import jax.core

    def _filter_tree(mask, arg):
        if isinstance(mask, jax.core.Tracer):
            raise ValueError("`filter_spec` leaf values cannot be traced arrays.")
        if isinstance(mask, bool):
            return jax.tree.map(lambda _: mask, arg, is_leaf=is_leaf)
        elif callable(mask):
            return jax.tree.map(mask, arg, is_leaf=is_leaf)
        else:
            raise ValueError(
                "`filter_spec` must consist of booleans and callables only."
            )

    return _filter_tree


def _is_array(element) -> bool:
    """Returns `True` if `element` is a JAX array or NumPy array."""
    import jax

    return isinstance(element, np.ndarray | np.generic | jax.Array)


def _combine(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None


def _is_none(x):
    return x is None
