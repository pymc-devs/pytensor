"""Convert a jax function to a pytensor compatible function."""

import logging
from collections.abc import Sequence
from functools import wraps

import numpy as np

import pytensor.tensor as pt
from pytensor.compile.function import function
from pytensor.gradient import DisconnectedType
from pytensor.graph import Apply, Op, Variable


log = logging.getLogger(__name__)


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
        The JAX function that computes outputs from inputs.
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
    ...     return jnp.sum(x + y)
    >>>
    >>> # Create the input and output types, input has a dynamic shape.
    >>> input_type = TensorType("float32", shape=(None,))
    >>> output_type = TensorType("float32", shape=(1,))
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
    >>> g = pt.grad(result[0], x)
    >>> grad_f = pytensor.function([x, y], [g])
    >>> print(
    ...     grad_f(
    ...         np.array([2.0, 3.0], dtype=np.float32),
    ...         np.array([4.0, 5.0], dtype=np.float32),
    ...     )
    ... )
    [array([1., 1.], dtype=float32)]
    """

    __props__ = ("input_types", "output_types", "jax_func", "name")

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
        if self.name is not None:
            base = f"{base}{self.name}"
        props = list(self.__props__)
        props.remove("name")
        props = ",".join(f"{prop}={getattr(self, prop, '?')}" for prop in props)
        return f"{base}({props})"

    def make_node(self, *inputs: Variable) -> Apply:
        """Create an Apply node with the given inputs and inferred outputs."""
        outputs = [output_type() for output_type in self.output_types]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        """Execute the JAX function and store results in output storage."""
        results = self.jitted_func(*inputs)
        if len(results) != len(outputs):
            raise ValueError(
                f"JAX function returned {len(results)} outputs, but "
                f"{len(outputs)} were expected."
            )
        for i, result in enumerate(results):
            outputs[i][0] = np.array(result, dtype=self.output_types[i].dtype)

    def perform_jax(self, *inputs):
        """Execute the JAX function directly, returning JAX arrays."""
        outputs = self.jitted_func(*inputs)
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def grad(self, inputs, output_gradients):
        """Compute gradients using JAX's vector-Jacobian product (VJP)."""
        import jax

        # Find indices of outputs that need gradients
        connected_output_indices = []
        for i, output_grad in enumerate(output_gradients):
            if not isinstance(output_grad.type, DisconnectedType):
                connected_output_indices.append(i)

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

        # Create VJP operation
        vjp_op = JAXOp(
            self.input_types
            + tuple(self.output_types[i] for i in connected_output_indices),
            [self.input_types[i] for i in range(num_inputs)],
            vjp_operation,
            name="VJP" + (self.name if self.name is not None else ""),
        )

        gradient_outputs = vjp_op(
            *[*inputs, *[output_gradients[i] for i in connected_output_indices]]
        )
        if not isinstance(gradient_outputs, Sequence):
            gradient_outputs = [gradient_outputs]
        return gradient_outputs


def as_jax_op(jax_function=None, *, allow_eval=True):
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
    >>> @as_jax_op
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
    >>> @as_jax_op
    ... def complex_function(x, y, scale=1.0):
    ...     return {
    ...         "sum": jnp.add(x, y) * scale,
    ...     }
    >>> x = pt.vector("x")
    >>> y = pt.vector("y")
    >>> result = complex_function(x, y, scale=2.0)
    >>> f = pytensor.function([x, y], [result["sum"]])

    Or Equinox modules:

    >>> x = pt.tensor("x", shape=(3,))
    >>> y = pt.tensor("y", shape=(3,))
    >>> import equinox as eqx
    >>> mlp = eqx.nn.MLP(3, 3, 3, depth=2, activation=jnp.tanh, key=jax.random.key(0))
    >>> mlp = eqx.tree_at(lambda m: m.layers[0].bias, mlp, y)
    >>> @as_jax_op
    ... def neural_network(x, mlp):
    ...     return mlp(x)
    >>> out = neural_network(x, mlp)

    Notes
    -----
    The function is based on a blog post by Ricardo Vieira and Adrian Seyboldt,
    available at
    `pymc-labs.io <https://www.pymc-labs.io/blog-posts/jax-functions-in-pymc-3-quick
    -examples/>`__.
    To accept functions and non-PyTensor variables as input, the function uses
    :func:`equinox.partition` and :func:`equinox.combine` to split and combine the
    variables. Shapes are inferred using
    :func:`pytensor.compile.builders.infer_shape` and :func:`jax.eval_shape`.

    """

    def decorator(func):
        name = func.__name__

        try:
            import equinox as eqx
            import jax
        except ImportError as e:
            raise ImportError(
                "The as_jax_op decorator requires both jax and equinox to be installed."
            ) from e

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Partition inputs into dynamic PyTensor variables and static variables.
            # Static variables don't participate in the computational graph.
            pytensor_variables, static_values = eqx.partition(
                (args, kwargs), lambda x: isinstance(x, pt.Variable)
            )

            # Flatten the PyTensor variables for processing
            variables_flat, variables_treedef = jax.tree.flatten(pytensor_variables)
            input_types = [var.type for var in variables_flat]

            # Determine output types by analyzing the function structure
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
                reconstructed_args, reconstructed_kwargs = eqx.combine(
                    variables, static_values
                )
                function_outputs = func(*reconstructed_args, **reconstructed_kwargs)
                array_outputs, _ = eqx.partition(function_outputs, eqx.is_array)
                flattened_outputs, _ = jax.tree.flatten(array_outputs)
                return flattened_outputs

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
            final_outputs = eqx.combine(output_variables, output_static)

            return final_outputs

        return wrapper

    if jax_function is None:
        return decorator
    else:
        return decorator(jax_function)


def _find_output_types(
    jax_function, inputs_flat, input_treedef, static_input, *, allow_eval=True
):
    """Determine output types by analyzing the JAX function structure."""
    import equinox as eqx
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
        _, inferred_shape = pt.basic.infer_static_shape(variable.shape)
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
                resolved_input_shapes,
                on_unused_input="ignore",
                mode="FAST_COMPILE",
            )
        except Exception as e:
            raise ValueError(
                "Could not compile a function to infer example shapes. "
                "Please provide inputs with fully determined shapes by "
                "calling pt.specify_shape."
            ) from e
        resolved_input_shapes = shape_evaluation_function()

    # Determine output types using jax.eval_shape with dummy inputs
    output_metadata_storage = {}

    dummy_input_arrays = [
        jnp.ones(shape, dtype=variable.type.dtype)
        for variable, shape in zip(inputs_flat, resolved_input_shapes, strict=True)
    ]

    def wrapped_jax_function(input_arrays):
        """Wrapper to extract output metadata during shape evaluation."""
        variables = jax.tree.unflatten(input_treedef, input_arrays)
        reconstructed_args, reconstructed_kwargs = eqx.combine(variables, static_input)
        function_outputs = jax_function(*reconstructed_args, **reconstructed_kwargs)
        array_outputs, static_outputs = eqx.partition(function_outputs, eqx.is_array)

        # Store metadata for later use
        output_metadata_storage["output_static"] = static_outputs
        flattened_outputs, output_structure = jax.tree.flatten(array_outputs)
        output_metadata_storage["output_treedef"] = output_structure
        return flattened_outputs

    output_shapes_flat = jax.eval_shape(wrapped_jax_function, dummy_input_arrays)
    output_treedef = output_metadata_storage["output_treedef"]
    output_static = output_metadata_storage["output_static"]

    output_types = [
        pt.TensorType(dtype=output_shape.dtype, shape=output_shape.shape)
        for output_shape in output_shapes_flat
    ]

    return output_types, output_treedef, output_static
