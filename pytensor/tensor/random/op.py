import abc
import warnings
from collections.abc import Sequence
from typing import Any, cast

import numpy as np

import pytensor
from pytensor.configdefaults import config
from pytensor.graph.basic import Apply, Variable, equal_computations
from pytensor.graph.op import Op
from pytensor.graph.replace import _vectorize_node
from pytensor.scalar import ScalarVariable
from pytensor.tensor.basic import (
    as_tensor_variable,
    concatenate,
    constant,
    get_vector_length,
    infer_static_shape,
)
from pytensor.tensor.blockwise import OpWithCoreShape
from pytensor.tensor.random.type import RandomGeneratorType, RandomType
from pytensor.tensor.random.utils import (
    compute_batch_shape,
    custom_rng_deepcopy,
    explicit_expand_dims,
    normalize_size_param,
)
from pytensor.tensor.shape import shape_tuple
from pytensor.tensor.type import TensorType
from pytensor.tensor.type_other import NoneConst, NoneTypeT
from pytensor.tensor.utils import _parse_gufunc_signature, safe_signature
from pytensor.tensor.variable import TensorVariable


class RNGConsumerOp(Op):
    """Baseclass for Ops that consume RNGs."""

    @abc.abstractmethod
    def update(self, node: Apply) -> dict[Variable, Variable]:
        """Symbolic update expression for input RNG variables.

        Returns a dictionary with the symbolic expressions required for correct updating
        of RNG variables in repeated function evaluations.
        """
        pass


class RandomVariable(RNGConsumerOp):
    """An `Op` that produces a sample from a random variable.

    This is essentially `RandomFunction`, except that it removes the
    `outtype` dependency and handles shape dimension information more
    directly.

    """

    _output_type_depends_on_input_value = True

    __props__ = ("name", "signature", "dtype", "inplace")
    default_output = 1

    def __init__(
        self,
        name=None,
        ndim_supp=None,
        ndims_params=None,
        dtype: str | None = None,
        inplace=None,
        signature: str | None = None,
    ):
        """Create a random variable `Op`.

        Parameters
        ----------
        name: str
            The `Op`'s display name.
        signature: str
            Numpy-like vectorized signature of the random variable.
        dtype: str (optional)
            The default dtype of the sampled output.  If the value ``"floatX"`` is
            given, then ``dtype`` is set to ``pytensor.config.floatX``.  If
            ``None`` (the default), the `dtype` keyword must be set when
            `RandomVariable.make_node` is called.
        inplace: boolean (optional)
            Determine whether the underlying rng state is mutated or copied.

        """
        super().__init__()

        self.name = name or getattr(self, "name")

        ndim_supp = (
            ndim_supp if ndim_supp is not None else getattr(self, "ndim_supp", None)
        )
        if ndim_supp is not None:
            warnings.warn(
                "ndim_supp is deprecated. Provide signature instead.", FutureWarning
            )
            self.ndim_supp = ndim_supp
        ndims_params = (
            ndims_params
            if ndims_params is not None
            else getattr(self, "ndims_params", None)
        )
        if ndims_params is not None:
            warnings.warn(
                "ndims_params is deprecated. Provide signature instead.", FutureWarning
            )
            if not isinstance(ndims_params, Sequence):
                raise TypeError("Parameter ndims_params must be sequence type.")
            self.ndims_params = tuple(ndims_params)

        self.signature = signature or getattr(self, "signature", None)
        if self.signature is not None:
            # Assume a single output. Several methods need to be updated to handle multiple outputs.
            self.inputs_sig, [self.output_sig] = _parse_gufunc_signature(self.signature)
            self.ndims_params = [len(input_sig) for input_sig in self.inputs_sig]
            self.ndim_supp = len(self.output_sig)
        else:
            if (
                getattr(self, "ndim_supp", None) is None
                or getattr(self, "ndims_params", None) is None
            ):
                raise ValueError("signature must be provided")
            else:
                self.signature = safe_signature(self.ndims_params, [self.ndim_supp])

        if isinstance(dtype, np.dtype):
            dtype = dtype.name
        self.dtype = dtype or getattr(self, "dtype", None)

        self.inplace = (
            inplace if inplace is not None else getattr(self, "inplace", False)
        )

        if self.inplace:
            self.destroy_map = {0: [0]}

    def update(self, node: Apply) -> dict[Variable, Variable]:
        return {node.inputs[0]: node.outputs[0]}

    def _supp_shape_from_params(self, dist_params, param_shapes=None):
        """Determine the support shape of a multivariate `RandomVariable`'s output given its parameters.

        This does *not* consider the extra dimensions added by the `size` parameter
        or independent (batched) parameters.

        When provided, `param_shapes` should be given preference over `[d.shape for d in dist_params]`,
        as it will avoid redundancies in PyTensor shape inference.

        Examples
        --------
        Common multivariate `RandomVariable`s derive their support shapes implicitly from the
        last dimension of some of their parameters. For example `multivariate_normal` support shape
        corresponds to the last dimension of the mean or covariance parameters, `support_shape=(mu.shape[-1])`.
        For this case the helper `pytensor.tensor.random.utils.supp_shape_from_ref_param_shape` can be used.

        Other variables have fixed support shape such as `support_shape=(2,)` or it is determined by the
        values (not shapes) of some parameters. For instance, a `gaussian_random_walk(steps, size=(2,))`,
        might have `support_shape=(steps,)`.
        """
        if self.signature is not None:
            # Signature could indicate fixed numerical shapes
            # As per https://numpy.org/neps/nep-0020-gufunc-signature-enhancement.html
            output_sig = self.output_sig
            core_out_shape = {
                dim: int(dim) if str.isnumeric(dim) else None for dim in self.output_sig
            }

            # Try to infer missing support dims from signature of params
            for param, param_sig, ndim_params in zip(
                dist_params, self.inputs_sig, self.ndims_params, strict=True
            ):
                if ndim_params == 0:
                    continue
                for param_dim, dim in zip(
                    param.shape[-ndim_params:], param_sig, strict=True
                ):
                    if dim in core_out_shape and core_out_shape[dim] is None:
                        core_out_shape[dim] = param_dim

            if all(dim is not None for dim in core_out_shape.values()):
                # We have all we need
                return [core_out_shape[dim] for dim in output_sig]

        raise NotImplementedError(
            "`_supp_shape_from_params` must be implemented for multivariate RVs "
            "when signature is not sufficient to infer the support shape"
        )

    def rng_fn(self, rng, *args, **kwargs) -> int | float | np.ndarray:
        """Sample a numeric random variate."""
        return getattr(rng, self.name)(*args, **kwargs)

    def __str__(self):
        # Only show signature from core props
        if signature := self.signature:
            # inp, out = signature.split("->")
            # extended_signature = f"[rng],[size],{inp}->[rng],{out}"
            # core_props = [extended_signature]
            core_props = [f'"{signature}"']
        else:
            # Far back compat
            core_props = [str(self.ndim_supp), str(self.ndims_params)]

        # Add any extra props that the subclass may have
        extra_props = [
            str(getattr(self, prop))
            for prop in self.__props__
            if prop not in RandomVariable.__props__
        ]

        props_str = ", ".join(core_props + extra_props)
        return f"{self.name}_rv{{{props_str}}}"

    def _infer_shape(
        self,
        size: TensorVariable | Variable,
        dist_params: Sequence[TensorVariable],
        param_shapes: Sequence[tuple[Variable, ...]] | None = None,
    ) -> tuple[ScalarVariable | TensorVariable, ...]:
        """Compute the output shape given the size and distribution parameters.

        Parameters
        ----------
        size
            The size parameter specified for this `RandomVariable`.
        dist_params
            The symbolic parameter for this `RandomVariable`'s distribution.
        param_shapes
            The shapes of the `dist_params` as given by `ShapeFeature`'s
            via `Op.infer_shape`'s `input_shapes` argument.  This parameter's
            values are essentially more accurate versions of ``[d.shape for d
            in dist_params]``.

        """

        from pytensor.tensor.extra_ops import broadcast_shape_iter

        supp_shape: tuple[Any]
        if self.ndim_supp == 0:
            supp_shape = ()
        else:
            supp_shape = tuple(
                self._supp_shape_from_params(dist_params, param_shapes=param_shapes)
            )

        if not isinstance(size.type, NoneTypeT):
            size_len = get_vector_length(size)

            # Fail early when size is incompatible with parameters
            for i, (param, param_ndim_supp) in enumerate(
                zip(dist_params, self.ndims_params, strict=True)
            ):
                param_batched_dims = getattr(param, "ndim", 0) - param_ndim_supp
                if param_batched_dims > size_len:
                    raise ValueError(
                        f"Size length is incompatible with batched dimensions of parameter {i} {param}:\n"
                        f"len(size) = {size_len}, len(batched dims {param}) = {param_batched_dims}. "
                        f"Size must be None or have length >= {param_batched_dims}"
                    )

            return tuple(size) + supp_shape

        # Size was not provided, we must infer it from the shape of the parameters
        if param_shapes is None:
            param_shapes = [shape_tuple(p) for p in dist_params]

        def extract_batch_shape(p, ps, n):
            shape = tuple(ps)

            if n == 0:
                return shape

            batch_shape = tuple(
                s if not b else constant(1, "int64")
                for s, b in zip(shape[:-n], p.type.broadcastable[:-n], strict=True)
            )
            return batch_shape

        # These are versions of our actual parameters with the anticipated
        # dimensions (i.e. support dimensions) removed so that only the
        # independent variate dimensions are left.
        params_batch_shape = tuple(
            extract_batch_shape(p, ps, n)
            for p, ps, n in zip(
                dist_params, param_shapes, self.ndims_params, strict=False
            )
        )

        if len(params_batch_shape) == 1:
            [batch_shape] = params_batch_shape
        elif len(params_batch_shape) > 1:
            # If there are multiple parameters, the dimensions of their
            # independent variates should broadcast together.
            batch_shape = broadcast_shape_iter(
                params_batch_shape,
                arrays_are_shapes=True,
            )
        else:
            # Distribution has no parameters
            batch_shape = ()

        shape = batch_shape + supp_shape

        return shape

    def infer_shape(self, fgraph, node, input_shapes):
        _, size, *dist_params = node.inputs
        _, _, *param_shapes = input_shapes

        shape = self._infer_shape(size, dist_params, param_shapes=param_shapes)

        return [None, list(shape)]

    def __call__(self, *args, size=None, name=None, rng=None, dtype=None, **kwargs):
        if dtype is None:
            dtype = self.dtype
        if dtype == "floatX":
            dtype = config.floatX

        # We need to recreate the Op with the right dtype
        if dtype != self.dtype:
            # Check we are not switching from float to int
            if self.dtype is not None:
                if dtype.startswith("float") != self.dtype.startswith("float"):
                    raise ValueError(
                        f"Cannot change the dtype of a {self.name} RV from {self.dtype} to {dtype}"
                    )
            props = self._props_dict()
            props["dtype"] = dtype
            new_op = type(self)(**props)
            return new_op.__call__(
                *args, size=size, name=name, rng=rng, dtype=dtype, **kwargs
            )

        res = super().__call__(rng, size, *args, **kwargs)

        if name is not None:
            res.name = name

        return res

    def make_node(self, rng, size, *dist_params):
        """Create a random variable node.

        Parameters
        ----------
        rng: RandomGeneratorType
            Existing PyTensor `Generator` object to be used.  Creates a new one, if `None`.
        size: int or Sequence
            NumPy-like size parameter.
        dtype: str
            The dtype of the sampled output.  If the value ``"floatX"`` is
            given, then `dtype` is set to ``pytensor.config.floatX``.  This value is
            only used when ``self.dtype`` isn't set.
        dist_params: list
            Distribution parameters.

        Results
        -------
        out: Apply
            A node with inputs ``(rng, size, dtype) + dist_args`` and outputs
            ``(rng_var, out_var)``.

        """
        size = normalize_size_param(size)

        dist_params = tuple(
            as_tensor_variable(p) if not isinstance(p, Variable) else p
            for p in dist_params
        )

        if rng is None:
            rng = pytensor.shared(np.random.default_rng())
        elif not isinstance(rng.type, RandomType):
            raise TypeError(
                "The type of rng should be an instance of RandomGeneratorType "
            )

        inferred_shape = self._infer_shape(size, dist_params)
        _, static_shape = infer_static_shape(inferred_shape)

        dist_params = explicit_expand_dims(
            dist_params,
            self.ndims_params,
            size_length=None
            if isinstance(size.type, NoneTypeT)
            else get_vector_length(size),
        )

        inputs = (rng, size, *dist_params)
        out_type = TensorType(dtype=self.dtype, shape=static_shape)
        outputs = (rng.type(), out_type())

        if self.dtype == "floatX":
            # Commit to a specific float type if the Op is still using "floatX"
            dtype = config.floatX
            props = self._props_dict()
            props["dtype"] = dtype
            self = type(self)(**props)

        return Apply(self, inputs, outputs)

    def batch_ndim(self, node: Apply) -> int:
        return cast(int, node.default_output().type.ndim - self.ndim_supp)

    def rng_param(self, node) -> Variable:
        """Return the node input corresponding to the rng"""
        return node.inputs[0]

    def size_param(self, node) -> Variable:
        """Return the node input corresponding to the size"""
        return node.inputs[1]

    def dist_params(self, node) -> Sequence[Variable]:
        """Return the node inpust corresponding to dist params"""
        return node.inputs[2:]

    def perform(self, node, inputs, outputs):
        rng, size, *args = inputs

        # Draw from `rng` if `self.inplace` is `True`, and from a copy of `rng` otherwise.
        if not self.inplace:
            rng = custom_rng_deepcopy(rng)

        outputs[0][0] = rng
        outputs[1][0] = np.asarray(
            self.rng_fn(rng, *args, None if size is None else tuple(size)),
            dtype=self.dtype,
        )

    def grad(self, inputs, outputs):
        return [
            pytensor.gradient.grad_undefined(
                self, k, inp, "No gradient defined for random variables"
            )
            for k, inp in enumerate(inputs)
        ]

    def R_op(self, inputs, eval_points):
        return [None for i in eval_points]


class AbstractRNGConstructor(Op):
    def make_node(self, seed=None):
        if seed is None:
            seed = NoneConst
        elif isinstance(seed, Variable) and isinstance(seed.type, NoneTypeT):
            pass
        else:
            seed = as_tensor_variable(seed)
        inputs = [seed]
        outputs = [self.random_type()]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, output_storage):
        (seed,) = inputs
        if seed is not None and seed.size == 1:
            seed = int(seed)
        output_storage[0][0] = getattr(np.random, self.random_constructor)(seed=seed)


class DefaultGeneratorMakerOp(AbstractRNGConstructor):
    random_type = RandomGeneratorType()
    random_constructor = "default_rng"


default_rng = DefaultGeneratorMakerOp()


@_vectorize_node.register(RandomVariable)
def vectorize_random_variable(
    op: RandomVariable, node: Apply, rng, size, *dist_params
) -> Apply:
    # If size was provided originally and a new size hasn't been provided,
    # We extend it to accommodate the new input batch dimensions.
    # Otherwise, we assume the new size already has the right values

    original_dist_params = op.dist_params(node)
    old_size = op.size_param(node)

    if not isinstance(old_size.type, NoneTypeT) and equal_computations(
        [old_size], [size]
    ):
        # If the original RV had a size variable and a new one has not been provided,
        # we need to define a new size as the concatenation of the original size dimensions
        # and the novel ones implied by new broadcasted batched parameters dimensions.
        new_ndim = dist_params[0].type.ndim - original_dist_params[0].type.ndim
        if new_ndim >= 0:
            new_size = compute_batch_shape(dist_params, ndims_params=op.ndims_params)
            new_size_dims = new_size[:new_ndim]
            size = concatenate([new_size_dims, size])

    return op.make_node(rng, size, *dist_params)


class RandomVariableWithCoreShape(OpWithCoreShape):
    """Generalizes a random variable `Op` to include a core shape parameter."""

    @property
    def core_op(self):
        [rv_node] = self.fgraph.apply_nodes
        return rv_node.op

    def __str__(self):
        [rv_node] = self.fgraph.apply_nodes
        return f"[{rv_node.op!s}]"
