from copy import copy
from typing import Optional, Sequence, Tuple, Union

import numpy as np

import pytensor
from pytensor.configdefaults import config
from pytensor.graph.basic import Apply, Variable, equal_computations
from pytensor.graph.op import Op
from pytensor.graph.replace import _vectorize_node
from pytensor.misc.safe_asarray import _asarray
from pytensor.scalar import ScalarVariable
from pytensor.tensor.basic import (
    as_tensor_variable,
    concatenate,
    constant,
    get_underlying_scalar_constant_value,
    get_vector_length,
    infer_static_shape,
)
from pytensor.tensor.random.type import RandomGeneratorType, RandomStateType, RandomType
from pytensor.tensor.random.utils import (
    broadcast_params,
    normalize_size_param,
    params_broadcast_shapes,
)
from pytensor.tensor.shape import shape_tuple
from pytensor.tensor.type import TensorType, all_dtypes
from pytensor.tensor.type_other import NoneConst
from pytensor.tensor.variable import TensorVariable


class RandomVariable(Op):
    """An `Op` that produces a sample from a random variable.

    This is essentially `RandomFunction`, except that it removes the
    `outtype` dependency and handles shape dimension information more
    directly.

    """

    _output_type_depends_on_input_value = True

    __props__ = ("name", "ndim_supp", "ndims_params", "dtype", "inplace")
    default_output = 1

    def __init__(
        self,
        name=None,
        ndim_supp=None,
        ndims_params=None,
        dtype=None,
        inplace=None,
    ):
        """Create a random variable `Op`.

        Parameters
        ----------
        name: str
            The `Op`'s display name.
        ndim_supp: int
            Total number of dimensions for a single draw of the random variable
            (e.g. a multivariate normal draw is 1D, so ``ndim_supp = 1``).
        ndims_params: list of int
            Number of dimensions for each distribution parameter when the
            parameters only specify a single drawn of the random variable
            (e.g. a multivariate normal's mean is 1D and covariance is 2D, so
            ``ndims_params = [1, 2]``).
        dtype: str (optional)
            The dtype of the sampled output.  If the value ``"floatX"`` is
            given, then ``dtype`` is set to ``pytensor.config.floatX``.  If
            ``None`` (the default), the `dtype` keyword must be set when
            `RandomVariable.make_node` is called.
        inplace: boolean (optional)
            Determine whether or not the underlying rng state is updated
            in-place or not (i.e. copied).

        """
        super().__init__()

        self.name = name or getattr(self, "name")
        self.ndim_supp = (
            ndim_supp if ndim_supp is not None else getattr(self, "ndim_supp")
        )
        self.ndims_params = (
            ndims_params if ndims_params is not None else getattr(self, "ndims_params")
        )
        self.dtype = dtype or getattr(self, "dtype", None)

        self.inplace = (
            inplace if inplace is not None else getattr(self, "inplace", False)
        )

        if not isinstance(self.ndims_params, Sequence):
            raise TypeError("Parameter ndims_params must be sequence type.")

        self.ndims_params = tuple(self.ndims_params)

        if self.inplace:
            self.destroy_map = {0: [0]}

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
        raise NotImplementedError(
            "`_supp_shape_from_params` must be implemented for multivariate RVs"
        )

    def rng_fn(self, rng, *args, **kwargs) -> Union[int, float, np.ndarray]:
        """Sample a numeric random variate."""
        return getattr(rng, self.name)(*args, **kwargs)

    def __str__(self):
        props_str = ", ".join(f"{getattr(self, prop)}" for prop in self.__props__[1:])
        return f"{self.name}_rv{{{props_str}}}"

    def _infer_shape(
        self,
        size: TensorVariable,
        dist_params: Sequence[TensorVariable],
        param_shapes: Optional[Sequence[Tuple[Variable, ...]]] = None,
    ) -> Union[TensorVariable, Tuple[ScalarVariable, ...]]:
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

        size_len = get_vector_length(size)

        if size_len > 0:
            # Fail early when size is incompatible with parameters
            for i, (param, param_ndim_supp) in enumerate(
                zip(dist_params, self.ndims_params)
            ):
                param_batched_dims = getattr(param, "ndim", 0) - param_ndim_supp
                if param_batched_dims > size_len:
                    raise ValueError(
                        f"Size length is incompatible with batched dimensions of parameter {i} {param}:\n"
                        f"len(size) = {size_len}, len(batched dims {param}) = {param_batched_dims}. "
                        f"Size length must be 0 or >= {param_batched_dims}"
                    )

            if self.ndim_supp == 0:
                return size
            else:
                supp_shape = self._supp_shape_from_params(
                    dist_params, param_shapes=param_shapes
                )
                return tuple(size) + tuple(supp_shape)

        # Broadcast the parameters
        param_shapes = params_broadcast_shapes(
            param_shapes or [shape_tuple(p) for p in dist_params],
            self.ndims_params,
        )

        def extract_batch_shape(p, ps, n):
            shape = tuple(ps)

            if n == 0:
                return shape

            batch_shape = [
                s if not b else constant(1, "int64")
                for s, b in zip(shape[:-n], p.type.broadcastable[:-n])
            ]
            return batch_shape

        # These are versions of our actual parameters with the anticipated
        # dimensions (i.e. support dimensions) removed so that only the
        # independent variate dimensions are left.
        params_batch_shape = tuple(
            extract_batch_shape(p, ps, n)
            for p, ps, n in zip(dist_params, param_shapes, self.ndims_params)
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

        if self.ndim_supp == 0:
            supp_shape = ()
        else:
            supp_shape = self._supp_shape_from_params(
                dist_params,
                param_shapes=param_shapes,
            )

        shape = tuple(batch_shape) + tuple(supp_shape)
        if not shape:
            shape = constant([], dtype="int64")

        return shape

    def infer_shape(self, fgraph, node, input_shapes):
        _, size, _, *dist_params = node.inputs
        _, size_shape, _, *param_shapes = input_shapes

        try:
            size_len = get_vector_length(size)
        except ValueError:
            size_len = get_underlying_scalar_constant_value(size_shape[0])

        size = tuple(size[n] for n in range(size_len))

        shape = self._infer_shape(size, dist_params, param_shapes=param_shapes)

        return [None, list(shape)]

    def __call__(self, *args, size=None, name=None, rng=None, dtype=None, **kwargs):
        res = super().__call__(rng, size, dtype, *args, **kwargs)

        if name is not None:
            res.name = name

        return res

    def make_node(self, rng, size, dtype, *dist_params):
        """Create a random variable node.

        Parameters
        ----------
        rng: RandomGeneratorType or RandomStateType
            Existing PyTensor `Generator` or `RandomState` object to be used.  Creates a
            new one, if `None`.
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
                "The type of rng should be an instance of either RandomGeneratorType or RandomStateType"
            )

        shape = self._infer_shape(size, dist_params)
        _, static_shape = infer_static_shape(shape)
        dtype = self.dtype or dtype

        if dtype == "floatX":
            dtype = config.floatX
        elif dtype is None or (isinstance(dtype, str) and dtype not in all_dtypes):
            raise TypeError("dtype is unspecified")

        if isinstance(dtype, str):
            dtype_idx = constant(all_dtypes.index(dtype), dtype="int64")
        else:
            dtype_idx = constant(dtype, dtype="int64")
            dtype = all_dtypes[dtype_idx.data]

        outtype = TensorType(dtype=dtype, shape=static_shape)
        out_var = outtype()
        inputs = (rng, size, dtype_idx) + dist_params
        outputs = (rng.type(), out_var)

        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        rng_var_out, smpl_out = outputs

        rng, size, dtype, *args = inputs

        out_var = node.outputs[1]

        # If `size == []`, that means no size is enforced, and NumPy is trusted
        # to draw the appropriate number of samples, NumPy uses `size=None` to
        # represent that.  Otherwise, NumPy expects a tuple.
        if np.size(size) == 0:
            size = None
        else:
            size = tuple(size)

        # Draw from `rng` if `self.inplace` is `True`, and from a copy of `rng`
        # otherwise.
        if not self.inplace:
            rng = copy(rng)

        rng_var_out[0] = rng

        smpl_val = self.rng_fn(rng, *(args + [size]))

        if (
            not isinstance(smpl_val, np.ndarray)
            or str(smpl_val.dtype) != out_var.type.dtype
        ):
            smpl_val = _asarray(smpl_val, dtype=out_var.type.dtype)

        smpl_out[0] = smpl_val

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


class RandomStateConstructor(AbstractRNGConstructor):
    random_type = RandomStateType()
    random_constructor = "RandomState"


RandomState = RandomStateConstructor()


class DefaultGeneratorMakerOp(AbstractRNGConstructor):
    random_type = RandomGeneratorType()
    random_constructor = "default_rng"


default_rng = DefaultGeneratorMakerOp()


@_vectorize_node.register(RandomVariable)
def vectorize_random_variable(
    op: RandomVariable, node: Apply, rng, size, dtype, *dist_params
) -> Apply:
    # If size was provided originally and a new size hasn't been provided,
    # We extend it to accommodate the new input batch dimensions.
    # Otherwise, we assume the new size already has the right values
    old_size = node.inputs[1]
    len_old_size = get_vector_length(old_size)
    if len_old_size and equal_computations([old_size], [size]):
        bcasted_param = broadcast_params(dist_params, op.ndims_params)[0]
        new_param_ndim = (bcasted_param.type.ndim - op.ndims_params[0]) - len_old_size
        if new_param_ndim >= 0:
            new_size_dims = bcasted_param.shape[:new_param_ndim]
            size = concatenate([new_size_dims, size])

    return op.make_node(rng, size, dtype, *dist_params)
