"""Provide a simple user friendly API to PyTensor-managed memory."""

import copy
import warnings
from contextlib import contextmanager
from functools import singledispatch
from typing import TYPE_CHECKING

from pytensor.graph.basic import Variable
from pytensor.graph.utils import add_tag_trace
from pytensor.link.backend_conversion import HOST, backend_conversion
from pytensor.link.basic import Container
from pytensor.link.c.type import generic
from pytensor.utils import add_lazy_dispatcher


if TYPE_CHECKING:
    from pytensor.graph.type import Type


__SHARED_CONTEXT__: list[Variable] | None = None


@contextmanager
def collect_new_shareds():
    r"""Return all the `SharedVariable`\s created within this context manager."""
    global __SHARED_CONTEXT__
    old_context = __SHARED_CONTEXT__
    context = []
    try:
        __SHARED_CONTEXT__ = context
        yield context
    finally:
        __SHARED_CONTEXT__ = old_context


class _BackendState:
    """Tracks backend-native views of a shared value, stored on its Container.

    Lives on the ``Container`` (shared across clones and across functions), not
    the ``SharedVariable``. ``fresh`` is the set of tags whose copy currently
    holds the authoritative value (``HOST`` = the canonical container); a write
    by tag ``W`` leaves ``fresh == {W}``, and a read by tag ``R`` translates only
    when ``R not in fresh``. So a single-backend loop never converts and never
    writes state. ``eager`` tags are re-materialized on host writes rather than
    invalidated (used by ``readback=False`` functions).
    """

    __slots__ = ("eager", "fresh", "views", "warned")

    def __init__(self):
        self.views: dict[str, list] = {}
        self.fresh: set[str] = {HOST}
        self.eager: set[str] = set()
        self.warned: set[str] = set()

    def reconcile_host(self, storage: list) -> str | None:
        """Reconcile ``storage[0]`` from the authoritative backend view if stale.

        Returns the source tag when a (possibly lossy) conversion ran, else
        ``None``. Silent: callers that want to warn inspect the return value.
        """
        if HOST in self.fresh:
            return None
        source = next(iter(self.fresh))
        storage[0] = backend_conversion(source).from_native(  # type: ignore[union-attr]
            self.views[source][0]
        )
        self.fresh.add(HOST)
        return source


class SharedVariable(Variable):
    """Variable that is shared between compiled functions."""

    def __init__(
        self,
        type: "Type",
        value,
        strict: bool,
        allow_downcast=None,
        container: Container | None = None,
        name: str | None = None,
    ):
        r"""
        Parameters
        ----------
        type
            The `Type` for this variable (see `Variable`).
        value
            A value to associate with this variable (a new container will be
            created).
        strict
            ``True`` means that values assigned to this variable will not be
            cast or copied, so they must have the correct `Type`\s.
        allow_downcast
            Only applies if `strict` is ``False``.
            ``True`` means that the assigned value can lose precision when cast
            during assignment. ``None`` means that only down-casting of a Python
            float to a scalar ``floatX`` is allowed.
        container
            The container to use for this variable. Illegal to pass this as well as
            a value.
        name
            The name for this variable (see `Variable`).

        """
        super().__init__(type=type, owner=None, index=None, name=name)

        if container is not None:
            self.container = container
            if (value is not None) or (strict is not None):
                raise TypeError(
                    "value and strict are ignored if you pass a container here"
                )
        else:
            self.container = Container(
                self,
                storage=[
                    type.filter(value, strict=strict, allow_downcast=allow_downcast)
                ],
                readonly=False,
                strict=strict,
                allow_downcast=allow_downcast,
            )

        global __SHARED_CONTEXT__

        if isinstance(__SHARED_CONTEXT__, list):
            __SHARED_CONTEXT__.append(self)

        self._default_update: Variable | None = None

    def _backend_state(self) -> _BackendState:
        state = self.container._backend_state
        if state is None:
            state = self.container._backend_state = _BackendState()
        return state

    def _host_value(self):
        """Return the current host value, reconciling from a fresh backend if stale.

        Warns once per source when the fresh backend's representation is lossy,
        so the reconciled host value does not continue that backend's stream.
        """
        state = self.container._backend_state
        if state is None:
            return self.container.value
        source = state.reconcile_host(self.container.storage)
        if (
            source is not None
            and backend_conversion(source).lossy
            and source not in state.warned
        ):
            warnings.warn(
                f"{self} was last advanced by the {source!r} backend; converting "
                f"its state back is lossy, so the reconciled value does not "
                f"continue the {source!r} stream.",
                UserWarning,
                stacklevel=3,
            )
            state.warned.add(source)
        return self.container.value

    def _view_storage(self, backend: str, *, eager: bool = False) -> list:
        """Return (materializing if needed) the native storage list for ``backend``.

        Pass ``eager=True`` to have host writes re-materialize this view eagerly
        (see ``readback=False``) instead of invalidating it lazily.
        """
        state = self._backend_state()
        if eager:
            state.eager.add(backend)
        storage = state.views.get(backend)
        if storage is None:
            native = backend_conversion(backend).to_native(  # type: ignore[union-attr]
                self._host_value()
            )
            storage = state.views[backend] = [native]
            state.fresh.add(backend)
        return storage

    def _mark_written(self, backend: str) -> None:
        """Record that ``backend`` produced the current authoritative value.

        Invalidates every other copy. A no-op (no allocation, no write) when
        ``backend`` is already the sole fresh copy, so a steady loop pays nothing.
        """
        state = self.container._backend_state
        if state is None:
            return
        fresh = state.fresh
        if len(fresh) != 1 or backend not in fresh:
            state.fresh = {backend}

    def _reconcile_into(self, backend: str) -> None:
        """Bring ``backend``'s copy up to the authoritative value, if it is stale.

        A no-op (single membership check) unless another backend wrote since this
        one last synced, so a single-backend loop never converts.
        """
        state = self.container._backend_state
        if state is None or backend in state.fresh:
            return
        if backend == HOST:
            self._host_value()
            return
        native = backend_conversion(backend).to_native(  # type: ignore[union-attr]
            self._host_value()
        )
        storage = state.views.get(backend)
        if storage is None:
            state.views[backend] = [native]
        else:
            storage[0] = native
        state.fresh.add(backend)

    def get_value(self, borrow=False, return_internal_type=False):
        """
        Get the non-symbolic value associated with this SharedVariable.

        Parameters
        ----------
        borrow : bool
            True to permit returning of an object aliased to internal memory.
        return_internal_type : bool
            True to permit the returning of an arbitrary type object used
            internally to store the shared variable.

        The host value is reconciled first if a compiled function on a backend
        with its own representation (e.g. JAX) advanced it more recently.

        Only with borrow=False and return_internal_type=True does this function
        guarantee that you actually get the internal object.
        But in that case, you may get different return types when using
        different compute devices.

        """
        value = self._host_value()
        return value if borrow else copy.deepcopy(value)

    def set_value(self, new_value, borrow=False):
        """
        Set the non-symbolic value associated with this SharedVariable.

        Parameters
        ----------
        borrow : bool
            True to use the new_value directly, potentially creating problems
            related to aliased memory.

        Changes to this value will be visible to all functions using
        this SharedVariable.
        """
        if borrow:
            self.container.value = new_value
        else:
            self.container.value = copy.deepcopy(new_value)
        # Invalidate the cached backend views. Eager ones (readback=False, which
        # skip the reconcile bracket) are refreshed now; the rest lazily on use.
        state = self.container._backend_state
        if state is not None:
            host = self.container.value
            for tag in state.eager:
                state.views[tag][0] = backend_conversion(tag).to_native(host)
            state.fresh = {HOST, *state.eager}

    def clone(self, **kwargs):
        name = kwargs.get("name", self.name)
        cp = self.__class__(
            name=name,
            type=self.type,
            value=None,
            strict=None,
            container=self.container,
        )
        cp.tag = copy.copy(self.tag)
        return cp

    @property
    def default_update(self) -> Variable | None:
        """A default update expression for this `Variable`.

        If this value is non-``None``, its value will be used as the `update`
        (see `pytensor.function`) for this `Variable` when no updates are
        provided through `pytensor.function` and `no_default_updates` isn't
        enabled.
        """
        return self._default_update

    @default_update.setter
    def default_update(self, value):
        warnings.warn(
            "Setting default_update is deprecated.", DeprecationWarning, stacklevel=2
        )
        if value is not None:
            self._default_update = self.type.filter_variable(value, allow_convert=True)
        else:
            self._default_update = value


def shared(value, name=None, strict=False, allow_downcast=None, **kwargs):
    r"""Create a `SharedVariable` initialized with a copy or reference of `value`.

    This function iterates over constructor functions to find a
    suitable `SharedVariable` subclass.  The suitable one is the first
    constructor that accept the given value.  See the documentation of
    :func:`shared_constructor` for the definition of a constructor
    function.

    This function is meant as a convenient default.  If you want to use a
    specific constructor, consider calling it directly.

    `pytensor.shared` is a shortcut to this function.

    Notes
    -----
    By passing kwargs, you effectively limit the set of potential constructors
    to those that can accept those kwargs.

    Some shared variable have `borrow` as a kwarg.

    `SharedVariable`\s of `TensorType` have `broadcastable` as a kwarg. As shared
    variable shapes can change, all dimensions default to not being
    broadcastable, even if `value` has a shape of 1 along some dimension.
    This parameter allows one to create for example a row or column tensor.

    """

    if isinstance(value, Variable):
        raise TypeError("Shared variable values can not be symbolic.")

    try:
        var = shared_constructor(
            value,
            name=name,
            strict=strict,
            allow_downcast=allow_downcast,
            **kwargs,
        )
        add_tag_trace(var)
        return var
    except MemoryError as e:
        e.args = (*e.args, "Consider using `pytensor.shared(..., borrow=True)`")
        raise


@singledispatch
def shared_constructor(value, name=None, strict=False, allow_downcast=None, **kwargs):
    return SharedVariable(
        type=generic,
        value=value,
        strict=strict,
        allow_downcast=allow_downcast,
        name=name,
    )


add_lazy_dispatcher(shared_constructor)
