import inspect
import sys
import time
import warnings
from dataclasses import dataclass
from io import StringIO
from typing import Any

import numpy as np

import pytensor
from pytensor.configdefaults import config
from pytensor.graph.basic import Variable
from pytensor.graph.traversal import toposort
from pytensor.graph.utils import InconsistencyError


@dataclass(slots=True)
class HistoryEntry:
    """A recorded edit to a `FunctionGraph` input.

    Used by `History` and `FullHistory` to remember a ``change_node_input``
    that can later be replayed (forward) or undone (backward).

    Not ``frozen=True`` despite being a logical record: the frozen-dataclass
    ``__init__`` routes through ``object.__setattr__`` to bypass the immutability
    check, which is ~4x slower to construct. We allocate one of these per
    ``change_node_input``, so we rely on convention for "don't mutate".
    """

    node: Any
    i: int
    var: Any
    reason: Any

    def __reduce__(self):
        # `reason` is typically a rewriter instance passed by
        # `replace_all_validate(..., reason=node_rewriter)`. Decorated
        # rewriters (`@graph_rewriter` / `@node_rewriter`) aren't picklable
        # — the decorator rebinds the module attribute to the wrapper, so
        # pickle can't resolve the inner function back to itself by qualname.
        # Since `reason` is only used for display in verbose revert, drop
        # the live object and keep a string at pickle time.
        reason = self.reason
        if reason is not None and not isinstance(reason, str):
            name = getattr(reason, "__name__", None)
            reason = name if isinstance(name, str) else str(reason)
        return type(self), (self.node, self.i, self.var, reason)


class AlreadyThere(Exception):
    """
    Raised by a Feature's on_attach callback method if the FunctionGraph
    attempting to attach the feature already has a functionally identical
    feature.

    """


class ReplacementDidNotRemoveError(Exception):
    """
    This exception should be thrown by replace_all_validate_remove
    when an optimization wanted to remove a Variable or a Node from
    the graph, but the replacement it gave didn't do that.

    """


class BadOptimization(Exception):
    """
    Exception: some variable and its substitute take different runtime values.

    Note: If there is only 1 parameter and it is a string, we will use
    it as the error message. This is needed when we catch, extend and
    reraise an error.

    """

    new_r = None
    """
    A `Variable` instance that took a different value from `old_r`,
    but which replaced `old_r`.

    """

    old_r = None
    """
    A `Variable` instance that was replaced by `new_r`.

    """

    old_r_val = None
    """
    The value computed for `old_r`.

    """

    new_r_val = None
    """
    The value computed for `new_r`.

    """

    reason = None
    """
    An object that indicates why old_r was turned into new_r.

    Convention is that this is the name of the optimization that
    requested the replacement.

    """

    old_graph = ""
    """
    A multiline string representation of the graph leading to
    old_r, at the time of the replacement.

    """

    new_graph = ""
    """
    A multiline string representation of the graph leading to
    new_r, at the time of the replacement.

    """

    def __init__(
        self,
        old_r,
        new_r=None,
        old_r_val=None,
        new_r_val=None,
        reason=None,
        old_graph=None,
        new_graph=None,
    ):
        super().__init__()

        self.old_r = old_r
        self.new_r = new_r
        self.old_r_val = old_r_val
        self.new_r_val = new_r_val
        self.reason = reason

        done = dict()
        used_ids = dict()

        if isinstance(old_r, Variable):
            self.old_graph = pytensor.printing._debugprint(
                old_r,
                prefix="  ",
                depth=6,
                file=StringIO(),
                done=done,
                print_type=True,
                used_ids=used_ids,
            ).getvalue()
        else:
            self.old_graph = None

        if isinstance(new_r, Variable):
            self.new_graph = pytensor.printing._debugprint(
                new_r,
                prefix="  ",
                depth=6,
                file=StringIO(),
                done=done,
                print_type=True,
                used_ids=used_ids,
            ).getvalue()
        else:
            self.new_graph = None

        # To allow extending the error message of an existing error.
        self.full_err = None
        if isinstance(old_r, str):
            assert (
                new_r is None
                and old_r_val is None
                and new_r_val is None
                and reason is None
                and old_graph is None
                and new_graph is None
            )
            self.full_err = old_r

    def __str__(self):
        return self.str_diagnostic()

    def str_diagnostic(self):
        """
        Return a pretty multiline string representing the cause of the exception.
        """
        # We have a pre-made message
        if getattr(self, "full_err", None) is not None:
            return self.full_err
        sio = StringIO()
        val_str_len_limit = 800
        print("BadOptimization Error", super().__str__(), file=sio)
        print("  Variable: id", id(self.new_r), self.new_r, file=sio)
        print("  Op", self.new_r.owner, file=sio)
        print("  Value Type:", type(self.new_r_val), file=sio)
        try:
            ssio = StringIO()
            print("  Old Value shape, dtype, strides:", end=" ", file=ssio)
            print(self.old_r_val.shape, end=" ", file=ssio)
            print(self.old_r_val.dtype, end=" ", file=ssio)
            print(self.old_r_val.strides, file=ssio)
            # only if all succeeds to we add anything to sio
            print(ssio.getvalue(), file=sio)
        except Exception:
            pass

        str_old_r_val = str(self.old_r_val)
        if len(str_old_r_val) > val_str_len_limit:
            print(
                "  Old Value: ",
                str(self.old_r_val)[:val_str_len_limit],
                "...",
                file=sio,
            )
        else:
            print("  Old Value: ", str(self.old_r_val), file=sio)

        try:
            ssio = StringIO()
            print("  New Value shape, dtype, strides:", end=" ", file=ssio)
            print(self.new_r_val.shape, end=" ", file=ssio)
            print(self.new_r_val.dtype, end=" ", file=ssio)
            print(self.new_r_val.strides, file=ssio)
            # only if all succeeds to we add anything to sio
            print(ssio.getvalue(), file=sio)
        except Exception:
            pass
        str_new_r_val = str(self.new_r_val)
        if len(str_new_r_val) > val_str_len_limit:
            print(
                "  New Value: ",
                str(self.new_r_val)[:val_str_len_limit],
                "...",
                file=sio,
            )
        else:
            print("  New Value: ", str(self.new_r_val), file=sio)

        try:
            ov = np.asarray(self.old_r_val)
            nv = np.asarray(self.new_r_val)
            ssio = StringIO()
            abs_diff = np.absolute(nv - ov)
            print("  Max Abs Diff: ", np.max(abs_diff), file=ssio)
            print("  Mean Abs Diff: ", np.mean(abs_diff), file=ssio)
            print("  Median Abs Diff: ", np.median(abs_diff), file=ssio)
            print("  Std Abs Diff: ", np.std(abs_diff), file=ssio)
            arg_max_val = np.argmax(abs_diff)
            values_at_max = (nv.flatten()[arg_max_val], ov.flatten()[arg_max_val])
            print("  Value at Max Diff: ", values_at_max, file=ssio)

            # N.B. the maximum(..., 1e-8) protects against div by 0 when
            #      nv == ov == 0
            reldiff = abs_diff / np.maximum(np.absolute(nv) + np.absolute(ov), 1e-8)
            print("  Max Rel Diff: ", np.max(reldiff), file=ssio)
            print("  Mean Rel Diff: ", np.mean(reldiff), file=ssio)
            print("  Median Rel Diff: ", np.median(reldiff), file=ssio)
            print("  Std Rel Diff: ", np.std(reldiff), file=ssio)
            arg_max_val = np.argmax(reldiff)
            values_at_max = (nv.flatten()[arg_max_val], ov.flatten()[arg_max_val])
            print("  Value at Max Diff: ", values_at_max, file=ssio)
            # only if all succeeds to we add anything to sio
            print(ssio.getvalue(), file=sio)
        except Exception:
            pass

        print("  Reason: ", str(self.reason), file=sio)
        print("  Old Graph:", file=sio)
        print(self.old_graph, file=sio)
        print("  New Graph:", file=sio)
        print(self.new_graph, file=sio)
        print("", file=sio)
        print("Hint: relax the tolerance by setting tensor__cmp_sloppy=1", file=sio)
        print("  or even tensor__cmp_sloppy=2 for less-strict comparison", file=sio)
        return sio.getvalue()


def register_feature_callback(method):
    """Mark a Feature method as dispatched by ``execute_callbacks``.

    The decorated method's name is collected into the owning class's
    ``_feature_callbacks`` set at class-definition time. Subclasses inherit
    the registration via the MRO walk in ``Feature.__init_subclass__`` —
    they can override the method without re-decorating; the override will
    still be invoked by ``execute_callbacks``.
    """
    method._is_feature_callback = True
    return method


class Feature:
    """
    Base class for FunctionGraph extensions.

    A Feature has two ways to integrate with a ``FunctionGraph``:

    1. **Callbacks.** Methods decorated with ``@register_feature_callback``
       are invoked by ``FunctionGraph.execute_callbacks`` (or
       ``collect_callbacks``) at well-defined points in the graph's lifecycle
       — attach/detach, import/prune, input change, validation, and toposort
       ordering queries. The registered name is what ``execute_callbacks``
       dispatches by.

    2. **Provided methods.** Names listed in ``provides`` become callable
       as ``fgraph.<name>(...)``, dispatched through
       ``FunctionGraph.__getattr__`` to ``feature.<name>(fgraph, ...)``.

    A name cannot appear in both ``provides`` and the callback registry —
    ``__init_subclass__`` enforces this at import time.

    See Also
    --------
    pytensor.graph.features : for common extensions.

    """

    provides: tuple[str, ...] = ()
    _feature_callbacks: frozenset[str] = frozenset()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        callbacks: set[str] = set()
        for base in cls.__mro__:
            for name, val in vars(base).items():
                if getattr(val, "_is_feature_callback", False):
                    callbacks.add(name)
        cls._feature_callbacks = frozenset(callbacks)
        clash = set(cls.provides) & callbacks
        if clash:
            raise TypeError(
                f"{cls.__name__}: names {sorted(clash)} appear in both "
                "`provides` and as callbacks; pick one role per name"
            )

    @register_feature_callback
    def on_attach(self, fgraph):
        """
        Called by `FunctionGraph.attach_feature`, the method that attaches the
        feature to the `FunctionGraph`. Since this is called after the
        `FunctionGraph` is initially populated, this is where you should run
        checks on the initial contents of the `FunctionGraph`.

        The on_attach method may raise the `AlreadyThere` exception to cancel
        the attach operation if it detects that another Feature instance
        implementing the same functionality is already attached to the
        `FunctionGraph`.

        """

    @register_feature_callback
    def on_detach(self, fgraph):
        """
        Called by `FunctionGraph.remove_feature`.  Should remove any
        dynamically-added functionality that it installed into the fgraph.

        """

    @register_feature_callback
    def on_import(self, fgraph, node, reason):
        """
        Called whenever a node is imported into `fgraph`, which is just before
        the node is actually connected to the graph.

        Note: this is not called when the graph is created. If you want to
        detect the first nodes to be implemented to the graph, you should do
        this by implementing `on_attach`.

        """

    @register_feature_callback
    def on_change_input(self, fgraph, node, i, var, new_var, reason=None):
        """
        Called whenever ``node.inputs[i]`` is changed from `var` to `new_var`.
        At the moment the callback is done, the change has already taken place.

        If you raise an exception in this function, the state of the graph
        might be broken for all intents and purposes.

        """

    @register_feature_callback
    def on_prune(self, fgraph, node, reason):
        """
        Called whenever a node is pruned (removed) from the `fgraph`, after it
        is disconnected from the graph.

        """

    @register_feature_callback
    def on_validate(self, fgraph):
        """
        Called by ``Validator.validate`` to give each Feature a chance to
        veto the current graph state. Implementations should raise
        ``InconsistencyError`` if the graph is invalid.

        """

    @register_feature_callback
    def orderings(self, fgraph):
        """
        Called by `FunctionGraph.toposort`. It should return a dictionary of
        ``{node: predecessors}`` where ``predecessors`` is a list of
        nodes that should be computed before the key node.

        If you raise an exception in this function, the state of the graph
        might be broken for all intents and purposes.

        """
        return {}

    def clone(self):
        """Create a clone that can be attached to a new `FunctionGraph`.

        This default implementation returns `self`, which carries the
        assumption that the `Feature` is essentially stateless.  If a subclass
        has state of its own that is in any way relative to a given
        `FunctionGraph`, this method should be overridden with an
        implementation that actually creates a fresh copy.
        """
        return self


class Bookkeeper(Feature):
    def on_attach(self, fgraph):
        for node in toposort(fgraph.outputs):
            self.on_import(fgraph, node, "on_attach")

    def on_detach(self, fgraph):
        for node in toposort(fgraph.outputs):
            self.on_prune(fgraph, node, "Bookkeeper.detach")


class History(Feature):
    """Keep an history of changes to an FunctionGraph.

    This history can be reverted up to the last checkpoint.. We can
    revert to only 1 point in the past. This limit was added to lower
    the memory usage.

    """

    provides: tuple[str, ...] = ("checkpoint", "revert")

    def __init__(self):
        self.history: dict = {}
        self._checkpoint_counters: dict = {}

    def on_attach(self, fgraph):
        if "checkpoint" in fgraph._feature_methods:
            raise AlreadyThere(
                "History feature is already present or in conflict with another plugin."
            )
        self.history[fgraph] = []
        self._checkpoint_counters[fgraph] = 0

    def clone(self):
        return type(self)()

    def on_detach(self, fgraph):
        del self.history[fgraph]
        del self._checkpoint_counters[fgraph]

    def on_change_input(self, fgraph, node, i, r, new_r, reason=None):
        h = self.history[fgraph]
        if h is None:
            return
        h.append(HistoryEntry(node, i, r, reason))

    def checkpoint(self, fgraph):
        self.history[fgraph] = []
        self._checkpoint_counters[fgraph] += 1
        return self._checkpoint_counters[fgraph]

    def revert(self, fgraph, checkpoint):
        """
        Reverts the graph to whatever it was at the provided
        checkpoint (undoes all replacements).

        """
        h = self.history[fgraph]
        self.history[fgraph] = None
        # Reject stale tokens: only the most recent checkpoint can be reverted to.
        assert self._checkpoint_counters[fgraph] == checkpoint
        while h:
            entry = h.pop()
            fgraph.change_node_input(
                entry.node,
                entry.i,
                entry.var,
                reason=("Revert", entry.reason),
                check=False,
            )
        self.history[fgraph] = h


class FullHistory(Feature):
    """Keeps track of all changes in FunctionGraph and allows arbitrary back and forth through intermediate states

    .. testcode::
        import pytensor
        import pytensor.tensor as pt
        from pytensor.graph.fg import FunctionGraph
        from pytensor.graph.features import FullHistory
        from pytensor.graph.rewriting.utils import rewrite_graph

        x = pt.scalar("x")
        out = pt.log(pt.exp(x) / pt.sum(pt.exp(x)))

        fg = FunctionGraph(outputs=[out])
        history = FullHistory()
        fg.attach_feature(history)

        rewrite_graph(fg, clone=False, include=("canonicalize", "stabilize"))

        # Replay rewrites
        history.start()
        pytensor.dprint(fg)
        with pytensor.config.change_flags(optimizer_verbose = True):
            for i in range(3):
                print(">> ", end="")
                pytensor.dprint(history.next())

    .. testoutput::
        Log [id A] 4
         └─ True_div [id B] 3
            ├─ Exp [id C] 2
            │  └─ x [id D]
            └─ Sum{axes=None} [id E] 1
               └─ Exp [id F] 0
                  └─ x [id D]
        >> MergeOptimizer
        Log [id A] 3
         └─ True_div [id B] 2
            ├─ Exp [id C] 0
            │  └─ x [id D]
            └─ Sum{axes=None} [id E] 1
               └─ Exp [id C] 0
                  └─ ···
        >> local_mul_canonizer
        Log [id A] 1
         └─ Softmax{axis=None} [id B] 0
            └─ x [id C]
        >> local_logsoftmax
        LogSoftmax{axis=None} [id A] 0
         └─ x [id B]


    .. testcode::
        # Or in reverse
        with pytensor.config.change_flags(optimizer_verbose=True):
            for i in range(3):
                print(">> ", end="")
                pytensor.dprint(history.prev())

    .. testoutput::
        >> local_logsoftmax
        Log [id A] 1
         └─ Softmax{axis=None} [id B] 0
            └─ x [id C]
        >> local_mul_canonizer
        Log [id A] 3
         └─ True_div [id B] 2
            ├─ Exp [id C] 0
            │  └─ x [id D]
            └─ Sum{axes=None} [id E] 1
               └─ Exp [id C] 0
                  └─ ···
        >> MergeOptimizer
        Log [id A] 4
         └─ True_div [id B] 3
            ├─ Exp [id C] 2
            │  └─ x [id D]
            └─ Sum{axes=None} [id E] 1
               └─ Exp [id F] 0
                  └─ x [id D]


    .. testcode::
        # Or go to any step
        pytensor.dprint(history.goto(2))

    .. testoutput::
        Log [id A] 1
         └─ Softmax{axis=None} [id B] 0
            └─ x [id C]


    """

    def __init__(self, callback=None):
        self.fw = []
        self.bw = []
        self.pointer = -1
        self.fg = None
        self.callback = callback

    def on_attach(self, fgraph):
        if self.fg is not None:
            raise ValueError("Full History already attached to another fgraph")
        self.fg = fgraph

    def on_change_input(self, fgraph, node, i, r, new_r, reason=None):
        self.bw.append(HistoryEntry(node, i, r, reason))
        self.fw.append(HistoryEntry(node, i, new_r, reason))
        self.pointer += 1
        if self.callback:
            self.callback()

    def goto(self, checkpoint):
        """
        Reverts the graph to whatever it was at the provided
        checkpoint (undoes all replacements). A checkpoint at any
        given time can be obtained using self.checkpoint().

        """
        history_len = len(self.bw)
        pointer = self.pointer
        assert 0 <= checkpoint <= history_len
        verbose = config.optimizer_verbose

        # Go backwards
        while pointer > checkpoint - 1:
            entry = self.bw[pointer]
            if verbose:
                print(entry.reason)  # noqa: T201
            self.fg.change_node_input(
                entry.node,
                entry.i,
                entry.var,
                reason=("Revert", entry.reason),
                check=False,
            )
            pointer -= 1

        # Go forward
        while pointer < checkpoint - 1:
            pointer += 1
            entry = self.fw[pointer]
            if verbose:
                print(entry.reason)  # noqa: T201
            self.fg.change_node_input(
                entry.node,
                entry.i,
                entry.var,
                reason=("Revert", entry.reason),
                check=False,
            )

        # Remove history changes caused by the foward/backward!
        self.bw = self.bw[:history_len]
        self.fw = self.fw[:history_len]
        self.pointer = pointer
        return self.fg

    def start(self):
        return self.goto(0)

    def end(self):
        return self.goto(len(self.bw))

    def prev(self):
        if self.pointer < 0:
            return self.fg
        else:
            return self.goto(self.pointer)

    def next(self):
        if self.pointer >= len(self.bw) - 1:
            return self.fg
        else:
            return self.goto(self.pointer + 2)


class Validator(Feature):
    provides: tuple[str, ...] = ("validate", "consistent")

    def on_attach(self, fgraph):
        if "validate" in fgraph._feature_methods:
            raise AlreadyThere(
                "Validator feature is already present or in"
                " conflict with another plugin."
            )

    def validate(self, fgraph):
        """
        If the caller is replace_all_validate, just raise the
        exception. replace_all_validate will print out the
        verbose output. Or it has to be done here before raise.
        """
        t0 = time.perf_counter()
        try:
            ret = fgraph.execute_callbacks("on_validate")
        except Exception as e:
            cf = inspect.currentframe()
            uf = cf.f_back
            uf_info = inspect.getframeinfo(uf)

            # If the caller is replace_all_validate, just raise the
            # exception. replace_all_validate will print out the
            # verbose output.
            # Or it has to be done here before raise.
            if uf_info.function == "replace_all_validate":
                raise
            else:
                verbose = uf.f_locals.get("verbose", False)
                if verbose:
                    r = uf.f_locals.get("r", "")
                    reason = uf_info.function
                    print(f"validate failed on node {r}.\n Reason: {reason}, {e}")  # noqa: T201
                raise
        t1 = time.perf_counter()
        if fgraph.profile:
            fgraph.profile.validate_time += t1 - t0
        return ret

    def consistent(self, fgraph):
        try:
            fgraph.validate()
            return True
        except Exception:
            return False


class ReplaceValidate(History, Validator):
    provides: tuple[str, ...] = (
        *History.provides,
        *Validator.provides,
        "replace_validate",
        "replace_all_validate",
        "replace_all_validate_remove",
    )

    def __init__(self):
        super().__init__()
        self._nodes_removed: set = set()
        self.fail_validate: bool = False

    def on_attach(self, fgraph):
        if "replace_validate" in fgraph._feature_methods:
            raise AlreadyThere(
                "ReplaceValidate feature is already present"
                " or in conflict with another plugin."
            )
        self._nodes_removed = set()
        self.fail_validate = False
        History.on_attach(self, fgraph)
        Validator.on_attach(self, fgraph)

    def clone(self):
        return type(self)()

    def on_detach(self, fgraph):
        History.on_detach(self, fgraph)
        Validator.on_detach(self, fgraph)

    def replace_validate(self, fgraph, r, new_r, reason=None, **kwargs):
        self.replace_all_validate(fgraph, [(r, new_r)], reason=reason, **kwargs)

    def replace_all_validate(
        self, fgraph, replacements, reason=None, verbose=None, **kwargs
    ):
        chk = fgraph.checkpoint()

        if verbose is None:
            verbose = config.optimizer_verbose

        if verbose:
            print_reason = True
            if config.optimizer_verbose_ignore:
                print_reason = str(reason) not in config.optimizer_verbose_ignore.split(
                    ","
                )

        for r, new_r in replacements:
            try:
                fgraph.replace(r, new_r, reason=reason, verbose=False, **kwargs)
            except Exception as e:
                msg = str(e)
                s1 = "The type of the replacement must be the same"
                s2 = "does not belong to this FunctionGraph"
                s3 = "maximum recursion depth exceeded"
                if s3 in msg:
                    # There is nothing safe we can do to recover from this.
                    # So don't revert as this raise a different error
                    # that isn't helpful.
                    e.args += (
                        " As a temporary work around, you can raise Python"
                        " stack limit with:"
                        " import sys; sys.setrecursionlimit(10000)",
                    )
                    raise
                elif s1 not in msg and s2 not in msg:
                    out = sys.stderr
                    print(
                        "<<!! BUG IN FGRAPH.REPLACE OR A LISTENER !!>>",
                        type(e),
                        e,
                        reason,
                        file=out,
                    )
                # this might fail if the error is in a listener:
                # (fgraph.replace kinda needs better internal error handling)
                fgraph.revert(chk)
                raise
        try:
            fgraph.validate()
        except Exception as e:
            fgraph.revert(chk)
            if verbose:
                print(  # noqa: T201
                    f"rewriting: validate failed on node {r}.\n Reason: {reason}, {e}"
                )
            raise

        if verbose and print_reason:
            print(  # noqa: T201
                f"rewriting: rewrite {reason} replaces {r} of {r.owner} with {new_r} of {new_r.owner}"
            )

        # The return is needed by replace_all_validate_remove
        return chk

    def replace_all_validate_remove(
        self, fgraph, replacements, remove, reason=None, warn=True, **kwargs
    ):
        """
        As replace_all_validate, revert the replacement if the ops
        in the list remove are still in the graph. Also print a warning.

        """
        chk = fgraph.replace_all_validate(replacements, reason=reason, **kwargs)
        self._nodes_removed.update(remove)
        for rm in remove:
            if rm in fgraph.apply_nodes or rm in fgraph.variables:
                fgraph.revert(chk)
                if warn:
                    warnings.warn(
                        "An optimization wanted to replace a Variable"
                        " in the graph, but the replacement for it doesn't"
                        " remove it. We disabled the optimization."
                        f"{reason}: {replacements}",
                    )
                raise ReplacementDidNotRemoveError()

    def on_import(self, fgraph, node, reason):
        if node in self._nodes_removed:
            self.fail_validate = True

    def on_validate(self, fgraph):
        if self.fail_validate:
            self.fail_validate = False
            raise InconsistencyError("Trying to reintroduce a removed node")


class PreserveVariableAttributes(Feature):
    """
    This preserve some variables attributes and tag during optimization.
    """

    def on_change_input(self, fgraph, node, i, r, new_r, reason=None):
        # Don't change the name of constants
        if r.owner and r.name is not None and new_r.name is None:
            new_r.name = r.name
        if (
            getattr(r.tag, "nan_guard_mode_check", False)
            and getattr(new_r.tag, "nan_guard_mode_check", False) is False
        ):
            new_r.tag.nan_guard_mode_check = r.tag.nan_guard_mode_check


class NoOutputFromInplace(Feature):
    """Prevent `FunctionGraph` outputs within a range from being altered in-place."""

    def __init__(self, protected_out_ids):
        self.protected_out_ids = tuple(protected_out_ids)

    def on_attach(self, fgraph):
        if hasattr(fgraph, "_no_output_from_inplace"):
            raise AlreadyThere(f"InnerGraphWatcher is already attached to {fgraph}.")

        fgraph._no_output_from_inplace = self

    def clone(self):
        return type(self)(self.protected_out_ids)

    def on_validate(self, fgraph):
        if not hasattr(fgraph, "destroyers"):
            return True

        for out in tuple(fgraph.outputs[i] for i in self.protected_out_ids):
            node = out.owner

            if node is None:
                continue

            # Validate that the node that produces the output does not produce
            # it by modifying something else in-place.
            op = node.op
            out_idx = node.outputs.index(out)
            if out_idx in op.destroy_map:
                raise InconsistencyError(
                    "A function graph Feature has requested that outputs of the graph "
                    "be prevented from being the result of in-place "
                    f"operations. This has prevented the output {out} from "
                    "being computed by modifying another variable in-place."
                )

        return True
