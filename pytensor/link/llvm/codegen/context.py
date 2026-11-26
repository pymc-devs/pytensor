"""The state of generating code (`Codegen`): errors, temporary memory, loops."""

from __future__ import annotations

import math
from collections.abc import (
    Callable,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pytensor.graph.basic import Constant, Variable
from pytensor.link.llvm.analysis.layout import Layout
from pytensor.link.llvm.codegen.array import Array, advance, size
from pytensor.link.llvm.codegen.types import ir, memory_type
from pytensor.scalar.basic import ScalarType


if TYPE_CHECKING:
    from pytensor.link.llvm.codegen.call import LinkTable
    from pytensor.link.llvm.codegen.graph import GraphFunction
    from pytensor.link.llvm.codegen.kernel_module import KernelModule


#: Temporaries of a static size up to this many bytes are stack memory (`temp_memory`),
#: as long as one generated function has no more than the budget of them.
STACK_TEMPORARY_BYTES = 1024
STACK_BUDGET_BYTES = 16 * 1024


#: The loop body of `Codegen.for_each` and `loop_nest`: called with the element
#: pointers of the arrays, once per element (the loop index is in the generated code).
type ElementBody = Callable[[list[ir.Value]], None]

#: The loop index at an element: the index along each axis that is looped over.
type LoopIndex = Mapping[int, ir.Value]

#: What the `at` function of an `InnerLoop` returns for an iteration of the innermost
#: loop: the element pointers of the arrays, and the loop index.
type Element = tuple[list[ir.Value], LoopIndex]

#: A caller-generated innermost loop of `Codegen.for_each`, called as
#: `inner(axes, n, at)`. See `for_each`.
type InnerLoop = Callable[
    [frozenset[int], ir.Value, Callable[[ir.Value], Element]], None
]

#: What is known about the values of an index: (minimum, maximum), None where unknown.
type IndexBounds = tuple[int | None, int | None]


class _ScratchScope(list[tuple[ir.Value, ir.Value, bool]]):
    """The memory to give back when a `scratch_scope` ends: (ctx, pointer, False) for
    scratch memory to free, (ctx, pointer, True) for a buffer reference (allocated by a
    kernel, or by `rt_alloc`) to release, stored at `pointer`."""

    def __init__(self, function: bool):
        super().__init__()
        self.function = function


@dataclass
class Codegen:
    """The state of generating one function's code: the builder, the kernel's
    parameters (`ctx`, its `config` = link table) and the module the function is in
    (`unit`: its options, its link table, how inner graphs are planned)."""

    b: ir.Builder
    ctx: ir.Value
    #: the kernel's config: its link table (call.py)
    config: ir.Value
    unit: KernelModule
    #: the name of the function being built (nested functions are named after it)
    name: str = ""
    #: bytes of stack temporaries so far (`temp_memory`)
    stack_bytes: int = 0
    #: may loops run in parallel? None: as the options say. Off inside a parallel
    #: loop (`for_range`): no nested parallel regions.
    parallel: bool | None = None
    #: the scratch memory taken in the scopes being emitted, innermost last: what to
    #: free at the end of each (`scratch_scope`)
    scratch_scopes: list[_ScratchScope] = field(default_factory=list)
    #: the index checks deferred in the scopes being emitted (`deferred_checks`),
    #: innermost last: per message, the flag set when one fails
    deferred: list[dict[str, ir.Value]] = field(default_factory=list)
    _checks: int = field(default=0)

    def __post_init__(self) -> None:
        if self.parallel is None:
            self.parallel = self.unit.options.parallel

    @property
    def check_layouts(self) -> bool:
        """Verify at run time that views have the layouts the planner relies on."""
        return self.unit.options.check_layouts

    @property
    def links(self) -> LinkTable:
        """The link table of the module (call.py)."""
        return self.unit.links

    @property
    def module(self) -> ir.Module:
        return self.unit.module

    def fail(self, message: str | None, exc: type[BaseException] = ValueError) -> None:
        """Report `message`, for the call to raise `exc` with (unless a called kernel
        already reported an error: `message` None), and return 1."""
        if message is not None:
            self.b.rt_fail(self.ctx, message, self.unit.exceptions.code(exc))
        self.b.ret(1)

    def check(
        self, ok: ir.Value, message: str | None, exc: type[BaseException] = ValueError
    ) -> None:
        """Fail with `message` and `exc` unless `ok` (i1). Constant conditions fold."""
        c = ok.const_int
        if c == 1:
            return
        with self.b.if_(self.b.not_(ok)):
            self.fail(message, exc)

    def graph_function(
        self,
        inputs: Sequence[Variable],
        outputs: Sequence[Variable],
        in_layouts: Sequence[Layout | None],
        name: str = "graph",
    ) -> GraphFunction:
        """Prepare an inner graph (a Scan's loop body, a Blockwise's core, ...) to be
        run from the function being generated. Run it with
        `GraphFunction.emit_call`.

        The graph goes from the variables `inputs` to the variables `outputs`.
        `in_layouts` gives the layout of each input as it will be passed (None for
        inputs that aren't tensors); the graph's code is specialized for them.

        This emits code for the graph's constants at the current point, so call it
        before a loop that runs the graph, not inside the loop. `name` is used to name
        the LLVM functions of its nodes.

        Raises NotImplementedError if inner graphs can't be run here (no `plan_graph`)."""
        from pytensor.link.llvm.codegen.graph import GraphFunction

        if self.unit.plan_graph is None:
            raise NotImplementedError("inner graphs here")
        body = self.unit.plan_graph(inputs, outputs, in_layouts)
        return GraphFunction.bind(self, body, f"{self.name}_{name}")

    def temp_memory(self, nbytes: ir.Value, escapes: bool = False) -> ir.Value:
        """Temporary memory of `nbytes` (i64), 64-byte aligned, uninitialized.

        Small memory of static size is on the stack of the function being generated:
        LLVM sees all of it, and may keep it in registers. Everything else is scratch
        memory of the context (`rt_scratch`). Of a static size, it is taken when the
        function starts, like stack memory (a loop asking for it in every iteration gets
        the same memory), and freed when the function's scope ends; else it is freed
        when the innermost `scratch_scope` ends.

        `escapes`: the memory is used after the function being generated returns (an
        output that is a temporary of the unit around it); then it is never on the stack,
        and the caller frees it.

        Each call is a new piece of scratch: ask before a loop, not inside it."""
        b = self.b
        n = nbytes.const_int
        if n is not None and not escapes:
            p = self._stack_memory(n)
            if p is not None:
                return p
        at_entry = n is not None and not escapes
        p = b.rt_scratch(self.ctx, nbytes, at_entry=at_entry)
        self.check(b.ne(p, b.null()), None)
        if at_entry:
            self.free_later(p, function=True)
        elif not escapes:
            self.free_later(p)
        return p

    def _stack_memory(self, n: int) -> ir.Value | None:
        """Stack memory for `n` bytes, or None if it shouldn't go on the stack.

        This is the rule `temp_memory` follows: memory goes on the stack if it is small
        (at most `STACK_TEMPORARY_BYTES`) and the function hasn't used up its stack
        budget (`STACK_BUDGET_BYTES`)."""
        if n > STACK_TEMPORARY_BYTES:
            return None
        rounded = max(64, -(-n // 64) * 64)
        if self.stack_bytes + rounded > STACK_BUDGET_BYTES:
            return None
        self.stack_bytes += rounded
        return self.b.alloca(ir.Type.array(ir.i8, max(n, 1)))

    def temporaries[K](
        self, sizes: dict[K, int], lives: dict[K, tuple[int, int]]
    ) -> tuple[dict[K, ir.Value], tuple[dict[K, int], int]]:
        """Memory for several temporary values of a known size.

        * `sizes`: the size in bytes of each value ({key: bytes});
        * `lives`: when each value is needed: from item `lives[key][0]` to item
          `lives[key][1]` (both included).

        Each value gets its own stack memory if `temp_memory` would put it on the stack
        (the values defined first get the stack first). The other values share one
        block of scratch memory, the arena: two values can use the same place in it if
        they are never needed at the same time (`analysis.liveness.first_fit`).

        Returns the memory of each value ({key: pointer}), and the arena's layout
        ({key: offset in the arena}, size of the arena in bytes)."""
        from pytensor.link.llvm.analysis.liveness import first_fit

        memory: dict[K, ir.Value] = {}
        shared: list[tuple[int, int, int, K]] = []
        for key in sorted(sizes, key=lambda k: lives[k]):
            p = self._stack_memory(sizes[key])
            if p is None:
                shared.append((*lives[key], sizes[key], key))
            else:
                memory[key] = p
        offsets, nbytes = first_fit(shared)
        if offsets:
            arena = self.temp_memory(self.b.i64(nbytes))
            memory.update((k, self.b.gep_bytes(arena, at)) for k, at in offsets.items())
        return memory, (offsets, nbytes)

    def free_later(self, p: ir.Value, function: bool = False) -> None:
        """Free the scratch memory `p` when the innermost `scratch_scope` ends, or with
        `function` the innermost one of the function being generated (anything else
        than scratch is ignored when running)."""
        scopes = [sc for sc in self.scratch_scopes if sc.function or not function]
        if not scopes:
            raise ValueError("scratch memory outside of a scratch_scope")
        scopes[-1].append((self.ctx, p, False))

    def release_later(self, owner: ir.Value) -> None:
        """Release the buffer reference stored at `owner` (a slot's owner field, ...)
        when the innermost `scratch_scope` ends. What is stored there then counts: a
        null reference is fine."""
        if not self.scratch_scopes:
            raise ValueError("a buffer to release outside of a scratch_scope")
        self.scratch_scopes[-1].append((self.ctx, owner, True))

    @contextmanager
    def scratch_scope(self, function: bool = False) -> Generator[None]:
        """The scratch memory taken inside is freed at its end (on the way out of the
        code emitted inside, unless that returned; on errors the caller's mark releases
        it). The code emitted inside must be one region of control flow: a function body
        (`function`, also a parallel body), a loop iteration, a branch."""
        taken = _ScratchScope(function)
        self.scratch_scopes.append(taken)
        try:
            yield
        finally:
            self.scratch_scopes.pop()
        if not self.b.terminated:
            b = self.b
            for ctx, p, is_owner in reversed(taken):
                if is_owner:
                    b.rt_decref(ctx, b.load(ir.ptr, p))
                    continue
                # (folds away for checked scratch, and where a view left null)
                with b.if_(b.ne(p, b.null())):
                    b.rt_scratch_free(ctx, p)

    def temporary(
        self,
        scalar_type: ScalarType,
        shape: Sequence[ir.Value | int],
        layout: Layout | None = None,
        escapes: bool = False,
    ) -> Array:
        """A temporary array (`temp_memory`), packed in `layout` (default C). `shape`:
        i64 values or Python ints."""
        b = self.b
        lengths = [b.i64(n) if isinstance(n, int) else n for n in shape]
        nbytes = b.mul(size(b, lengths), np.dtype(scalar_type.dtype).itemsize, nsw=True)
        data = self.temp_memory(nbytes, escapes)
        return Array.packed(b, scalar_type, data, lengths, layout)

    # ---- loops and indices

    def for_each(
        self,
        shape: Sequence[ir.Value],
        arrays: Sequence[Array],
        body: ElementBody | None = None,
        *,
        inner: InnerLoop | None = None,
        flat: bool = True,
        parallel_cost: float | None = None,
    ) -> None:
        """Loop over all elements of `shape`, moving a pointer through each of `arrays`.

        The arrays have strides over `shape`, 0 along axes where they are broadcast.
        The loops follow the order the arrays are laid out in memory (`loop_order`).
        If every array is packed in that order, there is only one flat loop over all
        elements, which the vectorizer likes. The layouts usually tell whether they
        are packed. If they don't, it is checked at run time, and both the flat loop
        and the nest of loops are generated. Arrays with stride 0 along every axis
        don't move, so they never prevent the flat loop. `flat=False` always uses the
        nest, for callers that need the loop index along each axis.

        Normally you pass `body`, and it is called once per element with the pointers
        to that element.

        Pass `inner` instead when one call per element isn't enough because you need
        to generate the innermost loop yourself. For example, a sum can then load its
        running total before the innermost loop, keep it in a register while the loop
        runs, and store it afterwards. for_each still chooses the loop order and
        generates the outer loops, but where it would generate the innermost loop it
        calls `inner(axes, n, at)` instead. `n` is the innermost loop's length, and
        `at(i)` returns the pointers and the loop index at its i-th element. `axes`
        are the axes the innermost loop moves along.

        For a shape (n0, n1) where axis 1 is innermost in memory, this generates:

            for i0 in range(n0):
                inner({1}, n1, at)    # at(i) gives the pointers and index at [i0, i]

        In the flat loop, `inner` is called once, with axes {0, 1} and n0 * n1, and
        the loop index from `at` is empty.

        With `parallel_cost`, the cost of one element (analysis/work.py), the loop (the
        flat one, or the outermost loop of the nest) runs in chunks that may run in
        parallel when there's enough work. Iterations must not depend on each other.
        If the parallel loop is also the innermost one, `inner` is called once per
        chunk."""
        b = self.b
        if inner is None:
            inner = self._per_element(body)
        moving = [a for a in arrays if not _stationary(a)]
        order = loop_order(shape, moving)
        policy = self._policy(lambda: size(self.b, shape), parallel_cost)
        contig = b.const_bool(flat)
        for a in moving:
            if contig.const_int == 0:
                break
            contig = b.and_(contig, a.is_contiguous(b, shape, order))
        c = contig.const_int
        if c == 1:
            self._flat_loop(shape, arrays, inner, policy)
        elif c == 0:
            self._loop_nest(shape, arrays, inner, order, policy=policy)
        else:
            b.if_else(
                contig,
                lambda: self._flat_loop(shape, arrays, inner, policy),
                lambda: self._loop_nest(shape, arrays, inner, order, policy=policy),
            )

    def loop_nest(
        self,
        shape: Sequence[ir.Value],
        arrays: Sequence[Array],
        body: ElementBody,
        order: Sequence[int],
        parallel_cost: float | None = None,
    ) -> None:
        """Loops over the axes in `order` (outermost first; may be a subset of the axes),
        calling `body(ptrs)` with the element pointers of `arrays` in the innermost
        one. Unlike `for_each`, this never uses a flat loop. With `parallel_cost`, the
        outermost loop runs in chunks that may run in parallel."""
        policy = self._policy(lambda: size(self.b, shape), parallel_cost)
        self._loop_nest(shape, arrays, self._per_element(body), order, policy=policy)

    def for_range(
        self,
        n: ir.Value,
        body: Callable[[ir.Value], None],
        parallel_cost: float | None = None,
        per_iteration: ir.Value | None = None,
    ) -> None:
        """`body(i)` for every i in 0..n: a loop over an index (`for_each` loops over
        elements).

        With `parallel_cost`, the cost of one element (analysis/work.py), and
        `per_iteration`, the elements an iteration works on (default 1), the iterations
        run in chunks that may run in parallel when there's enough work. They must not
        depend on each other. Code emitted in `body` doesn't start parallel loops of
        its own then (no nested parallel regions)."""
        b = self.b
        per = b.i64(1) if per_iteration is None else per_iteration
        policy = self._policy(lambda: b.mul(n, per), parallel_cost)
        if policy is None:
            with b.loop(n) as i:
                body(i)
            return
        parallel, self.parallel = self.parallel, False
        try:
            policy.range(b, n, per, body)
        finally:
            self.parallel = parallel

    def checked_index(
        self,
        i: ir.Value,
        n: ir.Value,
        message: str,
        bounds: IndexBounds = (None, None),
        defer: bool = False,
    ) -> ir.Value:
        """Index `i` (i64) into an axis of length `n`: negative values wrap, out-of-range
        ones fail with `message`. Generates only what `bounds` (`index_bounds`), a
        constant `i` and a constant `n` don't already guarantee.

        With `defer`, inside `deferred_checks` for `message`, an index out of range
        doesn't leave the loop it is in: it is clamped into the axis, and the failure is
        reported when the scope ends. The axis must not be empty then (check that
        before the loop)."""
        b = self.b
        if i.const_int is not None:
            bounds = (i.const_int, i.const_int)
        lo, hi = bounds
        if lo is None or lo < 0:
            i = b.select(b.slt(i, 0), b.add(i, n), i)
        nc = n.const_int
        if not (lo is None or hi is None or nc is None or lo < -nc or hi >= nc):
            return i
        ok = b.ult(i, n)  # also catches what is still negative
        if not defer:
            self.check(ok, message, IndexError)
            return i
        flag = self.deferred[-1].get(message) if self.deferred else None
        if flag is None:
            raise ValueError(f"a deferred check outside of deferred_checks: {message}")
        # (a clamp stays a vector operation where the loads use the index one lane at a
        # time; a select of 0 would be done lane by lane with them)
        clamped = b.umin(i, b.sub(n, 1))
        b.store(b.or_(b.load(ir.i1, flag), b.ne(clamped, i)), flag)
        return clamped

    @contextmanager
    def deferred_checks(self, messages: Iterable[str]) -> Iterator[None]:
        """Index checks with these `messages` deferred inside (`checked_index` with
        `defer`) fail when the scope ends, not in the loop they are in: loops without
        an exit in the middle can be vectorized. Use it around a loop whose results
        aren't used when it fails. Like `scratch_scope`, the code inside must be one
        region of control flow, of one function (a parallel body has scopes of its
        own)."""
        b = self.b
        flags = {}
        for message in messages:
            flags[message] = b.alloca(ir.i1)
            b.store(b.const_bool(False), flags[message])
        self.deferred.append(flags)
        try:
            yield
        finally:
            self.deferred.pop()
        if not self.b.terminated:
            for message, flag in flags.items():
                self.check(self.b.not_(self.b.load(ir.i1, flag)), message, IndexError)

    def _policy(
        self, n_elements: Callable[[], ir.Value], parallel_cost: float | None
    ) -> _ParallelPolicy | None:
        """How a loop over `n_elements()` elements costing `parallel_cost` each
        (analysis/work.py units) runs in parallel; None: serially (not asked for, not
        allowed, or too little work)."""
        if parallel_cost is None or parallel_cost <= 0 or not self.parallel:
            return None
        policy = _ParallelPolicy(self, parallel_cost)
        return policy if policy.worth_it(n_elements()) else None

    def _per_element(self, body: ElementBody | None) -> InnerLoop:
        """The innermost loop `for_each` generates when the caller passes `body`: it
        calls `body` with the pointers at each element."""
        if body is None:
            raise TypeError("for_each needs a body or an inner loop")
        b = self.b

        def inner(
            axes: frozenset[int], n: ir.Value, at: Callable[[ir.Value], Element]
        ) -> None:
            if n.const_int == 1:
                body(at(b.i64(0))[0])
                return
            with b.loop(n) as i:
                body(at(i)[0])

        return inner

    def _flat_loop(
        self,
        shape: Sequence[ir.Value],
        arrays: Sequence[Array],
        inner: InnerLoop,
        policy: _ParallelPolicy | None = None,
    ) -> None:
        b = self.b
        axes = frozenset(range(len(shape)))

        def at(i: ir.Value) -> Element:
            ptrs = [
                a.data
                if _stationary(a)
                else b.gep(memory_type(a.scalar_type), a.data, i)
                for a in arrays
            ]
            return ptrs, {}

        if policy is None:
            inner(axes, size(b, shape), at)
            return
        policy.chunks(
            b,
            size(b, shape),
            b.i64(1),
            lambda start, n: inner(axes, n, lambda j: at(b.add(start, j))),
        )

    def _loop_nest(
        self,
        shape: Sequence[ir.Value],
        arrays: Sequence[Array],
        inner: InnerLoop,
        order: Sequence[int],
        level: int = 0,
        base: list[ir.Value] | None = None,
        outer: LoopIndex | None = None,
        policy: _ParallelPolicy | None = None,
    ) -> None:
        b = self.b
        if base is None:
            base = [a.data for a in arrays]
        if outer is None:
            outer = {}
        if not order:
            inner(frozenset(), b.i64(1), lambda i: (base, outer))
            return
        axis = order[level]
        pointers, index = base, outer

        def at(i: ir.Value) -> Element:
            ptrs = [
                advance(b, p, i, a.strides[axis], a.scalar_type)
                for p, a in zip(pointers, arrays, strict=True)
            ]
            return ptrs, {**index, axis: i}

        if level == len(order) - 1:
            if policy is not None:
                policy.chunks(
                    b,
                    shape[axis],
                    b.i64(1),
                    lambda start, n: inner(
                        frozenset({axis}), n, lambda j: at(b.add(start, j))
                    ),
                )
            else:
                inner(frozenset({axis}), shape[axis], at)
            return

        def step(i: ir.Value) -> None:
            ptrs, deeper = at(i)
            self._loop_nest(shape, arrays, inner, order, level + 1, ptrs, deeper)

        if policy is not None:
            rest = size(b, [shape[a] for a in order[1:]])
            policy.range(b, shape[axis], rest, step)
            return
        with b.loop(shape[axis]) as i:
            step(i)


def _stationary(a: Array) -> bool:
    """Does `a` have stride 0 along every axis of the loop, so that its pointer never
    moves? A sum over all axes, for example."""
    return all(st.const_int == 0 for st in a.strides)


# ------------------------------------------------------------------ loops


def loop_order(shape: Sequence[ir.Value], arrays: Sequence[Array]) -> list[int]:
    """The axis order (outermost first) to loop in: the memory order most arrays share
    (by their packed axes). On a tie, the one with the longer innermost axis (static
    lengths; an inner loop that is longer vectorizes better and costs less around
    it), then the one of the later array, usually an output. Identity when nothing is
    known."""
    static = [n.const_int for n in shape]
    votes: dict[
        tuple[int, ...], tuple[int, int]
    ] = {}  # packed block (innermost first) -> (count, last position)
    for pos, a in enumerate(arrays):
        block = tuple(a.layout.packed_block(static))
        if block:
            count, _ = votes.get(block, (0, -1))
            votes[block] = (count + len(block), pos)
    if not votes:
        return list(range(len(shape)))

    def rank(block: tuple[int, ...]) -> tuple[int, int, int]:
        count, pos = votes[block]
        return count, static[block[0]] or 0, pos

    block = max(votes, key=rank)
    rest = [ax for ax in range(len(shape)) if ax not in block]
    return rest + list(reversed(block))


# ------------------------------------------------------------------ parallel loops

#: cost (analysis/work.py units) of one chunk of a parallel loop: enough to make the
#: overhead of a parallel region (microseconds) small
CHUNK_COST = 50_000
#: loops cheaper than this stay serial (decided when generating the code when the
#: shapes are static; at run time by the runtime otherwise: one chunk runs directly)
MIN_PARALLEL_COST = 2 * CHUNK_COST


@dataclass
class _ParallelPolicy:
    """A loop whose iterations may run in parallel: the context, the cost of one item."""

    cg: Codegen
    per_element: float

    def worth_it(self, n_elements: ir.Value) -> bool:
        """False if the loop is statically too small for a parallel region."""
        n = n_elements.const_int
        return n is None or n * self.per_element >= MIN_PARALLEL_COST

    def grain(self, b: ir.Builder, elements_per_iteration: ir.Value) -> ir.Value:
        """Iterations per chunk (i64), for iterations of `elements_per_iteration` items."""
        per = elements_per_iteration.const_int
        if per is not None:
            return b.i64(
                max(1, math.ceil(CHUNK_COST / (self.per_element * max(per, 1))))
            )
        target = max(1, math.ceil(CHUNK_COST / self.per_element))
        at_least_one = b.smax(elements_per_iteration, 1)
        return b.smax(b.sdiv(target, at_least_one), 1)

    def range(
        self,
        b: ir.Builder,
        n: ir.Value,
        elements_per_iteration: ir.Value,
        body: Callable[[ir.Value], None],
    ) -> None:
        """`body(i)` for i in 0..n, in chunks that may run in parallel; failures in a
        chunk fail the function."""

        def chunk(start: ir.Value, count: ir.Value) -> None:
            with b.loop(count) as j:
                body(b.add(start, j))

        self.chunks(b, n, elements_per_iteration, chunk)

    def chunks(
        self,
        b: ir.Builder,
        n: ir.Value,
        elements_per_iteration: ir.Value,
        body: Callable[[ir.Value, ir.Value], None],
    ) -> None:
        """Split the iterations 0..n into chunks that may run in parallel, and call
        `body(start, count)` to generate the code of a chunk: the iterations
        start..start + count. Use it instead of `range` to generate the loop over a
        chunk yourself. Failures in a chunk fail the function."""
        self.cg.unit.uses_pool = True
        with b.parallel(self.cg.ctx, n, self.grain(b, elements_per_iteration)) as chunk:
            # A parallel body is a function of its own, with its own scratch scope and
            # its own flags for the deferred checks of the loop around it.
            with (
                self.cg.scratch_scope(function=True),
                self.cg.deferred_checks(
                    self.cg.deferred[-1] if self.cg.deferred else ()
                ),
            ):
                body(chunk.start, b.sub(chunk.end, chunk.start))
        self.cg.check(b.eq(chunk.result, 0), None)


# ------------------------------------------------------------------ indices


def index_bounds(var: Variable) -> IndexBounds:
    """(min, max) known of the values of the integer index variable `var` (None where
    unknown): a constant's values, or at least 0 for an unsigned dtype."""
    if isinstance(var, Constant):
        if var.data.size == 0:
            return 0, -1
        return int(var.data.min()), int(var.data.max())
    if var.type.dtype.startswith("uint"):
        return 0, None
    return None, None


__all__ = [
    "Codegen",
    "ElementBody",
    "IndexBounds",
    "index_bounds",
    "loop_order",
]
