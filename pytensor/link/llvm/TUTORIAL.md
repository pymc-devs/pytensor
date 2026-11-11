# Adding an op to the LLVM backend

This tutorial implements a new op for the LLVM backend, start to finish. It first
introduces the interface the implementation is written with, `pytensor_llvm.ir`, and
then uses it for a small tensor op and a scalar op, and explains layouts (what the
backend knows about the strides of each array, and how an op takes part in that),
views and writing over inputs, temporary memory and random variables. `DESIGN.md`
explains why things are the way they are; this file only shows how to use them.

Every `python` block below is run, in order, by `tests/link/llvm/test_tutorial.py`, so
the code is known to work. You can paste the blocks into one session and follow along.

## 1. Three ways to implement an op

A node of the graph becomes a *task* of the program. The planner looks for the kernel
of that task in this order:

1. **Generated code**: an emitter, registered with `llvm_emit`, writes LLVM IR for the
   node. The code is specialized to the node's dtypes, static shapes and layouts, and
   several nodes can be compiled into one function. This is what the tutorial covers,
   and what most ops should use.
2. **A Rust kernel** (`rust_kernels.py`): precompiled code that looks at its arguments
   when it runs. For ops whose work is a library call (a Cholesky factorization, a
   sort), where there is nothing to specialize.
3. **A numba kernel**: PyTensor's numba implementation of the op, called through a
   wrapper. This happens by itself for every op that has neither of the above, so a
   graph with an op the backend doesn't know still runs. It costs numba's compile time
   and the op can't be compiled together with its neighbors.

An emitter that can't handle a particular node raises `NotImplementedError`, and the
planner moves on to the next way. Supporting only the easy cases of an op is therefore
fine.

## 2. The builder

Emitters write code through a *builder*, a thin and checked wrapper around LLVM's own
IR builder. Before using it inside the backend, here is a complete function built by
hand: the sum of squares of `n` doubles.

```python
import ctypes

import numpy as np

from pytensor_llvm import ir

module = ir.Module("hello")
function = module.function("sum_squares", [ir.ptr, ir.i64], ir.f64, exported=True)
b = function.builder()
x, n = function.params
x.named("x")  # names only show in the IR text
n.named("n")


def add_square(i, total):
    value = b.load(ir.f64, b.gep(ir.f64, x, i))  # x[i]
    return b.fadd(total, b.fmul(value, value))


(total,) = b.fold(n, [b.f64(0.0)], add_square)
b.ret(total)
unoptimized = module.ir()

compiled = module.compile(cache=False)
sum_squares = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_void_p, ctypes.c_int64)(
    compiled.address("sum_squares")
)
data = np.arange(5.0)
assert sum_squares(data.ctypes.data, 5) == 30.0
```

What happened:

* A `Module` holds functions. `module.function(name, parameter types, return type)`
  declares one, and `function.builder()` gives the builder for its body.
* Each builder method appends one instruction and returns its result, a `Value`. Python
  runs once, while the code is *generated*; the values are what the generated code
  computes later. `b.fmul(value, value)` multiplies nothing now.
* `b.fold(n, init, body)` is a loop with carried values. It calls `add_square` once, to
  generate the loop body, with the loop index and the carried value, and returns the
  values after the loop.
* `module.compile()` verifies, optimizes and compiles the module and loads it into the
  process; `compiled.address(name)` is the function's address. A compiled module is
  closed: nothing can be added to it.

`module.ir()` is the module as LLVM IR text, the first thing to look at when something
misbehaves. Before `compile()` it is what the builder wrote; afterwards it is the
optimized code (here: a vectorized loop).

```python
print(unoptimized)
```

```text
define double @sum_squares(ptr %x, i64 %n) {
  ...
fold:                                             ; preds = %fold, %body
  %1 = phi i64 [ 0, %body ], [ %7, %fold ]
  %2 = phi double [ 0.000000e+00, %body ], [ %6, %fold ]
  %3 = getelementptr inbounds double, ptr %x, i64 %1
  %4 = load double, ptr %3, align 8
  %5 = fmul reassoc nsz arcp contract afn double %4, %4
  %6 = fadd reassoc nsz arcp contract afn double %2, %5
  %7 = add nsw i64 %1, 1
  %8 = icmp slt i64 %7, %n
  br i1 %8, label %fold, label %fold.exit
  ...
```

The flags on `fmul` and `fadd` are the fast-math flags the backend uses by default
(`Module(name, fastmath=False)` and the linker option `fastmath` turn them off).

### What the builder offers

| | |
|---|---|
| Types | `ir.i1`, `ir.i8`, `ir.i16`, `ir.i32`, `ir.i64`, `ir.f32`, `ir.f64`, `ir.ptr`; `ir.Type.array(elem, n)`, `ir.Type.struct_([...])` |
| Constants | `b.i64(3)`, `b.i32(3)`, `b.i8(3)`, `b.f64(0.5)`, `b.f32(0.5)`, `b.const_bool(True)`, `b.null()` |
| Integers | `add`, `sub`, `mul` (each with `nsw=True`: "this doesn't overflow"), `sdiv`, `udiv`, `srem`, `urem`, `shl`, `lshr`, `ashr`, `and_`, `or_`, `xor`, `not_`, `neg`, `smin`, `smax`, `umin`, `umax` |
| Floats | `fadd`, `fsub`, `fmul`, `fdiv`, `frem`, `fneg`; `b.intrinsic(ir.Intrinsic.Exp, [x])`, `b.libm(ir.Libm.Erf, [x])` |
| Comparisons | integers: `eq`, `ne`, `slt`, `sle`, `sgt`, `sge` (signed), `ult`, `ule`, `ugt`, `uge` (unsigned); floats: `b.fcmp(ir.FCmp.OLT, x, y)`. The result is an `i1`. |
| Choosing | `b.select(cond, x, y)` |
| Conversions | `sext`, `zext`, `trunc`, `sitofp`, `uitofp`, `fptosi`, `fptoui`, `fpext`, `fptrunc` |
| Memory | `b.load(type, ptr)`, `b.store(value, ptr)`, `b.gep(elem_type, ptr, index)`, `b.gep_bytes(ptr, offset)`, `b.alloca(type)` |
| Control flow | `with b.loop(n) as i:`, `with b.if_(cond):`, `with b.else_():`, `b.fold(n, init, body)`, `b.if_else(cond, then, else_)`, `b.ret(value)` |

Five things are worth knowing before writing an emitter.

**Integers have no sign, and floats have their own instructions.** An `i64` is 64 bits;
whether it is signed is decided by the instruction (`sdiv` or `udiv`, `slt` or `ult`).
Adding floats is `fadd`, adding integers is `add`. Code that must work for every dtype
would have to choose each time, which is what `Num` (section 4) is for.

**Operands may be Python numbers.** A literal takes the type of the other operand, or
the type the method asks for:

```python
scratch = ir.Module("scratch")
probe = scratch.function("probe", [ir.i64, ir.f64], ir.i64)
b = probe.builder()
i, x = probe.params

j = b.add(b.mul(i, 2), 1)  # i64 constants
half = b.fmul(x, 0.5)  # an f64 constant
with b.loop(4) as k:  # loop counts are i64
    pass
```

At least one operand must be a value, because a literal alone has no type: write
`b.select(cond, b.i64(0), -1)`, not `b.select(cond, 0, -1)`.

**Mistakes raise.** Every method checks its operands, so a wrong type is a Python
exception at the line that made the mistake, not a crash in LLVM:

```python
import pytest

with pytest.raises(ValueError, match="operand types differ"):
    b.add(i, b.i32(1))
with pytest.raises(ValueError, match="add needs integers"):
    b.add(x, 1.0)
```

**Arithmetic on constants gives constants.** `value.const_int` is the Python integer if
the value is a constant, else `None`. Integer arithmetic keeps constants constant and
drops operations that do nothing:

```python
assert b.add(b.i64(2), 3).const_int == 5
assert b.mul(i, 0).const_int == 0
assert repr(b.mul(i, 1)) == repr(i)  # no instruction: this is `i`
assert b.add(i, 1).const_int is None
```

This matters because the lengths of an array with a static shape are constants. An
emitter can do its index arithmetic without thinking about it, and ask `.const_int`
when it wants to generate different code for a known size.

**A value is not a Python condition.** `if b.slt(i, n):` would test a Python object,
which is always true. A decision the generated code must make is a `b.select`, or a
block:

```python
with b.if_(b.slt(i, 0)):
    b.ret(0)  # the literal takes the function's return type
with b.else_():
    b.ret(j)
```

`b.if_else(cond, then, else_)` is the form for branches that produce values: both
callables return a value (or a list of values), and the merged values are returned.

## 3. A tensor op: the running maximum

The op for the rest of this tutorial computes the running maximum of a vector. It is an
ordinary PyTensor op:

```python
import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import get_mode
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op


class CumMax(Op):
    """The running maximum of a vector."""

    __props__ = ()

    def make_node(self, x):
        x = pt.as_tensor_variable(x)
        if x.type.ndim != 1:
            raise TypeError("CumMax needs a vector")
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        output_storage[0][0] = np.maximum.accumulate(inputs[0])


cummax = CumMax()
```

The backend doesn't know it. With the numba fallback switched off, which is how the
backend's own tests run, compiling fails:

```python
LLVM = get_mode("LLVM")
STRICT = LLVM.clone(link_kwargs={"numba_fallback": False})

x = pt.vector("x")
with pytest.raises(
    NotImplementedError, match="No LLVM backend implementation for CumMax"
):
    pytensor.function([x], cummax(x), mode=STRICT)
```

### What an emitter gets

An emitter is a function registered for the op's class:

```py
@llvm_emit.register(CumMax)
def _(op, node, cg, inputs, outputs): ...
```

* `cg` is the code generation context. `cg.b` is the builder; `cg.check(ok, message)`
  makes the task fail with `message` unless `ok` holds.
* `inputs` has one entry per input of the node. A tensor is an `Array`:
  * `a.dtype`: the dtype as a string,
  * `a.data`: the pointer to the first element,
  * `a.shape`: one `i64` value per axis, constants where the shape is static,
  * `a.strides`: one `i64` value per axis, in bytes, as in numpy; constants where the
    layout fixes them,
  * `a.layout`: what is known about the strides while generating (section 5).

  Inputs must not be written.
* `outputs` has one entry per output, to be filled. For a tensor,
  `outputs[k].allocate(shape)` returns the `Array` to write: either memory the caller
  provides (static shapes), or memory allocated here with that shape. The emitter
  doesn't need to know which. (An op whose `view_map` lets an output be a view of an
  input can hand over a view instead; section 5.)

Elements are read and written through the array:

* `a.load(b, i, j)` is the element at an index, as a `Num`: a value that knows its
  dtype (section 4),
* `a.store(b, value, i, j)` writes one; `value` is a `Num`, an IR value or a Python
  number,
* `a.at(b, i)` is the sub-array at index `i` of the first axis (`a.at(b, j, axis=1)`
  for another one): a row of a matrix, as an `Array` again,
* `a.ptr(b, i, j)` is the address of an element.

Indices are `i64` values or Python ints, one per axis. All four follow the strides, so
they work for any view, and none checks bounds. Use them and not `b.load`: a bool is a
byte in memory and an `i1` in a register, and they know.

### The emitter

```python
from pytensor.link.llvm.codegen.function import llvm_emit
from pytensor.link.llvm.codegen.num import Num


@llvm_emit.register(CumMax)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    (x,) = inputs
    (n,) = x.shape
    out = outputs[0].allocate(x.shape)

    def step(i, best):
        """out[i] = max(best, x[i]); returns the new maximum."""
        value = x.load(b, i)
        if best is not None:
            value = Num(b, best, x.dtype).maximum(value)
        out.store(b, value, i)
        return value.value

    with b.if_(b.sgt(n, 0)):
        first = step(0, None)
        b.fold(b.sub(n, 1), [first], lambda i, best: step(b.add(i, 1), best))
```

Line by line:

* `outputs[0].allocate` gets the output's memory, of the shape of `x`.
* `x.load(b, i)` reads element `i` through the stride of `x`. The input may be any view
  (`x[::-2]`), so the stride is never assumed. Where the planner knows the stride, it
  is a constant here and the multiplication disappears (section 5).
* `step` generates the code for one element. It is plain Python that is called twice
  and so generates code twice: once for the first element, which has no maximum before
  it (`if best is not None` is decided while generating, which is fine: `best` is
  either a value or Python's `None`), and once inside the loop.
* The maximum itself depends on the dtype: an intrinsic for floats, a signed or an
  unsigned compare for integers, an `or` for bools. `x.load` returns a `Num`, whose
  `maximum` picks. The carried value of a `fold` is a plain IR value, so it is wrapped
  with `Num(b, best, x.dtype)` on the way in and unwrapped with `.value` on the way
  out.
* An empty vector has no first element, hence `b.if_(b.sgt(n, 0))`. For a static shape
  `n` is a constant and LLVM removes the branch.
* `b.fold(count, [first], body)` runs `body` for `i = 0 .. count - 1` with the running
  maximum carried along. The body returns the new carried value.

The emitter doesn't return anything. When it ends, the generated function returns 0
(success) unless the emitter already returned.

### Running it

```python
f = pytensor.function([x], cummax(x), mode=STRICT)
values = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
np.testing.assert_array_equal(f(values), [1.0, 3.0, 3.0, 5.0, 5.0])
```

`f.maker.linker.program` is the compiled program. Its dump lists the values and the
tasks, and `f.maker.linker.stats` counts what the planner used:

```python
program = f.maker.linker.program
print(program.dump())
assert program.tasks == ["CumMax"]
assert f.maker.linker.stats.numba == []  # the ops that fell back to numba
```

```text
program: 1 inputs, 1 outputs, 1 tasks, arena 0 bytes
values:
  %0    float64[?]               input 0
  %1    float64[?]               allocated by its task
tasks:
  0    (%0) -> (%1)  CumMax
outputs: %1 (allocated)
```

The length of `x` is unknown here, so the task allocates its output. With a static
shape the caller provides the memory and the lengths are constants in the generated
code; the emitter is the same.

### Testing it

An emitter should be tried on what it claims to handle: every dtype, inputs that are
views, the empty case, static shapes.

```python
rng = np.random.default_rng(0)
for dtype in ["float64", "float32", "int64", "int8", "uint8", "bool"]:
    xd = pt.vector("x", dtype=dtype)
    fd = pytensor.function([xd], [cummax(xd), cummax(xd[::-2])], mode=STRICT)
    data = (rng.normal(size=11) * 50).astype(dtype)
    whole, view = fd(data)
    assert whole.dtype == data.dtype
    np.testing.assert_array_equal(whole, np.maximum.accumulate(data))
    np.testing.assert_array_equal(view, np.maximum.accumulate(data[::-2]))

assert f(np.array([])).shape == (0,)
np.testing.assert_array_equal(f(np.array([1.0, np.nan, 2.0])), [1.0, np.nan, np.nan])

static = pt.vector("x", shape=(4,))
fs = pytensor.function([static], cummax(static), mode=STRICT)
np.testing.assert_array_equal(fs(np.array([1.0, 0.0, 2.0, 1.0])), [1.0, 1.0, 2.0, 2.0])
```

`compile_node` compiles one node alone, without planning a program around it. That is
the quickest way to look at the code of one emitter:

```python
from pytensor.link.llvm.codegen import compile_node

node = cummax(x).owner
alone = compile_node(node)
(result,) = ir.call_kernel(alone.address, [values], [(np.dtype("float64"), 1)])
np.testing.assert_array_equal(result, [1.0, 3.0, 3.0, 5.0, 5.0])
```

Setting `PYTENSOR_LLVM_DUMP_IR=<directory>` writes the optimized IR of every module
that is compiled there (not of those taken from the cache), which shows whether a loop
was vectorized.

### Errors

Conditions only known when the code runs are checked with `cg.check`. If the op
required a non-empty vector, the emitter would start with

```py
cg.check(b.sgt(n, 0), "CumMax of an empty vector")
```

and calling the function with an empty vector would raise an error with that message. A check on a constant that holds generates nothing.

Conditions known while generating are ordinary Python. To decline a node, raise
`NotImplementedError`:

```py
if x.dtype == "bool":
    raise NotImplementedError("CumMax of booleans")
```

## 4. A scalar op, and `Num`

Scalar ops are what `Elemwise` loops over. Their emitters only compute one value from
others; the loop, broadcasting and the output belong to `Elemwise`. Here is a new one,
`x / (1 + |x|)`:

```python
from pytensor.scalar.basic import UnaryScalarOp, upgrade_to_float
from pytensor.tensor.elemwise import Elemwise


class Softsign(UnaryScalarOp):
    def impl(self, x):
        return x / (1 + abs(x))


softsign = Elemwise(Softsign(upgrade_to_float, name="softsign"))
```

```python
from pytensor.link.llvm.codegen.scalar import nums, register


@register(Softsign)
def _(b, args, out):
    (x,) = nums(b, args, out)
    return x / (1.0 + abs(x))


g = pytensor.function([x], softsign(x), mode=STRICT)
np.testing.assert_allclose(g(np.array([-3.0, 0.0, 1.0])), [-0.75, 0.0, 0.5])
```

`args` are the arguments as `(value, dtype)` pairs and `out` is the output dtype
PyTensor computed. `nums(b, args, out)` converts the arguments to the output dtype and
wraps them as `Num`s, and the emitter returns a `Num` (or a plain value) of that dtype.

A `Num` is a value together with its dtype. Its operators choose the instruction for
the dtype and follow numpy:

| | |
|---|---|
| `x + y`, `x - y`, `x * y`, `-x`, `abs(x)` | for booleans `+` is `or` and `*` is `and` |
| `x / y` | floats only |
| `x // y`, `x % y` | rounding to the floor; integers give 0 for a zero divisor |
| `x < y`, `x <= y`, `x > y`, `x >= y`, `x.eq(y)`, `x.ne(y)` | a `Num` of dtype `bool` |
| `x & y`, `x \| y`, `x ^ y`, `~x` | integers and booleans |
| `cond.select(x, y)` | `x` where the boolean `cond` holds, else `y` |
| `x.minimum(y)`, `x.maximum(y)` | like `numpy.minimum` and `numpy.maximum` |
| `x.astype(dtype)` | like numpy's `astype` |
| `x.value`, `x.dtype`, `x.const_value` | the IR value, the dtype, the Python value of a constant |

Three rules keep mistakes visible:

```python
i = Num(b, probe.params[0], "int64")
with pytest.raises(TypeError, match="convert one with astype"):
    i + i.astype("int32")  # dtypes are never mixed silently
with pytest.raises(TypeError, match="not a Python condition"):
    if i < 3:  # this decision belongs to the generated code
        pass
assert (i == i) is True  # `==` is identity; `i.eq(j)` compares
```

Use `Num` where the code depends on the dtype; elements loaded from an `Array` already
are `Num`s. Index arithmetic is always `i64` and reads fine with the builder directly,
as in the `CumMax` emitter.

## 5. Layouts

Every array has strides, and code that reads the strides from memory is slower than
code that knows them: a loop over a stride of 8 bytes can be vectorized, a loop over "a
stride" can't. The backend therefore tracks, for every value of the graph, what is known
about its strides *before the program runs*. That knowledge is the value's **layout**.

### What a layout says

A `Layout` has one entry per axis:

* `PACKED`: the axis is part of a contiguous block. Its stride follows from the
  itemsize and the lengths of the packed axes inside it, in the memory order `order`.
* an `int`: the stride is exactly this many bytes (0 for a broadcast axis, negative
  for a reversed one),
* `DYNAMIC`: not known; the stride is read when the code runs.

```python
from pytensor.link.llvm.layout import DYNAMIC, PACKED, Layout

c = Layout.c(2)  # C-contiguous: all packed, last axis innermost
unknown = Layout.dynamic(2)
fortran = Layout.make((None, None), order=(1, 0), strides=(PACKED, PACKED))
rows = Layout.make((None, None), order=(0, 1), strides=(DYNAMIC, PACKED))

assert [str(c), str(unknown), str(fortran), str(rows)] == [
    "C",
    "[?, ?]",
    "[P0, P1]",
    "[?, P0]",
]
```

The printed form is what program dumps show: `C`, or one entry per axis, where `?` is
dynamic, a number is a static stride in bytes, and `P0`, `P1`, ... are the packed axes
counted from the innermost. `[P0, P1]` is Fortran order: axis 0 is the innermost.
`[?, P0]` describes rows that are contiguous while nothing is known about the step from
row to row. The first argument of `Layout.make` is the static shape (`None` for unknown
lengths); an axis of static length 1 never matters for addressing and is always packed.

With lengths, a layout gives strides:

```python
assert c.static_stride(1, (3, 4), 8) == 8  # itemsize
assert c.static_stride(0, (3, 4), 8) == 32  # 4 elements of 8 bytes
assert fortran.static_stride(1, (3, 4), 8) == 24
assert c.static_stride(0, (None, None), 8) is None  # depends on an unknown length

a = np.zeros((3, 4))
assert c.satisfied_by(a.shape, a.strides, 8)
assert not c.satisfied_by(a.T.shape, a.T.strides, 8)
assert fortran.satisfied_by(a.T.shape, a.T.strides, 8)
```

A layout is a claim the generated code relies on. A value claimed `PACKED` is read
without looking at its strides, so a wrong claim gives wrong results, not an error.

### Where layouts come from

* Memory the backend allocates itself (outputs of tasks, temporaries) is C-contiguous,
  or in the order the op asked for.
* A view gets its layout from its input's: the transpose of a C-contiguous matrix is
  Fortran-contiguous.
* Graph inputs are `DYNAMIC`, unless the function was compiled with
  `input_layout="c"`. Then they are C-contiguous: calls from Python copy arrays that
  aren't, and the C API rejects them.
* Outputs of numba kernels are `DYNAMIC`.

`infer_layouts` shows what the backend derives for a graph:

```python
from pytensor.graph.fg import FunctionGraph
from pytensor.link.llvm.layout import infer_layouts

A = pt.matrix("A")
outs = [A.T, A[1], A[:, ::2], A[::-1], pt.exp(A), pt.exp(A.T)]
graph = FunctionGraph([A], outs, clone=False)


def layouts(input_layouts):
    found = infer_layouts(graph, input_layouts)
    return [str(found[o]) for o in outs]


assert layouts(None) == ["[?, ?]", "[?]", "[?, ?]", "[?, ?]", "C", "C"]
assert layouts([Layout.c(2)]) == [
    "[P0, P1]",
    "C",
    "[?, 16]",
    "[?, P0]",
    "C",
    "[P0, P1]",
]
```

| | input unknown | input C-contiguous |
|---|---|---|
| `A.T` | `[?, ?]` | `[P0, P1]`: Fortran order |
| `A[1]` | `[?]` | `C`: a row |
| `A[:, ::2]` | `[?, ?]` | `[?, 16]`: every second double; rows no longer packed |
| `A[::-1]` | `[?, ?]` | `[?, P0]`: rows contiguous, row step unknown (it depends on a length) |
| `pt.exp(A)` | `C` | `C`: new memory |
| `pt.exp(A.T)` | `C` | `[P0, P1]`: `Elemwise` allocates in the memory order of its input |

Views of an unknown input stay unknown, and anything that allocates starts fresh. This
is why `input_layout="c"` pays off for functions whose inputs are always contiguous.

### What a layout does for an emitter

Nothing has to be done to benefit. The planner generates each task's code for the
layouts of its inputs, and what the layout fixes arrives in the emitter as a constant:
`x.strides[0].const_int` is `8` and not `None`, and `x.load(b, i)` then addresses its
element with a constant step. A packed stride is not even passed to the generated function:

```python
from pytensor.link.llvm.codegen.function import build_node_function

node = cummax(x).owner
any_strides = build_node_function(ir.Module("any"), node, "node")
contiguous = build_node_function(ir.Module("c"), node, "node", in_layouts=[Layout.c(1)])

assert any_strides.signature.inputs[0].stride_axes == (
    0,
)  # the stride of axis 0 is an argument
assert contiguous.signature.inputs[0].stride_axes == ()  # nothing to pass
```

The dump of a program shows, in braces, the input layouts each task was generated for:

```python
C_INPUTS = STRICT.clone(link_kwargs={"input_layout": "c"})
h = pytensor.function([A], cummax(A[:, ::2][1]), mode=C_INPUTS)
print(h.maker.linker.program.dump())
np.testing.assert_array_equal(
    h(np.array([[0.0, 9.0, 0.0], [1.0, 7.0, 0.5]])), [1.0, 1.0]
)
```

```text
  0    (%0) -> (%1)  Subtensor{i, ::step} {C}
  1    (%1) -> (%2)  CumMax {[16]}  [release %1]
```

The indexing takes every second element of a row of a C-contiguous matrix, so `CumMax`
was generated for a vector with a stride of 16 bytes. Nothing in its emitter
mentions that.

An emitter may also decide by the layout while generating, for instance to use a
simpler loop when its inputs are contiguous:

```py
if x.layout.is_c(x.static_shape()):
    ...  # one flat loop over all elements
else:
    ...  # a loop nest following the strides
```

For loops that visit every element of several arrays, `for_each` in `codegen/emit.py`
does exactly this: it picks a flat loop when the layouts allow it, a nest when they
rule it out, and generates both with a check when they leave it open.

### Views

Whether an output may share memory with an input is not something the implementation
declares. It is a property of the op: PyTensor's `view_map`. `view_map = {0: [0]}` says
output 0 may be a view of input 0, and the backend plans accordingly: it keeps the
input's memory alive as long as the output is used, and it never hands the op memory
to write that output into.

Here is an op with a view: a vector backwards.

```python
class Reverse(Op):
    """A vector backwards, as a view."""

    __props__ = ()
    view_map = {0: [0]}

    def make_node(self, x):
        x = pt.as_tensor_variable(x)
        if x.type.ndim != 1:
            raise TypeError("Reverse needs a vector")
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        output_storage[0][0] = inputs[0][::-1]


reverse = Reverse()
```

An output that may be a view is filled in one of two ways, and the emitter chooses
freely, even when the code runs:

* `outputs[k].set_view(array)` hands over a view of an input the `view_map` lists.
  `array` must come from that input: build it with `x.view(data, shape, strides,
  layout)`, or with `x.at(...)`, which remember where they come from.
* `outputs[k].allocate(shape)` allocates instead, as for any other output. An op
  that only sometimes can return a view (a reshape of a non-contiguous array) does
  this in a branch.

```python
@llvm_emit.register(Reverse)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    (x,) = inputs
    (n,) = x.shape
    (stride,) = x.strides
    # the last element (the first, for an empty vector: stay inside the array)
    last = b.select(b.sgt(n, 0), b.mul(b.sub(n, 1), stride, nsw=True), 0)
    outputs[0].set_view(x.view(b.gep_bytes(x.data, last), [n], [b.neg(stride)]))


r = pytensor.function([x], cummax(reverse(x)), mode=STRICT)
np.testing.assert_array_equal(r(values), [4.0, 5.0, 5.0, 5.0, 5.0])
```

Mistakes are caught while generating. An output the `view_map` doesn't list can't be a
view, and inputs can't be written:

```python
class Careless(Op):
    __props__ = ()

    def make_node(self, x):
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        output_storage[0][0] = inputs[0].copy()


@llvm_emit.register(Careless)
def _(op, node, cg, inputs, outputs):
    outputs[0].set_view(inputs[0])  # no view_map


with pytest.raises(ValueError, match="this output can't be a view"):
    pytensor.function([x], Careless()(x), mode=STRICT)


@llvm_emit.register(Careless)
def _(op, node, cg, inputs, outputs):
    inputs[0].store(cg.b, 0.0, 0)  # an input


with pytest.raises(ValueError, match="storing into an input"):
    pytensor.function([x], Careless()(x), mode=STRICT)
```

These checks cover the array helpers, not raw pointers: the backend trusts an
implementation to do what its op declares.

### Writing over an input

The other thing an op says about memory is `destroy_map`: `{0: [0]}` means computing
output 0 invalidates input 0. PyTensor's inplace rewrites put such ops into a graph
where it is safe, and they are an offer, not an obligation: an implementation that
ignores `destroy_map` and produces new memory is correct. That is what happens unless
the emitter opts in.

Many results are "the input, with a few changes". `outputs[k].overwrite(cg, x)` gives
memory that already holds the values of `x`: a copy, or, in place, `x` itself.

```python
class ZeroFirst(Op):
    """A vector with its first element set to zero."""

    __props__ = ("inplace",)

    def __init__(self, inplace=False):
        self.inplace = inplace
        if inplace:
            self.destroy_map = {0: [0]}

    def make_node(self, x):
        x = pt.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0] if self.inplace else inputs[0].copy()
        x[:1] = 0
        output_storage[0][0] = x


@llvm_emit.register(ZeroFirst)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    (x,) = inputs
    out = outputs[0].overwrite(cg, x)
    with b.if_(b.sgt(out.shape[0], 0)):
        out.store(b, 0.0, 0)


z = pytensor.function([x], ZeroFirst()(x), mode=STRICT)
np.testing.assert_array_equal(z(values), [0.0, *values[1:]])
```

This emitter is already right for both versions of the op, and both copy. To let the
in-place version skip the copy, say which output can be written over which input:

```python
from pytensor.link.llvm.codegen.function import llvm_inplace


@llvm_inplace.register(ZeroFirst)
def _(op, node):
    return {0: 0} if op.inplace else {}
```

The planner then decides, node by node. It lets a node overwrite an input when the
graph was checked for it (it has PyTensor's `DestroyHandler`, as every graph that went
through the inplace rewrites has) and nothing else can see the memory: the value is
not a constant and shares no memory with the node's other inputs. When it does,
`overwrite` returns `x`'s memory and no copy is made; `allocate` would return the same
memory, still holding `x`'s values, for emitters that read each element before writing
it (Elemwise does that). Otherwise the output is new memory as before, so the emitter
never needs to know.

A rewrite would normally put `ZeroFirst(inplace=True)` into the graph. Here we build
such a graph by hand:

```python
from pytensor.graph.destroyhandler import DestroyHandler
from pytensor.link.llvm.program import Options, build_program

graph = FunctionGraph([x], [ZeroFirst(inplace=True)(cummax(x))], clone=False)
graph.attach_feature(DestroyHandler())
program, stats = build_program(graph, Options(numba_fallback=False))
print(stats.inplace)
np.testing.assert_array_equal(
    program(values)[0], [0.0, *np.maximum.accumulate(values)[1:]]
)
```

```text
['ZeroFirst{inplace=True}']
```

Two things to keep in mind when opting in:

* `overwrite` takes the destroyed input and nothing else (anything else is an error
  when the code is generated), and the result must be correct when the output and that
  input are the same memory. Reading other inputs is always safe: the planner doesn't
  grant in-place work when they might overlap.
* Only opt in for pairs in the op's `destroy_map`; others are ignored.

### Declaring the layout of an output

`llvm_layout(op, node, in_layouts)` returns the layout of each output, whether it is a
view or new memory. The defaults are C-contiguous for new memory, which was right for
`CumMax`, and `DYNAMIC` for an output that may be a view, which is always true and
never fast. The `CumMax` after `Reverse` above was generated for an unknown stride.

An op that returns a view should say what it knows. The stride of a reversed vector is
minus the stride of the input, so if the one is known, so is the other:

```python
from pytensor.link.llvm.codegen.function import llvm_layout
from pytensor.link.llvm.layout import itemsize_of, shape_of


def reverse_layout(node, x: Layout) -> Layout:
    var = node.inputs[0]
    stride = x.static_stride(0, shape_of(var), itemsize_of(var))
    return Layout.make(
        shape_of(node.outputs[0]), (0,), (DYNAMIC if stride is None else -stride,)
    )


@llvm_layout.register(Reverse)
def _(op, node, in_layouts):
    return [reverse_layout(node, in_layouts[0])]
```

A declared layout is binding, so the emitter must now produce it: the view it hands
over carries the same layout.

```python
@llvm_emit.register(Reverse)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    (x,) = inputs
    (n,) = x.shape
    (stride,) = x.strides
    last = b.select(b.sgt(n, 0), b.mul(b.sub(n, 1), stride, nsw=True), 0)
    layout = reverse_layout(node, x.layout)
    outputs[0].set_view(x.view(b.gep_bytes(x.data, last), [n], [b.neg(stride)], layout))
```

The layout function is shared by the declaration and the emitter, which is how the
backend's own view ops are written (`subtensor_layout`, `dimshuffle_layout` in
`layout.py`).

```python
CHECKED = STRICT.clone(link_kwargs={"input_layout": "c", "check_layouts": True})
r = pytensor.function([x], cummax(reverse(x)), mode=CHECKED)
np.testing.assert_array_equal(r(values), [4.0, 5.0, 5.0, 5.0, 5.0])
np.testing.assert_array_equal(
    r(values[::2]), [4.0, 4.0, 4.0]
)  # not contiguous: copied on the way in
print(r.maker.linker.program.dump())
```

```text
  0    (%0) -> (%1)  Reverse {C}
  1    (%1) -> (%2)  CumMax {[-8]}  [release %1]
```

`CumMax` now runs backwards over the input with a constant stride of -8 bytes, with no
copy in between.

`llvm_layout` is also how an op asks for new memory in another order than C: `Elemwise`
declares the memory order of its inputs, so that an operation on transposed matrices
gives a transposed result instead of a strided loop. For new memory the layout must be
packed.

Two things guard against wrong declarations:

* `set_view` compares the layout of the array it is given with the declaration, while
  generating. An emitter that forgets the layout is caught:

  ```python
  class Forgetful(Reverse):
      pass


  @llvm_emit.register(Forgetful)
  def _(op, node, cg, inputs, outputs):
      (x,) = inputs
      outputs[0].set_view(x.view(x.data, x.shape, x.strides))  # no layout: all DYNAMIC


  with pytest.raises(ValueError, match="llvm_layout declared"):
      pytensor.function([x], Forgetful()(x), mode=CHECKED)
  ```

* The linker option `check_layouts=True` (or `PYTENSOR_LLVM_CHECK_LAYOUTS=1`) makes
  every generated task verify, when it runs, that its arguments have the layouts it
  was generated for. It costs a few comparisons per task. Use it in the tests of any
  op with its own `llvm_layout`: it catches a layout that is declared consistently
  but is not what the code produces.

## 6. Temporary memory

Some ops need memory besides their outputs while they run. `cg.temporary(dtype, shape)`
returns an `Array` for that: C-contiguous, uninitialized, and gone when the task ends.

The example computes the mean of every column of a matrix. To read the matrix in
memory order it goes row by row, so it needs one running sum per column, and it keeps
those in double precision whatever the dtype of the matrix:

```python
class ColumnMeans(Op):
    """The mean of every column of a matrix."""

    __props__ = ()

    def make_node(self, x):
        x = pt.as_tensor_variable(x)
        if x.type.ndim != 2 or x.type.dtype not in ("float32", "float64"):
            raise TypeError("ColumnMeans needs a float matrix")
        return Apply(
            self, [x], [pt.tensor(dtype=x.type.dtype, shape=(x.type.shape[1],))]
        )

    def perform(self, node, inputs, output_storage):
        (x,) = inputs
        output_storage[0][0] = x.mean(axis=0, dtype="float64").astype(x.dtype)


column_means = ColumnMeans()


@llvm_emit.register(ColumnMeans)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    (x,) = inputs
    rows, columns = x.shape
    out = outputs[0].allocate([columns])
    sums = cg.temporary("float64", [columns])
    with b.loop(columns) as j:
        sums.store(b, 0.0, j)
    with b.loop(rows) as i:
        row = x.at(b, i)
        with b.loop(columns) as j:
            sums.store(b, sums.load(b, j) + row.load(b, j).astype("float64"), j)
    count = Num(b, rows, "int64").astype("float64")
    with b.loop(columns) as j:
        out.store(b, (sums.load(b, j) / count).astype(out.dtype), j)
```

`x.at(b, i)` is row `i` as a vector, so the inner loop reads `row.load(b, j)`; the
address of the row is computed once per row, not once per element.

```python
single = pt.matrix("M", dtype="float32")
m = pytensor.function(
    [single], [column_means(single), column_means(single.T[::2])], mode=STRICT
)
data = (rng.normal(size=(1000, 3)) + 1e4).astype("float32")
by_column, of_a_view = m(data)
np.testing.assert_allclose(by_column, data.mean(axis=0, dtype="float64"), rtol=1e-6)
np.testing.assert_allclose(
    of_a_view, data.T[::2].mean(axis=0, dtype="float64"), rtol=1e-6
)
```

Where the memory comes from depends on its size, and the emitter doesn't choose:

* A small temporary of static size (up to 1 KiB; here, a matrix whose number of
  columns is known and small) is stack memory of the generated function. LLVM sees
  all of it and often keeps it in registers.
* Anything else is *scratch memory* of the context, valid until the task ends.

Two rules follow:

* Ask for temporaries before a loop, not inside it. Every request for scratch is a new
  piece of memory, and it is only given back when the task ends.
* A temporary is uninitialized. `ColumnMeans` zeroes its sums first.

`cg.temp_memory(nbytes)` is the same without the `Array`: a pointer to that many bytes,
64-byte aligned.

## 7. Random variables

Random numbers don't come from the generator in the graph. Every session (one caller of
a compiled function) has one random state, and generated code draws from it through the
context. The generator input and the next-generator output of a `RandomVariable` node
have no value at all in the backend. Seeding is done on the program:

```py
program.seed(123)
```

The draws are the same for every number of threads, so a seeded function is
reproducible.

`codegen/random.py` implements `RandomVariable` once, for all distributions: it works
out the shape of the draws, broadcasts the parameters, checks them and loops. A
distribution only supplies the function for one draw, as a `Sampler`:

```py
Sampler(base, draw, param_dtypes, checks)
```

* `base` is `"uniform"` or `"normal"` for distributions that are a transformation of
  uniform numbers in `[0, 1)` or of standard normal numbers. Those are drawn for the
  whole array in one call, and `draw` then turns one of them into one draw. The loop
  doing that has no calls in it and can be vectorized.
* `base=None` is for distributions that need their own sampler. `draw` then calls the
  builder, `b.rng(ir.Rng.Gamma, cg.ctx, [shape])`; the runtime has samplers for
  uniform, normal, gamma, beta, Poisson and binomial numbers.
* `draw(b, cg, z, params)` returns the value of one draw: `z` is the base number (or
  `None`), `params` are the parameters as IR values of `param_dtypes`.
* `checks` has one entry per parameter: `None`, or `(kind, bound, message)` with kind
  `"ge"`, `"gt"` or `"unit"` (between 0 and 1).

The Gumbel distribution is not implemented:

```python
import pytensor.tensor.random.basic as rb

generator = pytensor.shared(np.random.default_rng(0))
draws = rb.gumbel(1.0, 2.0, size=(100_000,), rng=generator)
with pytest.raises(
    NotImplementedError, match="no implementation of the random variable gumbel"
):
    pytensor.function([], draws, mode=STRICT)
```

A Gumbel number is `loc - scale * log(e)` for a standard exponential number `e`, and
`e` is `-log(1 - u)` for a uniform `u`:

```python
from pytensor.link.llvm.codegen.random import SAMPLERS, Sampler


def gumbel(b, cg, u, params):
    loc, scale = params
    exponential = b.fneg(b.libm(ir.Libm.Log1p, [b.fneg(u)]))  # -log(1 - u)
    return b.fsub(loc, b.fmul(scale, b.intrinsic(ir.Intrinsic.Log, [exponential])))


SAMPLERS[rb.GumbelRV] = Sampler(
    "uniform", gumbel, ["float64", "float64"], [None, ("ge", 0.0, "scale < 0")]
)
```

The parameters arrive as doubles, so the builder's float instructions are used
directly; there is no dtype to dispatch on.

```python
sample = pytensor.function([], draws, mode=STRICT)
program = sample.maker.linker.program

program.seed(1)
first = sample()
np.testing.assert_allclose(first.mean(), 1.0 + 2.0 * np.euler_gamma, atol=0.05)
np.testing.assert_allclose(first.var(), 4.0 * np.pi**2 / 6, rtol=0.05)

assert not np.array_equal(sample(), first)  # the state advances
program.seed(1)
np.testing.assert_array_equal(sample(), first)  # and can be set again
```

A failed check is an error when the function runs:

```python
scale = pt.scalar("scale")
checked = pytensor.function(
    [scale], rb.gumbel(0.0, scale, size=(3,), rng=generator), mode=STRICT
)
with pytest.raises(Exception, match="scale < 0"):
    checked(-1.0)
```

Only random variables whose draws are scalars (`ndim_supp == 0`) can be implemented
this way. The backend has no implementation for the others (the multivariate normal,
for one), and random variables never fall back to numba, whose generators are a
different state: compiling a graph with one raises `NotImplementedError`.

## 8. Estimating the work

**`llvm_work(op, node)`** estimates the work as a number of elements times a cost per
element, in units of one float addition. The planner uses it to decide whether a loop
is worth running in parallel and whether a node is small enough to be compiled together
with its neighbors. The default is one unit per output element; a comparison and a
store per element is about two:

```python
from pytensor.link.llvm.work import Work, llvm_work


@llvm_work.register(CumMax)
def _(op, node):
    return Work(2.0, node.inputs[0].type.shape)


assert llvm_work(cummax, cummax(static).owner).total == 8.0
```

## 9. Putting it into the backend

The registrations above work from anywhere, which is how this tutorial runs. An op that
belongs to PyTensor gets its emitter in a module of `pytensor/link/llvm/codegen/`:

1. Write the emitter in the module that fits (`basic.py`, `shape.py`, `extra.py`, ...),
   or in a new one. A new module must be added to the imports at the top of
   `codegen/__init__.py`, which is what registers its emitters.
2. Scalar ops go into `codegen/scalar.py`, with their cost in `SCALAR_COSTS` in
   `work.py`; distributions into the `SAMPLERS` table of `codegen/random.py`.
3. Register `llvm_work` in `work.py` if one unit per output element is far off, and
   `llvm_layout` next to the emitter if an output is a view (its layout function goes
   into `layout.py`), and `llvm_inplace` if the emitter can write over an input its op
   destroys.
4. Add a test to `tests/link/llvm/test_ops.py` that compiles with
   `numba_fallback=False` and compares with numpy, over the dtypes, with a strided
   input and with static and unknown shapes; for a view, also with
   `input_layout="c"` and `check_layouts=True`.

Things that commonly go wrong:

* **A Python `if` on a value.** It is always true. Use `b.select`, `b.if_`, or a `Num`,
  which raises.
* **Two literals.** `b.select(cond, 0, 1)` has no type to give them; make one a
  constant (`b.i64(0)`).
* **`b.load` on a tensor element.** Use `array.load` and `array.store`, which handle
  booleans and strides.
* **Assuming contiguous inputs.** Always go through the array (`load`, `store`, `at`),
  which follows the strides. The planner passes the strides it knows as constants, so
  this costs nothing when the input is contiguous.
* **A temporary inside a loop.** Each request is new memory until the task ends. Ask
  once, before the loop.
* **Writing an input.** The planner assumes inputs are only read. The one way to write
  over an input is through an output: `outputs[k].overwrite(cg, x)`.
* **Declaring more than is true.** A layout is trusted. Declare `DYNAMIC` for whatever
  the op can't guarantee.
* **Leaving an output unset.** Every output must be written, or set with `set_view` for
  a view; the backend raises when a node function ends with an output never set.
