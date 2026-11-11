# Adding an op to the LLVM backend

This tutorial implements a new op for the LLVM backend, start to finish. It first
introduces the interface the implementation is written with, `pytensor_llvm.ir`, and
then uses it for a small tensor op and a scalar op, and explains layouts: what the
backend knows about the strides of each array, and how an op takes part in that. `DESIGN.md` explains why things are
the way they are; this file only shows how to use them.

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
with pytest.raises(NotImplementedError, match="No LLVM backend implementation for CumMax"):
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
* `outputs` has one entry per output, to be filled. For a tensor in new memory,
  `output_array(out, shape)` returns the `Array` to write: either memory the caller
  provides (static shapes), or memory allocated here with that shape. The emitter
  doesn't need to know which.

Elements are read and written with three helpers from `codegen/emit.py`:

* `advance(b, ptr, i, stride, dtype)` is the pointer `i` strides after `ptr`,
* `load(b, dtype, ptr)` and `store(b, dtype, value, ptr)` read and write one element.
  Use them and not `b.load`: a bool is a byte in memory and an `i1` in a register.

### The emitter

```python
from pytensor.link.llvm.codegen.elemwise import output_array
from pytensor.link.llvm.codegen.emit import advance, load, store
from pytensor.link.llvm.codegen.function import llvm_emit
from pytensor.link.llvm.codegen.num import Num


@llvm_emit.register(CumMax)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    (x,) = inputs
    (n,) = x.shape
    out = output_array(outputs[0], x.shape)
    dtype = x.dtype

    def element(array, i):
        return advance(b, array.data, i, array.strides[0], dtype)

    def step(i, best):
        """out[i] = max(best, x[i]); returns the new maximum."""
        value = Num(b, load(b, dtype, element(x, i)), dtype)
        if best is not None:
            value = Num(b, best, dtype).maximum(value)
        store(b, dtype, value.value, element(out, i))
        return value.value

    with b.if_(b.sgt(n, 0)):
        first = step(b.i64(0), None)
        b.fold(b.sub(n, 1), [first], lambda i, best: step(b.add(i, 1), best))
```

Line by line:

* `output_array` gets the output's memory, of the shape of `x`.
* `element` addresses element `i` of an array through its stride. The input may be any
  view (`x[::-2]`), so the stride is never assumed. Where the planner knows the stride,
  it is a constant here and the multiplication disappears (section 5).
* `step` generates the code for one element. It is plain Python that is called twice
  and so generates code twice: once for the first element, which has no maximum before
  it (`if best is not None` is decided while generating, which is fine: `best` is
  either a value or Python's `None`), and once inside the loop.
* The maximum itself depends on the dtype: an intrinsic for floats, a signed or an
  unsigned compare for integers, an `or` for bools. `Num(b, value, dtype).maximum(...)`
  picks; `.value` is the IR value again.
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

Use `Num` where the code depends on the dtype. Index arithmetic is always `i64` and
reads fine with the builder directly, as in the `CumMax` emitter.

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

assert [str(c), str(unknown), str(fortran), str(rows)] == ["C", "[?, ?]", "[P0, P1]", "[?, P0]"]
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
assert layouts([Layout.c(2)]) == ["[P0, P1]", "C", "[?, 16]", "[?, P0]", "C", "[P0, P1]"]
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
`x.strides[0].const_int` is `8` and not `None`, and `advance` then addresses elements
with a constant step. A packed stride is not even passed to the generated function:

```python
from pytensor.link.llvm.codegen.function import build_node_function

node = cummax(x).owner
any_strides = build_node_function(ir.Module("any"), node, "node")
contiguous = build_node_function(ir.Module("c"), node, "node", in_layouts=[Layout.c(1)])

assert any_strides.signature.inputs[0].stride_axes == (0,)  # the stride of axis 0 is an argument
assert contiguous.signature.inputs[0].stride_axes == ()  # nothing to pass
```

The dump of a program shows, in braces, the input layouts each task was generated for:

```python
C_INPUTS = STRICT.clone(link_kwargs={"input_layout": "c"})
h = pytensor.function([A], cummax(A[:, ::2][1]), mode=C_INPUTS)
print(h.maker.linker.program.dump())
np.testing.assert_array_equal(h(np.array([[0.0, 9.0, 0.0], [1.0, 7.0, 0.5]])), [1.0, 1.0])
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

### Declaring the layout of an output

`llvm_outputs(op, node, in_layouts)` declares how each output of a node comes to be,
and with which layout. The default is right for ops that write new memory, like
`CumMax`: the output is C-contiguous memory, provided by the caller if the shape is
static and allocated by the task otherwise. An output listed in the op's `view_map`
is declared a view with a `DYNAMIC` layout, which is always true and never fast.

An op that returns a view should say what it knows. Here is one: a vector backwards.

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

The stride of the result is minus the stride of the input. If the input's stride is
known, so is the output's:

```python
from pytensor.link.llvm.codegen.function import known_layouts, llvm_outputs, view_output
from pytensor.link.llvm.layout import itemsize_of, shape_of


def reverse_layout(node, x: Layout) -> Layout:
    var = node.inputs[0]
    stride = x.static_stride(0, shape_of(var), itemsize_of(var))
    return Layout.make(shape_of(node.outputs[0]), (0,), (DYNAMIC if stride is None else -stride,))


@llvm_outputs.register(Reverse)
def _(op, node, in_layouts):
    (x,) = known_layouts(node, in_layouts)  # unknown layouts as DYNAMIC
    return [view_output(node.outputs[0], reverse_layout(node, x))]
```

`view_output(var, layout, view_of=0)` declares a view of input `view_of`. Its
counterpart for new memory is `new_output(var, layout)`, for an op that wants its
output in another order than C.

The emitter must produce what was declared. It builds the `Array` of the view, with
the same layout, and hands it over with `set_view`:

```python
from pytensor.link.llvm.codegen.emit import Array, copy


@llvm_emit.register(Reverse)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    (x,) = inputs
    (n,) = x.shape
    (stride,) = x.strides
    # the last element (the first, for an empty vector: stay inside the array)
    last = b.select(b.sgt(n, 0), b.mul(b.sub(n, 1), stride, nsw=True), 0)
    view = Array(x.dtype, b.gep_bytes(x.data, last), [n], [b.neg(stride)], reverse_layout(node, x.layout))
    out = outputs[0]
    if hasattr(out, "set_view"):
        out.set_view(view)
    else:  # asked to write into given memory (inside a compiled unit)
        copy(cg, view, out.array)
```

The layout function is shared by the declaration and the emitter, which is how the
backend's own view ops are written (`subtensor_layout`, `dimshuffle_layout` in
`layout.py`).

```python
CHECKED = STRICT.clone(link_kwargs={"input_layout": "c", "check_layouts": True})
r = pytensor.function([x], cummax(reverse(x)), mode=CHECKED)
np.testing.assert_array_equal(r(values), [4.0, 5.0, 5.0, 5.0, 5.0])
np.testing.assert_array_equal(r(values[::2]), [4.0, 4.0, 4.0])  # not contiguous: copied on the way in
print(r.maker.linker.program.dump())
```

```text
  0    (%0) -> (%1)  Reverse {C}
  1    (%1) -> (%2)  CumMax {[-8]}  [release %1]
```

`CumMax` now runs backwards over the input with a constant stride of -8 bytes, with no
copy in between.

Two things guard against wrong declarations:

* `set_view` compares the layout of the array it is given with the declaration, while
  generating. An emitter that forgets the layout is caught:

  ```python
  class Forgetful(Reverse):
      pass


  @llvm_emit.register(Forgetful)
  def _(op, node, cg, inputs, outputs):
      (x,) = inputs
      outputs[0].set_view(Array(x.dtype, x.data, x.shape, x.strides))  # no layout: all DYNAMIC


  with pytest.raises(ValueError, match="llvm_outputs declared"):
      pytensor.function([x], Forgetful()(x), mode=CHECKED)
  ```

* The linker option `check_layouts=True` (or `PYTENSOR_LLVM_CHECK_LAYOUTS=1`) makes
  every generated task verify, when it runs, that its arguments have the layouts it
  was generated for. It costs a few comparisons per task. Use it in the tests of any
  op with its own `llvm_outputs`: it catches a layout that is declared consistently
  but is not what the code produces.

## 6. Estimating the work

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

## 7. Putting it into the backend

The registrations above work from anywhere, which is how this tutorial runs. An op that
belongs to PyTensor gets its emitter in a module of `pytensor/link/llvm/codegen/`:

1. Write the emitter in the module that fits (`basic.py`, `shape.py`, `extra.py`, ...),
   or in a new one. A new module must be added to the imports at the top of
   `codegen/__init__.py`, which is what registers its emitters.
2. Scalar ops go into `codegen/scalar.py`, with their cost in `SCALAR_COSTS` in
   `work.py`.
3. Register `llvm_work` in `work.py` if one unit per output element is far off, and
   `llvm_outputs` next to the emitter if an output is a view (its layout function goes
   into `layout.py`).
4. Add a test to `tests/link/llvm/test_ops.py` that compiles with
   `numba_fallback=False` and compares with numpy, over the dtypes, with a strided
   input and with static and unknown shapes; for a view, also with
   `input_layout="c"` and `check_layouts=True`.

Things that commonly go wrong:

* **A Python `if` on a value.** It is always true. Use `b.select`, `b.if_`, or a `Num`,
  which raises.
* **Two literals.** `b.select(cond, 0, 1)` has no type to give them; make one a
  constant (`b.i64(0)`).
* **`b.load` on a tensor element.** Use `load` and `store` from `codegen/emit.py`, which
  handle booleans.
* **Assuming contiguous inputs.** Always go through the strides (`advance`). The
  planner passes the strides it knows as constants, so this costs nothing when the
  input is contiguous.
* **Writing an input.** The planner assumes inputs are only read.
* **Declaring more than is true.** A layout is trusted. Declare `DYNAMIC` for whatever
  the op can't guarantee.
* **Leaving an output unset.** Every output must be written, or set with `set_view` for
  a view; the backend raises when a node function ends with an output never set.
