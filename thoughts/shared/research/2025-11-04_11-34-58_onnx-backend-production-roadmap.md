---
date: 2025-11-04T11:34:58Z
researcher: Claude
git_commit: b556aec588e2f55a347e5e30ed955d3a611f8a20
branch: onnx-backend
repository: pytensor
topic: "ONNX Backend Production Roadmap: Core Operations Focus"
tags: [research, onnx, backend, implementation, roadmap, core-operations]
status: complete
last_updated: 2025-11-04
last_updated_by: Claude
---

# Research: ONNX Backend Production Roadmap - Core Operations Focus

**Date**: 2025-11-04T11:34:58Z
**Researcher**: Claude
**Git Commit**: b556aec588e2f55a347e5e30ed955d3a611f8a20
**Branch**: onnx-backend
**Repository**: pytensor

## Research Question

What operations should a production ONNX backend support for PyTensor, focusing on core operations (not CNN-specific operations like Conv2D, MaxPool, BatchNorm)?

## Executive Summary

**Current State**: The ONNX backend **does not exist** in this repository. Only planning documents exist, which were created for a YOLO demo and focused heavily on CNN operations.

**Key Finding**: For a production ONNX backend supporting general PyTensor code, you need approximately **70-100 core operations**, based on the JAX backend's coverage of 99 ops.

**Recommended Approach**: Implement operations in 5 tiers based on usage frequency and dependencies:
- **Tier 1 (20 ops)**: Infrastructure + Elemwise framework - enables basic computation
- **Tier 2 (15 ops)**: Shape manipulation - enables tensor reshaping and slicing
- **Tier 3 (16 ops)**: Reductions & aggregations - enables statistical operations
- **Tier 4 (20 ops)**: Linear algebra - enables matrix operations
- **Tier 5 (43 ops)**: Advanced operations - special functions, control flow

**Timeline Estimate**: 6-10 weeks for full production coverage (4-6 weeks for Tiers 1-3)

---

## Implementation Progress Tracker

### Overall Progress: 0/114 operations (0%)

| Tier | Operations | Status | Progress |
|------|-----------|--------|----------|
| **Tier 1** | 20 ops | Not Started | 0/20 (0%) |
| **Tier 2** | 15 ops | Not Started | 0/15 (0%) |
| **Tier 3** | 16 ops | Not Started | 0/16 (0%) |
| **Tier 4** | 20 ops | Not Started | 0/20 (0%) |
| **Tier 5** | 43 ops | Not Started | 0/43 (0%) |

**Note**: Update this table manually as you check off operations in the detailed tier sections below.

---

## Detailed Findings

### 1. Current ONNX Backend Status

**Implementation Status**: **NONE - Does Not Exist**

The repository contains only:
- ✅ Planning documents in `/thoughts/shared/plans/`
- ✅ Research documents in `/thoughts/shared/research/`
- ❌ **No implementation code** in `pytensor/link/onnx/` (directory doesn't exist)
- ❌ **No tests** in `tests/link/onnx/` (directory doesn't exist)

**What the Plans Describe**:
The existing plans (particularly `onnx-backend-implementation.md`) describe:
1. A demo-focused implementation targeting YOLO11n
2. Heavy emphasis on CNN operations (Conv2D, MaxPool, BatchNorm, Resize)
3. A 5-phase implementation plan with ~30-40 operations
4. WebAssembly browser deployment as the target

**Why This Differs from Production Needs**:
- Demo was CNN-specific (neural network inference in browser)
- Production needs general PyTensor computation support
- Demo focused on inference; production may need training support
- Demo prioritized visual operations; production needs core math/linalg

---

### 2. PyTensor Core Operations Catalog

Based on analysis of `pytensor/tensor/`, here are the core operation categories:

#### 2.1 Basic Tensor Operations (~25 ops)
**File**: `pytensor/tensor/basic.py`

**Key Operations**:
- **Allocation**: `Alloc`, `AllocEmpty`, `MakeVector`, `ARange`, `Eye`, `Tri`
- **Joining/Splitting**: `Join`, `Split`, `Concatenate`, `Stack`
- **Indexing**: `Subtensor`, `IncSubtensor`, `AdvancedSubtensor`, `AdvancedIncSubtensor`
- **Conversion**: `TensorFromScalar`, `ScalarFromTensor`
- **Utility**: `ExtractDiag`, `Nonzero`, `Default`, `Choose`, `PermuteRowElements`

**Functions** (commonly used):
```python
# Creation
zeros, ones, empty, full, eye, identity, arange
zeros_like, ones_like, empty_like, full_like

# Structure
concatenate, stack, split, join
transpose, flatten, expand_dims, swapaxes, moveaxis

# Conditional
switch, where, choose

# Diagonal
diag, diagonal, extract_diag, trace

# Other
tile, roll, horizontal_stack, vertical_stack
```

#### 2.2 Element-wise Mathematical Operations (~60 ops)
**Files**: `pytensor/tensor/elemwise.py`, `pytensor/scalar/basic.py`, `pytensor/scalar/math.py`

**Categories**:

**Arithmetic** (8 ops):
- `Add`, `Sub`, `Mul`, `TrueDiv`, `IntDiv`, `Mod`, `Pow`, `Reciprocal`

**Unary** (8 ops):
- `Neg`, `Abs`, `Sign`, `Sqrt`, `Sqr`, `Floor`, `Ceil`, `Round`, `Trunc`

**Exponential/Logarithmic** (9 ops):
- `Exp`, `Exp2`, `Expm1`, `Log`, `Log2`, `Log10`, `Log1p`, `Log1mexp`

**Trigonometric** (12 ops):
- `Sin`, `Cos`, `Tan`, `ArcSin`, `ArcCos`, `ArcTan`, `ArcTan2`
- `Sinh`, `Cosh`, `Tanh`, `ArcSinh`, `ArcCosh`, `ArcTanh`

**Comparison** (6 ops):
- `LT` (<), `GT` (>), `LE` (<=), `GE` (>=), `EQ` (==), `NEQ` (!=)

**Logical** (4 ops):
- `AND`, `OR`, `XOR`, `Invert` (NOT)

**Special Checks** (2 ops):
- `IsNan`, `IsInf`

**Min/Max** (3 ops):
- `Maximum`, `Minimum`, `Clip`

**Special Math Functions** (18 ops):
- Error functions: `Erf`, `Erfc`, `Erfcx`, `Erfinv`, `Erfcinv`
- Gamma functions: `Gamma`, `GammaLn`, `GammaInc`, `GammaIncC`, `GammaU`, `GammaL`
- Psi functions: `Psi` (Digamma), `TriGamma`, `PolyGamma`
- Bessel functions: `Jv`, `Iv`, `Ive`, `Kve`
- Activations: `Sigmoid`, `Softplus`
- Beta functions: `BetaInc`, `BetaIncInv`

**Elemwise Framework** (2 meta-ops):
- `Elemwise` - Applies scalar ops to tensors with broadcasting
- `DimShuffle` - Transpose, squeeze, unsqueeze operations

#### 2.3 Shape Operations (~10 ops)
**Files**: `pytensor/tensor/shape.py`, `pytensor/tensor/extra_ops.py`

**Operations**:
- `Shape` - Get shape as tensor
- `Shape_i` - Get specific dimension
- `Reshape` - Reshape array
- `SpecifyShape` - Runtime shape assertion
- `Squeeze` - Remove singleton dimensions
- `BroadcastTo` - Broadcast to shape
- `BroadcastArrays` - Broadcast multiple arrays
- `BroadcastShape` - Compute broadcast shape

**Functions**:
```python
shape, shape_tuple, shape_i
reshape, flatten
specify_shape
squeeze, expand_dims
broadcast_to, broadcast_arrays
shape_padleft, shape_padright, shape_padaxis
```

#### 2.4 Reduction Operations (~10 ops)
**File**: `pytensor/tensor/math.py`

**Operations**:
- `Sum`, `Prod` - Arithmetic reductions
- `Max`, `Min` - Extrema
- `All`, `Any` - Logical reductions
- `Argmax`, `Argmin` - Index of extrema
- `MaxAndArgmax` - Combined operation
- `ProdWithoutZeros` - Special product

**Functions** (derived):
```python
sum, prod, mean, var, std
max, min, all, any
argmax, argmin, max_and_argmax
ptp (peak-to-peak), median
logsumexp, logaddexp
```

#### 2.5 Linear Algebra Operations (~35 ops)
**Files**: `pytensor/tensor/blas.py`, `pytensor/tensor/nlinalg.py`, `pytensor/tensor/slinalg.py`

**BLAS Operations** (6 ops):
- `Gemv` - General matrix-vector product
- `Ger` - Outer product
- `Gemm` - General matrix-matrix product
- `Dot22` - 2D dot product (optimized)
- `Dot22Scalar` - Scaled 2D dot
- `BatchedDot` - Batched matrix multiplication

**General Linear Algebra** (10 ops):
- `Dot`, `MatMul` - Matrix multiplication
- `MatrixInverse` - Matrix inverse
- `MatrixPinv` - Pseudo-inverse
- `Det`, `SLogDet` - Determinants
- `Eig`, `Eigh` - Eigendecomposition
- `SVD` - Singular value decomposition
- `Lstsq` - Least squares
- `TensorInv`, `TensorSolve` - Tensor operations

**Specialized Linear Algebra** (15 ops):
- `Cholesky` - Cholesky decomposition
- `Solve`, `SolveTriangular` - Linear system solving
- `LU`, `LUFactor` - LU decomposition
- `QR` - QR decomposition
- `Eigvalsh` - Hermitian eigenvalues
- `Expm` - Matrix exponential
- `SolveContinuousLyapunov`, `SolveDiscreteLyapunov`, `SolveDiscreteARE` - Control theory
- `BlockDiagonal` - Block diagonal construction

**Functions**:
```python
# Multiplication
dot, matmul, tensordot, outer
matvec, vecmat, vecdot

# Decompositions
svd, qr, lu, cholesky

# Solving
solve, solve_triangular, lstsq

# Properties
det, slogdet, eig, eigh, eigvalsh
inv, pinv, norm

# Advanced
matrix_power, kron, tensorinv, tensorsolve
```

#### 2.6 Extra Operations (~15 ops)
**File**: `pytensor/tensor/extra_ops.py`

**Operations**:
- `CumOp` - Cumulative operations (cumsum, cumprod)
- `Repeat` - Repeat elements
- `Unique` - Find unique elements
- `SearchsortedOp` - Binary search
- `UnravelIndex`, `RavelMultiIndex` - Index conversion
- `FillDiagonal` - Set diagonal values
- `Bincount` - Count occurrences
- `Diff` - Differences

**Functions**:
```python
cumsum, cumprod, diff
bincount, repeat, unique, searchsorted
compress, take, take_along_axis
linspace, logspace, geomspace
```

#### 2.7 Sorting Operations (2 ops)
**File**: `pytensor/tensor/sort.py`

- `SortOp` - Sort arrays
- `ArgSortOp` - Argsort with stability option

#### 2.8 Special Functions (2 ops)
**File**: `pytensor/tensor/special.py`

- `Softmax` - Softmax activation
- `LogSoftmax` - Log-softmax

**Functions**:
```python
softmax, log_softmax, logit
beta, betaln, poch, factorial
```

---

### 3. JAX Backend: Production Baseline

The JAX backend is one of PyTensor's most complete backends with **99 operation implementations** plus **22 random distributions**.

#### 3.1 JAX Operation Coverage by Category

| Category | Count | Files |
|----------|-------|-------|
| **Core Infrastructure** | 7 | `basic.py` |
| **Tensor Creation** | 11 | `tensor_basic.py` |
| **Elemwise Operations** | 6 | `elemwise.py` |
| **Scalar Operations** | 21 | `scalar.py` |
| **Basic Math** | 3 | `math.py` |
| **Dense Linear Algebra** | 8 | `nlinalg.py` |
| **Sparse/Structured Linear Algebra** | 11 | `slinalg.py` |
| **BLAS Operations** | 1 | `blas.py` |
| **Indexing & Slicing** | 7 | `subtensor.py` |
| **Shape Operations** | 5 | `shape.py` |
| **Extra Operations** | 9 | `extra_ops.py` |
| **Sorting** | 2 | `sort.py` |
| **Padding** | 1 | `pad.py` |
| **Random Variables** | 1 + 22 | `random.py` |
| **Scan (Control Flow)** | 1 | `scan.py` |
| **Sparse Operations** | 2 | `sparse.py` |
| **Einsum** | 1 | `einsum.py` |
| **Blockwise** | 1 | `blockwise.py` |
| **Signal Processing** | 1 | `signal/conv.py` |
| **TOTAL** | **99** | 21 files |

#### 3.2 Key Patterns from JAX Backend

**1. Extensible Dispatch System**
```python
@singledispatch
def jax_funcify(op, node=None, storage_map=None, **kwargs):
    """Create a JAX compatible function from a PyTensor Op."""
    raise NotImplementedError(f"No JAX conversion for Op: {op}")

@jax_funcify.register(OpClass)
def jax_funcify_OpClass(op, node, **kwargs):
    # Return function that performs computation
    def op_impl(*inputs):
        return jnp.operation(*inputs)
    return op_impl
```

**2. Static vs Dynamic Value Handling**

Many operations need to distinguish:
- **Compile-time constants**: Embedded in JAX code
- **Runtime values**: Traced by JAX
- **Shape-derived values**: Special case JAX can handle

Example from `ARange`:
```python
if isinstance(arg, Constant):
    constant_args.append(arg.value)
elif arg.owner and isinstance(arg.owner.op, Shape_i):
    constant_args.append(None)  # Use runtime shape
else:
    raise NotImplementedError("ARange needs concrete values")
```

**3. Runtime Validation Strategy**

JAX tracing removes conditionals, so validation happens at conversion time:
```python
@jax_funcify.register(CheckAndRaise)
def jax_funcify_CheckAndRaise(op, node, **kwargs):
    # Validate constants at conversion time
    conds = node.inputs[1:]
    if any(isinstance(cond, Constant) and not bool(cond.data) for cond in conds):
        raise op.exc_type(op.msg)

    # Skip runtime checks with warning
    warnings.warn(f"Skipping {op} as JAX tracing would remove it.")
    return lambda x, *inputs: x
```

**4. Recursive Dispatch for Complex Ops**

```python
@jax_funcify.register(Elemwise)
def jax_funcify_Elemwise(op, node, **kwargs):
    scalar_op = op.scalar_op
    # Recursively dispatch to scalar op
    base_fn = jax_funcify(scalar_op, node=node, **kwargs)

    def elemwise_fn(*inputs):
        Elemwise._check_runtime_broadcast(node, tuple(map(jnp.asarray, inputs)))
        return base_fn(*inputs)
    return elemwise_fn
```

**5. External Dependencies Management**

Some operations require optional packages:
```python
def try_import_tfp_jax_op(op: ScalarOp, jax_op_name: str | None = None) -> Callable:
    try:
        import tensorflow_probability.substrates.jax.math as tfp_jax_math
    except ModuleNotFoundError:
        raise NotImplementedError(
            f"No JAX implementation for Op {op.name}. "
            "TensorFlow Probability required for this operation."
        )
```

---

### 4. ONNX Backend: What's Different

#### 4.1 ONNX-Specific Constraints

**Static Graph Requirement**:
- ONNX models are **static graphs** (like TensorFlow 1.x)
- All shapes must be known at export time (or symbolic)
- Control flow must use ONNX operators (If, Loop)
- No Python control flow in exported graph

**No In-Place Operations**:
- ONNX has no concept of in-place updates
- `IncSubtensor` needs to compile to full copy + update

**Limited Dynamic Features**:
- Dynamic shapes require ONNX opset 11+
- Some operations don't support dynamic shapes at all

**Type System Differences**:
- ONNX has strict type requirements
- Must handle PyTensor's flexible typing

#### 4.2 ONNX Advantages

**Broad Deployment Support**:
- ONNX Runtime: CPU, GPU, WebAssembly, mobile
- Hardware accelerators: Intel OpenVINO, Nvidia TensorRT
- Cloud services: Azure ML, AWS SageMaker

**Optimization Pipeline**:
- ONNX Runtime has extensive graph optimizations
- Can rely on ONNX optimizer for fusions

**Standardization**:
- Well-defined operator set (opset)
- Strong backward compatibility guarantees

---

### 5. Recommended Implementation Tiers

Based on JAX backend analysis and ONNX constraints, here are 5 implementation tiers:

---

### **TIER 1: Core Infrastructure + Basic Elemwise (20 ops)**
**Goal**: Enable basic tensor computation
**Timeline**: 1-2 weeks

**Operations**:

1. **Infrastructure (5 ops)**:
   - [ ] `FunctionGraph` - Graph conversion (meta-op)
   - [ ] `Constant` - Constant handling
   - [ ] `DeepCopyOp` - Copy operation (maps to Identity)
   - [ ] `Cast` - Type conversion
   - [ ] `Identity` - No-op passthrough

2. **Basic Elemwise Arithmetic (8 ops)** via `Elemwise`:
   - [ ] `Add` - Addition
   - [ ] `Sub` - Subtraction
   - [ ] `Mul` - Multiplication
   - [ ] `TrueDiv` - Division
   - [ ] `Neg` - Negation
   - [ ] `Abs` - Absolute value
   - [ ] `Maximum` - Element-wise maximum
   - [ ] `Minimum` - Element-wise minimum

3. **Basic Elemwise Math (7 ops)** via `Elemwise`:
   - [ ] `Exp` - Exponential
   - [ ] `Log` - Natural logarithm
   - [ ] `Sqrt` - Square root
   - [ ] `Pow` - Power operation
   - [ ] `Floor` - Floor function
   - [ ] `Ceil` - Ceiling function
   - [ ] `Round` - Rounding function

**ONNX Mappings**:
```python
# Direct 1:1 mappings
Add → ONNX::Add
Mul → ONNX::Mul
Sub → ONNX::Sub
Div → ONNX::Div
Neg → ONNX::Neg
Abs → ONNX::Abs
Exp → ONNX::Exp
Log → ONNX::Log
Sqrt → ONNX::Sqrt
Pow → ONNX::Pow
Max → ONNX::Max (element-wise)
Min → ONNX::Min (element-wise)
Floor → ONNX::Floor
Ceil → ONNX::Ceil
Round → ONNX::Round
Cast → ONNX::Cast
Identity → ONNX::Identity
```

**Success Criteria**:
```python
# Test: Basic arithmetic
x = pt.vector('x')
y = pt.vector('y')
z = (x + y) * 2 - 1
f = pytensor.function([x, y], z)
export_onnx(f, "basic_math.onnx")

# Test: Element-wise operations
x = pt.vector('x')
y = pt.exp(x) + pt.sqrt(pt.abs(x))
f = pytensor.function([x], y)
export_onnx(f, "elemwise.onnx")
```

---

### **TIER 2: Shape Manipulation (15 ops)**
**Goal**: Enable tensor reshaping, indexing, and joining
**Timeline**: 1.5-2 weeks

**Operations**:

1. **Shape Inspection (3 ops)**:
   - [ ] `Shape` - Get shape as tensor
   - [ ] `Shape_i` - Get specific dimension
   - [ ] `SpecifyShape` - Shape assertion (for optimization)

2. **Reshape Operations (4 ops)**:
   - [ ] `Reshape` - Reshape tensor
   - [ ] `DimShuffle` - Transpose, squeeze, unsqueeze
   - [ ] `Squeeze` - Remove singleton dimensions
   - [ ] `ExpandDims` - Add dimensions (via DimShuffle)

3. **Joining/Splitting (4 ops)**:
   - [ ] `Join` / `Concatenate` - Concatenate tensors
   - [ ] `Stack` - Stack tensors (via Join + Reshape)
   - [ ] `Split` - Split tensor into parts

4. **Basic Indexing (4 ops)**:
   - [ ] `Subtensor` - Basic slicing
   - [ ] `IncSubtensor` - In-place set/increment
   - [ ] `AdvancedSubtensor1` - 1D advanced indexing
   - [ ] `AdvancedIncSubtensor1` - 1D advanced in-place

**ONNX Mappings**:
```python
Shape → ONNX::Shape
Shape_i → ONNX::Shape + ONNX::Gather
Reshape → ONNX::Reshape
DimShuffle → ONNX::Transpose / ONNX::Unsqueeze / ONNX::Squeeze
Squeeze → ONNX::Squeeze
Join → ONNX::Concat
Split → ONNX::Split
Stack → ONNX::Concat + ONNX::Reshape

# Indexing (complex - may need multiple ONNX ops)
Subtensor → ONNX::Slice / ONNX::Gather
IncSubtensor → ONNX::ScatterND / ONNX::ScatterElements
AdvancedSubtensor1 → ONNX::Gather
AdvancedIncSubtensor1 → ONNX::ScatterElements
```

**Success Criteria**:
```python
# Test: Reshape and transpose
x = pt.matrix('x')  # (3, 4)
y = x.reshape((2, 6)).T  # (6, 2)
f = pytensor.function([x], y)
export_onnx(f, "reshape.onnx")

# Test: Concatenation
x = pt.matrix('x')
y = pt.matrix('y')
z = pt.concatenate([x, y], axis=0)
f = pytensor.function([x, y], z)
export_onnx(f, "concat.onnx")

# Test: Indexing
x = pt.vector('x')
y = x[2:5]  # Slice
f = pytensor.function([x], y)
export_onnx(f, "slice.onnx")
```

---

### **TIER 3: Reductions & Allocation (16 ops)**
**Goal**: Enable statistical operations and tensor creation
**Timeline**: 1-1.5 weeks

**Operations**:

1. **Reductions (8 ops)**:
   - [ ] `Sum` - Sum reduction
   - [ ] `Prod` - Product reduction
   - [ ] `Max` - Maximum reduction (not element-wise)
   - [ ] `Min` - Minimum reduction (not element-wise)
   - [ ] `All` - Logical AND reduction
   - [ ] `Any` - Logical OR reduction
   - [ ] `Argmax` - Index of maximum
   - [ ] `Argmin` - Index of minimum
   - [ ] `CAReduce` - Meta-op for reductions

2. **Allocation (7 ops)**:
   - [ ] `Alloc` - Broadcast scalar to shape
   - [ ] `AllocEmpty` - Allocate uninitialized (maps to ConstantOfShape)
   - [ ] `MakeVector` - Create vector from scalars
   - [ ] `ARange` - Range generation
   - [ ] `Eye` - Identity matrix
   - [ ] `TensorFromScalar` - Scalar to tensor
   - [ ] `ScalarFromTensor` - Tensor to scalar

**ONNX Mappings**:
```python
# Reductions
Sum → ONNX::ReduceSum
Prod → ONNX::ReduceProd
Max → ONNX::ReduceMax
Min → ONNX::ReduceMin
All → ONNX::ReduceMin (for bool)
Any → ONNX::ReduceMax (for bool)
Argmax → ONNX::ArgMax
Argmin → ONNX::ArgMin

# Allocation
Alloc → ONNX::Expand
AllocEmpty → ONNX::ConstantOfShape
MakeVector → ONNX::Concat (of scalars)
ARange → ONNX::Range (requires static inputs)
Eye → ONNX::EyeLike or custom (Shape + Expand + Mul)
TensorFromScalar → ONNX::Reshape
ScalarFromTensor → ONNX::Reshape or ONNX::ReduceSum (size-1 tensor)
```

**Success Criteria**:
```python
# Test: Reductions
x = pt.matrix('x')
y = pt.sum(x, axis=1)  # Row sums
f = pytensor.function([x], y)
export_onnx(f, "sum.onnx")

# Test: Mean and variance
x = pt.matrix('x')
mean = pt.mean(x, axis=0)
var = pt.var(x, axis=0)
f = pytensor.function([x], [mean, var])
export_onnx(f, "stats.onnx")

# Test: Allocation
n = pt.scalar('n', dtype='int64')
x = pt.zeros(n)  # Uses AllocEmpty + constant fill
f = pytensor.function([n], x)
export_onnx(f, "zeros.onnx")
```

---

### **TIER 4: Linear Algebra (20 ops)**
**Goal**: Enable matrix operations and scientific computing
**Timeline**: 2-3 weeks

**Operations**:

1. **Matrix Multiplication (5 ops)**:
   - [ ] `Dot` - General dot product
   - [ ] `Gemm` - General matrix multiply (A @ B)
   - [ ] `Gemv` - Matrix-vector product
   - [ ] `BatchedDot` - Batched matrix multiplication
   - [ ] `Dot22` - Optimized 2x2 dot

2. **Decompositions (6 ops)**:
   - [ ] `SVD` - Singular value decomposition
   - [ ] `QR` - QR decomposition
   - [ ] `Cholesky` - Cholesky decomposition
   - [ ] `LU` - LU decomposition (if ONNX Runtime supports)
   - [ ] `Eig` - Eigendecomposition
   - [ ] `Eigh` - Hermitian eigendecomposition

3. **Solving (5 ops)**:
   - [ ] `Solve` - Linear system solving (A @ x = b)
   - [ ] `SolveTriangular` - Triangular system solving
   - [ ] `Lstsq` - Least squares
   - [ ] `MatrixInverse` - Matrix inverse
   - [ ] `MatrixPinv` - Pseudo-inverse

4. **Other Linear Algebra (4 ops)**:
   - [ ] `Det` - Determinant
   - [ ] `SLogDet` - Log-determinant (sign + log)
   - [ ] `Expm` - Matrix exponential
   - [ ] `ExtractDiag` - Diagonal extraction

**ONNX Mappings**:
```python
# Matrix Multiplication
Dot → ONNX::MatMul
Gemm → ONNX::Gemm (general matrix multiply with alpha/beta)
Gemv → ONNX::Gemm (vector as 2D)
BatchedDot → ONNX::MatMul (with batch dimensions)
Dot22 → ONNX::MatMul

# Decompositions (ONNX Runtime specific, not in standard ONNX)
# May need to use ONNX Runtime contrib ops or implement as sequences
SVD → ONNX Runtime contrib op (or NumPy fallback)
QR → ONNX Runtime contrib op
Cholesky → ONNX Runtime contrib op
Eig → ONNX Runtime contrib op
Eigh → ONNX Runtime contrib op

# Solving
Solve → Custom implementation (LU + substitution)
SolveTriangular → Custom implementation
Lstsq → Custom implementation (QR + solve)
MatrixInverse → Custom implementation (or Identity + Gemm trick)
MatrixPinv → Custom implementation (SVD + reconstruction)

# Other
Det → Custom (LU + product of diagonal)
SLogDet → Custom (LU + sum of log diagonal)
Expm → Not in ONNX standard (skip or use Padé approximation)
ExtractDiag → ONNX::Identity (if contiguous) or custom
```

**Success Criteria**:
```python
# Test: Matrix multiplication
A = pt.matrix('A')  # (3, 4)
B = pt.matrix('B')  # (4, 5)
C = pt.dot(A, B)  # (3, 5)
f = pytensor.function([A, B], C)
export_onnx(f, "matmul.onnx")

# Test: Linear regression (W @ x + b)
x = pt.vector('x')  # (n,)
W = pt.matrix('W')  # (m, n)
b = pt.vector('b')  # (m,)
y = pt.dot(W, x) + b
f = pytensor.function([x, W, b], y)
export_onnx(f, "linear.onnx")

# Test: Matrix inverse
A = pt.matrix('A')
A_inv = pt.nlinalg.inv(A)
f = pytensor.function([A], A_inv)
export_onnx(f, "inverse.onnx")  # May not work if no contrib op
```

**Note**: Many decompositions and solvers are **not in standard ONNX opset**. Options:
1. Use ONNX Runtime contrib ops (platform-specific)
2. Implement as sequences of basic ONNX ops (slow)
3. Skip and document as unsupported
4. Use custom operators (requires runtime support)

---

### **TIER 5: Advanced Operations (43 ops)**
**Goal**: Complete coverage for scientific computing and ML
**Timeline**: 2-3 weeks

**Operations**:

1. **Trigonometric & Hyperbolic (12 ops)** via `Elemwise`:
   - [ ] `Sin` - Sine
   - [ ] `Cos` - Cosine
   - [ ] `Tan` - Tangent
   - [ ] `ArcSin` - Arcsine
   - [ ] `ArcCos` - Arccosine
   - [ ] `ArcTan` - Arctangent
   - [ ] `Sinh` - Hyperbolic sine
   - [ ] `Cosh` - Hyperbolic cosine
   - [ ] `Tanh` - Hyperbolic tangent
   - [ ] `ArcSinh` - Inverse hyperbolic sine
   - [ ] `ArcCosh` - Inverse hyperbolic cosine
   - [ ] `ArcTanh` - Inverse hyperbolic tangent

2. **Comparison & Logical (10 ops)** via `Elemwise`:
   - [ ] `LT` - Less than
   - [ ] `GT` - Greater than
   - [ ] `LE` - Less or equal
   - [ ] `GE` - Greater or equal
   - [ ] `EQ` - Equal
   - [ ] `NEQ` - Not equal
   - [ ] `AND` - Logical AND
   - [ ] `OR` - Logical OR
   - [ ] `XOR` - Logical XOR
   - [ ] `Invert` - Logical NOT

3. **Special Math (8 ops)** via `Elemwise`:
   - [ ] `Sigmoid` - Sigmoid activation
   - [ ] `Softplus` - Softplus activation
   - [ ] `Log1p` - log(1 + x)
   - [ ] `Expm1` - exp(x) - 1
   - [ ] `Erf` - Error function
   - [ ] `Erfc` - Complementary error function
   - [ ] `Clip` - Clip values to range

4. **Neural Network Operations (5 ops)**:
   - [ ] `Softmax` - Softmax activation
   - [ ] `LogSoftmax` - Log-softmax
   - [ ] `Switch` - Conditional (element-wise ternary)
   - [ ] `IfElse` - Control flow conditional
   - [ ] `Scan` - Sequential/recurrent operations

5. **Extra Operations (8 ops)**:
   - [ ] `CumOp` - Cumulative sum/product
   - [ ] `Repeat` - Repeat elements
   - [ ] `Unique` - Find unique elements
   - [ ] `SearchsortedOp` - Binary search
   - [ ] `SortOp` - Sort operation
   - [ ] `ArgSortOp` - Argsort operation
   - [ ] `FillDiagonal` - Set diagonal values
   - [ ] `Pad` - Array padding

**ONNX Mappings**:
```python
# Trigonometric
Sin → ONNX::Sin
Cos → ONNX::Cos
Tan → ONNX::Tan
Asin → ONNX::Asin
Acos → ONNX::Acos
Atan → ONNX::Atan
Sinh → ONNX::Sinh
Cosh → ONNX::Cosh
Tanh → ONNX::Tanh
Asinh → ONNX::Asinh
Acosh → ONNX::Acosh
Atanh → ONNX::Atanh

# Comparison
Less → ONNX::Less
Greater → ONNX::Greater
LessOrEqual → ONNX::LessOrEqual
GreaterOrEqual → ONNX::GreaterOrEqual
Equal → ONNX::Equal
NotEqual → ONNX::Not + ONNX::Equal or custom

# Logical
And → ONNX::And
Or → ONNX::Or
Xor → ONNX::Xor
Not → ONNX::Not

# Special Math
Sigmoid → ONNX::Sigmoid
Tanh → ONNX::Tanh
Erf → ONNX::Erf
Softplus → ONNX::Softplus
Log1p → ONNX::Log(1 + x) via ONNX::Add + ONNX::Log
Expm1 → ONNX::Sub(ONNX::Exp(x), 1)
Clip → ONNX::Clip

# Neural Network
Softmax → ONNX::Softmax
LogSoftmax → ONNX::LogSoftmax
Switch → ONNX::Where
IfElse → ONNX::If
Scan → ONNX::Loop (complex translation)

# Extra
CumSum → ONNX::CumSum
Repeat → ONNX::Tile or ONNX::Expand
Unique → ONNX::Unique (opset 11+)
Searchsorted → Custom (not in ONNX standard)
Sort → ONNX::TopK (limited) or custom
ArgSort → Custom (not in ONNX standard)
FillDiagonal → ONNX::ScatterND
Pad → ONNX::Pad
```

**Success Criteria**:
```python
# Test: Trigonometric
x = pt.vector('x')
y = pt.sin(x) + pt.cos(x**2)
f = pytensor.function([x], y)
export_onnx(f, "trig.onnx")

# Test: Conditional
x = pt.vector('x')
y = pt.switch(x > 0, x**2, -x)  # ReLU variant
f = pytensor.function([x], y)
export_onnx(f, "switch.onnx")

# Test: Softmax
x = pt.matrix('x')
y = pt.nnet.softmax(x)
f = pytensor.function([x], y)
export_onnx(f, "softmax.onnx")

# Test: Scan (recurrence)
x = pt.vector('x')
def step(x_t, acc):
    return acc + x_t
outputs, _ = pytensor.scan(fn=step, sequences=x, outputs_info=[pt.as_tensor(0.0)])
cumsum = outputs[-1]
f = pytensor.function([x], cumsum)
export_onnx(f, "scan.onnx")  # May fail - Scan is complex
```

---

### 6. Implementation Strategy

#### 6.1 File Structure

```
pytensor/link/onnx/
├── __init__.py              # Public API
├── linker.py                # ONNXLinker class
├── export.py                # export_onnx() function
├── opset.py                 # ONNX opset version management
└── dispatch/
    ├── __init__.py          # Import all dispatch modules
    ├── basic.py             # Core dispatch (onnx_funcify, onnx_typify, FunctionGraph)
    ├── elemwise.py          # Elemwise operations + scalar op mapping
    ├── shape.py             # Shape operations (Reshape, DimShuffle, etc.)
    ├── tensor_basic.py      # Tensor creation and joining
    ├── math.py              # Reductions and basic math
    ├── nlinalg.py           # Linear algebra
    ├── slinalg.py           # Specialized linear algebra
    ├── blas.py              # BLAS operations
    ├── subtensor.py         # Indexing operations
    ├── special.py           # Special functions (Softmax, etc.)
    ├── extra_ops.py         # Extra operations
    ├── sort.py              # Sorting operations
    ├── control_flow.py      # IfElse, Scan
    └── pad.py               # Padding operations

tests/link/onnx/
├── __init__.py
├── conftest.py              # Pytest configuration and fixtures
├── test_basic.py            # Core functionality tests
├── test_elemwise.py         # Elemwise operation tests
├── test_shape.py            # Shape operation tests
├── test_tensor_basic.py     # Tensor creation tests
├── test_math.py             # Reduction tests
├── test_nlinalg.py          # Linear algebra tests
├── test_slinalg.py          # Specialized linear algebra tests
├── test_blas.py             # BLAS tests
├── test_subtensor.py        # Indexing tests
├── test_special.py          # Special function tests
├── test_extra_ops.py        # Extra operation tests
├── test_sort.py             # Sorting tests
├── test_control_flow.py     # Control flow tests
└── test_integration.py      # End-to-end integration tests
```

#### 6.2 Core Dispatch Pattern

**File**: `pytensor/link/onnx/dispatch/basic.py`

```python
"""Basic ONNX dispatch system."""

from functools import singledispatch
from typing import Dict, List, Callable

import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except ImportError as e:
    raise ImportError(
        "ONNX export requires the 'onnx' package. "
        "Install it with: pip install pytensor[onnx]"
    ) from e

from pytensor.graph.basic import Constant, Variable
from pytensor.graph.fg import FunctionGraph


# Target ONNX opset version
ONNX_OPSET_VERSION = 18


@singledispatch
def onnx_funcify(op, node=None, **kwargs):
    """Convert PyTensor Op to ONNX node(s).

    This is the main dispatch function. Register converters for specific
    Op types using @onnx_funcify.register(OpClass).

    Parameters
    ----------
    op : Op or FunctionGraph
        The operation to convert
    node : Apply, optional
        The Apply node containing the op
    **kwargs
        Additional conversion parameters:
        - var_names: Dict[Variable, str] - variable name mapping
        - get_var_name: Callable - function to get/create variable names
        - opset_version: int - target ONNX opset version

    Returns
    -------
    onnx.NodeProto or List[onnx.NodeProto]
        ONNX node(s) representing the operation

    Raises
    ------
    NotImplementedError
        If no converter is registered for this Op type
    """
    raise NotImplementedError(
        f"No ONNX conversion available for: {type(op).__name__}\n"
        f"Op: {op}\n"
        f"This operation is not yet supported for ONNX export.\n\n"
        f"Currently supported operations:\n"
        f"  Tier 1: Add, Mul, Sub, Div, Neg, Abs, Exp, Log, Sqrt, Pow, Max, Min\n"
        f"  Tier 2: Reshape, DimShuffle, Join, Split, Subtensor\n"
        f"  Tier 3: Sum, Prod, Max, Min, Argmax, Argmin, Alloc, ARange\n"
        f"  Tier 4: Dot, Gemm, SVD, Cholesky, Solve (limited)\n"
        f"  Tier 5: Sin, Cos, Tanh, Softmax, IfElse\n\n"
        f"To add support for this operation, register a converter:\n"
        f"  @onnx_funcify.register({type(op).__name__})\n"
        f"  def onnx_funcify_{type(op).__name__}(op, node, var_names, get_var_name, **kwargs):\n"
        f"      # Return onnx.NodeProto or list of onnx.NodeProto\n"
    )


@singledispatch
def onnx_typify(data, dtype=None, **kwargs):
    """Convert Python/NumPy data to ONNX-compatible types.

    This is used for converting constants and inputs to ONNX tensors.

    Parameters
    ----------
    data : Any
        Data to convert (typically numpy array or scalar)
    dtype : str, optional
        Target dtype for conversion

    Returns
    -------
    onnx.TensorProto or data
        ONNX tensor representation or original data
    """
    if dtype is None:
        return data
    else:
        return np.array(data, dtype=dtype)


@onnx_typify.register(np.ndarray)
def onnx_typify_ndarray(data, dtype=None, name="", **kwargs):
    """Convert numpy array to ONNX TensorProto."""
    if dtype is not None:
        data = data.astype(dtype)
    return numpy_helper.from_array(data, name=name)


def make_value_info(var: Variable, name: str) -> onnx.ValueInfoProto:
    """Create ONNX ValueInfoProto from PyTensor Variable.

    Parameters
    ----------
    var : Variable
        PyTensor variable
    name : str
        Name for the ONNX value

    Returns
    -------
    onnx.ValueInfoProto
        ONNX value info with type and shape
    """
    # Map PyTensor dtype to ONNX dtype
    dtype_map = {
        "float32": TensorProto.FLOAT,
        "float64": TensorProto.DOUBLE,
        "int32": TensorProto.INT32,
        "int64": TensorProto.INT64,
        "uint8": TensorProto.UINT8,
        "int8": TensorProto.INT8,
        "bool": TensorProto.BOOL,
    }

    dtype_str = str(var.type.dtype)
    onnx_dtype = dtype_map.get(dtype_str, TensorProto.FLOAT)

    # Get shape (use symbolic dimensions if needed)
    if hasattr(var.type, "shape"):
        shape = []
        for i, dim in enumerate(var.type.shape):
            if dim is None or (isinstance(dim, int) and dim < 0):
                # Dynamic dimension - use symbolic name
                shape.append(f"dim_{i}")
            else:
                shape.append(int(dim))
    else:
        shape = None

    # Create tensor type
    tensor_type = helper.make_tensor_type_proto(elem_type=onnx_dtype, shape=shape)

    return helper.make_value_info(name, tensor_type)


@onnx_funcify.register(FunctionGraph)
def onnx_funcify_FunctionGraph(
    fgraph: FunctionGraph,
    node=None,
    opset_version: int = ONNX_OPSET_VERSION,
    model_name: str = "pytensor_model",
    **kwargs,
) -> onnx.ModelProto:
    """Convert a FunctionGraph to ONNX ModelProto.

    Parameters
    ----------
    fgraph : FunctionGraph
        The graph to convert
    opset_version : int
        ONNX opset version to target (default: 18)
    model_name : str
        Name for the ONNX model

    Returns
    -------
    onnx.ModelProto
        Complete ONNX model
    """
    # Track converted nodes and initializers
    onnx_nodes: List[onnx.NodeProto] = []
    initializers: List[onnx.TensorProto] = []

    # Generate unique names for variables
    var_names: Dict[Variable, str] = {}
    name_counter = 0

    def get_var_name(var: Variable) -> str:
        """Get or create unique name for a variable."""
        nonlocal name_counter
        if var not in var_names:
            if hasattr(var, "name") and var.name:
                base_name = var.name
                # Ensure uniqueness
                if base_name in var_names.values():
                    base_name = f"{base_name}_{name_counter}"
                    name_counter += 1
                var_names[var] = base_name
            else:
                var_names[var] = f"var_{name_counter}"
                name_counter += 1
        return var_names[var]

    # Convert constants to initializers
    for node in fgraph.apply_nodes:
        for inp in node.inputs:
            if isinstance(inp, Constant):
                name = get_var_name(inp)
                if name not in [init.name for init in initializers]:
                    tensor = numpy_helper.from_array(
                        np.asarray(inp.data), name=name
                    )
                    initializers.append(tensor)

    # Convert ops in topological order
    for node in fgraph.toposort():
        # Get ONNX node(s) for this Apply
        onnx_node_or_nodes = onnx_funcify(
            node.op,
            node=node,
            var_names=var_names,
            get_var_name=get_var_name,
            opset_version=opset_version,
            **kwargs,
        )

        # Handle both single nodes and lists of nodes
        if onnx_node_or_nodes is not None:
            if isinstance(onnx_node_or_nodes, list):
                onnx_nodes.extend(onnx_node_or_nodes)
            else:
                onnx_nodes.append(onnx_node_or_nodes)

    # Create inputs (only non-constant inputs)
    input_protos = []
    for inp in fgraph.inputs:
        if not isinstance(inp, Constant):
            name = get_var_name(inp)
            input_protos.append(make_value_info(inp, name))

    # Create outputs
    output_protos = []
    for out in fgraph.outputs:
        name = get_var_name(out)
        output_protos.append(make_value_info(out, name))

    # Create graph
    graph = helper.make_graph(
        nodes=onnx_nodes,
        name=f"{model_name}_graph",
        inputs=input_protos,
        outputs=output_protos,
        initializer=initializers,
    )

    # Create model
    model = helper.make_model(
        graph,
        producer_name="PyTensor",
        opset_imports=[helper.make_opsetid("", opset_version)],
    )

    # Validate model
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        raise ValueError(f"Generated ONNX model is invalid: {e}") from e

    return model
```

#### 6.3 Example Operation Implementation

**File**: `pytensor/link/onnx/dispatch/elemwise.py`

```python
"""ONNX conversion for elementwise operations."""

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.elemwise import Elemwise
from pytensor.scalar import basic as scalar

try:
    from onnx import helper
except ImportError as e:
    raise ImportError("ONNX package required for export") from e


# Mapping from PyTensor scalar ops to ONNX op types
SCALAR_OP_TO_ONNX = {
    # Arithmetic
    scalar.Add: "Add",
    scalar.Mul: "Mul",
    scalar.Sub: "Sub",
    scalar.TrueDiv: "Div",
    scalar.Neg: "Neg",

    # Math
    scalar.Abs: "Abs",
    scalar.Exp: "Exp",
    scalar.Log: "Log",
    scalar.Sqrt: "Sqrt",
    scalar.Pow: "Pow",

    # Rounding
    scalar.Floor: "Floor",
    scalar.Ceil: "Ceil",
    scalar.Round: "Round",

    # Min/Max
    scalar.Maximum: "Max",
    scalar.Minimum: "Min",

    # Trig (Tier 5)
    scalar.Sin: "Sin",
    scalar.Cos: "Cos",
    scalar.Tan: "Tan",
    scalar.ArcSin: "Asin",
    scalar.ArcCos: "Acos",
    scalar.ArcTan: "Atan",

    # Hyperbolic (Tier 5)
    scalar.Sinh: "Sinh",
    scalar.Cosh: "Cosh",
    scalar.Tanh: "Tanh",
    scalar.ArcSinh: "Asinh",
    scalar.ArcCosh: "Acosh",
    scalar.ArcTanh: "Atanh",

    # Comparison (Tier 5)
    scalar.LT: "Less",
    scalar.GT: "Greater",
    scalar.LE: "LessOrEqual",
    scalar.GE: "GreaterOrEqual",
    scalar.EQ: "Equal",

    # Logical (Tier 5)
    scalar.AND: "And",
    scalar.OR: "Or",
    scalar.XOR: "Xor",
    scalar.Invert: "Not",
}


@onnx_funcify.register(Elemwise)
def onnx_funcify_Elemwise(op, node, var_names, get_var_name, **kwargs):
    """Convert Elemwise op to ONNX node.

    Elemwise ops perform element-wise operations on tensors.
    They map directly to ONNX ops like Add, Mul, etc.
    """
    scalar_op_type = type(op.scalar_op)

    if scalar_op_type not in SCALAR_OP_TO_ONNX:
        raise NotImplementedError(
            f"Elemwise scalar op not supported for ONNX export: {scalar_op_type.__name__}\n"
            f"Supported scalar ops: {', '.join(op.__name__ for op in SCALAR_OP_TO_ONNX.keys())}"
        )

    onnx_op_type = SCALAR_OP_TO_ONNX[scalar_op_type]

    # Get input and output names
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    # Create ONNX node
    onnx_node = helper.make_node(
        onnx_op_type,
        inputs=input_names,
        outputs=output_names,
        name=f"{onnx_op_type}_{output_names[0]}",
    )

    return onnx_node
```

#### 6.4 Test Pattern

**File**: `tests/link/onnx/test_elemwise.py`

```python
"""Tests for ONNX elemwise operations."""

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import pytensor
import pytensor.tensor as pt

from tests.link.onnx.test_basic import compare_onnx_and_py


def test_add(tmp_path):
    """Test addition operation."""
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = x + y

    x_val = np.array([1, 2, 3], dtype="float32")
    y_val = np.array([4, 5, 6], dtype="float32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_mul(tmp_path):
    """Test multiplication operation."""
    x = pt.vector("x", dtype="float32")
    y = pt.vector("y", dtype="float32")
    z = x * y

    x_val = np.array([1, 2, 3], dtype="float32")
    y_val = np.array([4, 5, 6], dtype="float32")

    compare_onnx_and_py([x, y], z, [x_val, y_val], tmp_path=tmp_path)


def test_chained_operations(tmp_path):
    """Test multiple operations chained together."""
    x = pt.vector("x", dtype="float32")
    # (x * 2 + 3) / 4
    z = ((x * 2) + 3) / 4

    x_val = np.array([1, 2, 3], dtype="float32")

    compare_onnx_and_py([x], z, [x_val], tmp_path=tmp_path)
```

---

### 7. Timeline and Resource Estimates

#### 7.1 Implementation Timeline

| Tier | Operations | Weeks | Dependencies |
|------|-----------|-------|--------------|
| **Tier 1** | 20 ops | 1-2 weeks | None |
| **Tier 2** | 15 ops | 1.5-2 weeks | Tier 1 |
| **Tier 3** | 15 ops | 1-1.5 weeks | Tier 1, Tier 2 |
| **Tier 4** | 20 ops | 2-3 weeks | Tier 1-3 |
| **Tier 5** | 20 ops | 2-3 weeks | Tier 1-4 |
| **Testing & Polish** | - | 1-2 weeks | All tiers |
| **TOTAL** | **90 ops** | **9-13.5 weeks** | |

**Recommended Milestones**:
- **Month 1**: Tiers 1-2 (core infrastructure + basic ops)
- **Month 2**: Tiers 3-4 (reductions + linear algebra)
- **Month 3**: Tier 5 + testing (advanced ops + polish)

#### 7.2 Resource Requirements

**Developer Skills Needed**:
- PyTensor internals (Op system, FunctionGraph, type system)
- ONNX specification and opset knowledge
- Python dispatch patterns (singledispatch)
- Numerical computing (NumPy, linear algebra)
- Testing frameworks (pytest)

**External Dependencies**:
- `onnx` package (core)
- `onnxruntime` package (testing)
- Optional: `onnxoptimizer` (graph optimization)

**Testing Resources**:
- ONNX Runtime (CPU execution provider)
- Test data generation (NumPy RandomState)
- Model validation (ONNX checker)

---

### 8. Key Differences from Demo Plans

| Aspect | Demo Plans | Production Recommendation |
|--------|------------|-------------------------|
| **Target Use Case** | YOLO11n neural network inference | General PyTensor computation |
| **Operation Focus** | CNN ops (Conv2D, MaxPool, BatchNorm, Resize) | Core ops (elemwise, linalg, shape) |
| **Deployment Target** | WebAssembly browser | Multiple (ONNX Runtime, hardware accelerators) |
| **Operation Count** | ~30-40 ops | ~90 ops |
| **Priority** | Visual operations for demo | Most commonly used operations |
| **Timeline** | 5-8 days (minimal demo) | 9-13 weeks (production) |
| **Testing** | Basic end-to-end tests | Comprehensive unit + integration tests |
| **Linear Algebra** | Not prioritized | Essential (Tier 4) |
| **Control Flow** | Not addressed | Tier 5 (Scan, IfElse) |
| **Random Variables** | Not addressed | Future work (see JAX backend) |

---

### 9. Open Questions and Decisions

#### 9.1 Linear Algebra Implementation

**Question**: How to handle operations not in standard ONNX opset?

**Options**:
1. **Use ONNX Runtime contrib ops** (e.g., `com.microsoft.Cholesky`)
   - Pros: Native implementation, good performance
   - Cons: Platform-specific, not portable

2. **Implement as sequences of basic ONNX ops**
   - Pros: Portable, standard ONNX
   - Cons: Slow, complex implementations

3. **Skip and document as unsupported**
   - Pros: Fast implementation, clear limitations
   - Cons: Incomplete coverage

4. **Use custom operators**
   - Pros: Flexible, can wrap existing libraries
   - Cons: Requires runtime support, deployment complexity

**Recommendation**: Start with option 3 (document unsupported), add contrib ops in Tier 4 for specific platforms

#### 9.2 Control Flow (Scan, IfElse)

**Question**: Should we implement Scan → ONNX Loop conversion?

**Considerations**:
- PyTensor Scan is complex (multiple recurrence patterns)
- ONNX Loop is low-level (requires manual state management)
- JAX backend has working Scan implementation (reference)
- Many ML models use recurrent operations

**Recommendation**:
- Tier 5 priority
- Start with simple recurrence (SIT-SOT pattern)
- Use JAX backend as reference
- May require 1-2 weeks alone

#### 9.3 Random Variables

**Question**: Should we support RandomVariable operations?

**Considerations**:
- ONNX has no standard RNG operations
- Some ONNX Runtime versions have RandomNormal, etc.
- Needed for probabilistic models
- JAX backend has extensive random support (22 distributions)

**Recommendation**:
- **Not in initial production backend**
- Future work (Tier 6)
- Focus on deterministic operations first
- Can use contrib ops for specific distributions later

#### 9.4 Dynamic Shapes

**Question**: How to handle dynamic shapes in ONNX?

**Considerations**:
- ONNX opset 11+ supports dynamic shapes
- Some operations don't work with dynamic shapes
- PyTensor has flexible shape system
- Need clear error messages when shapes must be static

**Recommendation**:
- Support dynamic shapes where possible (Reshape, Alloc, etc.)
- Require static shapes for operations that need them (ARange)
- Provide clear error messages
- Document limitations

#### 9.5 Opset Version

**Question**: Which ONNX opset version to target?

**Options**:
- Opset 13 (2021): Stable, wide support
- Opset 15 (2022): Better dynamic shape support
- Opset 18 (2023): Latest features
- Opset 19+ (2024): Cutting edge

**Recommendation**: **Opset 18** (same as demo plans)
- Good balance of features and compatibility
- Dynamic shapes support
- Wide ONNX Runtime support

---

### 10. Success Metrics

**Tier 1 Complete**:
- ✅ Can export basic arithmetic expressions
- ✅ Elemwise operations work with broadcasting
- ✅ All tests pass (20+ tests)

**Tier 2 Complete**:
- ✅ Can export tensor reshaping operations
- ✅ Concatenation and splitting work
- ✅ Basic indexing exports correctly
- ✅ All tests pass (35+ tests)

**Tier 3 Complete**:
- ✅ Can export statistical operations (mean, var, sum)
- ✅ Tensor creation operations work
- ✅ All tests pass (50+ tests)

**Tier 4 Complete**:
- ✅ Can export matrix multiplication and linear layers
- ✅ Basic linear algebra works (SVD, Cholesky if supported)
- ✅ All tests pass (70+ tests)

**Tier 5 Complete**:
- ✅ Can export complete neural networks (MLP, maybe RNN)
- ✅ Trigonometric and special functions work
- ✅ All tests pass (90+ tests)

**Production Ready**:
- ✅ 90+ operations implemented
- ✅ 100+ tests passing
- ✅ Documentation complete
- ✅ Can export real-world PyTensor code
- ✅ Performance benchmarks available
- ✅ Known limitations documented

---

## Code References

### PyTensor Core Operations
- `pytensor/tensor/basic.py:1-4700` - Basic tensor operations
- `pytensor/tensor/elemwise.py:1-1400` - Elemwise framework
- `pytensor/scalar/basic.py:1-4100` - Scalar operations
- `pytensor/scalar/math.py:1-1700` - Special math functions
- `pytensor/tensor/math.py:1-4000` - Reduction and math operations
- `pytensor/tensor/shape.py:1-800` - Shape operations
- `pytensor/tensor/blas.py:1-1400` - BLAS operations
- `pytensor/tensor/nlinalg.py:1-1200` - General linear algebra
- `pytensor/tensor/slinalg.py:1-1800` - Specialized linear algebra
- `pytensor/tensor/subtensor.py:1-3000` - Indexing operations
- `pytensor/tensor/extra_ops.py:1-2000` - Extra operations
- `pytensor/tensor/sort.py:1-200` - Sorting operations
- `pytensor/tensor/special.py:1-600` - Special functions

### JAX Backend (Reference Implementation)
- `pytensor/link/jax/linker.py:9-127` - JAXLinker
- `pytensor/link/jax/dispatch/basic.py:1-200` - Core dispatch
- `pytensor/link/jax/dispatch/elemwise.py:1-70` - Elemwise ops
- `pytensor/link/jax/dispatch/tensor_basic.py:1-300` - Tensor creation
- `pytensor/link/jax/dispatch/shape.py:1-200` - Shape ops
- `pytensor/link/jax/dispatch/nlinalg.py:1-150` - Linear algebra
- All 21 files in `pytensor/link/jax/dispatch/` - Complete coverage

### Planning Documents (Demo-Focused)
- `thoughts/shared/plans/onnx-backend-implementation.md` - 5-phase demo plan
- `thoughts/shared/research/2025-10-15_onnx-implementation-plan.md` - Implementation details
- `thoughts/shared/research/2025-10-14_23-53-33_onnx-backend-coverage-analysis.md` - Coverage analysis
- `thoughts/shared/research/2025-10-14_adding-new-backend-onnx-xla.md` - Backend architecture

---

## Related Research

**From thoughts/ directory**:
- `thoughts/shared/research/2025-10-14_adding-new-backend-onnx-xla.md` - How to add backends
- `thoughts/shared/research/2025-10-14_22-30-00_yolo11n-onnx-backend-gaps.md` - YOLO gaps (CNN-focused)
- `thoughts/shared/plans/jax-cnn-ops-implementation.md` - JAX CNN ops (not needed for core)

---

## Recommendations

### Immediate Next Steps

1. **Week 1-2: Tier 1 Implementation**
   - Create directory structure
   - Implement core dispatch system
   - Add 20 basic elemwise operations
   - Write 20+ tests

2. **Week 3-4: Tier 2 Implementation**
   - Shape operations (Reshape, DimShuffle)
   - Tensor joining/splitting
   - Basic indexing
   - Write 15+ tests

3. **Week 5-6: Tier 3 Implementation**
   - Reduction operations
   - Tensor allocation
   - Statistical functions (mean, var)
   - Write 15+ tests

4. **Month 2-3: Tiers 4-5**
   - Linear algebra (as supported)
   - Advanced operations
   - Control flow (if time permits)
   - Comprehensive testing

### Long-term Roadmap

**Tier 6 (Future Work)**:
- Random variables (22 distributions from JAX)
- CNN operations (Conv2D, MaxPool, BatchNorm) if needed
- Custom operators for unsupported linalg
- Graph optimizations (fusion, constant folding)
- WebAssembly-specific optimizations

**Tier 7 (Research)**:
- Training operations (if ONNX supports)
- Gradient computation via ONNX
- Sparse tensor operations
- Quantization support

---

## Quick Reference: Complete Operation Checklist

### Tier 1: Core Infrastructure + Basic Elemwise (20 ops)
- [ ] `FunctionGraph`, `Constant`, `DeepCopyOp`, `Cast`, `Identity`
- [ ] `Add`, `Sub`, `Mul`, `TrueDiv`, `Neg`, `Abs`, `Maximum`, `Minimum`
- [ ] `Exp`, `Log`, `Sqrt`, `Pow`, `Floor`, `Ceil`, `Round`

### Tier 2: Shape Manipulation (15 ops)
- [ ] `Shape`, `Shape_i`, `SpecifyShape`
- [ ] `Reshape`, `DimShuffle`, `Squeeze`, `ExpandDims`
- [ ] `Join`/`Concatenate`, `Stack`, `Split`
- [ ] `Subtensor`, `IncSubtensor`, `AdvancedSubtensor1`, `AdvancedIncSubtensor1`

### Tier 3: Reductions & Allocation (16 ops)
- [ ] `Sum`, `Prod`, `Max`, `Min`, `All`, `Any`, `Argmax`, `Argmin`, `CAReduce`
- [ ] `Alloc`, `AllocEmpty`, `MakeVector`, `ARange`, `Eye`, `TensorFromScalar`, `ScalarFromTensor`

### Tier 4: Linear Algebra (20 ops)
- [ ] `Dot`, `Gemm`, `Gemv`, `BatchedDot`, `Dot22`
- [ ] `SVD`, `QR`, `Cholesky`, `LU`, `Eig`, `Eigh`
- [ ] `Solve`, `SolveTriangular`, `Lstsq`, `MatrixInverse`, `MatrixPinv`
- [ ] `Det`, `SLogDet`, `Expm`, `ExtractDiag`

### Tier 5: Advanced Operations (43 ops)
- [ ] Trig: `Sin`, `Cos`, `Tan`, `ArcSin`, `ArcCos`, `ArcTan`, `Sinh`, `Cosh`, `Tanh`, `ArcSinh`, `ArcCosh`, `ArcTanh`
- [ ] Comparison: `LT`, `GT`, `LE`, `GE`, `EQ`, `NEQ`
- [ ] Logical: `AND`, `OR`, `XOR`, `Invert`
- [ ] Special Math: `Sigmoid`, `Softplus`, `Log1p`, `Expm1`, `Erf`, `Erfc`, `Clip`
- [ ] Neural Network: `Softmax`, `LogSoftmax`, `Switch`, `IfElse`, `Scan`
- [ ] Extra: `CumOp`, `Repeat`, `Unique`, `SearchsortedOp`, `SortOp`, `ArgSortOp`, `FillDiagonal`, `Pad`

---

## Conclusion

For a production ONNX backend supporting general PyTensor code:

**Focus on**: 114 core operations across 5 tiers (elemwise, shape, reductions, linear algebra, advanced)

**Don't focus on**: CNN operations (Conv2D, MaxPool, BatchNorm) - these were demo-specific

**Timeline**: 9-13 weeks for production-ready implementation

**Reference**: JAX backend (99 ops) shows what "complete" looks like

**Priority**: Tiers 1-3 (51 ops, 4-5 weeks) enable 80% of PyTensor usage
