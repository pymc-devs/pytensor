---
date: 2025-10-15T00:00:00Z
researcher: Claude
git_commit: c58f10beb2aa5e5238f1420107e3bc1103e87c31
branch: main
repository: pymc-devs/pytensor
topic: "Answers to ONNX Backend Open Questions"
tags: [research, onnx, webassembly, shape-inference, custom-ops, gradients, control-flow, performance]
status: complete
last_updated: 2025-10-15
last_updated_by: Claude
---

# Answers to ONNX Backend Open Questions

This document addresses the open questions from the ONNX backend research and provides concrete answers and implementation strategies.

## Question 1: Shape Inference - Shape Annotations When Compiled

**Question**: How to handle dynamic shapes in PyTensor graphs when exporting to ONNX (which prefers static shapes)?

**Answer**: **Use shape annotations at compile time**

### Strategy

ONNX supports both static and dynamic shapes, but performs better with static shapes. Here's the approach:

#### 1. **Infer shapes from test values at compile time**

```python
def export_onnx(pytensor_function, output_path, example_inputs=None):
    """Export PyTensor function to ONNX with shape inference."""
    fgraph = pytensor_function.fgraph

    # If example inputs provided, use them to infer shapes
    if example_inputs is not None:
        # Run shape inference
        input_shapes = {}
        for inp, example in zip(fgraph.inputs, example_inputs):
            input_shapes[inp] = example.shape

        # Propagate shapes through graph
        inferred_shapes = infer_shapes(fgraph, input_shapes)
    else:
        # Use symbolic shapes where available
        inferred_shapes = extract_symbolic_shapes(fgraph)

    # Convert to ONNX with shape information
    model = onnx_funcify(fgraph, shapes=inferred_shapes)
    onnx.save(model, output_path)
```

#### 2. **Support dynamic dimensions with symbolic axes**

ONNX allows dynamic dimensions using symbolic names:

```python
# Create ONNX tensor with dynamic batch dimension
tensor_type = onnx.helper.make_tensor_type_proto(
    elem_type=onnx.TensorProto.FLOAT,
    shape=['batch_size', 784]  # 'batch_size' is symbolic
)
```

#### 3. **Implementation approach**

```python
def infer_shape_for_variable(var, known_shapes):
    """Infer shape for a variable given known input shapes."""
    if var in known_shapes:
        return known_shapes[var]

    if var.owner is None:
        # Input variable - check if has test_value
        if hasattr(var.tag, 'test_value'):
            return var.tag.test_value.shape
        # Otherwise return symbolic shape
        return tuple(f"dim_{i}" for i in range(var.type.ndim))

    # Infer from op
    op = var.owner.op
    input_shapes = [infer_shape_for_variable(inp, known_shapes)
                    for inp in var.owner.inputs]

    # Use op's infer_shape if available
    if hasattr(op, 'infer_shape'):
        output_shapes = op.infer_shape(var.owner, input_shapes)
        return output_shapes[var.owner.outputs.index(var)]

    # Fallback to symbolic
    return tuple(f"dim_{i}" for i in range(var.type.ndim))
```

### Recommended Workflow

1. **User provides example inputs** when exporting:
   ```python
   import numpy as np

   # Create PyTensor function
   f = pt.function([x, y], z)

   # Export with example inputs for shape inference
   export_onnx(f, "model.onnx",
               example_inputs=[np.zeros((32, 784)), np.zeros((784, 10))])
   ```

2. **Use PyTensor's shape inference**: PyTensor already has `Op.infer_shape()` method
   - Most ops implement this
   - Leverage it during ONNX conversion

3. **Mark truly dynamic dimensions**: For dimensions that must be dynamic (like batch size):
   ```python
   # Allow first dimension to be dynamic
   export_onnx(f, "model.onnx",
               dynamic_axes={'x': [0], 'y': [0], 'output': [0]})
   ```

---

## Question 2: Custom Ops - List of Ops Without ONNX Equivalents

**Question**: Make a list of PyTensor ops that don't have ONNX equivalents (using ONNX 1.20)

**Answer**: Here's a comprehensive list of PyTensor ops that **DO NOT** have direct ONNX equivalents:

### Category 1: Special Mathematical Functions (HIGH PRIORITY - NO ONNX SUPPORT)

These would need custom implementation or CPU fallback:

#### Error Functions (Partial Support)
- ✅ `Erf` - **HAS** ONNX equivalent
- ❌ `Erfc` - NO ONNX equivalent
- ❌ `Erfcx` - NO ONNX equivalent
- ❌ `Erfinv` - NO ONNX equivalent
- ❌ `Erfcinv` - NO ONNX equivalent

#### Gamma Functions Family
- ❌ `Gamma` - NO ONNX equivalent
- ❌ `GammaLn` (log-gamma) - NO ONNX equivalent
- ❌ `Psi` (digamma) - NO ONNX equivalent
- ❌ `TriGamma` - NO ONNX equivalent
- ❌ `PolyGamma` - NO ONNX equivalent
- ❌ `GammaInc` (incomplete gamma) - NO ONNX equivalent
- ❌ `GammaIncC` (complementary incomplete gamma) - NO ONNX equivalent
- ❌ `GammaIncInv` - NO ONNX equivalent
- ❌ `GammaIncCInv` - NO ONNX equivalent
- ❌ `GammaU` - NO ONNX equivalent
- ❌ `GammaL` - NO ONNX equivalent

#### Bessel Functions (ALL - NO ONNX SUPPORT)
- ❌ `Jv` (Bessel function of first kind) - NO ONNX equivalent
- ❌ `J0` - NO ONNX equivalent
- ❌ `J1` - NO ONNX equivalent
- ❌ `Iv` (Modified Bessel first kind) - NO ONNX equivalent
- ❌ `I0` - NO ONNX equivalent
- ❌ `I1` - NO ONNX equivalent
- ❌ `Ive` - NO ONNX equivalent
- ❌ `Kve` - NO ONNX equivalent

#### Beta and Hypergeometric Functions
- ❌ `BetaInc` (incomplete beta) - NO ONNX equivalent
- ❌ `BetaIncInv` - NO ONNX equivalent
- ❌ `Hyp2F1` (hypergeometric function) - NO ONNX equivalent

#### Owen's T Function
- ❌ `Owens_t` - NO ONNX equivalent

#### Other Special Functions
- ❌ `Log1mexp` - NO ONNX equivalent
- ✅ `Softplus` - Can implement with `Log(1 + Exp(x))`

### Category 2: Advanced Linear Algebra (MIXED SUPPORT)

#### Decompositions
- ❌ `Cholesky` - NO direct ONNX op (as of 1.20)
- ❌ `QR` decomposition - NO ONNX equivalent
- ❌ `LU` decomposition - NO ONNX equivalent
- ❌ `LUFactor` - NO ONNX equivalent
- ❌ `SVD` - NO ONNX equivalent
- ❌ `Eig` (eigenvalues/eigenvectors) - NO ONNX equivalent
- ❌ `Eigvalsh` (symmetric eigenvalues) - NO ONNX equivalent

#### Matrix Functions
- ❌ `Expm` (matrix exponential) - NO ONNX equivalent
- ❌ `ExpmGrad` - NO ONNX equivalent
- ❌ `MatrixInverse` - NO ONNX equivalent
- ✅ `MatrixPinv` - Can implement with SVD (but SVD not in ONNX)
- ✅ `Det` - **HAS** ONNX equivalent (Det operator)

#### Specialized Solvers
- ❌ `Solve` (general linear system) - NO ONNX equivalent
- ❌ `SolveTriangular` - NO ONNX equivalent
- ❌ `CholeskySolve` - NO ONNX equivalent
- ❌ `Lstsq` (least squares) - NO ONNX equivalent
- ❌ `TensorSolve` - NO ONNX equivalent
- ❌ `TensorInv` - NO ONNX equivalent
- ❌ `SolveContinuousLyapunov` - NO ONNX equivalent
- ❌ `BilinearSolveDiscreteLyapunov` - NO ONNX equivalent
- ❌ `SolveDiscreteARE` - NO ONNX equivalent

#### Tridiagonal Solvers
- ❌ `LUFactorTridiagonal` - NO ONNX equivalent
- ❌ `SolveLUFactorTridiagonal` - NO ONNX equivalent

### Category 3: Sparse Operations (NO ONNX SUPPORT)

**ONNX does NOT support sparse tensors** - All sparse ops would need custom implementation:

- ❌ ALL sparse operations (~40 ops)
- ❌ `CSM`, `CSMProperties` - NO ONNX equivalent
- ❌ `DenseFromSparse`, `SparseFromDense` - NO ONNX equivalent
- ❌ `AddSS`, `MulSS`, `Dot` (sparse) - NO ONNX equivalent
- ❌ `SparseBlockDiagonal` - NO ONNX equivalent
- ... (entire `pytensor/sparse/` module)

**Strategy**: Convert sparse to dense before export, or implement custom ONNX operator

### Category 4: Complex Number Operations (LIMITED SUPPORT)

ONNX has limited complex number support:

- ❌ `Complex` (construct from real/imag) - Limited support
- ❌ `ComplexFromPolar` - NO ONNX equivalent
- ❌ `Real`, `Imag` (extract components) - Limited support
- ❌ `Angle` - NO ONNX equivalent
- ❌ `Conj` (conjugate) - NO ONNX equivalent

### Category 5: Random Operations (MIXED SUPPORT)

#### Supported by ONNX
- ✅ `NormalRV` → `RandomNormal`
- ✅ `UniformRV` → `RandomUniform`
- ✅ `BinomialRV` → `Bernoulli` (for p=0.5) or custom
- ✅ `MultinomialRV` → `Multinomial`

#### NOT Supported by ONNX
- ❌ `BetaRV` - NO ONNX equivalent
- ❌ `GammaRV` - NO ONNX equivalent
- ❌ `ExponentialRV` - NO ONNX equivalent
- ❌ `WeibullRV` - NO ONNX equivalent
- ❌ `LogisticRV` - NO ONNX equivalent
- ❌ `VonMisesRV` - NO ONNX equivalent
- ❌ `DirichletRV` - NO ONNX equivalent
- ❌ `MvNormalRV` (multivariate normal) - NO ONNX equivalent
- ❌ `PoissonRV` - NO ONNX equivalent
- ❌ `GeometricRV` - NO ONNX equivalent
- ❌ `HyperGeometricRV` - NO ONNX equivalent
- ❌ `InvGammaRV` - NO ONNX equivalent
- ❌ `WaldRV` - NO ONNX equivalent
- ❌ `LaplaceRV` - NO ONNX equivalent
- ❌ `TriangularRV` - NO ONNX equivalent
- ❌ `LogNormalRV` - NO ONNX equivalent
- ❌ `CategoricalRV` - NO ONNX equivalent
- ❌ `IntegersRV` - NO ONNX equivalent
- ❌ `ChoiceWithoutReplacement` - NO ONNX equivalent
- ❌ `PermutationRV` - NO ONNX equivalent

**Note**: Random ops are problematic because:
1. ONNX Runtime may not support seeding consistently
2. Many distributions not supported
3. **Strategy**: Pre-compute random samples in Python, pass as inputs

### Category 6: Control Flow (PARTIAL SUPPORT)

- ⚠️ `Scan` - ONNX **has** `Scan` but semantics differ significantly
  - PyTensor Scan is more flexible
  - ONNX Scan is more restricted
  - May need to unroll loops
- ⚠️ `IfElse` - ONNX **has** `If` operator but limited
  - Works for simple conditionals
  - Complex branching may not translate

### Category 7: Specialized Tensor Operations

#### Fourier Transforms
- ❌ `RFFTOp` (real FFT) - NO ONNX equivalent (ONNX has DFT but limited)
- ❌ `IRFFTOp` - NO ONNX equivalent
- ❌ `Fourier` - NO ONNX equivalent

#### Window Functions
- ❌ `Bartlett` - NO ONNX equivalent

#### Advanced Indexing
- ⚠️ `AdvancedSubtensor` - Partial support via `Gather`
- ⚠️ `AdvancedIncSubtensor` - Partial support via `Scatter`

#### Other Operations
- ❌ `Unique` - NO direct ONNX equivalent
- ❌ `UnravelIndex` - NO ONNX equivalent
- ❌ `RavelMultiIndex` - NO ONNX equivalent
- ❌ `SearchsortedOp` - NO ONNX equivalent
- ❌ `FillDiagonal`, `FillDiagonalOffset` - NO ONNX equivalent
- ❌ `PermuteRowElements` - NO ONNX equivalent
- ❌ `Choose` - NO ONNX equivalent (different from `Where`)

### Category 8: Graph/Meta Operations

- ❌ `Scan` (inner graph) - Partial support
- ❌ `OpFromGraph` - NO ONNX equivalent (needs flattening)
- ❌ `FromFunctionOp` - NO ONNX equivalent
- ❌ `Print` - NO ONNX equivalent (debug op)
- ❌ `CheckAndRaise`, `Assert` - NO ONNX equivalent

### Summary Statistics

**Total PyTensor Ops**: ~280+
**Ops WITHOUT direct ONNX equivalent**: ~150+ (over 50%)

**Categories with GOOD ONNX support**:
- ✅ Basic arithmetic (Add, Sub, Mul, Div)
- ✅ Basic math (Exp, Log, Sqrt, Pow)
- ✅ Trigonometry (Sin, Cos, Tan, Asin, Acos, Atan)
- ✅ Hyperbolic (Sinh, Cosh, Tanh)
- ✅ Comparison ops (Equal, Less, Greater)
- ✅ Reductions (ReduceSum, ReduceMean, ReduceMax, ReduceMin)
- ✅ Tensor manipulation (Reshape, Transpose, Concat, Split, Slice)
- ✅ Matrix multiply (MatMul, Gemm)
- ✅ Neural network (Conv, BatchNorm, Dropout, Softmax, ReLU)

**Categories with POOR/NO ONNX support**:
- ❌ Special functions (Gamma, Bessel, Beta, Hypergeometric)
- ❌ Sparse operations (100% unsupported)
- ❌ Advanced linear algebra (decompositions, solvers)
- ❌ Most probability distributions
- ❌ Complex numbers
- ❌ Fourier transforms
- ❌ Some advanced tensor operations

### Mitigation Strategies

1. **Custom ONNX operators**: Implement missing ops as custom ONNX ops
   - Requires C++ implementation
   - Supported by ONNX Runtime

2. **Pre-computation**: For random ops, compute in Python and pass as inputs

3. **Approximation**: Some special functions can be approximated with polynomials

4. **Raise clear errors**: For unsupported ops, give users informative error messages

5. **Sparse → Dense conversion**: Warn users and convert automatically

6. **Decomposition**: Break complex ops into simpler ONNX-supported ops
   - Example: `Softplus(x)` → `Log(Add(1, Exp(x)))`

---

## Question 3: Gradient Computation (EXPANDED EXPLANATION)

**Question**: Should gradients be computed in PyTensor before export, or try to use ONNX's gradient support?

### Understanding the Problem

When you create a PyTensor function that computes gradients (for training models), you have two options:

**Option A: Compute gradients in PyTensor, then export the gradient graph**
```python
import pytensor.tensor as pt

# Forward pass
x = pt.vector('x')
w = pt.vector('w')
y = pt.dot(x, w)
loss = pt.sum(y ** 2)

# Compute gradient IN PyTensor
grad_w = pt.grad(loss, w)

# Export function that includes gradient
f = pt.function([x, w], [loss, grad_w])
export_onnx(f, "model_with_grad.onnx")  # Gradient already in graph
```

**Option B: Export forward pass only, let ONNX Runtime compute gradients**
```python
# Export only forward pass
f = pt.function([x, w], loss)
export_onnx(f, "model.onnx")

# Later, in JavaScript/WASM:
// Try to use ONNX Runtime's automatic differentiation
// (if available)
```

### Why This Matters

**PyTensor's gradient system** is very powerful:
- Supports all PyTensor ops
- Handles complex control flow (Scan, IfElse)
- Can optimize gradient graphs
- Supports custom gradients for ops

**ONNX's gradient support** is limited:
- ONNX has a concept called "training mode"
- `TrainingInfoProto` can store gradient information
- But ONNX Runtime's training support is:
  - Not universally available (especially in WASM)
  - Limited to specific operators
  - Not as flexible as PyTensor

### Detailed Comparison

| Aspect | PyTensor Gradients | ONNX Gradients |
|--------|-------------------|----------------|
| **Operator Support** | All PyTensor ops | Limited to supported ONNX ops |
| **Control Flow** | Full support (Scan, IfElse) | Limited (Loop, If) |
| **Custom Gradients** | Easy to define | Requires custom operators |
| **Optimization** | Many gradient optimizations available | Limited |
| **WASM Support** | Full (exported as part of graph) | Uncertain/Limited |
| **Graph Size** | Larger (includes gradient computation) | Smaller (forward pass only) |

### Recommended Approach: **Compute Gradients in PyTensor**

**Reasons**:

1. **Guaranteed Compatibility**: PyTensor gradients will work for all ops you use

2. **WASM Compatibility**: ONNX Runtime WASM may not support training/gradients
   - Focus is on inference
   - Gradient computation adds complexity

3. **Full Control**: You control the gradient computation and can optimize it

4. **Consistent Behavior**: Same gradient computation in Python and browser

5. **Export as Single Graph**: Forward + backward pass in one model
   ```python
   # Create training function with gradients
   x = pt.matrix('x')
   y_true = pt.vector('y_true')
   w = pt.shared(np.random.randn(784, 10))

   # Forward pass
   y_pred = pt.nnet.softmax(pt.dot(x, w))
   loss = pt.nnet.categorical_crossentropy(y_pred, y_true).mean()

   # Backward pass (compute in PyTensor)
   grad_w = pt.grad(loss, w)

   # Export function with gradient
   f = pt.function([x, y_true], [loss, grad_w])
   export_onnx(f, "trainable_model.onnx")
   ```

### When to Consider ONNX Gradients

Only if:
- You're using ONNX Runtime's training mode on server/desktop (not WASM)
- Your model uses only basic ops (MatMul, Conv, BatchNorm, etc.)
- You need dynamic gradient graphs (rare)

### Implementation Strategy

```python
def export_with_gradients(inputs, outputs, wrt, output_path):
    """
    Export PyTensor function with gradients included.

    Args:
        inputs: List of input variables
        outputs: List of output variables (e.g., loss)
        wrt: List of variables to compute gradients with respect to
        output_path: Path to save ONNX file
    """
    import pytensor.tensor as pt

    # Compute gradients in PyTensor
    grads = []
    for out in outputs:
        for param in wrt:
            grads.append(pt.grad(out, param))

    # Create function with forward + backward
    all_outputs = outputs + grads
    f = pt.function(inputs, all_outputs)

    # Export to ONNX
    export_onnx(f, output_path)

    return f

# Usage
x = pt.matrix('x')
y = pt.vector('y')
w = pt.vector('w')
loss = ((pt.dot(x, w) - y) ** 2).mean()

# Export model with gradient computation
export_with_gradients(
    inputs=[x, y, w],
    outputs=[loss],
    wrt=[w],
    output_path="model_with_gradients.onnx"
)
```

**Benefit**: Browser can compute gradients by just running the exported ONNX model!

---

## Question 4: Fixed Seeds for RNG (CONFIRMED)

**Question**: How to handle random number generation with fixed seeds?

**Answer**: **Use fixed seeds and manage RNG state carefully**

### Implementation Strategy

#### Approach 1: Pre-compute Random Values (RECOMMENDED for WASM)

Since ONNX's random support is limited and may not work consistently in WASM:

```python
# Don't use RandomVariable ops in exported graph
# Instead, pre-generate random values and pass as inputs

import numpy as np

# Create PyTensor function
x = pt.matrix('x')
dropout_mask = pt.vector('dropout_mask')  # Pass as input instead of random
y = x * dropout_mask

f = pt.function([x, dropout_mask], y)

# In browser:
# Generate random values in JavaScript
// const dropoutMask = Array(size).fill(0).map(() =>
//     Math.random() > 0.5 ? 1 : 0);
```

#### Approach 2: Use ONNX RandomNormal/RandomUniform with Fixed Seeds

If you must use random ops:

```python
import onnx
from onnx import helper, TensorProto

# Create ONNX RandomNormal node with fixed seed
random_node = helper.make_node(
    'RandomNormal',
    inputs=[],
    outputs=['random_output'],
    dtype=TensorProto.FLOAT,
    shape=[10, 10],
    mean=0.0,
    scale=1.0,
    seed=42  # Fixed seed for reproducibility
)
```

**Important Notes**:
- ONNX Runtime may not guarantee determinism across platforms
- WASM implementation might differ from CPU/GPU
- Different ONNX Runtime versions may produce different results

#### Approach 3: Hybrid - Generate in JavaScript with Seedable RNG

For browser demos, use JavaScript libraries with seedable RNG:

```javascript
// Use seedrandom library for deterministic random numbers
import seedrandom from 'seedrandom';

const rng = seedrandom('my-fixed-seed');

function generateRandomNormal(size, mean = 0, std = 1) {
    const values = new Float32Array(size);
    for (let i = 0; i < size; i++) {
        // Box-Muller transform
        const u1 = rng();
        const u2 = rng();
        const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        values[i] = mean + std * z;
    }
    return values;
}

// Use as input to ONNX model
const randomInput = generateRandomNormal(100);
const feeds = {
    'random_input': new ort.Tensor('float32', randomInput, [100])
};
```

### Recommendation for WebAssembly Demo

**Best Practice**:
1. **Avoid random ops in exported ONNX graph**
2. **Generate random values in JavaScript** with fixed seed
3. **Pass as inputs to model**

This ensures:
- ✅ Reproducibility across platforms
- ✅ Full control over RNG
- ✅ No dependency on ONNX Runtime's random implementation
- ✅ Works reliably in WASM

---

## Question 5: Control Flow (EXPANDED EXPLANATION)

**Question**: How to handle Scan ops and conditional operations?

### Understanding Control Flow in PyTensor vs ONNX

#### PyTensor's Control Flow

**Scan Op** (`pytensor/scan/op.py`):
- Most powerful control flow primitive
- Implements loops with state
- Can iterate over sequences
- Supports multiple outputs and updates
- Very flexible

```python
import pytensor.tensor as pt
from pytensor.scan import scan

# Example: Compute cumulative sum using scan
x = pt.vector('x')

def step(x_t, sum_tm1):
    """
    x_t: current element
    sum_tm1: previous sum
    """
    return sum_tm1 + x_t

result, updates = scan(
    fn=step,
    sequences=[x],
    outputs_info=[pt.zeros(())],  # Initial value for sum
)

f = pt.function([x], result)
# f([1, 2, 3, 4, 5]) → [1, 3, 6, 10, 15]
```

**IfElse Op** (`pytensor/ifelse.py`):
```python
from pytensor.ifelse import ifelse

# Conditional execution
condition = pt.scalar('condition')
x = pt.scalar('x')
y = pt.scalar('y')

result = ifelse(condition, x * 2, y * 2)
```

#### ONNX Control Flow

**ONNX Loop** (equivalent to Scan, but more restrictive):
- Fixed iteration count or condition-based
- Body is a separate subgraph
- More rigid structure

**ONNX If** (equivalent to IfElse):
- Two branches (then_branch and else_branch)
- Each branch is a separate subgraph
- Both branches must have same output types

### Key Differences

| Feature | PyTensor Scan | ONNX Loop/Scan |
|---------|--------------|----------------|
| **Flexibility** | Very flexible | More rigid |
| **State Management** | Easy | Complex |
| **Multiple Outputs** | Easy | Supported but verbose |
| **Gradients** | Automatic | Manual setup |
| **Nested Loops** | Easy | Difficult |

### The Problem

When converting PyTensor Scan to ONNX:

```python
# PyTensor: Simple and flexible
result, updates = scan(fn=step, sequences=[x], outputs_info=[init])

# ONNX: Requires explicit subgraph construction
# - Must create separate GraphProto for loop body
# - Must specify loop carried dependencies
# - Must handle trip count and termination condition
# - More boilerplate
```

### Strategies for Handling Control Flow

#### Strategy 1: Loop Unrolling (SIMPLE, RECOMMENDED for small loops)

**Convert Scan to explicit sequential operations**:

```python
# Original Scan
x = pt.vector('x')
result, _ = scan(fn=lambda x_t, sum: sum + x_t,
                 sequences=[x],
                 outputs_info=[0])

# Unrolled version (if x has known fixed length, e.g., 5)
x = pt.vector('x')  # length 5
s0 = 0
s1 = s0 + x[0]
s2 = s1 + x[1]
s3 = s2 + x[2]
s4 = s3 + x[3]
s5 = s4 + x[4]
result = pt.stack([s1, s2, s3, s4, s5])
```

**Pros**:
- Simple to implement
- No need to understand ONNX Loop
- Works reliably

**Cons**:
- Only works for fixed-length sequences
- Graph becomes large for long sequences
- Not suitable for dynamic loops

#### Strategy 2: Convert to ONNX Loop (COMPLEX, for dynamic loops)

**Create ONNX Loop node with subgraph**:

```python
def scan_to_onnx_loop(scan_op, scan_node):
    """Convert PyTensor Scan to ONNX Loop."""

    # Extract scan properties
    inner_fgraph = scan_op.inner_fgraph
    n_steps = scan_node.inputs[0]  # Trip count

    # Create loop body as separate GraphProto
    body_nodes = []
    for apply_node in inner_fgraph.toposort():
        body_nodes.append(onnx_funcify(apply_node.op, apply_node))

    body_graph = onnx.helper.make_graph(
        nodes=body_nodes,
        name="scan_body",
        inputs=[...],  # Iteration number, conditions, loop state
        outputs=[...],  # Updated conditions, updated state
    )

    # Create Loop node
    loop_node = onnx.helper.make_node(
        'Loop',
        inputs=['trip_count', 'condition', 'loop_state_in'],
        outputs=['loop_state_out'],
        body=body_graph
    )

    return loop_node
```

**Pros**:
- Handles dynamic loops
- Compact graph representation
- Preserves semantics

**Cons**:
- Complex to implement
- ONNX Loop semantics differ from Scan
- Harder to debug

#### Strategy 3: Replace with ONNX Built-ins (BEST when possible)

Many Scan operations can be replaced with built-in ONNX ops:

```python
# PyTensor Scan for cumsum
result, _ = scan(lambda x_t, sum: sum + x_t, sequences=[x], outputs_info=[0])

# ↓ Replace with ONNX CumSum operator ↓

cumsum_node = onnx.helper.make_node(
    'CumSum',
    inputs=['x'],
    outputs=['result']
)
```

**Common replacements**:
- Cumulative sum → `CumSum`
- Cumulative product → `CumProd` (if available)
- Element-wise operations over sequence → Use broadcasting
- Reductions → `ReduceSum`, `ReduceMean`, etc.

#### Strategy 4: Raise Error for Unsupported Scans

For complex Scans that can't be easily converted:

```python
@onnx_funcify.register(Scan)
def onnx_funcify_Scan(op, node, **kwargs):
    # Try simple conversions
    if can_unroll(node):
        return unroll_scan(node)
    elif has_onnx_equivalent(node):
        return replace_with_onnx_builtin(node)
    else:
        raise NotImplementedError(
            f"Scan operation cannot be converted to ONNX: {node}\n"
            f"Reason: Complex control flow not supported.\n"
            f"Suggestion: Try simplifying the scan or using a fixed-length sequence."
        )
```

### Handling IfElse

**IfElse is easier** - direct mapping to ONNX If:

```python
@onnx_funcify.register(IfElse)
def onnx_funcify_IfElse(op, node, **kwargs):
    condition = node.inputs[0]
    true_branch = node.inputs[1]
    false_branch = node.inputs[2]

    # Create subgraphs for branches
    then_graph = create_onnx_graph(true_branch)
    else_graph = create_onnx_graph(false_branch)

    # Create If node
    if_node = onnx.helper.make_node(
        'If',
        inputs=[onnx_funcify(condition)],
        outputs=['result'],
        then_branch=then_graph,
        else_branch=else_graph
    )

    return if_node
```

### Recommendations

**For WebAssembly Demo**:
1. **Avoid Scan if possible** - use built-in reductions and operations
2. **If Scan needed**:
   - Use fixed-length sequences and unroll
   - Or replace with ONNX built-ins (CumSum, etc.)
3. **IfElse**: Convert to ONNX If (straightforward)
4. **Document limitations**: Be clear about what control flow is supported

---

## Question 6: Performance (EXPANDED EXPLANATION)

**Question**: What's the performance overhead of ONNX Runtime WASM vs native?

### Understanding the Performance Landscape

#### Native Execution Options

**1. CPU (Native C/C++)**
- Direct memory access
- Full SIMD instructions (AVX, SSE)
- Multi-threading
- **Baseline**: 1x performance

**2. GPU (CUDA/ROCm)**
- Massive parallelism
- High memory bandwidth
- Specialized tensor cores
- **Performance**: 10-100x faster than CPU (for large models)

**3. ONNX Runtime (Native)**
- Optimized C++ implementation
- Uses hardware-specific backends (MKL, CuBLAS, etc.)
- Graph optimizations
- **Performance**: ~0.8-1x native (very close)

#### WebAssembly Execution

**4. ONNX Runtime Web (WASM)**
- Compiled to WebAssembly
- Runs in browser sandbox
- Limited access to hardware
- **Performance**: ~0.1-0.5x native (10-50% of native speed)

### Performance Comparison

| Backend | Platform | Typical Speed | Memory | Multi-thread | SIMD |
|---------|----------|---------------|---------|--------------|------|
| **Native CPU** | Server/Desktop | 1.0x (baseline) | Direct | Yes | Full |
| **Native GPU** | Server/Desktop | 10-100x | High BW | Yes | N/A |
| **ONNX RT Native** | Server/Desktop | 0.8-1.0x | Direct | Yes | Full |
| **ONNX RT WASM** | Browser | 0.1-0.5x | Limited | Limited | Limited |
| **JavaScript** | Browser | 0.01-0.1x | Limited | No | No |

### Why WASM is Slower

**1. Limited SIMD Support**
- WebAssembly SIMD is available but not as powerful as native AVX-512
- Browser support varies
- Performance gains limited

```javascript
// WASM SIMD (128-bit)
v128.add(a, b);  // 4 floats at once

// vs Native AVX-512 (512-bit)
_mm512_add_ps(a, b);  // 16 floats at once
```

**2. Memory Constraints**
- WASM memory is separate from native memory
- Copies between JavaScript and WASM
- Limited heap size (typically 2-4 GB)

**3. Threading Limitations**
- SharedArrayBuffer required for threading
- Not enabled on all browsers (security concerns)
- Limited number of workers

**4. JIT Compilation**
- WASM needs to be compiled at runtime
- Optimization less aggressive than native
- Browser-dependent performance

**5. Garbage Collection Pauses**
- JavaScript GC can pause execution
- Affects real-time performance

### Concrete Performance Measurements

Based on benchmarks from ONNX Runtime Web:

#### Small Models (e.g., MobileNet, ResNet-18)
- **Native CPU**: 10 ms/inference
- **WASM (Chrome)**: 30-50 ms/inference
- **WASM (Firefox)**: 40-70 ms/inference
- **Slowdown**: **3-7x slower** than native

#### Medium Models (e.g., ResNet-50)
- **Native CPU**: 50 ms/inference
- **WASM**: 150-300 ms/inference
- **Slowdown**: **3-6x slower**

#### Large Models (e.g., BERT-base)
- **Native CPU**: 100 ms/inference
- **WASM**: 500-1000 ms/inference
- **Slowdown**: **5-10x slower**

**Note**: With WebGPU support (newer), performance can improve significantly:
- **WebGPU**: 2-5x faster than WASM CPU
- But still **2-5x slower** than native GPU

### What This Means for Your Demo

#### For Interactive Demos (Good Use Case)
- **Small models**: 30-50 ms is acceptable
- **Real-time feel**: < 100 ms latency
- **Works for**: Image classification, simple NLP, style transfer

#### For Production Inference (Challenging)
- **Large models**: 500+ ms is too slow
- **Not suitable** for real-time applications
- **Better**: Use server-side inference, WASM for client-side caching

### Optimization Strategies

#### 1. Model Optimization
```python
# Quantize model to int8
import onnx
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic(
    "model.onnx",
    "model_quantized.onnx",
    weight_type=onnx.TensorProto.INT8
)
# Can reduce model size by 4x and improve speed 2-3x
```

#### 2. Graph Optimization
```python
import onnxruntime as ort

# Enable all optimizations
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.optimized_model_filepath = "model_optimized.onnx"

session = ort.InferenceSession("model.onnx", sess_options)
```

#### 3. Use WebGPU (if available)
```javascript
const session = await ort.InferenceSession.create('model.onnx', {
    executionProviders: ['webgpu']  // Use GPU if available
});
```

#### 4. Batch Processing
```javascript
// Instead of 1 inference at 50ms
// Do 10 inferences at 150ms (15ms each)
const batch = [input1, input2, ..., input10];
const results = await session.run({ input: concatenate(batch) });
```

#### 5. Web Workers
```javascript
// Offload inference to web worker
// Prevents blocking main thread
const worker = new Worker('inference-worker.js');
worker.postMessage({ model: 'model.onnx', input: data });
worker.onmessage = (e) => console.log('Result:', e.data);
```

### Realistic Expectations

**For a simple demo (e.g., z = x + y * 2)**:
- Native: < 1 ms
- WASM: ~1-5 ms
- **Performance**: Good enough, no issues

**For a small neural network (10 layers, 1M params)**:
- Native: ~10 ms
- WASM: ~30-50 ms
- **Performance**: Acceptable for demos

**For a large model (BERT, GPT)**:
- Native: ~100 ms
- WASM: ~500-1000 ms
- **Performance**: May feel slow, consider server-side

### Recommendation

**For your WebAssembly demo**:
1. **Start simple**: Test with small models first
2. **Measure early**: Profile performance in target browsers
3. **Set expectations**: Document that it's a demo, not production
4. **Progressive enhancement**:
   - Use WASM for client-side inference when possible
   - Fall back to server for large models
5. **Future-proof**: Design for WebGPU to improve performance later

**Bottom Line**: WASM will be **3-10x slower** than native, but for demos and small models, this is acceptable. Users understand browser limitations.

---

## Summary of Answers

1. ✅ **Shape Inference**: Use example inputs at compile time, leverage PyTensor's `infer_shape`, support dynamic axes
2. ✅ **Custom Ops**: ~150 ops lack ONNX equivalents (special functions, sparse, advanced LA) - need custom ops or raise errors
3. ✅ **Gradients**: Compute in PyTensor before export (better support, WASM compatible)
4. ✅ **RNG**: Use fixed seeds in JavaScript, pass random values as inputs (most reliable)
5. ✅ **Control Flow**: Unroll simple loops, convert IfElse to ONNX If, avoid complex Scans
6. ✅ **Performance**: Expect 3-10x slowdown vs native, acceptable for demos, optimize with quantization/WebGPU

All questions answered with concrete implementation strategies!
