---
date: 2025-10-15T07:28:53Z
researcher: Claude Code
git_commit: c58f10beb2aa5e5238f1420107e3bc1103e87c31
branch: onnx-backend
repository: pymc-devs/pytensor
topic: "What do I need to do to support training on GPUs with PyTensor natively"
tags: [research, codebase, gpu, cuda, training, backends, jax, pytorch, mlx, device-management]
status: complete
last_updated: 2025-10-15
last_updated_by: Claude Code
---

# Research: What do I need to do to support training on GPUs with PyTensor natively

**Date**: 2025-10-15T07:28:53Z
**Researcher**: Claude Code
**Git Commit**: c58f10beb2aa5e5238f1420107e3bc1103e87c31
**Branch**: onnx-backend
**Repository**: pymc-devs/pytensor

## Research Question

What do I need to do to support training on GPUs with PyTensor natively?

## Summary

PyTensor **does not have native CUDA/GPU support** like its predecessor Theano. Instead, PyTensor uses a **backend abstraction model** where GPU acceleration is delegated to external frameworks (JAX, PyTorch, MLX, Numba). This is a fundamental architectural decision.

**To support GPU training in PyTensor, you have three main options:**

1. **Use JAX Backend** (Recommended) - Most mature, supports NVIDIA GPUs and Google TPUs via XLA
2. **Use PyTorch Backend** - Native CUDA support, extensive GPU testing infrastructure
3. **Use MLX Backend** - For Apple Silicon (M1/M2/M3) GPU acceleration
4. **Implement Native CUDA Backend** - Major undertaking, would require creating new linker and dispatch system

**Training Infrastructure Status:**
- ✅ Complete automatic differentiation (grad, jacobian, hessian)
- ✅ Gradient computation for all operations (L_op, R_op)
- ✅ Shared variables and updates mechanism
- ✅ Scan operations for RNNs
- ❌ **No built-in optimizers** (SGD, Adam, etc.) - must implement manually

## Detailed Findings

### 1. Backend Architecture and GPU Support

#### Current Architecture
PyTensor uses a **Linker + Dispatch** pattern for backends:
- **Linker**: Compiles PyTensor graph into executable function
- **Dispatch**: Translates PyTensor ops to backend-specific operations

**6 Existing Backends:**
1. **Python** (`PerformLinker`) - CPU only, uses `.perform()` methods
2. **C** (`CLinker`) - CPU only, compiles to C code
3. **JAX** (`JAXLinker`) - GPU/TPU capable via XLA
4. **Numba** (`NumbaLinker`) - LLVM JIT, theoretical CUDA support
5. **PyTorch** (`PytorchLinker`) - CUDA GPU support
6. **MLX** (`MLXLinker`) - Apple Silicon GPU

#### Backend Files
- `pytensor/link/jax/linker.py:9` - JAXLinker class
- `pytensor/link/pytorch/linker.py:5-70` - PytorchLinker with GPU support (line 69-70)
- `pytensor/link/numba/linker.py:4` - NumbaLinker class
- `pytensor/link/mlx/linker.py:4-52` - MLXLinker for Apple GPU
- `pytensor/compile/mode.py:464-524` - Mode definitions (NUMBA, JAX, PYTORCH, MLX)

### 2. JAX Backend - Recommended for GPU Training

#### Why JAX?
- **Most mature GPU support** via Google's XLA compiler
- Supports NVIDIA GPUs and Google TPUs
- Automatic differentiation built-in
- Extensive PyTensor integration (45+ test files)

#### Implementation
**Dispatch System:**
- `pytensor/link/jax/dispatch/__init__.py` - `jax_funcify` and `jax_typify` registries
- `pytensor/link/jax/dispatch/basic.py:28-46` - Core dispatch implementations
- 20+ dispatch files for operations (elemwise, math, linalg, conv, etc.)

**Usage Pattern:**
```python
import pytensor
import pytensor.tensor as pt

# Set JAX backend for GPU acceleration
with pytensor.config.change_flags(mode="JAX"):
    x = pt.vector("x")
    y = pt.vector("y")
    z = x + y
    f = pytensor.function([x, y], z)

    # JAX automatically uses GPU if available
    result = f([1, 2, 3], [4, 5, 6])
```

**Device Management:**
JAX handles GPU placement automatically via `jax.config`:
- `jax.config.update("jax_platform_name", "gpu")` - Force GPU
- `jax.config.update("jax_enable_x64", True)` - Enable float64 on GPU

**Testing Infrastructure:**
- `tests/link/jax/test_basic.py:36-96` - `compare_jax_and_py()` testing helper
- Verifies results are `jax.Array` (device arrays)
- 45 test files covering all operations

### 3. PyTorch Backend - Native CUDA Support

#### Why PyTorch?
- **Native CUDA support** with extensive testing
- Familiar API for PyTorch users
- Automatic CPU↔GPU conversion
- Active development

#### Implementation
**Automatic GPU Handling:**
```python
# From pytensor/link/pytorch/linker.py:40-85
class PytorchLinker(JITLinker):
    def jit_compile(self, fn):
        class wrapper:
            def __call__(self, *inputs, **kwargs):
                # Convert NumPy → PyTorch tensors (GPU if available)
                outs = self.fn(*(pytorch_typify(inp) for inp in inputs), **kwargs)

                # Convert GPU tensors → CPU → NumPy
                return tuple(out.cpu().numpy() for out in outs)
```

**GPU Testing Pattern:**
```python
# From tests/link/pytorch/test_basic.py:88-155
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pytorch_operation(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    with torch.device(device):
        # Operations run on specified device
        x = vector("x")
        f = function([x], x * 2, mode="PYTORCH")
        result = f([1, 2, 3])
```

**Key Features:**
- Transparent device management
- Automatic memory transfers
- Shared variables work on GPU
- Results automatically converted to NumPy

**Testing Infrastructure:**
- `tests/link/pytorch/test_basic.py:88-189` - CUDA device tests
- 14 test files total
- Parametrized tests for CPU/CUDA

### 4. MLX Backend - Apple Silicon GPU

#### Why MLX?
- **Apple Silicon GPU acceleration** (M1/M2/M3)
- Unified memory architecture
- Metal-based performance
- Similar API to JAX

#### Implementation
- `pytensor/link/mlx/linker.py:4-52` - MLXLinker implementation
- 10 dispatch files
- `tests/link/mlx/test_basic.py:30-105` - Testing utilities

**Usage Pattern:**
```python
with pytensor.config.change_flags(mode="MLX"):
    # Operations run on Apple Silicon GPU
    f = pytensor.function([x], x * 2)
    result = f([1, 2, 3])  # Returns mx.array
```

### 5. Training Infrastructure

#### Automatic Differentiation (Complete ✅)
**Core Gradient Module:**
- `pytensor/gradient.py` - Main AD infrastructure
  - `grad()` - Reverse mode (backpropagation)
  - `Lop()` - Linear operator (reverse mode)
  - `Rop()` - R-operator (forward mode)
  - `jacobian()` - Jacobian matrix computation
  - `hessian()` - Hessian matrix computation
  - `verify_grad()` - Numerical gradient verification

**Operator-Level Gradients:**
- `pytensor/graph/op.py` - Base Op class with `L_op` and `R_op` methods
- All operations implement gradients via `L_op` for backprop

**Testing:**
- `tests/test_gradient.py` - Comprehensive gradient tests
- `tests/test_rop.py` - Forward mode tests
- Operation-specific gradient tests in `tests/tensor/`

#### Loss Functions and Activations (Complete ✅)
**Neural Network Operations:**
- `pytensor/tensor/special.py` - Softmax, LogSoftmax
- `pytensor/tensor/xlogx.py` - Cross-entropy components (XlogX, XlogY0)
- `pytensor/tensor/math.py` - Activations (sigmoid, tanh, softplus)

**Reduction Operations:**
- `sum()`, `mean()`, `var()`, `std()` - Loss computation
- All support gradients

#### Update Mechanism (Complete ✅)
**Shared Variables:**
- `pytensor/compile/sharedvalue.py` - SharedVariable class
  - `get_value()` / `set_value()` - Access/modify parameters
  - Works transparently with GPU backends

**Updates:**
- `pytensor/updates.py` - OrderedUpdates class
- `pytensor/compile/io.py` - In/Out classes for updates
- `pytensor/compile/function/pfunc.py` - Function compilation with updates

**Pattern:**
```python
# Manual optimizer implementation required
W = pytensor.shared(np.random.randn(100, 10))
b = pytensor.shared(np.zeros(10))

x = pt.matrix('x')
y_pred = pt.nnet.softmax(pt.dot(x, W) + b)
loss = pt.nnet.categorical_crossentropy(y_pred, y_true).mean()

# Compute gradients
grads = pytensor.grad(loss, [W, b])

# Define updates (manual SGD)
learning_rate = 0.01
updates = OrderedUpdates()
updates[W] = W - learning_rate * grads[0]
updates[b] = b - learning_rate * grads[1]

# Compile training function
train_fn = pytensor.function([x, y_true], loss, updates=updates, mode="JAX")
```

#### Optimizers (Missing ❌)
**No built-in optimizers.** Users must implement:
- SGD (Stochastic Gradient Descent)
- Adam
- RMSprop
- Momentum
- etc.

**Example Implementation:**
```python
class SGDOptimizer:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def get_updates(self, params, grads):
        updates = OrderedUpdates()
        for param, grad in zip(params, grads):
            updates[param] = param - self.lr * grad
        return updates

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Timestep

    def get_updates(self, params, grads):
        updates = OrderedUpdates()
        self.t += 1

        for param, grad in zip(params, grads):
            if param not in self.m:
                self.m[param] = pytensor.shared(np.zeros_like(param.get_value()))
                self.v[param] = pytensor.shared(np.zeros_like(param.get_value()))

            m = self.m[param]
            v = self.v[param]

            m_new = self.beta1 * m + (1 - self.beta1) * grad
            v_new = self.beta2 * v + (1 - self.beta2) * grad**2

            m_hat = m_new / (1 - self.beta1**self.t)
            v_hat = v_new / (1 - self.beta2**self.t)

            updates[m] = m_new
            updates[v] = v_new
            updates[param] = param - self.lr * m_hat / (pt.sqrt(v_hat) + self.epsilon)

        return updates
```

#### Convolutional Operations (Complete ✅)
- `pytensor/tensor/conv/abstract_conv.py` - Convolution with gradients
- `pytensor/tensor/signal/conv.py` - Signal processing convolutions
- `pytensor/tensor/pool.py` - Pooling operations (newly added)
- `pytensor/tensor/batchnorm.py` - Batch normalization (newly added)
- `pytensor/tensor/resize.py` - Resize operations (newly added)

#### Recurrent Operations (Complete ✅)
**Scan Infrastructure:**
- `pytensor/scan/basic.py` - Main scan implementation
- `pytensor/scan/op.py` - Scan operator with gradient support
- `pytensor/scan/checkpoints.py` - Memory-efficient gradients
- `pytensor/scan/views.py` - Higher-level interfaces (map, reduce, foldl, foldr)

**Pattern:**
```python
# RNN cell example
def rnn_step(x_t, h_prev, W_h, W_x):
    return pt.tanh(pt.dot(h_prev, W_h) + pt.dot(x_t, W_x))

outputs, updates = pytensor.scan(
    fn=rnn_step,
    sequences=X,
    outputs_info=h0,
    non_sequences=[W_h, W_x]
)
```

### 6. Configuration and Device Management

#### Current Device Configuration
**Config File:**
- `pytensor/configdefaults.py:263-265` - Device parameter
```python
# Currently only accepts "cpu"
device = "cpu"
```

**Config System:**
- `pytensor/configparser.py:515` - DeviceParam class
- `pytensor/configparser.py:48-60` - Context manager for config changes

**Environment Variables:**
- `PYTENSOR_FLAGS` - Comma-separated config overrides
- `PYTENSORRC` - Colon-delimited list of config files

**Usage:**
```bash
# Set backend via environment variable
PYTENSOR_FLAGS='mode=JAX' python train.py

# Or in .pytensorrc file
[global]
mode = JAX
floatX = float32
```

```python
# Or via context manager
with pytensor.config.change_flags(mode="JAX", floatX="float32"):
    # GPU operations here
    pass
```

#### Mode Configuration
**Available Modes:**
- `pytensor/compile/mode.py:464-524` - Mode definitions
- Supported: "Mode", "DebugMode", "FAST_RUN", "FAST_COMPILE", "JAX", "NUMBA", "PYTORCH", "MLX"

#### Profiling GPU Memory
**Memory Tracking:**
- `pytensor/compile/profiling.py:875-1000` - ProfileStats class
- Tracks separate CPU and GPU memory (infrastructure in place)
- `config.profile = True` - Enable profiling
- `config.profile_memory = True` - Enable memory profiling

### 7. Implementing Native CUDA Backend (Major Undertaking)

If you want to implement a **native CUDA backend** (not using JAX/PyTorch), you would need:

#### Required Components

**1. New Linker**
- Create `pytensor/link/cuda/linker.py`
- Extend `JITLinker` base class
- Implement CUDA kernel compilation
- Handle device memory management

**2. Dispatch System**
- Create `pytensor/link/cuda/dispatch/__init__.py`
- Implement `cuda_funcify` and `cuda_typify` registries
- Convert each PyTensor op to CUDA kernel

**3. Operation Implementations**
- ~50+ dispatch files needed (see JAX/PyTorch as reference)
- Elemwise, math, linalg, conv, pool, etc.
- CUDA kernel code for each operation

**4. Device Management**
- Extend `DeviceParam` in `pytensor/configdefaults.py`
- Add "cuda", "cuda0", "cuda1" support
- Implement device transfer operations

**5. Type System**
- Create CUDA-specific types
- Handle device memory representation
- Automatic CPU↔GPU transfers

**6. Testing Infrastructure**
- Create `tests/link/cuda/` directory
- Implement parameterized CPU/GPU tests
- Follow PyTorch backend test patterns

#### Estimated Effort
- **6-12 months** full-time development
- **10,000+ lines of code**
- Deep CUDA and PyTensor expertise required

#### Risks
- Maintenance burden (CUDA API changes)
- Performance optimization complexity
- Limited value (JAX/PyTorch already provide GPU support)

## Code References

### GPU Backend Implementations
- `pytensor/link/jax/linker.py:9` - JAXLinker (GPU via XLA)
- `pytensor/link/pytorch/linker.py:5-70` - PytorchLinker (CUDA support, line 69-70)
- `pytensor/link/mlx/linker.py:4-52` - MLXLinker (Apple Silicon)
- `pytensor/compile/mode.py:464-524` - Backend mode definitions

### Training Infrastructure
- `pytensor/gradient.py` - Automatic differentiation (grad, Lop, Rop, jacobian, hessian)
- `pytensor/updates.py` - OrderedUpdates for parameter updates
- `pytensor/compile/sharedvalue.py` - SharedVariable for parameters
- `pytensor/scan/basic.py` - Scan for RNNs
- `pytensor/tensor/special.py` - Softmax and neural network operations
- `pytensor/tensor/xlogx.py` - Cross-entropy components

### Configuration
- `pytensor/configdefaults.py:263-265` - Device parameter (CPU only currently)
- `pytensor/configdefaults.py:307-311` - Mode configuration
- `pytensor/configparser.py:515` - DeviceParam class
- `pytensor/compile/profiling.py:875-1000` - Memory profiling with GPU tracking

### GPU Testing
- `tests/link/pytorch/test_basic.py:88-189` - CUDA device tests
- `tests/link/jax/test_basic.py:36-96` - JAX GPU testing utilities
- `tests/link/mlx/test_basic.py:30-105` - MLX testing utilities

### Examples
- `examples/onnx/onnx-mnist-demo/train_mnist_cnn.py` - Complete CNN training example

## Architecture Insights

### Backend Abstraction Design
PyTensor uses a **delegation model** for GPU support rather than implementing CUDA directly:

**Advantages:**
1. ✅ Leverages mature GPU ecosystems (JAX/XLA, PyTorch/CUDA)
2. ✅ Reduces maintenance burden
3. ✅ Supports multiple hardware backends (NVIDIA, Google TPU, Apple Silicon)
4. ✅ Benefits from upstream optimizations

**Trade-offs:**
1. ⚠️ Depends on external frameworks
2. ⚠️ Less control over GPU-specific optimizations
3. ⚠️ Multiple installation paths (jax, torch, mlx)

### Linker + Dispatch Pattern
All backends follow the same pattern:
```
PyTensor Graph → Linker → Backend-Specific Graph → Execute
                    ↓
                 Dispatch
                (op translation)
```

**Key Files:**
- `pytensor/link/basic.py:576-596` - JITLinker base class
- `pytensor/compile/mode.py` - Mode selection
- `pytensor/link/*/dispatch/__init__.py` - Dispatch registries

### Memory Management
- **Shared variables** work transparently across devices
- Backend linkers handle CPU↔GPU transfers
- `sharedvalue.py` provides unified interface
- Results automatically converted to NumPy

## Historical Context (from thoughts/)

Found **0 documents** specifically about GPU/CUDA support in the thoughts/ directory.

Found **3 documents** about backend architecture:
- `thoughts/shared/research/2025-10-14_adding-new-backend-onnx-xla.md` - How to add new backends (XLA is JAX's GPU backend)
- `thoughts/shared/research/2025-10-14_backend-comparison-dataflow.md` - Comparison of all 6 backends
- `thoughts/shared/research/2025-10-14_backend-dataflow-example.md` - Backend execution patterns

Found **1 document** about training:
- `thoughts/shared/plans/yolo11n-pytensor-training.md` - YOLO training plan (no GPU discussion)

**Key Finding:** PyTensor has GPU-capable backends (JAX, PyTorch, MLX) but no dedicated documentation about GPU usage, best practices, or implementation details in the thoughts/ directory.

## Related Research

- Backend architecture: `thoughts/shared/research/2025-10-14_adding-new-backend-onnx-xla.md`
- Backend comparison: `thoughts/shared/research/2025-10-14_backend-comparison-dataflow.md`
- Training example: `thoughts/shared/plans/yolo11n-pytensor-training.md`

## Recommendations

### For Immediate GPU Training Support

**Option 1: Use JAX Backend (Recommended)**
```python
import pytensor
import pytensor.tensor as pt
from pytensor import function, grad, shared
from pytensor.updates import OrderedUpdates
import numpy as np

# Configure JAX backend
with pytensor.config.change_flags(mode="JAX"):
    # Define model
    W = shared(np.random.randn(784, 10).astype('float32'))
    b = shared(np.zeros(10).astype('float32'))

    x = pt.matrix('x')
    y_true = pt.matrix('y_true')

    y_pred = pt.nnet.softmax(pt.dot(x, W) + b)
    loss = pt.nnet.categorical_crossentropy(y_pred, y_true).mean()

    # Compute gradients (on GPU)
    grads = grad(loss, [W, b])

    # Manual optimizer
    lr = 0.01
    updates = OrderedUpdates()
    updates[W] = W - lr * grads[0]
    updates[b] = b - lr * grads[1]

    # Compile (JAX uses GPU automatically)
    train_fn = function([x, y_true], loss, updates=updates)

    # Train
    for epoch in range(10):
        batch_loss = train_fn(X_train, Y_train)
        print(f"Epoch {epoch}, Loss: {batch_loss}")
```

**Option 2: Use PyTorch Backend**
```python
with pytensor.config.change_flags(mode="PYTORCH"):
    # Same code as above
    # PyTorch uses CUDA automatically if available
    pass
```

**Option 3: Use MLX Backend (Apple Silicon)**
```python
with pytensor.config.change_flags(mode="MLX"):
    # Same code as above
    # MLX uses Apple GPU automatically
    pass
```

### For Advanced Users

**Create Optimizer Library:**
1. Implement common optimizers (SGD, Adam, RMSprop)
2. Package as `pytensor.optimizers` module
3. Contribute back to PyTensor

**Example Structure:**
```python
# pytensor/optimizers/__init__.py
from .sgd import SGD
from .adam import Adam
from .rmsprop import RMSprop

# pytensor/optimizers/base.py
class Optimizer:
    def get_updates(self, params, grads):
        raise NotImplementedError
```

### For Core Contributors

**Native CUDA Backend:**
Only pursue if:
- JAX/PyTorch don't meet requirements
- Team has CUDA expertise
- 6-12 month timeline acceptable
- Willing to maintain long-term

**Steps:**
1. Study JAX/PyTorch linker implementations
2. Create `pytensor/link/cuda/` directory
3. Implement linker and dispatch system
4. Add CUDA kernels for operations
5. Create extensive test suite
6. Document GPU-specific features

## Open Questions

1. **Should PyTensor implement built-in optimizers?**
   - Pro: Easier for users, consistent API
   - Con: Adds maintenance burden, overlaps with higher-level libraries

2. **Should device parameter support "cuda0", "cuda1", etc.?**
   - Currently only "cpu" is supported
   - Backend frameworks handle device selection
   - May add confusion vs. simplicity

3. **Should PyTensor add GPU-specific optimizations?**
   - E.g., fused kernels, memory pooling
   - Or rely on backend frameworks?

4. **Documentation gaps:**
   - No GPU usage guide
   - No backend selection documentation
   - No training examples with GPU

5. **Should there be a native CUDA backend?**
   - Large engineering effort
   - Limited value given JAX/PyTorch exist
   - But could enable PyTensor-specific optimizations
