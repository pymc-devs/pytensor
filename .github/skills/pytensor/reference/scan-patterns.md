# Scan Patterns and Examples

`pytensor.scan()` is how you express loops, recurrences, and sequential computations in PyTensor's static graph. After reading this reference you can: implement cumulative operations (running sums, products), build recurrent models (RNN, AR, Fibonacci), write state-space models (Kalman filter), create while-loops with early stopping via `until`, generate random walks, and manage memory efficiently with `scan_checkpoints`. Covers the full function signature, the strict `fn` parameter ordering convention, and all common scan idioms (SITSOT, MITSOT, no-sequence iteration).

## Contents
- Full function signature
- fn parameter order convention
- Basic map (no recurrence)
- Cumulative sum (SITSOT)
- Fibonacci (MITSOT — multiple taps)
- RNN-style recurrence
- Power iteration (no sequences)
- While loop with until
- AR(2) time series
- Kalman filter
- Random walk
- Key rules and performance tips
- scan_checkpoints for memory efficiency

## Function Signature

```python
pytensor.scan(
    fn,                      # step function (one iteration)
    sequences=None,          # variables to iterate over
    outputs_info=None,       # initial states for recurrent outputs
    non_sequences=None,      # constants passed at every step
    n_steps=None,            # number of iterations (if no sequences)
    truncate_gradient=-1,    # truncated BPTT (steps to backprop, -1=all)
    go_backwards=False,      # iterate in reverse
    mode=None,               # compilation mode for internal graph
    name=None,
    strict=False,            # raise error for unused inputs
    return_list=False,
)
# Returns: (outputs, updates)
```

## fn Parameter Order Convention

The step function receives arguments in this strict order:

```
1. Elements from sequences (in order provided)
2. Taps from recurrent outputs (oldest to newest, first output first)
3. Non-sequences (in order provided)
```

```python
def step(seq1_t, seq2_t, out_tm2, out_tm1, non_seq1, non_seq2):
    #      ^sequences^   ^recurrent output taps^  ^non-sequences^
    return new_output
```

## Basic Map (No Recurrence)

Apply function elementwise to sequences:

```python
output, updates = pytensor.scan(
    fn=lambda a, b: a * b,
    sequences=[vector1, vector2]
)
```

## Cumulative Sum (SITSOT)

Single Input Tap, Single Output Tap — the simplest recurrence:

```python
# y[t] = y[t-1] + x[t]
results, updates = pytensor.scan(
    fn=lambda x_t, y_tm1: y_tm1 + x_t,
    sequences=[x],
    outputs_info=[np.float64(0)]  # initial value
)
```

## Fibonacci (MITSOT)

Multiple Input Taps — need t-2 and t-1:

```python
def fibonacci_step(f_tm2, f_tm1):
    return f_tm2 + f_tm1

results, updates = pytensor.scan(
    fn=fibonacci_step,
    outputs_info=[dict(
        initial=pt.as_tensor_variable(np.array([1., 1.])),
        taps=[-2, -1]
    )],
    n_steps=10
)
```

The `taps` list specifies which past steps are needed. `initial` must have `len(taps)` leading elements.

## RNN-Style Recurrence

```python
def rnn_step(x_t, h_tm1, W_xh, W_hh, b_h):
    return pt.tanh(pt.dot(x_t, W_xh) + pt.dot(h_tm1, W_hh) + b_h)

hidden_states, updates = pytensor.scan(
    fn=rnn_step,
    sequences=[X],                   # input sequence (T, input_dim)
    outputs_info=[h0],               # initial hidden state (hidden_dim,)
    non_sequences=[W_xh, W_hh, b_h]  # weight matrices
)
# hidden_states shape: (T, hidden_dim)
```

## Power Iteration (No Sequences, Fixed Steps)

```python
def power_step(x_prev, A):
    Ax = pt.dot(A, x_prev)
    return Ax / pt.sqrt(pt.sum(Ax ** 2))

results, updates = pytensor.scan(
    fn=power_step,
    outputs_info=[x0],
    non_sequences=[A],
    n_steps=100
)
dominant_eigvec = results[-1]
```

## While Loop with until

```python
from pytensor.scan import until

def step(x_prev):
    x_new = x_prev / 2
    return x_new, until(x_new < 1e-6)  # stop condition

results, updates = pytensor.scan(
    fn=step,
    outputs_info=[x0],
    n_steps=1000   # max iterations (safety bound)
)
```

`until()` returns a second output that signals early termination.

## AR(2) Time Series

```python
from pytensor.tensor.random.utils import RandomStream
srng = RandomStream(seed=42)

def ar2_step(y_tm1, y_tm2, phi1, phi2, sigma):
    noise = srng.normal(0, sigma)
    return phi1 * y_tm1 + phi2 * y_tm2 + noise

y_init = pt.as_tensor_variable(np.array([0.0, 0.0]))
results, updates = pytensor.scan(
    fn=ar2_step,
    outputs_info=[dict(initial=y_init, taps=[-1, -2])],
    non_sequences=[phi1, phi2, sigma],
    n_steps=200
)
generate = pytensor.function([phi1, phi2, sigma], results, updates=updates)
```

## Kalman Filter

```python
import pytensor.tensor.nlinalg as nla

def kalman_step(y_t, x_prev, P_prev, F, H, Q, R):
    # Predict
    x_pred = pt.dot(F, x_prev)
    P_pred = pt.dot(pt.dot(F, P_prev), F.T) + Q
    # Update
    innovation = y_t - pt.dot(H, x_pred)
    S = pt.dot(pt.dot(H, P_pred), H.T) + R
    K = pt.dot(pt.dot(P_pred, H.T), nla.matrix_inverse(S))
    x_new = x_pred + pt.dot(K, innovation)
    P_new = P_pred - pt.dot(pt.dot(K, H), P_pred)
    return x_new, P_new

(filtered_states, filtered_covs), updates = pytensor.scan(
    fn=kalman_step,
    sequences=[y_obs],
    outputs_info=[x0, P0],
    non_sequences=[F, H, Q, R]
)
```

Multiple outputs: `outputs_info` has one entry per output. Each gets its own tap history.

## Random Walk

```python
from pytensor.tensor.random.utils import RandomStream
srng = RandomStream(seed=42)

def random_walk_step(x_prev, sigma):
    return x_prev + srng.normal(0, sigma)

x0 = pt.dscalar("x0")
sigma = pt.dscalar("sigma")

results, updates = pytensor.scan(
    fn=random_walk_step,
    outputs_info=[x0],
    non_sequences=[sigma],
    n_steps=500
)

generate_walk = pytensor.function([x0, sigma], results, updates=updates)
walk = generate_walk(0.0, 0.1)
```

## Key Rules

1. **Always pass `updates` to `pytensor.function`** — required for random number state management
2. **Prefer vectorized ops** when computation can be parallelized (no recurrence)
3. **Use `n_steps`** when there are no sequences to iterate over
4. **Scan compiles its own sub-graph** — adds compile-time overhead but reduces graph size for large loops
5. Gradients propagate through scan automatically

## Memory Efficiency

### ScanSaveMem (Automatic)

When only the last N timesteps are needed, PyTensor automatically allocates a circular buffer:

- `results[-1]` → buffer of size 1 (not `n_steps`)
- Reduces memory from O(n_steps) to O(1)

### scan_checkpoints (Manual)

For memory-efficient gradient computation through long sequences:

```python
from pytensor.scan import scan_checkpoints

results, updates = scan_checkpoints(
    fn=step_fn,
    sequences=[x],
    outputs_info=[y0],
    n_steps=10000
)
```

Trades compute for memory: reduces gradient memory from O(n_steps) to O(sqrt(n_steps)) at ~2x compute cost.

## External Docs

| Topic | URL |
|---|---|
| Scan API | https://pytensor.readthedocs.io/en/latest/library/scan.html |
| Scan Tutorial | https://pytensor.readthedocs.io/en/latest/gallery/scan/scan_tutorial.html |
| Loop Tutorial | https://pytensor.readthedocs.io/en/latest/tutorial/loop.html |
| Scan Developer Docs | https://pytensor.readthedocs.io/en/latest/extending/scan.html |
