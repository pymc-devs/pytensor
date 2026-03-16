# PyMC Integration Patterns

How PyMC uses PyTensor as its computational backend and the patterns for working at the boundary between the two. After reading this you can: structure models with coords/dims for labeled InferenceData, sample with nutpie (default, 2-5x faster) or NumPyro/JAX, understand the two-layer distribution architecture (PyTensor `RandomVariable` Op + PyMC `Distribution` class), inspect compiled logp/dlogp graphs, build custom distributions with `CustomDist`/`DensityDist`, add soft constraints via `pm.Potential`, avoid PyTensor gotchas inside models (Python `if/else`, hard clipping, in-place assignment, dimension mismatches), implement hierarchical models with centered vs non-centered parameterization, write state-space and time series models using `pytensor.scan` inside PyMC, profile logp bottlenecks, and debug models before sampling.

## Contents
- How PyMC builds computational graphs
- Coords and dims for labeled InferenceData
- Sampling: nutpie (default), NumPyro/JAX, compile pipeline
- Distribution architecture (two layers)
- compile_logp / compile_dlogp internals
- Transform system
- CustomDist — three approaches
- DensityDist and Potential patterns
- PyTensor gotchas inside PyMC models
- Hierarchical model patterns (centered vs non-centered)
- Time series with scan
- Performance patterns
- Model debugging and graph inspection

## How PyMC Builds Computational Graphs

Inside `pm.Model()`, every distribution call creates a symbolic `TensorVariable` — no computation happens:

```python
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=2)        # symbolic RandomVariable node
    sigma = pm.HalfNormal("sigma", sigma=3)     # another symbolic node
    x = pm.Normal("x", mu=mu, sigma=sigma, observed=data)
```

Each `pm.Distribution` call returns a `pytensor.tensor.var.TensorVariable`.

## Coords and Dims

Use coords/dims for interpretable InferenceData. This is the modern PyMC pattern:

```python
coords = {
    "obs": np.arange(n_obs),
    "features": ["intercept", "age", "income"],
    "group": group_labels,
}

with pm.Model(coords=coords) as model:
    beta = pm.Normal("beta", 0, 1, dims="features")
    alpha = pm.Normal("alpha", 0, 1, dims="group")
    sigma = pm.HalfNormal("sigma", sigma=1)

    mu = pt.dot(X, beta) + alpha[group_idx]
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs, dims="obs")
```

Benefits: labeled xarray dimensions in InferenceData, readable `az.summary()`, proper `az.plot_forest()` labels.

**Pitfall**: Don't use the same name for a variable and a dimension. `pm.Normal("group", ..., dims="group")` causes a `ValueError`.

## Sampling

### nutpie (Default — Always Use)

nutpie is Rust-based and 2-5x faster than PyMC's default NUTS:

```python
with model:
    idata = pm.sample(
        draws=1000, tune=1000, chains=4,
        nuts_sampler="nutpie",
        random_seed=42,
    )
idata.to_netcdf("results.nc")  # save immediately after sampling
```

**Critical**: nutpie doesn't store log_likelihood automatically. Compute explicitly for LOO-CV:

```python
pm.compute_log_likelihood(idata, model=model)
```

### NumPyro/JAX (GPU or Vectorized Chains)

```python
with model:
    idata = pm.sample(nuts_sampler="numpyro")     # NumPyro's NUTS via JAX
    idata = pm.sample(nuts_sampler="blackjax")     # BlackJAX's NUTS via JAX

    # Vectorized chains on GPU
    idata = pm.sample(
        nuts_sampler="numpyro",
        nuts_sampler_kwargs={"chain_method": "vectorized"},
    )
```

Pipeline: `model.logp()` → `jax_funcify(fgraph)` → native JAX → NumPyro/BlackJAX NUTS

### Compile Pipeline (What Happens Inside pm.sample)

1. Construct symbolic logp graph from model's random variables
2. Replace each RandomVariable with its logp expression at a "value variable" (via `clone_replace`)
3. Apply transforms (e.g., log-transform for HalfNormal) + add Jacobian terms
4. Flatten all value variables into a single 1D vector (for NUTS)
5. Apply PyTensor graph rewrites for stability/performance
6. Compile to selected backend (C for nutpie, JAX for numpyro/blackjax)

## Distribution Architecture (Two Layers)

### Layer 1: PyTensor RandomVariable Op

Handles random sampling, parameter broadcasting, shape inference:

```python
from pytensor.tensor.random.op import RandomVariable

class BlahRV(RandomVariable):
    name = "blah"
    signature = "(),()->()"   # numpy-style gufunc signature
    dtype = "floatX"

    @classmethod
    def rng_fn(cls, rng, param1, param2, size):
        return scipy.stats.blah.rvs(param1, param2,
                                     random_state=rng, size=size)

blah = BlahRV()
```

### Layer 2: PyMC Distribution Class

Links the RandomVariable Op with logp, logcdf, transforms:

```python
class Blah(PositiveContinuous):
    rv_op = blah

    @classmethod
    def dist(cls, param1, param2, **kwargs):
        param1 = pt.as_tensor_variable(param1)
        param2 = pt.as_tensor_variable(param2)
        return super().dist([param1, param2], **kwargs)

    def logp(value, param1, param2):
        return -param1 * pt.log(value) + ...

    def logcdf(value, param1, param2):
        return pt.log(pt.gammainc(param1, param2 * value))

    def support_point(rv, size, param1, param2):
        return param1 / param2
```

## compile_logp / compile_dlogp

```python
with pm.Model() as model:
    mu = pm.Normal("mu", 0, 1)
    obs = pm.Normal("obs", mu=mu, sigma=1, observed=data)

logp_fn = model.compile_logp()
logp_fn(model.initial_point())

dlogp_fn = model.compile_dlogp()
dlogp_fn(model.initial_point())
```

### Inspecting the graph

```python
import pytensor
from pytensor.printing import debugprint

# See the symbolic logp expression
logp_expr = model.logp()
debugprint(logp_expr)

# See the value variables (including transforms)
model.rvs_to_values
# {mu ~ Normal(0, 1): mu, sigma ~ HalfNormal(3): sigma_log__}
```

## Transform System

Constrained parameters are automatically transformed to unconstrained space:

```python
# HalfNormal(sigma=3) → log-transformed value variable "sigma_log__"
# Jacobian determinant added to logp automatically
model.rvs_to_values
# {mu ~ Normal(0, 2): mu, sigma ~ HalfNormal(0, 3): sigma_log__}
```

NUTS samples in unconstrained space. The sampler never sees the original constrained parameter — it sees the transformed version.

## CustomDist — Three Approaches

### Approach 1: dist function (Recommended)

Build from existing distributions; logp auto-derived:

```python
def dist(mu, sigma, size):
    return pm.Normal.dist(mu=mu, sigma=sigma, size=size)

with pm.Model():
    pm.CustomDist("y", mu, sigma, dist=dist, observed=data)
```

### Approach 2: Black-box logp (PyTensor expression)

```python
def logp(value, mu):
    return -(value - mu)**2

with pm.Model():
    pm.CustomDist("y", mu, logp=logp, observed=data)
```

### Approach 3: NumPy random + PyTensor logp

```python
def random(mu, sigma, rng=None, size=None):
    return rng.normal(mu, sigma, size=size)

def logp(value, mu, sigma):
    return -0.5 * pt.log(2 * np.pi * sigma**2) - (value - mu)**2 / (2 * sigma**2)

with pm.Model():
    pm.CustomDist("y", mu, sigma, random=random, logp=logp, observed=data)
```

## DensityDist and Potential Patterns

### pm.DensityDist (Custom Likelihoods)

When the likelihood isn't a built-in PyMC distribution:

```python
def custom_logp(value, param1, param2):
    """Must return a PyTensor tensor, not a Python float."""
    return -0.5 * ((value - param1) / param2) ** 2 - pt.log(param2)

with pm.Model() as model:
    param1 = pm.Normal("param1", 0, 1)
    param2 = pm.HalfNormal("param2", 1)

    y = pm.DensityDist(
        "y",
        param1, param2,
        logp=custom_logp,
        observed=y_obs,
    )
```

Add random generation for prior/posterior predictive:

```python
def custom_random(mu, sigma, rng=None, size=None):
    return rng.normal(loc=mu, scale=sigma, size=size)

y = pm.DensityDist("y", mu, sigma,
                    logp=custom_logp, random=custom_random,
                    observed=y_obs)
```

### pm.Potential (Soft Constraints)

Adds arbitrary log-probability terms. Does NOT generate samples — only modifies logp.

```python
# Sum-to-zero constraint (hierarchical models)
alpha = pm.Normal("alpha", 0, 1, dims="group")
pm.Potential("sum_to_zero", -100 * pt.sqr(alpha.sum()))

# Truncation
pm.Potential("truncation", pt.switch(
    pt.and_(x >= a, x <= b), 0, -np.inf
))

# Jacobian adjustment for custom transforms
log_sigma = pm.Normal("log_sigma", 0, 1)
sigma = pm.Deterministic("sigma", pt.exp(log_sigma))
pm.Potential("jacobian", log_sigma)
```

## PyTensor Gotchas Inside PyMC Models

### Python if/else vs pt.switch

Python conditionals evaluate at graph construction time, not during sampling:

```python
# WRONG — evaluated once
if x > threshold:
    mu = a
else:
    mu = b

# CORRECT — symbolic, evaluated per sample
mu = pt.switch(x > threshold, a, b)

# For complex conditionals
from pytensor.ifelse import ifelse
result = ifelse(condition, true_branch, false_branch)
```

For iterative logic, use `pytensor.scan`.

### Hard Clipping Kills Gradients

Clipping creates flat gradient regions where NUTS cannot navigate:

```python
# BAD — gradient is zero in clipped regions
mu = pt.clip(linear_pred, 0, np.inf)

# GOOD — smooth, positive
mu = pt.softplus(linear_pred)

# Or use naturally constrained distributions
sigma = pm.HalfNormal("sigma", 1)
rate = pm.LogNormal("rate", 0, 1)
```

### No In-Place Assignment

PyTensor is functional. Use `set_subtensor`/`inc_subtensor`:

```python
# WRONG
x[0] = new_value

# CORRECT
x_new = x[0].set(new_value)
```

### Dimension Mismatch: Group Parameters vs Observations

```python
# WRONG — alpha has K groups, y_obs has N observations
alpha = pm.Normal("alpha", 0, 1, dims="group")
y = pm.Normal("y", mu=alpha, sigma=1, observed=y_obs)  # Shape error!

# CORRECT — index into group parameters
group_idx = df["group"].cat.codes
y = pm.Normal("y", mu=alpha[group_idx], sigma=1, observed=y_obs, dims="obs")
```

## Hierarchical Model Patterns

### Non-Centered (Use When Data Is Weak)

Avoids funnel geometry that causes divergences:

```python
with pm.Model(coords={"group": groups}) as model:
    mu_alpha = pm.Normal("mu_alpha", 0, 1)
    sigma_alpha = pm.HalfNormal("sigma_alpha", 1)

    # Non-centered: sample offset, construct alpha
    alpha_offset = pm.Normal("alpha_offset", 0, 1, dims="group")
    alpha = pm.Deterministic("alpha",
        mu_alpha + sigma_alpha * alpha_offset, dims="group")

    y = pm.Normal("y", alpha[group_idx], sigma, observed=y_obs)
```

### Centered (Use When Data Is Strong)

More efficient when each group has substantial data:

```python
alpha = pm.Normal("alpha", mu_alpha, sigma_alpha, dims="group")
```

### Sum-to-Zero Constraint

```python
alpha_raw = pm.Normal("alpha_raw", 0, 1, shape=n_groups - 1)
alpha = pm.Deterministic("alpha",
    pt.concatenate([alpha_raw, -alpha_raw.sum(keepdims=True)]))
```

## Time Series Patterns Using PyTensor

### State-Space Models with scan

When PyMC's built-in `pm.AR` or `pm.GaussianRandomWalk` aren't flexible enough:

```python
def kalman_step(y_t, x_prev, P_prev, F, H, Q, R):
    import pytensor.tensor.nlinalg as nla
    x_pred = pt.dot(F, x_prev)
    P_pred = pt.dot(pt.dot(F, P_prev), F.T) + Q
    innovation = y_t - pt.dot(H, x_pred)
    S = pt.dot(pt.dot(H, P_pred), H.T) + R
    K = pt.dot(pt.dot(P_pred, H.T), nla.matrix_inverse(S))
    x_new = x_pred + pt.dot(K, innovation)
    P_new = P_pred - pt.dot(pt.dot(K, H), P_pred)
    log_lik = -0.5 * (pt.log(nla.det(S)) + pt.dot(innovation, sla.solve(S, innovation)))
    return x_new, P_new, log_lik

with pm.Model() as ssm:
    F = pt.eye(state_dim)
    sigma_state = pm.HalfNormal("sigma_state", 1)
    Q = sigma_state**2 * pt.eye(state_dim)

    sigma_obs = pm.HalfNormal("sigma_obs", 1)
    R = sigma_obs**2 * pt.eye(obs_dim)

    (states, covs, log_liks), updates = pytensor.scan(
        fn=kalman_step,
        sequences=[y_obs],
        outputs_info=[x0, P0, None],
        non_sequences=[F, H, Q, R]
    )

    pm.Potential("log_likelihood", log_liks.sum())
    idata = pm.sample(nuts_sampler="nutpie")
```

### Custom AR Process with scan

```python
def ar_step(noise_t, y_tm1, y_tm2, phi1, phi2):
    return phi1 * y_tm1 + phi2 * y_tm2 + noise_t

with pm.Model() as ar2:
    phi1 = pm.Uniform("phi1", -1, 1)
    phi2 = pm.Uniform("phi2", -1, 1)
    sigma = pm.HalfNormal("sigma", 1)

    innovations = pm.Normal("innovations", 0, sigma, shape=T)

    y_init = pt.as_tensor_variable(np.array([0.0, 0.0]))
    results, updates = pytensor.scan(
        fn=ar_step,
        sequences=[innovations],
        outputs_info=[dict(initial=y_init, taps=[-1, -2])],
        non_sequences=[phi1, phi2],
    )

    y = pm.Normal("y", mu=results, sigma=0.1, observed=y_obs)
```

## Performance Patterns

### Avoid Large Deterministics

```python
# BAD — stores n_obs x n_draws array in trace
mu = pm.Deterministic("mu", pt.dot(X, beta), dims="obs")

# GOOD — don't save intermediate; use posterior_predictive if needed
mu = pt.dot(X, beta)
```

### Use pm.Data to Avoid Recompilation

```python
# BAD — recompiles every iteration
for dataset in datasets:
    with pm.Model() as model:
        ...
        idata = pm.sample()

# GOOD — compile once, swap data
with pm.Model() as model:
    x = pm.Data("x", x_initial)
    ...

for dataset in datasets:
    pm.set_data({"x": dataset["x"]})
    idata = pm.sample()
```

### Profile the Logp Graph

```python
# Identify bottlenecks in log-probability computation
profile = model.profile(model.logp())
profile.summary()

# Profile gradient computation (what NUTS uses)
import pytensor
grad_profile = model.profile(
    pytensor.grad(model.logp(), model.continuous_value_vars))
grad_profile.summary()
```

## Model Debugging and Graph Inspection

```python
# Validate model before sampling
model.debug()               # checks for common issues
model.point_logps()          # log-prob at initial point per variable

# Inspect model structure
print(model)                 # variables, shapes, distributions
pm.model_to_graphviz(model)  # visual DAG

# Inspect compiled logp graph
pytensor.dprint(model.logp())

# Symbolic logp evaluation for specific values
z = pm.Normal("z", 0, 1)
logp_expr = pm.logp(z, 2.5)
print(logp_expr.eval())
```

### Common Debugging Symptoms

| Symptom | Likely Cause | Fix |
|---|---|---|
| `ValueError: Shape mismatch` | Parameter vs observation dimensions | Index: `alpha[group_idx]` |
| `Initial evaluation failed` | Data outside distribution support | `init="adapt_diag"`, check bounds |
| `Mass matrix contains zeros` | Unscaled predictors or flat priors | Standardize features |
| High divergence count | Funnel geometry | Non-centered parameterization |
| `NaN` in log-probability | Invalid parameter combinations | Check constraints |
| Slow discrete sampling | NUTS incompatible with discrete | Marginalize discrete variables |

## Mixed Discrete-Continuous Models

PyTensor cannot compute gradients through discrete variables. PyMC handles this with compound stepping:

- **Continuous variables**: NUTS (gradient-based, uses `pytensor.grad`)
- **Discrete variables**: Metropolis-Hastings (gradient-free)

Alternatives for better performance:
- **Marginalization**: `import pymc_extras as pmx; pmx.marginalize(model, ["discrete_var"])`
- **NumPyro**: Auto-enumeration for discrete variables with finite support

## External Docs

| Topic | URL |
|---|---|
| PyMC + PyTensor Notebook | https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_pytensor.html |
| Implementing Distributions | https://www.pymc.io/projects/docs/en/stable/contributing/implementing_distribution.html |
| CustomDist API | https://www.pymc.io/projects/docs/en/stable/api/distributions/custom.html |
| JAX/Numba Sampling | https://www.pymc.io/projects/examples/en/latest/samplers/fast_sampling_with_jax_and_numba.html |
| Data Containers | https://www.pymc.io/projects/examples/en/latest/fundamentals/data_container.html |
| Wrapping JAX Functions | https://www.pymc.io/projects/examples/en/latest/howto/wrapping_jax_function.html |
