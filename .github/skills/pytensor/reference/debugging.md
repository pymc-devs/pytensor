# Debugging PyTensor

Tools and techniques for inspecting, validating, and profiling PyTensor computational graphs. After reading this you can: print graph structure with `debugprint`/`pprint`, generate interactive HTML visualizations with `d3viz`, insert `Print` ops to trace values inside compiled functions, catch NaN/Inf with `NanGuardMode` and `MonitorMode`, validate Op correctness with `DebugMode`, use test values for eager-like shape checking during graph construction, set error verbosity flags, create reproducible bug reports with `function_dump`, and profile compiled functions to identify performance bottlenecks.

## Contents
- Graph printing (debugprint, pprint, pydotprint)
- Interactive visualization (d3viz)
- Print Op (not Python print)
- DebugMode
- NanGuardMode
- MonitorMode — custom NaN detection
- Test values (eager-like shape checking)
- Error debugging flags
- Function dump for reproducible bug reports
- Profiling compiled functions

## Graph Printing

```python
from pytensor.printing import pprint, debugprint, pydotprint

x = pt.dmatrix("x")
y = pt.tanh(pt.dot(x, x.T))

print(pprint(y))        # compact math-like notation
debugprint(y)            # verbose structure with [id X] identifiers
y.dprint()               # shortcut for debugprint

# After compilation — shows optimized graph:
f = pytensor.function([x], y)
debugprint(f)

# Visual graph (requires graphviz + pydot):
pydotprint(f, outfile="graph.png", var_with_name_simple=True)
```

## Interactive Visualization (d3viz)

```python
import pytensor.d3viz as d3v
d3v.d3viz(f, "graph.html")
# Open in browser: drag/drop nodes, zoom, hover for details
# Green = inputs, Blue = outputs, Ellipses = Apply nodes
```

## Print Op

Python's `print()` does NOT work inside compiled functions. Use the `Print` Op:

```python
from pytensor.printing import Print

x = pt.dmatrix("x")
x_printed = Print("x value")(x)        # prints when evaluated
y = pt.dot(x_printed, x_printed.T)
f = pytensor.function([x], y)
# Output: "x value __str__ = [[1. 2.] [3. 4.]]"

# Print specific attributes:
x_printed = Print("x", attrs=["shape"])(x)
```

**Important**: Print ops can prevent graph rewrites. Remove after debugging.

## DebugMode

Thorough checking: NaN/Inf detection, memory aliasing validation, Op correctness:

```python
f = pytensor.function([x], y, mode="DebugMode")
```

## NanGuardMode

Full FAST_RUN optimizations with NaN/Inf checking on every output:

```python
f = pytensor.function([x], y, mode="NanGuardMode")
```

## MonitorMode — Custom NaN Detection

Execute custom callbacks after each node:

```python
from pytensor.compile.monitormode import MonitorMode

def detect_nan(fgraph, i, node, fn):
    for output in fn.outputs:
        if np.isnan(output[0]).any():
            print(f"*** NaN detected at node: {node}")
            print(f"    Inputs: {[inp[0] for inp in fn.inputs]}")

f = pytensor.function([x], y,
    mode=MonitorMode(post_func=detect_nan).excluding(
        'local_elemwise_fusion', 'inplace'
    )
)
```

Excluding fusion/inplace makes individual nodes inspectable.

## Test Values (Eager-Like Shape Checking)

Catch shape errors during graph construction rather than at runtime:

```python
pytensor.config.compute_test_value = "raise"  # or "warn", "ignore"

x = pt.dmatrix("x")
x.tag.test_value = np.random.randn(3, 4)
y = pt.dot(x, x.T)  # shape errors caught immediately
```

## Error Debugging Flags

```bash
# Detailed error messages
export PYTENSOR_FLAGS='exception_verbosity=high'

# Better line info (skip optimizations)
export PYTENSOR_FLAGS='optimizer=fast_compile'

# Readable graphs for development
export PYTENSOR_FLAGS='optimizer_excluding=fusion:inplace'
```

## Function Dump

Save everything needed to reproduce a compilation:

```python
pytensor.function_dump("bug_report.pkl", [x], y)
```

## Profiling Compiled Functions

### Per-function profiling

```python
f = pytensor.function([x], y, profile=True)
f(data)
f.profile.summary()
```

### Global profiling

```python
pytensor.config.profile = True
pytensor.config.profile_memory = True       # include memory stats
pytensor.config.profile_optimizer = True    # include rewrite timing
```

### Profiler output sections

1. **Global Info**: Function name, call count, total time, compilation overhead
2. **Class Info**: Time aggregated by Op class
3. **Ops Info**: Time grouped by identical operations
4. **Apply Node Info**: Individual node-level timing

**Bottleneck identification**: Focus on "Ops Info" section. Check the "implementation type" column — Python fallbacks are a common slowness source. Fix by: providing C implementation, switching backends, or writing a graph rewrite.

## External Docs

| Topic | URL |
|---|---|
| Debug FAQ | https://pytensor.readthedocs.io/en/latest/tutorial/debug_faq.html |
| Printing/Drawing | https://pytensor.readthedocs.io/en/latest/tutorial/printing_drawing.html |
| Printing Library | https://pytensor.readthedocs.io/en/latest/library/printing.html |
| d3viz | https://pytensor.readthedocs.io/en/latest/library/d3viz/index.html |
| Modes | https://pytensor.readthedocs.io/en/latest/tutorial/modes.html |
| Profiling | https://pytensor.readthedocs.io/en/latest/tutorial/profiling.html |
| Memory Aliasing | https://pytensor.readthedocs.io/en/latest/tutorial/aliasing.html |
