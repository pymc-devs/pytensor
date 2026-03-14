# PyTensor Development Guide

## Debugging Graphs

Use `pytensor.dprint()` to understand the computational graph before and after transformations:

```python
import pytensor
import pytensor.tensor as pt

x = pt.vector('x')
y = x ** 2 + x * 3

# Before compilation/rewriting
pytensor.dprint(y)

# Useful dprint arguments:
#   print_shape=True   — show output shapes
#   print_type=True    — show output types (dtype + shape)
#   print_memory_map=True — show memory layout info
#   depth=N            — limit tree depth to N levels

# After compilation
f = pytensor.function([x], y)
pytensor.dprint(f)

# After a specific rewrite pass
from pytensor.graph.rewriting.utils import rewrite_graph
y_rewritten = rewrite_graph(y, include=['canonicalize'])
pytensor.dprint(y_rewritten)
```

Use `pytensor.config.change_flags(optimizer_verbose=True)` to see which rewrites are applied during compilation:

```python
with pytensor.config.change_flags(optimizer_verbose=True):
    f = pytensor.function([x], y)
```
