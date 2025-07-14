(libdoc_xtensor)=
# `xtensor` -- XTensor operations

This module implements as abstraction layer on regular tensor operations, that behaves like Xarray.

A new type {class}`pytensor.xtensor.type.XTensorType`, generalizes the {class}`pytensor.tensor.TensorType` 
with the addition of a `dims` attribute, that labels the dimensions of the tensor. 

Variables of XTensorType (i.e.,  {class}`pytensor.xtensor.type.XTensorVariable`s) are the symbolic counterpart
to xarray DataArray objects.

The module implements several PyTensor operations {class}`pytensor.xtensor.basic.XOp`s, whose signature mimics that of 
xarray (and xarray_einstats) DataArray operations. These operations, unlike most regular PyTensor operations, cannot 
be directly evaluated, but require a rewrite (lowering) into a regular tensor graph that can itself be evaluated as usual.

Like regular PyTensor, we don't need an Op for every possible method or function in the public API of xarray.
If the existing XOps can be composed to produce the desired result, then we can use them directly.

## Coordinates
For now, there's no analogous of xarray coordinates, so you won't be able to do coordinate operations like `.sel`.
The graphs produced by an xarray program without coords are much more amenable to the numpy-like backend of PyTensor.
Coords involve aspects of Pandas/database query and joining that are not trivially expressible in PyTensor.

## Example


```{testcode}

import pytensor.tensor as pt
import pytensor.xtensor as ptx

a = pt.tensor("a", shape=(3,))
b = pt.tensor("b", shape=(4,))

ax = ptx.as_xtensor(a, dims=["x"])
bx = ptx.as_xtensor(b, dims=["y"])

zx = ax + bx
assert zx.type == ptx.type.XTensorType("float64", dims=["x", "y"], shape=(3, 4))

z = zx.values
z.dprint()
```


```{testoutput}

TensorFromXTensor [id A]
 └─ XElemwise{scalar_op=Add()} [id B]
    ├─ XTensorFromTensor{dims=('x',)} [id C]
    │  └─ a [id D]
    └─ XTensorFromTensor{dims=('y',)} [id E]
       └─ b [id F]
```

Once we compile the graph, no XOps are left.

```{testcode}

import pytensor

with pytensor.config.change_flags(optimizer_verbose=True):
    fn = pytensor.function([a, b], z)

```

```{testoutput}

rewriting: rewrite lower_elemwise replaces XElemwise{scalar_op=Add()}.0 of XElemwise{scalar_op=Add()}(XTensorFromTensor{dims=('x',)}.0, XTensorFromTensor{dims=('y',)}.0) with XTensorFromTensor{dims=('x', 'y')}.0 of XTensorFromTensor{dims=('x', 'y')}(Add.0)
rewriting: rewrite useless_tensor_from_xtensor replaces TensorFromXTensor.0 of TensorFromXTensor(XTensorFromTensor{dims=('x',)}.0) with a of None
rewriting: rewrite useless_tensor_from_xtensor replaces TensorFromXTensor.0 of TensorFromXTensor(XTensorFromTensor{dims=('y',)}.0) with b of None
rewriting: rewrite useless_tensor_from_xtensor replaces TensorFromXTensor.0 of TensorFromXTensor(XTensorFromTensor{dims=('x', 'y')}.0) with Add.0 of Add(ExpandDims{axis=1}.0, ExpandDims{axis=0}.0)

```

```{testcode}

fn.dprint()
```

```{testoutput}

Add [id A] 2
 ├─ ExpandDims{axis=1} [id B] 1
 │  └─ a [id C]
 └─ ExpandDims{axis=0} [id D] 0
    └─ b [id E]
```


## Index

:::{toctree}
:maxdepth: 1

module_functions
math
linalg
random
type
:::