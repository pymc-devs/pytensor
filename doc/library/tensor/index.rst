.. _libdoc_tensor:

===============================================
:mod:`tensor`  -- Tensor operations in PyTensor
===============================================

.. module:: tensor

PyTensor's strength is in expressing symbolic calculations involving tensors.

PyTensor tries to emulate the numpy interface as much as possible in the tensor module.
This means that once TensorVariables are created, it should be possibly to define
symbolic expressions using calls that look just like numpy calls, such as
`pt.exp(x).transpose(0, 1)[:, None]`



.. toctree::
    :maxdepth: 1

    basic
    random/index
    utils
    elemwise
    extra_ops
    io
    slinalg
    nlinalg
    fft
    conv
    math_opt
    basic_opt
    functional
