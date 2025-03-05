.. _libdoc_tensor_conv:

==========================================================
:mod:`conv` -- Ops for convolutional neural nets
==========================================================

.. module:: conv
   :platform: Unix, Windows
   :synopsis: ops for signal processing
.. moduleauthor:: LISA


The recommended user interface are:

- :func:`pytensor.tensor.conv.conv2d` for 2d convolution
- :func:`pytensor.tensor.conv.conv3d` for 3d convolution

Pytensor will automatically use the fastest implementation in many cases.
On the CPU, the implementation is a GEMM based one.

This auto-tuning has the inconvenience that the first call is much
slower as it tries and times each implementation it has. So if you
benchmark, it is important that you remove the first call from your
timing.

Implementation Details
======================

This section gives more implementation detail. Most of the time you do
not need to read it. Pytensor will select it for you.


- Implemented operators for neural network 2D / image convolution:
    - :func:`conv.conv2d <pytensor.tensor.conv.conv2d>`.
      old 2d convolution. DO NOT USE ANYMORE.

      For each element in a batch, it first creates a
      `Toeplitz <http://en.wikipedia.org/wiki/Toeplitz_matrix>`_ matrix in a CUDA kernel.
      Then, it performs a ``gemm`` call to multiply this Toeplitz matrix and the filters
      (hence the name: MM is for matrix multiplication).
      It needs extra memory for the Toeplitz matrix, which is a 2D matrix of shape
      ``(no of channels * filter width * filter height, output width * output height)``.

    - :func:`CorrMM <pytensor.tensor.nnet.corr.CorrMM>`
      This is a CPU-only 2d correlation implementation taken from
      `caffe's cpp implementation <https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cpp>`_.
      It does not flip the kernel.

- Implemented operators for neural network 3D / video convolution:
    - :func:`Corr3dMM <pytensor.tensor.nnet.corr3d.Corr3dMM>`
      This is a CPU-only 3d correlation implementation based on
      the 2d version (:func:`CorrMM <pytensor.tensor.nnet.corr.CorrMM>`).
      It does not flip the kernel. As it provides a gradient, you can use it as a
      replacement for nnet.conv3d. For convolutions done on CPU,
      nnet.conv3d will be replaced by Corr3dMM.

    - :func:`conv3d2d <pytensor.tensor.nnet.conv3d2d.conv3d>`
      Another conv3d implementation that uses the conv2d with data reshaping.
      It is faster in some corner cases than conv3d. It flips the kernel.

.. autofunction:: pytensor.tensor.conv.conv2d
.. autofunction:: pytensor.tensor.conv.conv2d_transpose
.. autofunction:: pytensor.tensor.conv.conv3d
.. autofunction:: pytensor.tensor.conv.conv3d2d.conv3d
.. autofunction:: pytensor.tensor.conv.conv.conv2d

.. automodule:: pytensor.tensor.conv.abstract_conv
    :members:
