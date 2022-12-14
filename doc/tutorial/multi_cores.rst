.. _tut_multi_cores:

===============================
Multi cores support in PyTensor
===============================

Convolution and Pooling
=======================

The convolution and pooling are parallelized on CPU.


BLAS operation
==============

BLAS is an interface for some mathematical operations between two
vectors, a vector and a matrix or two matrices (e.g. the dot product
between vector/matrix and matrix/matrix). Many different
implementations of that interface exist and some of them are
parallelized.

PyTensor tries to use that interface as frequently as possible for
performance reasons. So if PyTensor links to a parallel implementation,
those operations will run in parallel in PyTensor.

The most frequent way to control the number of threads used is via the
``OMP_NUM_THREADS`` environment variable. Set it to the number of
threads you want to use before starting the Python process. Some BLAS
implementations support other environment variables.

To test if you BLAS supports OpenMP/Multiple cores, you can use the pytensor/misc/check_blas.py script from the command line like this::

    OMP_NUM_THREADS=1 python pytensor/misc/check_blas.py -q
    OMP_NUM_THREADS=2 python pytensor/misc/check_blas.py -q



Parallel element wise ops with OpenMP
=====================================

Because element wise ops work on every tensor entry independently they
can be easily parallelized using OpenMP.

To use OpenMP you must set the ``openmp`` :ref:`flag <libdoc_config>`
to ``True``.

You can use the flag ``openmp_elemwise_minsize`` to set the minimum
tensor size for which the operation is parallelized because for short
tensors using OpenMP can slow down the operation. The default value is
``200000``.

For simple (fast) operations you can obtain a speed-up with very large
tensors while for more complex operations you can obtain a good speed-up
also for smaller tensors.

There is a script ``elemwise_openmp_speedup.py`` in ``pytensor/misc/``
which you can use to tune the value of ``openmp_elemwise_minsize`` for
your machine.  The script runs two elemwise operations (a fast one and
a slow one) for a vector of size ``openmp_elemwise_minsize`` with and
without OpenMP and shows the time difference between the cases.

The only way to control the number of threads used is via the
``OMP_NUM_THREADS`` environment variable. Set it to the number of
threads you want to use before starting the Python process. You can
test this with this command::


    OMP_NUM_THREADS=2 python pytensor/misc/elemwise_openmp_speedup.py
    #The output

    Fast op time without openmp 0.000533s with openmp 0.000474s speedup 1.12
    Slow op time without openmp 0.002987s with openmp 0.001553s speedup 1.92
