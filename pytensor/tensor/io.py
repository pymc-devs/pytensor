import numpy as np

from pytensor.graph.basic import Apply, Constant
from pytensor.graph.op import Op
from pytensor.link.c.type import Generic
from pytensor.tensor.type import tensor


class LoadFromDisk(Op):
    """
    An operation to load an array from disk.

    See Also
    --------
    load

    Notes
    -----
    Non-differentiable.

    """

    __props__ = ("dtype", "shape", "mmap_mode")

    def __init__(self, dtype, shape, mmap_mode=None):
        self.dtype = np.dtype(dtype)  # turn "float64" into np.float64
        self.shape = shape
        if mmap_mode not in (None, "c"):
            raise ValueError(
                "The only supported values for mmap_mode "
                "are None and 'c', got %s" % mmap_mode
            )
        self.mmap_mode = mmap_mode

    def make_node(self, path):
        if isinstance(path, str):
            path = Constant(Generic(), path)
        return Apply(self, [path], [tensor(dtype=self.dtype, shape=self.shape)])

    def perform(self, node, inp, out):
        path = inp[0]
        if path.split(".")[-1] == "npz":
            raise ValueError(f"Expected a .npy file, got {path} instead")
        result = np.load(path, mmap_mode=self.mmap_mode)
        if result.dtype != self.dtype:
            raise TypeError(
                f"Expected an array of type {self.dtype}, got {result.dtype} instead"
            )
        out[0][0] = result

    def __str__(self):
        return "Load{{dtype: {}, shape: {}, mmep: {}}}".format(
            self.dtype,
            self.shape,
            self.mmap_mode,
        )


def load(path, dtype, shape, mmap_mode=None):
    """
    Load an array from an .npy file.

    Parameters
    ----------
    path
        A Generic symbolic variable, that will contain a string
    dtype : data-type
        The data type of the array to be read.
    shape
        The static shape information of the loaded array.
    mmap_mode
        How the file will be loaded. None means that the
        data will be copied into an array in memory, 'c' means that the file
        will be mapped into virtual memory, so only the parts that are
        needed will be actually read from disk and put into memory.
        Other modes supported by numpy.load ('r', 'r+', 'w+') cannot
        be supported by PyTensor.

    Examples
    --------
    >>> from pytensor import *
    >>> path = Variable(Generic(), None)
    >>> x = tensor.load(path, 'int64', (None,))
    >>> y = x*2
    >>> fn = function([path], y)
    >>> fn("stored-array.npy")  # doctest: +SKIP
    array([0, 2, 4, 6, 8], dtype=int64)

    """

    return LoadFromDisk(dtype, shape, mmap_mode)(path)


__all__ = ["load"]
