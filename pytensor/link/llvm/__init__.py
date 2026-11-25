"""The LLVM backend (``mode="LLVM"``).

.. warning:: This backend is experimental: it may compute wrong results, crash, or
   change incompatibly without notice. Do not rely on it for real work. A warning is
   issued the first time a function is compiled with it.

How it works is described in "The LLVM backend" in the library documentation (the
``link`` section), how to add an op in "Adding an op to the LLVM backend" (Extending
PyTensor).
"""

from pytensor.link.llvm.linker import LLVMLinker
