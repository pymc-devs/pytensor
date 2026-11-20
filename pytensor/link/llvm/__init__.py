"""The LLVM backend (``mode="LLVM"``).

How it works is described in "The LLVM backend" in the library documentation (the
``link`` section), how to add an op in "Adding an op to the LLVM backend" (Extending
PyTensor).
"""

from pytensor.link.llvm.linker import LLVMLinker
