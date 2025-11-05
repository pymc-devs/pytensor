"""High-level ONNX export API for PyTensor."""

import onnx

from pytensor.compile.function import function
from pytensor.compile.mode import Mode
from pytensor.graph.fg import FunctionGraph
from pytensor.link.onnx.dispatch import onnx_funcify
from pytensor.link.onnx.linker import ONNXLinker


def export_onnx(inputs, outputs, filename, *, opset_version=18, **kwargs):
    """Export a PyTensor graph to an ONNX file.

    Parameters
    ----------
    inputs : list of Variable
        Input variables for the graph
    outputs : Variable or list of Variable
        Output variable(s) for the graph
    filename : str or Path
        Path where the ONNX model will be saved
    opset_version : int, default=18
        ONNX opset version to use
    **kwargs : dict
        Additional keyword arguments

    Returns
    -------
    onnx.ModelProto
        The created ONNX model

    Examples
    --------
    >>> import pytensor.tensor as pt
    >>> x = pt.vector('x', dtype='float32')
    >>> y = x * 2 + 1
    >>> model = export_onnx([x], y, 'model.onnx')
    """
    # Ensure outputs is a list
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    # Create a FunctionGraph (without cloning to preserve structure)
    from pytensor.compile.builders import construct_nominal_fgraph

    fgraph = construct_nominal_fgraph(inputs, outputs)

    # Convert to ONNX ModelProto
    onnx_model = onnx_funcify(fgraph, opset_version=opset_version, **kwargs)

    # Save to file
    onnx.save(onnx_model, filename)

    return onnx_model


def compile_onnx(inputs, outputs, *, opset_version=18, **kwargs):
    """Compile a PyTensor graph using the ONNX backend.

    This creates a function that executes the graph via ONNX Runtime.

    Parameters
    ----------
    inputs : list of Variable
        Input variables for the graph
    outputs : Variable or list of Variable
        Output variable(s) for the graph
    opset_version : int, default=18
        ONNX opset version to use
    **kwargs : dict
        Additional keyword arguments passed to pytensor.function

    Returns
    -------
    Function
        Compiled function that executes via ONNX Runtime

    Examples
    --------
    >>> import pytensor.tensor as pt
    >>> import numpy as np
    >>> x = pt.vector('x', dtype='float32')
    >>> y = x * 2 + 1
    >>> fn = compile_onnx([x], y)
    >>> result = fn(np.array([1, 2, 3], dtype='float32'))
    """
    # Create ONNX mode
    onnx_linker = ONNXLinker(opset_version=opset_version)
    onnx_mode = Mode(linker=onnx_linker, optimizer=None)

    # Compile the function
    return function(inputs, outputs, mode=onnx_mode, **kwargs)


def export_function_onnx(fn, filename, *, opset_version=18):
    """Export an already-compiled PyTensor function to ONNX.

    Parameters
    ----------
    fn : Function
        A compiled PyTensor function
    filename : str or Path
        Path where the ONNX model will be saved
    opset_version : int, default=18
        ONNX opset version to use (if the function wasn't compiled with ONNX)

    Returns
    -------
    onnx.ModelProto
        The created ONNX model

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector('x', dtype='float32')
    >>> y = pt.sqrt(x)
    >>> fn = pytensor.function([x], y)
    >>> model = export_function_onnx(fn, 'sqrt_model.onnx')
    """
    # Check if the function was already compiled with ONNX linker
    if isinstance(fn.maker.linker, ONNXLinker):
        # Already have ONNX model
        onnx_model = fn.maker.linker.onnx_model
    else:
        # Need to convert the FunctionGraph to ONNX
        fgraph = fn.maker.fgraph
        onnx_model = onnx_funcify(fgraph, opset_version=opset_version)

    # Save to file
    onnx.save(onnx_model, filename)

    return onnx_model
