import pytensor.tensor as pt
from pytensor import function


def detected_and_numeric_patterns(cases, inputs, values):
    """Evaluate each case's detected pattern and the nonzeros of its numeric Jacobian.

    Parameters
    ----------
    cases : dict of str to tuple
        Maps a title to ``(outputs, wrt)``.
    inputs : list of Variable
        Every root input of the cases.
    values : list of ndarray
        A value per input.

    Returns
    -------
    patterns : dict of str to tuple of ndarray
        Maps each title to ``(detected, numeric)``, both boolean of shape
        ``(total output elements, total wrt elements)``.
    """
    graphs = []
    for outputs, wrt in cases.values():
        flat_outputs = pt.concatenate([output.ravel() for output in outputs])
        blocks = pt.jacobian(flat_outputs, wrt, disconnected_inputs="ignore")
        jacobian = pt.concatenate(
            [block.reshape((flat_outputs.shape[0], -1)) for block in blocks], axis=1
        )
        graphs.extend([pt.jacobian_sparsity(outputs, wrt), pt.abs(jacobian) > 1e-12])

    results = function(inputs, graphs, on_unused_input="ignore")(*values)
    return dict(zip(cases, zip(results[::2], results[1::2], strict=True), strict=True))
