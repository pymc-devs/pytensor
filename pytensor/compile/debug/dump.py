"""Dump function compilation arguments for debugging."""

from pathlib import Path

import pytensor.misc.pkl_utils


def function_dump(
    filename: str | Path,
    inputs,
    outputs=None,
    mode=None,
    updates=None,
    givens=None,
    no_default_updates=False,
    accept_inplace=False,
    name=None,
    rebuild_strict=True,
    allow_input_downcast=None,
    profile=None,
    on_unused_input=None,
    extra_tag_to_remove=None,
    trust_input=False,
):
    """
    This is helpful to make a reproducible case for problems during PyTensor
    compilation.

    Ex:

    replace `pytensor.function(...)` by
    `pytensor.function_dump('filename.pkl', ...)`.

    If you see this, you were probably asked to use this function to
    help debug a particular case during the compilation of an PyTensor
    function. `function_dump` allows you to easily reproduce your
    compilation without generating any code. It pickles all the objects and
    parameters needed to reproduce a call to `pytensor.function()`. This
    includes shared variables and their values. If you do not want
    that, you can choose to replace shared variables values with zeros by
    calling set_value(...) on them before calling `function_dump`.

    To load such a dump and do the compilation:

    >>> import pickle
    >>> import pytensor
    >>> d = pickle.load(open("func_dump.bin", "rb"))  # doctest: +SKIP
    >>> f = pytensor.function(**d)  # doctest: +SKIP

    Note:
    The parameter `extra_tag_to_remove` is passed to the StripPickler used.
    To pickle graph made by Blocks, it must be:
    `['annotations', 'replacement_of', 'aggregation_scheme', 'roles']`

    """
    d = {
        "inputs": inputs,
        "outputs": outputs,
        "mode": mode,
        "updates": updates,
        "givens": givens,
        "no_default_updates": no_default_updates,
        "accept_inplace": accept_inplace,
        "name": name,
        "rebuild_strict": rebuild_strict,
        "allow_input_downcast": allow_input_downcast,
        "profile": profile,
        "on_unused_input": on_unused_input,
        "trust_input": trust_input,
    }
    with Path(filename).open("wb") as f:
        pickler = pytensor.misc.pkl_utils.StripPickler(
            f, protocol=-1, extra_tag_to_remove=extra_tag_to_remove
        )
        pickler.dump(d)
