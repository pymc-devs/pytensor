"""This module provides a convenient constructor for the `Scan` `Op`."""

import logging

from pytensor.scan import scan


_logger = logging.getLogger("pytensor.scan.views")


def map(
    fn,
    sequences,
    non_sequences=None,
    truncate_gradient=-1,
    go_backwards=False,
    mode=None,
    name=None,
    return_updates=True,
):
    """Construct a `Scan` `Op` that functions like `map`.

    Parameters
    ----------
    fn
        The function that ``map`` applies at each iteration step
        (see ``scan`` for more info).
    sequences
        List of sequences over which ``map`` iterates
        (see ``scan`` for more info).
    non_sequences
        List of arguments passed to ``fn``. ``map`` will not iterate over
        these arguments (see ``scan`` for more info).
    truncate_gradient
        See ``scan``.
    go_backwards : bool
        Decides the direction of iteration. True means that sequences are parsed
        from the end towards the beginning, while False is the other way around.
    mode
        See ``scan``.
    name
        See ``scan``.

    """
    return scan(
        fn=fn,
        sequences=sequences,
        outputs_info=[],
        non_sequences=non_sequences,
        truncate_gradient=truncate_gradient,
        go_backwards=go_backwards,
        mode=mode,
        name=name,
        return_updates=return_updates,
    )


def reduce(
    fn,
    sequences,
    outputs_info,
    non_sequences=None,
    go_backwards=False,
    mode=None,
    name=None,
    return_updates=True,
):
    """Construct a `Scan` `Op` that functions like `reduce`.

    Parameters
    ----------
    fn
        The function that ``reduce`` applies at each iteration step
        (see ``scan``  for more info).
    sequences
        List of sequences over which ``reduce`` iterates
        (see ``scan`` for more info).
    outputs_info
        List of dictionaries describing the outputs of
        reduce (see ``scan`` for more info).
    non_sequences
        List of arguments passed to ``fn``. ``reduce`` will
                          not iterate over these arguments (see ``scan`` for
                          more info).
    go_backwards : bool
        Decides the direction of iteration. True means that sequences are parsed
        from the end towards the beginning, while False is the other way around.
    mode
        See ``scan``.
    name
        See ``scan``.

    """
    rval = scan(
        fn=fn,
        sequences=sequences,
        outputs_info=outputs_info,
        non_sequences=non_sequences,
        go_backwards=go_backwards,
        truncate_gradient=-1,
        mode=mode,
        name=name,
        return_updates=return_updates,
    )
    if return_updates:
        if isinstance(rval[0], list | tuple):
            return [x[-1] for x in rval[0]], rval[1]
        else:
            return rval[0][-1], rval[1]
    else:
        if isinstance(rval, list | tuple):
            return [x[-1] for x in rval]
        else:
            return rval[-1]


def foldl(
    fn,
    sequences,
    outputs_info,
    non_sequences=None,
    mode=None,
    name=None,
    return_updates=True,
):
    """Construct a `Scan` `Op` that functions like Haskell's `foldl`.

    Parameters
    ----------
    fn
        The function that ``foldl`` applies at each iteration step
        (see ``scan`` for more info).
    sequences
        List of sequences over which ``foldl`` iterates
        (see ``scan`` for more info).
    outputs_info
        List of dictionaries describing the outputs of reduce
        (see ``scan`` for more info).
    non_sequences
        List of arguments passed to `fn`. ``foldl`` will not iterate over
        these arguments (see ``scan`` for more info).
    mode
        See ``scan``.
    name
        See ``scan``.

    """
    return reduce(
        fn=fn,
        sequences=sequences,
        outputs_info=outputs_info,
        non_sequences=non_sequences,
        go_backwards=False,
        mode=mode,
        name=name,
        return_updates=return_updates,
    )


def foldr(
    fn,
    sequences,
    outputs_info,
    non_sequences=None,
    mode=None,
    name=None,
    return_updates=True,
):
    """Construct a `Scan` `Op` that functions like Haskell's `foldr`.

    Parameters
    ----------
    fn
        The function that ``foldr`` applies at each iteration step
        (see ``scan`` for more info).
    sequences
        List of sequences over which ``foldr`` iterates
        (see ``scan`` for more info).
    outputs_info
        List of dictionaries describing the outputs of reduce
        (see ``scan`` for more info).
    non_sequences
        List of arguments passed to `fn`. ``foldr`` will not iterate over these
        arguments (see ``scan`` for more info).
    mode
        See ``scan``.
    name
        See ``scan``.

    """
    return reduce(
        fn=fn,
        sequences=sequences,
        outputs_info=outputs_info,
        non_sequences=non_sequences,
        go_backwards=True,
        mode=mode,
        name=name,
        return_updates=return_updates,
    )


def filter(
    fn,
    sequences,
    non_sequences=None,
    go_backwards=False,
    mode=None,
    name=None,
):
    """Construct a `Scan` `Op` that functions like `filter`.

    Parameters
    ----------
    fn : callable
        Predicate function returning a boolean tensor.
    sequences : list
        Sequences to filter.
    non_sequences : list
        Non-iterated arguments passed to `fn`.
    go_backwards : bool
        Whether to iterate in reverse.
    mode : str or None
        See ``scan``.
    name : str or None
        See ``scan``.

    Notes
    -----
    If the predicate function `fn` returns multiple boolean masks (one per sequence),
    each mask will be applied to its corresponding sequence. If it returns a single mask,
    that mask will be broadcast to all sequences.
    """
    mask, _ = scan(
        fn=fn,
        sequences=sequences,
        outputs_info=None,
        non_sequences=non_sequences,
        go_backwards=go_backwards,
        mode=mode,
        name=name,
    )

    if isinstance(mask, (list, tuple)):
        # One mask per sequence
        if not isinstance(sequences, (list, tuple)):
            raise TypeError(
                "If multiple masks are returned, sequences must be a list or tuple."
            )
        if len(mask) != len(sequences):
            raise ValueError("Number of masks must match number of sequences.")
        filtered_sequences = [seq[m] for seq, m in zip(sequences, mask)]
    else:
        # Single mask applied to all sequences
        if isinstance(sequences, (list, tuple)):
            filtered_sequences = [seq[mask] for seq in sequences]
        else:
            filtered_sequences = sequences[mask]

    return filtered_sequences
