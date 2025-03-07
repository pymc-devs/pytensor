import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.printing import _try_pydot_import, debugprint, pydotprint
from pytensor.tensor.type import dvector, iscalar, scalar, vector


@config.change_flags(floatX="float64")
def test_debugprint_sitsot():
    k = iscalar("k")
    A = dvector("A")

    # Symbolic description of the result
    result, updates = pytensor.scan(
        fn=lambda prior_result, A: prior_result * A,
        outputs_info=pt.ones_like(A),
        non_sequences=A,
        n_steps=k,
    )

    final_result = result[-1]
    output_str = debugprint(final_result, file="str", print_op_info=True)
    lines = output_str.split("\n")

    expected_output = """Subtensor{i} [id A]
     ├─ Subtensor{start:} [id B]
     │  ├─ Scan{scan_fn, while_loop=False, inplace=none} [id C] (outer_out_sit_sot-0)
     │  │  ├─ k [id D] (n_steps)
     │  │  ├─ SetSubtensor{:stop} [id E] (outer_in_sit_sot-0)
     │  │  │  ├─ AllocEmpty{dtype='float64'} [id F]
     │  │  │  │  ├─ Add [id G]
     │  │  │  │  │  ├─ k [id D]
     │  │  │  │  │  └─ Subtensor{i} [id H]
     │  │  │  │  │     ├─ Shape [id I]
     │  │  │  │  │     │  └─ Unbroadcast{0} [id J]
     │  │  │  │  │     │     └─ ExpandDims{axis=0} [id K]
     │  │  │  │  │     │        └─ Second [id L]
     │  │  │  │  │     │           ├─ A [id M]
     │  │  │  │  │     │           └─ ExpandDims{axis=0} [id N]
     │  │  │  │  │     │              └─ 1.0 [id O]
     │  │  │  │  │     └─ 0 [id P]
     │  │  │  │  └─ Subtensor{i} [id Q]
     │  │  │  │     ├─ Shape [id I]
     │  │  │  │     │  └─ ···
     │  │  │  │     └─ 1 [id R]
     │  │  │  ├─ Unbroadcast{0} [id J]
     │  │  │  │  └─ ···
     │  │  │  └─ ScalarFromTensor [id S]
     │  │  │     └─ Subtensor{i} [id H]
     │  │  │        └─ ···
     │  │  └─ A [id M] (outer_in_non_seqs-0)
     │  └─ 1 [id T]
     └─ -1 [id U]

    Inner graphs:

    Scan{scan_fn, while_loop=False, inplace=none} [id C]
     ← Mul [id V] (inner_out_sit_sot-0)
        ├─ *0-<Vector(float64, shape=(?,))> [id W] -> [id E] (inner_in_sit_sot-0)
        └─ *1-<Vector(float64, shape=(?,))> [id X] -> [id M] (inner_in_non_seqs-0)
    """

    for truth, out in zip(expected_output.split("\n"), lines, strict=True):
        assert truth.strip() == out.strip()


def test_debugprint_sitsot_no_extra_info():
    k = iscalar("k")
    A = dvector("A")

    # Symbolic description of the result
    result, updates = pytensor.scan(
        fn=lambda prior_result, A: prior_result * A,
        outputs_info=pt.ones_like(A),
        non_sequences=A,
        n_steps=k,
    )

    final_result = result[-1]
    output_str = debugprint(final_result, file="str", print_op_info=False)
    lines = output_str.split("\n")

    expected_output = """Subtensor{i} [id A]
     ├─ Subtensor{start:} [id B]
     │  ├─ Scan{scan_fn, while_loop=False, inplace=none} [id C]
     │  │  ├─ k [id D]
     │  │  ├─ SetSubtensor{:stop} [id E]
     │  │  │  ├─ AllocEmpty{dtype='float64'} [id F]
     │  │  │  │  ├─ Add [id G]
     │  │  │  │  │  ├─ k [id D]
     │  │  │  │  │  └─ Subtensor{i} [id H]
     │  │  │  │  │     ├─ Shape [id I]
     │  │  │  │  │     │  └─ Unbroadcast{0} [id J]
     │  │  │  │  │     │     └─ ExpandDims{axis=0} [id K]
     │  │  │  │  │     │        └─ Second [id L]
     │  │  │  │  │     │           ├─ A [id M]
     │  │  │  │  │     │           └─ ExpandDims{axis=0} [id N]
     │  │  │  │  │     │              └─ 1.0 [id O]
     │  │  │  │  │     └─ 0 [id P]
     │  │  │  │  └─ Subtensor{i} [id Q]
     │  │  │  │     ├─ Shape [id I]
     │  │  │  │     │  └─ ···
     │  │  │  │     └─ 1 [id R]
     │  │  │  ├─ Unbroadcast{0} [id J]
     │  │  │  │  └─ ···
     │  │  │  └─ ScalarFromTensor [id S]
     │  │  │     └─ Subtensor{i} [id H]
     │  │  │        └─ ···
     │  │  └─ A [id M]
     │  └─ 1 [id T]
     └─ -1 [id U]

    Inner graphs:

    Scan{scan_fn, while_loop=False, inplace=none} [id C]
     ← Mul [id V]
        ├─ *0-<Vector(float64, shape=(?,))> [id W] -> [id E]
        └─ *1-<Vector(float64, shape=(?,))> [id X] -> [id M]
    """

    for truth, out in zip(expected_output.split("\n"), lines, strict=True):
        assert truth.strip() == out.strip()


@config.change_flags(floatX="float64")
def test_debugprint_nitsot():
    coefficients = vector("coefficients")
    x = scalar("x")

    max_coefficients_supported = 10000

    # Generate the components of the polynomial
    components, updates = pytensor.scan(
        fn=lambda coefficient, power, free_variable: coefficient
        * (free_variable**power),
        outputs_info=None,
        sequences=[coefficients, pt.arange(max_coefficients_supported)],
        non_sequences=x,
    )
    # Sum them up
    polynomial = components.sum()

    output_str = debugprint(polynomial, file="str", print_op_info=True)
    lines = output_str.split("\n")

    expected_output = """Sum{axes=None} [id A]
     └─ Scan{scan_fn, while_loop=False, inplace=none} [id B] (outer_out_nit_sot-0)
        ├─ Minimum [id C] (outer_in_nit_sot-0)
        │  ├─ Subtensor{i} [id D]
        │  │  ├─ Shape [id E]
        │  │  │  └─ Subtensor{start:} [id F] 'coefficients[0:]'
        │  │  │     ├─ coefficients [id G]
        │  │  │     └─ 0 [id H]
        │  │  └─ 0 [id I]
        │  └─ Subtensor{i} [id J]
        │     ├─ Shape [id K]
        │     │  └─ Subtensor{start:} [id L]
        │     │     ├─ ARange{dtype='int64'} [id M]
        │     │     │  ├─ 0 [id N]
        │     │     │  ├─ 10000 [id O]
        │     │     │  └─ 1 [id P]
        │     │     └─ 0 [id Q]
        │     └─ 0 [id R]
        ├─ Subtensor{:stop} [id S] (outer_in_seqs-0)
        │  ├─ Subtensor{start:} [id F] 'coefficients[0:]'
        │  │  └─ ···
        │  └─ ScalarFromTensor [id T]
        │     └─ Minimum [id C]
        │        └─ ···
        ├─ Subtensor{:stop} [id U] (outer_in_seqs-1)
        │  ├─ Subtensor{start:} [id L]
        │  │  └─ ···
        │  └─ ScalarFromTensor [id V]
        │     └─ Minimum [id C]
        │        └─ ···
        ├─ Minimum [id C] (outer_in_nit_sot-0)
        │  └─ ···
        └─ x [id W] (outer_in_non_seqs-0)

    Inner graphs:

    Scan{scan_fn, while_loop=False, inplace=none} [id B]
     ← Mul [id X] (inner_out_nit_sot-0)
        ├─ *0-<Scalar(float64, shape=())> [id Y] -> [id S] (inner_in_seqs-0)
        └─ Pow [id Z]
           ├─ *2-<Scalar(float64, shape=())> [id BA] -> [id W] (inner_in_non_seqs-0)
           └─ *1-<Scalar(int64, shape=())> [id BB] -> [id U] (inner_in_seqs-1)
   """

    for truth, out in zip(expected_output.split("\n"), lines, strict=True):
        assert truth.strip() == out.strip()


@config.change_flags(floatX="float64")
def test_debugprint_nested_scans():
    c = dvector("c")
    n = 10

    k = iscalar("k")
    A = dvector("A")

    def compute_A_k(A, k):
        result, updates = pytensor.scan(
            fn=lambda prior_result, A: prior_result * A,
            outputs_info=pt.ones_like(A),
            non_sequences=A,
            n_steps=k,
        )

        A_k = result[-1]

        return A_k

    components, updates = pytensor.scan(
        fn=lambda c, power, some_A, some_k: c * (compute_A_k(some_A, some_k) ** power),
        outputs_info=None,
        sequences=[c, pt.arange(n)],
        non_sequences=[A, k],
    )
    final_result = components.sum()

    output_str = debugprint(final_result, file="str", print_op_info=True)
    lines = output_str.split("\n")

    expected_output = """Sum{axes=None} [id A]
     └─ Scan{scan_fn, while_loop=False, inplace=none} [id B] (outer_out_nit_sot-0)
        ├─ Minimum [id C] (outer_in_nit_sot-0)
        │  ├─ Subtensor{i} [id D]
        │  │  ├─ Shape [id E]
        │  │  │  └─ Subtensor{start:} [id F] 'c[0:]'
        │  │  │     ├─ c [id G]
        │  │  │     └─ 0 [id H]
        │  │  └─ 0 [id I]
        │  └─ Subtensor{i} [id J]
        │     ├─ Shape [id K]
        │     │  └─ Subtensor{start:} [id L]
        │     │     ├─ ARange{dtype='int64'} [id M]
        │     │     │  ├─ 0 [id N]
        │     │     │  ├─ 10 [id O]
        │     │     │  └─ 1 [id P]
        │     │     └─ 0 [id Q]
        │     └─ 0 [id R]
        ├─ Subtensor{:stop} [id S] (outer_in_seqs-0)
        │  ├─ Subtensor{start:} [id F] 'c[0:]'
        │  │  └─ ···
        │  └─ ScalarFromTensor [id T]
        │     └─ Minimum [id C]
        │        └─ ···
        ├─ Subtensor{:stop} [id U] (outer_in_seqs-1)
        │  ├─ Subtensor{start:} [id L]
        │  │  └─ ···
        │  └─ ScalarFromTensor [id V]
        │     └─ Minimum [id C]
        │        └─ ···
        ├─ Minimum [id C] (outer_in_nit_sot-0)
        │  └─ ···
        ├─ A [id W] (outer_in_non_seqs-0)
        └─ k [id X] (outer_in_non_seqs-1)

    Inner graphs:

    Scan{scan_fn, while_loop=False, inplace=none} [id B]
     ← Mul [id Y] (inner_out_nit_sot-0)
        ├─ ExpandDims{axis=0} [id Z]
        │  └─ *0-<Scalar(float64, shape=())> [id BA] -> [id S] (inner_in_seqs-0)
        └─ Pow [id BB]
           ├─ Subtensor{i} [id BC]
           │  ├─ Subtensor{start:} [id BD]
           │  │  ├─ Scan{scan_fn, while_loop=False, inplace=none} [id BE] (outer_out_sit_sot-0)
           │  │  │  ├─ *3-<Scalar(int32, shape=())> [id BF] -> [id X] (inner_in_non_seqs-1) (n_steps)
           │  │  │  ├─ SetSubtensor{:stop} [id BG] (outer_in_sit_sot-0)
           │  │  │  │  ├─ AllocEmpty{dtype='float64'} [id BH]
           │  │  │  │  │  ├─ Add [id BI]
           │  │  │  │  │  │  ├─ *3-<Scalar(int32, shape=())> [id BF] -> [id X] (inner_in_non_seqs-1)
           │  │  │  │  │  │  └─ Subtensor{i} [id BJ]
           │  │  │  │  │  │     ├─ Shape [id BK]
           │  │  │  │  │  │     │  └─ Unbroadcast{0} [id BL]
           │  │  │  │  │  │     │     └─ ExpandDims{axis=0} [id BM]
           │  │  │  │  │  │     │        └─ Second [id BN]
           │  │  │  │  │  │     │           ├─ *2-<Vector(float64, shape=(?,))> [id BO] -> [id W] (inner_in_non_seqs-0)
           │  │  │  │  │  │     │           └─ ExpandDims{axis=0} [id BP]
           │  │  │  │  │  │     │              └─ 1.0 [id BQ]
           │  │  │  │  │  │     └─ 0 [id BR]
           │  │  │  │  │  └─ Subtensor{i} [id BS]
           │  │  │  │  │     ├─ Shape [id BK]
           │  │  │  │  │     │  └─ ···
           │  │  │  │  │     └─ 1 [id BT]
           │  │  │  │  ├─ Unbroadcast{0} [id BL]
           │  │  │  │  │  └─ ···
           │  │  │  │  └─ ScalarFromTensor [id BU]
           │  │  │  │     └─ Subtensor{i} [id BJ]
           │  │  │  │        └─ ···
           │  │  │  └─ *2-<Vector(float64, shape=(?,))> [id BO] -> [id W] (inner_in_non_seqs-0) (outer_in_non_seqs-0)
           │  │  └─ 1 [id BV]
           │  └─ -1 [id BW]
           └─ ExpandDims{axis=0} [id BX]
              └─ *1-<Scalar(int64, shape=())> [id BY] -> [id U] (inner_in_seqs-1)

    Scan{scan_fn, while_loop=False, inplace=none} [id BE]
     ← Mul [id BZ] (inner_out_sit_sot-0)
        ├─ *0-<Vector(float64, shape=(?,))> [id CA] -> [id BG] (inner_in_sit_sot-0)
        └─ *1-<Vector(float64, shape=(?,))> [id CB] -> [id BO] (inner_in_non_seqs-0)
    """

    for truth, out in zip(expected_output.split("\n"), lines, strict=True):
        assert truth.strip() == out.strip()

    fg = FunctionGraph([c, k, A], [final_result])

    output_str = debugprint(
        fg, file="str", print_op_info=True, print_fgraph_inputs=True
    )
    lines = output_str.split("\n")

    expected_output = """→ c [id A]
    → k [id B]
    → A [id C]
    Sum{axes=None} [id D] 13
     └─ Scan{scan_fn, while_loop=False, inplace=none} [id E] 12 (outer_out_nit_sot-0)
        ├─ Minimum [id F] 7 (outer_in_nit_sot-0)
        │  ├─ Subtensor{i} [id G] 6
        │  │  ├─ Shape [id H] 5
        │  │  │  └─ Subtensor{start:} [id I] 'c[0:]' 4
        │  │  │     ├─ c [id A]
        │  │  │     └─ 0 [id J]
        │  │  └─ 0 [id K]
        │  └─ Subtensor{i} [id L] 3
        │     ├─ Shape [id M] 2
        │     │  └─ Subtensor{start:} [id N] 1
        │     │     ├─ ARange{dtype='int64'} [id O] 0
        │     │     │  ├─ 0 [id P]
        │     │     │  ├─ 10 [id Q]
        │     │     │  └─ 1 [id R]
        │     │     └─ 0 [id S]
        │     └─ 0 [id T]
        ├─ Subtensor{:stop} [id U] 11 (outer_in_seqs-0)
        │  ├─ Subtensor{start:} [id I] 'c[0:]' 4
        │  │  └─ ···
        │  └─ ScalarFromTensor [id V] 10
        │     └─ Minimum [id F] 7
        │        └─ ···
        ├─ Subtensor{:stop} [id W] 9 (outer_in_seqs-1)
        │  ├─ Subtensor{start:} [id N] 1
        │  │  └─ ···
        │  └─ ScalarFromTensor [id X] 8
        │     └─ Minimum [id F] 7
        │        └─ ···
        ├─ Minimum [id F] 7 (outer_in_nit_sot-0)
        │  └─ ···
        ├─ A [id C] (outer_in_non_seqs-0)
        └─ k [id B] (outer_in_non_seqs-1)

    Inner graphs:

    Scan{scan_fn, while_loop=False, inplace=none} [id E]
     → *0-<Scalar(float64, shape=())> [id Y] -> [id U] (inner_in_seqs-0)
     → *1-<Scalar(int64, shape=())> [id Z] -> [id W] (inner_in_seqs-1)
     → *2-<Vector(float64, shape=(?,))> [id BA] -> [id C] (inner_in_non_seqs-0)
     → *3-<Scalar(int32, shape=())> [id BB] -> [id B] (inner_in_non_seqs-1)
     ← Mul [id BC] (inner_out_nit_sot-0)
        ├─ ExpandDims{axis=0} [id BD]
        │  └─ *0-<Scalar(float64, shape=())> [id Y] (inner_in_seqs-0)
        └─ Pow [id BE]
           ├─ Subtensor{i} [id BF]
           │  ├─ Subtensor{start:} [id BG]
           │  │  ├─ Scan{scan_fn, while_loop=False, inplace=none} [id BH] (outer_out_sit_sot-0)
           │  │  │  ├─ *3-<Scalar(int32, shape=())> [id BB] (inner_in_non_seqs-1) (n_steps)
           │  │  │  ├─ SetSubtensor{:stop} [id BI] (outer_in_sit_sot-0)
           │  │  │  │  ├─ AllocEmpty{dtype='float64'} [id BJ]
           │  │  │  │  │  ├─ Add [id BK]
           │  │  │  │  │  │  ├─ *3-<Scalar(int32, shape=())> [id BB] (inner_in_non_seqs-1)
           │  │  │  │  │  │  └─ Subtensor{i} [id BL]
           │  │  │  │  │  │     ├─ Shape [id BM]
           │  │  │  │  │  │     │  └─ Unbroadcast{0} [id BN]
           │  │  │  │  │  │     │     └─ ExpandDims{axis=0} [id BO]
           │  │  │  │  │  │     │        └─ Second [id BP]
           │  │  │  │  │  │     │           ├─ *2-<Vector(float64, shape=(?,))> [id BA] (inner_in_non_seqs-0)
           │  │  │  │  │  │     │           └─ ExpandDims{axis=0} [id BQ]
           │  │  │  │  │  │     │              └─ 1.0 [id BR]
           │  │  │  │  │  │     └─ 0 [id BS]
           │  │  │  │  │  └─ Subtensor{i} [id BT]
           │  │  │  │  │     ├─ Shape [id BM]
           │  │  │  │  │     │  └─ ···
           │  │  │  │  │     └─ 1 [id BU]
           │  │  │  │  ├─ Unbroadcast{0} [id BN]
           │  │  │  │  │  └─ ···
           │  │  │  │  └─ ScalarFromTensor [id BV]
           │  │  │  │     └─ Subtensor{i} [id BL]
           │  │  │  │        └─ ···
           │  │  │  └─ *2-<Vector(float64, shape=(?,))> [id BA] (inner_in_non_seqs-0) (outer_in_non_seqs-0)
           │  │  └─ 1 [id BW]
           │  └─ -1 [id BX]
           └─ ExpandDims{axis=0} [id BY]
              └─ *1-<Scalar(int64, shape=())> [id Z] (inner_in_seqs-1)

    Scan{scan_fn, while_loop=False, inplace=none} [id BH]
     → *0-<Vector(float64, shape=(?,))> [id BZ] -> [id BI] (inner_in_sit_sot-0)
     → *1-<Vector(float64, shape=(?,))> [id CA] -> [id BA] (inner_in_non_seqs-0)
     ← Mul [id CB] (inner_out_sit_sot-0)
        ├─ *0-<Vector(float64, shape=(?,))> [id BZ] (inner_in_sit_sot-0)
        └─ *1-<Vector(float64, shape=(?,))> [id CA] (inner_in_non_seqs-0)
    """

    for truth, out in zip(expected_output.split("\n"), lines, strict=True):
        assert truth.strip() == out.strip()


@config.change_flags(floatX="float64")
def test_debugprint_mitsot():
    def fn(a_m2, a_m1, b_m2, b_m1):
        return a_m1 + a_m2, b_m1 + b_m2

    a0 = pytensor.shared(np.arange(2, dtype="int64"))
    b0 = pytensor.shared(np.arange(2, dtype="int64"))

    (a, b), _ = pytensor.scan(
        fn,
        outputs_info=[
            {"initial": a0, "taps": [-2, -1]},
            {"initial": b0, "taps": [-2, -1]},
        ],
        n_steps=5,
    )

    final_result = a + b
    output_str = debugprint(final_result, file="str", print_op_info=True)
    lines = output_str.split("\n")

    expected_output = """Add [id A]
     ├─ Subtensor{start:} [id B]
     │  ├─ Scan{scan_fn, while_loop=False, inplace=none}.0 [id C] (outer_out_mit_sot-0)
     │  │  ├─ 5 [id D] (n_steps)
     │  │  ├─ SetSubtensor{:stop} [id E] (outer_in_mit_sot-0)
     │  │  │  ├─ AllocEmpty{dtype='int64'} [id F]
     │  │  │  │  └─ Add [id G]
     │  │  │  │     ├─ 5 [id D]
     │  │  │  │     └─ Subtensor{i} [id H]
     │  │  │  │        ├─ Shape [id I]
     │  │  │  │        │  └─ Subtensor{:stop} [id J]
     │  │  │  │        │     ├─ <Vector(int64, shape=(?,))> [id K]
     │  │  │  │        │     └─ 2 [id L]
     │  │  │  │        └─ 0 [id M]
     │  │  │  ├─ Subtensor{:stop} [id J]
     │  │  │  │  └─ ···
     │  │  │  └─ ScalarFromTensor [id N]
     │  │  │     └─ Subtensor{i} [id H]
     │  │  │        └─ ···
     │  │  └─ SetSubtensor{:stop} [id O] (outer_in_mit_sot-1)
     │  │     ├─ AllocEmpty{dtype='int64'} [id P]
     │  │     │  └─ Add [id Q]
     │  │     │     ├─ 5 [id D]
     │  │     │     └─ Subtensor{i} [id R]
     │  │     │        ├─ Shape [id S]
     │  │     │        │  └─ Subtensor{:stop} [id T]
     │  │     │        │     ├─ <Vector(int64, shape=(?,))> [id U]
     │  │     │        │     └─ 2 [id V]
     │  │     │        └─ 0 [id W]
     │  │     ├─ Subtensor{:stop} [id T]
     │  │     │  └─ ···
     │  │     └─ ScalarFromTensor [id X]
     │  │        └─ Subtensor{i} [id R]
     │  │           └─ ···
     │  └─ 2 [id Y]
     └─ Subtensor{start:} [id Z]
        ├─ Scan{scan_fn, while_loop=False, inplace=none}.1 [id C] (outer_out_mit_sot-1)
        │  └─ ···
        └─ 2 [id BA]

    Inner graphs:

    Scan{scan_fn, while_loop=False, inplace=none} [id C]
     ← Add [id BB] (inner_out_mit_sot-0)
        ├─ *1-<Scalar(int64, shape=())> [id BC] -> [id E] (inner_in_mit_sot-0-1)
        └─ *0-<Scalar(int64, shape=())> [id BD] -> [id E] (inner_in_mit_sot-0-0)
     ← Add [id BE] (inner_out_mit_sot-1)
        ├─ *3-<Scalar(int64, shape=())> [id BF] -> [id O] (inner_in_mit_sot-1-1)
        └─ *2-<Scalar(int64, shape=())> [id BG] -> [id O] (inner_in_mit_sot-1-0)
    """

    for truth, out in zip(expected_output.split("\n"), lines, strict=True):
        assert truth.strip() == out.strip()


@config.change_flags(floatX="float64")
def test_debugprint_mitmot():
    k = iscalar("k")
    A = dvector("A")

    # Symbolic description of the result
    result, updates = pytensor.scan(
        fn=lambda prior_result, A: prior_result * A,
        outputs_info=pt.ones_like(A),
        non_sequences=A,
        n_steps=k,
    )

    final_result = pytensor.grad(result[-1].sum(), A)

    output_str = debugprint(final_result, file="str", print_op_info=True)
    lines = output_str.split("\n")

    expected_output = """Subtensor{i} [id A]
     ├─ Scan{grad_of_scan_fn, while_loop=False, inplace=none}.1 [id B] (outer_out_sit_sot-0)
     │  ├─ Sub [id C] (n_steps)
     │  │  ├─ Subtensor{i} [id D]
     │  │  │  ├─ Shape [id E]
     │  │  │  │  └─ Scan{scan_fn, while_loop=False, inplace=none} [id F] (outer_out_sit_sot-0)
     │  │  │  │     ├─ k [id G] (n_steps)
     │  │  │  │     ├─ SetSubtensor{:stop} [id H] (outer_in_sit_sot-0)
     │  │  │  │     │  ├─ AllocEmpty{dtype='float64'} [id I]
     │  │  │  │     │  │  ├─ Add [id J]
     │  │  │  │     │  │  │  ├─ k [id G]
     │  │  │  │     │  │  │  └─ Subtensor{i} [id K]
     │  │  │  │     │  │  │     ├─ Shape [id L]
     │  │  │  │     │  │  │     │  └─ Unbroadcast{0} [id M]
     │  │  │  │     │  │  │     │     └─ ExpandDims{axis=0} [id N]
     │  │  │  │     │  │  │     │        └─ Second [id O]
     │  │  │  │     │  │  │     │           ├─ A [id P]
     │  │  │  │     │  │  │     │           └─ ExpandDims{axis=0} [id Q]
     │  │  │  │     │  │  │     │              └─ 1.0 [id R]
     │  │  │  │     │  │  │     └─ 0 [id S]
     │  │  │  │     │  │  └─ Subtensor{i} [id T]
     │  │  │  │     │  │     ├─ Shape [id L]
     │  │  │  │     │  │     │  └─ ···
     │  │  │  │     │  │     └─ 1 [id U]
     │  │  │  │     │  ├─ Unbroadcast{0} [id M]
     │  │  │  │     │  │  └─ ···
     │  │  │  │     │  └─ ScalarFromTensor [id V]
     │  │  │  │     │     └─ Subtensor{i} [id K]
     │  │  │  │     │        └─ ···
     │  │  │  │     └─ A [id P] (outer_in_non_seqs-0)
     │  │  │  └─ 0 [id W]
     │  │  └─ 1 [id X]
     │  ├─ Subtensor{:stop} [id Y] (outer_in_seqs-0)
     │  │  ├─ Subtensor{::step} [id Z]
     │  │  │  ├─ Subtensor{:stop} [id BA]
     │  │  │  │  ├─ Scan{scan_fn, while_loop=False, inplace=none} [id F] (outer_out_sit_sot-0)
     │  │  │  │  │  └─ ···
     │  │  │  │  └─ -1 [id BB]
     │  │  │  └─ -1 [id BC]
     │  │  └─ ScalarFromTensor [id BD]
     │  │     └─ Sub [id C]
     │  │        └─ ···
     │  ├─ Subtensor{:stop} [id BE] (outer_in_seqs-1)
     │  │  ├─ Subtensor{:stop} [id BF]
     │  │  │  ├─ Subtensor{::step} [id BG]
     │  │  │  │  ├─ Scan{scan_fn, while_loop=False, inplace=none} [id F] (outer_out_sit_sot-0)
     │  │  │  │  │  └─ ···
     │  │  │  │  └─ -1 [id BH]
     │  │  │  └─ -1 [id BI]
     │  │  └─ ScalarFromTensor [id BJ]
     │  │     └─ Sub [id C]
     │  │        └─ ···
     │  ├─ Subtensor{::step} [id BK] (outer_in_mit_mot-0)
     │  │  ├─ IncSubtensor{start:} [id BL]
     │  │  │  ├─ Second [id BM]
     │  │  │  │  ├─ Scan{scan_fn, while_loop=False, inplace=none} [id F] (outer_out_sit_sot-0)
     │  │  │  │  │  └─ ···
     │  │  │  │  └─ ExpandDims{axes=[0, 1]} [id BN]
     │  │  │  │     └─ 0.0 [id BO]
     │  │  │  ├─ IncSubtensor{i} [id BP]
     │  │  │  │  ├─ Second [id BQ]
     │  │  │  │  │  ├─ Subtensor{start:} [id BR]
     │  │  │  │  │  │  ├─ Scan{scan_fn, while_loop=False, inplace=none} [id F] (outer_out_sit_sot-0)
     │  │  │  │  │  │  │  └─ ···
     │  │  │  │  │  │  └─ 1 [id BS]
     │  │  │  │  │  └─ ExpandDims{axes=[0, 1]} [id BT]
     │  │  │  │  │     └─ 0.0 [id BU]
     │  │  │  │  ├─ Second [id BV]
     │  │  │  │  │  ├─ Subtensor{i} [id BW]
     │  │  │  │  │  │  ├─ Subtensor{start:} [id BR]
     │  │  │  │  │  │  │  └─ ···
     │  │  │  │  │  │  └─ -1 [id BX]
     │  │  │  │  │  └─ ExpandDims{axis=0} [id BY]
     │  │  │  │  │     └─ Second [id BZ]
     │  │  │  │  │        ├─ Sum{axes=None} [id CA]
     │  │  │  │  │        │  └─ Subtensor{i} [id BW]
     │  │  │  │  │        │     └─ ···
     │  │  │  │  │        └─ 1.0 [id CB]
     │  │  │  │  └─ -1 [id BX]
     │  │  │  └─ 1 [id BS]
     │  │  └─ -1 [id CC]
     │  ├─ Alloc [id CD] (outer_in_sit_sot-0)
     │  │  ├─ 0.0 [id CE]
     │  │  ├─ Add [id CF]
     │  │  │  ├─ Sub [id C]
     │  │  │  │  └─ ···
     │  │  │  └─ 1 [id CG]
     │  │  └─ Subtensor{i} [id CH]
     │  │     ├─ Shape [id CI]
     │  │     │  └─ A [id P]
     │  │     └─ 0 [id CJ]
     │  └─ A [id P] (outer_in_non_seqs-0)
     └─ -1 [id CK]

    Inner graphs:

    Scan{grad_of_scan_fn, while_loop=False, inplace=none} [id B]
     ← Add [id CL] (inner_out_mit_mot-0-0)
        ├─ Mul [id CM]
        │  ├─ *2-<Vector(float64, shape=(?,))> [id CN] -> [id BK] (inner_in_mit_mot-0-0)
        │  └─ *5-<Vector(float64, shape=(?,))> [id CO] -> [id P] (inner_in_non_seqs-0)
        └─ *3-<Vector(float64, shape=(?,))> [id CP] -> [id BK] (inner_in_mit_mot-0-1)
     ← Add [id CQ] (inner_out_sit_sot-0)
        ├─ Mul [id CR]
        │  ├─ *2-<Vector(float64, shape=(?,))> [id CN] -> [id BK] (inner_in_mit_mot-0-0)
        │  └─ *0-<Vector(float64, shape=(?,))> [id CS] -> [id Y] (inner_in_seqs-0)
        └─ *4-<Vector(float64, shape=(?,))> [id CT] -> [id CD] (inner_in_sit_sot-0)

    Scan{scan_fn, while_loop=False, inplace=none} [id F]
     ← Mul [id CU] (inner_out_sit_sot-0)
        ├─ *0-<Vector(float64, shape=(?,))> [id CS] -> [id H] (inner_in_sit_sot-0)
        └─ *1-<Vector(float64, shape=(?,))> [id CV] -> [id P] (inner_in_non_seqs-0)
    """

    for truth, out in zip(expected_output.split("\n"), lines, strict=True):
        assert truth.strip() == out.strip()


def test_debugprint_compiled_fn():
    M = pt.tensor(dtype=np.float64, shape=(20000, 2, 2))
    one = pt.as_tensor(1, dtype=np.int64)
    zero = pt.as_tensor(0, dtype=np.int64)

    def no_shared_fn(n, x_tm1, M):
        p = M[n, x_tm1]
        return pt.switch(pt.lt(zero, p[0]), one, zero)

    out, updates = pytensor.scan(
        no_shared_fn,
        outputs_info=[{"initial": zero, "taps": [-1]}],
        sequences=[pt.arange(M.shape[0])],
        non_sequences=[M],
        allow_gc=False,
        mode="FAST_RUN",
    )

    # In this case, `debugprint` should print the compiled inner-graph
    # (i.e. from `Scan._fn`)
    out = pytensor.function([M], out, updates=updates, mode="FAST_RUN")

    expected_output = """Subtensor{start:} [id A] 3
 ├─ Scan{scan_fn, while_loop=False, inplace=all} [id B] 2 (outer_out_sit_sot-0)
 │  ├─ 20000 [id C] (n_steps)
 │  ├─ [    0 ... 998 19999] [id D] (outer_in_seqs-0)
 │  ├─ SetSubtensor{:stop} [id E] 1 (outer_in_sit_sot-0)
 │  │  ├─ AllocEmpty{dtype='int64'} [id F] 0
 │  │  │  └─ 20001 [id G]
 │  │  ├─ [0] [id H]
 │  │  └─ 1 [id I]
 │  └─ <Tensor3(float64, shape=(20000, 2, 2))> [id J] (outer_in_non_seqs-0)
 └─ 1 [id I]

Inner graphs:

Scan{scan_fn, while_loop=False, inplace=all} [id B]
 ← Composite{switch(lt(0, i0), 1, 0)} [id K] (inner_out_sit_sot-0)
    └─ Subtensor{i, j, k} [id L]
       ├─ *2-<Tensor3(float64, shape=(20000, 2, 2))> [id M] -> [id J] (inner_in_non_seqs-0)
       ├─ ScalarFromTensor [id N]
       │  └─ *0-<Scalar(int64, shape=())> [id O] -> [id D] (inner_in_seqs-0)
       ├─ ScalarFromTensor [id P]
       │  └─ *1-<Scalar(int64, shape=())> [id Q] -> [id E] (inner_in_sit_sot-0)
       └─ 0 [id R]

Composite{switch(lt(0, i0), 1, 0)} [id K]
 ← Switch [id S] 'o0'
    ├─ LT [id T]
    │  ├─ 0 [id U]
    │  └─ i0 [id V]
    ├─ 1 [id W]
    └─ 0 [id U]
    """

    output_str = debugprint(out, file="str", print_op_info=True)
    lines = output_str.split("\n")

    for truth, out in zip(expected_output.split("\n"), lines, strict=True):
        assert truth.strip() == out.strip()


try:
    _try_pydot_import()
    pydot_imported = True
except Exception:
    pydot_imported = False


@pytest.mark.skipif(not pydot_imported, reason="pydot not available")
def test_pydotprint():
    def f_pow2(x_tm1):
        return 2 * x_tm1

    state = scalar("state")
    n_steps = iscalar("nsteps")
    output, updates = pytensor.scan(
        f_pow2, [], state, [], n_steps=n_steps, truncate_gradient=-1, go_backwards=False
    )
    f = pytensor.function(
        [state, n_steps], output, updates=updates, allow_input_downcast=True
    )
    pydotprint(output, scan_graphs=True)
    pydotprint(f, scan_graphs=True)
