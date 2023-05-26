import numpy as np
import pytest

import pytensor
import pytensor.tensor as at
from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.printing import debugprint, pydot_imported, pydotprint
from pytensor.tensor.type import dvector, iscalar, scalar, vector


@config.change_flags(floatX="float64")
def test_debugprint_sitsot():
    k = iscalar("k")
    A = dvector("A")

    # Symbolic description of the result
    result, updates = pytensor.scan(
        fn=lambda prior_result, A: prior_result * A,
        outputs_info=at.ones_like(A),
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
     │  │  │  │  │     │              └─ TensorConstant{1.0} [id O]
     │  │  │  │  │     └─ ScalarConstant{0} [id P]
     │  │  │  │  └─ Subtensor{i} [id Q]
     │  │  │  │     ├─ Shape [id R]
     │  │  │  │     │  └─ Unbroadcast{0} [id J]
     │  │  │  │     │     └─ ···
     │  │  │  │     └─ ScalarConstant{1} [id S]
     │  │  │  ├─ Unbroadcast{0} [id J]
     │  │  │  │  └─ ···
     │  │  │  └─ ScalarFromTensor [id T]
     │  │  │     └─ Subtensor{i} [id H]
     │  │  │        └─ ···
     │  │  └─ A [id M] (outer_in_non_seqs-0)
     │  └─ ScalarConstant{1} [id U]
     └─ ScalarConstant{-1} [id V]

    Inner graphs:

    Scan{scan_fn, while_loop=False, inplace=none} [id C]
     ← Mul [id W] (inner_out_sit_sot-0)
        ├─ *0-<TensorType(float64, (?,))> [id X] -> [id E] (inner_in_sit_sot-0)
        └─ *1-<TensorType(float64, (?,))> [id Y] -> [id M] (inner_in_non_seqs-0)"""

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip()


def test_debugprint_sitsot_no_extra_info():
    k = iscalar("k")
    A = dvector("A")

    # Symbolic description of the result
    result, updates = pytensor.scan(
        fn=lambda prior_result, A: prior_result * A,
        outputs_info=at.ones_like(A),
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
     │  │  │  │  │     │              └─ TensorConstant{1.0} [id O]
     │  │  │  │  │     └─ ScalarConstant{0} [id P]
     │  │  │  │  └─ Subtensor{i} [id Q]
     │  │  │  │     ├─ Shape [id R]
     │  │  │  │     │  └─ Unbroadcast{0} [id J]
     │  │  │  │     │     └─ ···
     │  │  │  │     └─ ScalarConstant{1} [id S]
     │  │  │  ├─ Unbroadcast{0} [id J]
     │  │  │  │  └─ ···
     │  │  │  └─ ScalarFromTensor [id T]
     │  │  │     └─ Subtensor{i} [id H]
     │  │  │        └─ ···
     │  │  └─ A [id M]
     │  └─ ScalarConstant{1} [id U]
     └─ ScalarConstant{-1} [id V]

    Inner graphs:

    Scan{scan_fn, while_loop=False, inplace=none} [id C]
     ← Mul [id W]
        ├─ *0-<TensorType(float64, (?,))> [id X] -> [id E]
        └─ *1-<TensorType(float64, (?,))> [id Y] -> [id M]"""

    for truth, out in zip(expected_output.split("\n"), lines):
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
        sequences=[coefficients, at.arange(max_coefficients_supported)],
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
        │  │  │     └─ ScalarConstant{0} [id H]
        │  │  └─ ScalarConstant{0} [id I]
        │  └─ Subtensor{i} [id J]
        │     ├─ Shape [id K]
        │     │  └─ Subtensor{start:} [id L]
        │     │     ├─ ARange{dtype='int64'} [id M]
        │     │     │  ├─ TensorConstant{0} [id N]
        │     │     │  ├─ TensorConstant{10000} [id O]
        │     │     │  └─ TensorConstant{1} [id P]
        │     │     └─ ScalarConstant{0} [id Q]
        │     └─ ScalarConstant{0} [id R]
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
        ├─ *0-<TensorType(float64, ())> [id Y] -> [id S] (inner_in_seqs-0)
        └─ Pow [id Z]
           ├─ *2-<TensorType(float64, ())> [id BA] -> [id W] (inner_in_non_seqs-0)
           └─ *1-<TensorType(int64, ())> [id BB] -> [id U] (inner_in_seqs-1)"""

    for truth, out in zip(expected_output.split("\n"), lines):
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
            outputs_info=at.ones_like(A),
            non_sequences=A,
            n_steps=k,
        )

        A_k = result[-1]

        return A_k

    components, updates = pytensor.scan(
        fn=lambda c, power, some_A, some_k: c * (compute_A_k(some_A, some_k) ** power),
        outputs_info=None,
        sequences=[c, at.arange(n)],
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
        │  │  │     └─ ScalarConstant{0} [id H]
        │  │  └─ ScalarConstant{0} [id I]
        │  └─ Subtensor{i} [id J]
        │     ├─ Shape [id K]
        │     │  └─ Subtensor{start:} [id L]
        │     │     ├─ ARange{dtype='int64'} [id M]
        │     │     │  ├─ TensorConstant{0} [id N]
        │     │     │  ├─ TensorConstant{10} [id O]
        │     │     │  └─ TensorConstant{1} [id P]
        │     │     └─ ScalarConstant{0} [id Q]
        │     └─ ScalarConstant{0} [id R]
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
        │  └─ *0-<TensorType(float64, ())> [id BA] -> [id S] (inner_in_seqs-0)
        └─ Pow [id BB]
           ├─ Subtensor{i} [id BC]
           │  ├─ Subtensor{start:} [id BD]
           │  │  ├─ Scan{scan_fn, while_loop=False, inplace=none} [id BE] (outer_out_sit_sot-0)
           │  │  │  ├─ *3-<TensorType(int32, ())> [id BF] -> [id X] (inner_in_non_seqs-1) (n_steps)
           │  │  │  ├─ SetSubtensor{:stop} [id BG] (outer_in_sit_sot-0)
           │  │  │  │  ├─ AllocEmpty{dtype='float64'} [id BH]
           │  │  │  │  │  ├─ Add [id BI]
           │  │  │  │  │  │  ├─ *3-<TensorType(int32, ())> [id BF] -> [id X] (inner_in_non_seqs-1)
           │  │  │  │  │  │  └─ Subtensor{i} [id BJ]
           │  │  │  │  │  │     ├─ Shape [id BK]
           │  │  │  │  │  │     │  └─ Unbroadcast{0} [id BL]
           │  │  │  │  │  │     │     └─ ExpandDims{axis=0} [id BM]
           │  │  │  │  │  │     │        └─ Second [id BN]
           │  │  │  │  │  │     │           ├─ *2-<TensorType(float64, (?,))> [id BO] -> [id W] (inner_in_non_seqs-0)
           │  │  │  │  │  │     │           └─ ExpandDims{axis=0} [id BP]
           │  │  │  │  │  │     │              └─ TensorConstant{1.0} [id BQ]
           │  │  │  │  │  │     └─ ScalarConstant{0} [id BR]
           │  │  │  │  │  └─ Subtensor{i} [id BS]
           │  │  │  │  │     ├─ Shape [id BT]
           │  │  │  │  │     │  └─ Unbroadcast{0} [id BL]
           │  │  │  │  │     │     └─ ···
           │  │  │  │  │     └─ ScalarConstant{1} [id BU]
           │  │  │  │  ├─ Unbroadcast{0} [id BL]
           │  │  │  │  │  └─ ···
           │  │  │  │  └─ ScalarFromTensor [id BV]
           │  │  │  │     └─ Subtensor{i} [id BJ]
           │  │  │  │        └─ ···
           │  │  │  └─ *2-<TensorType(float64, (?,))> [id BO] -> [id W] (inner_in_non_seqs-0) (outer_in_non_seqs-0)
           │  │  └─ ScalarConstant{1} [id BW]
           │  └─ ScalarConstant{-1} [id BX]
           └─ ExpandDims{axis=0} [id BY]
              └─ *1-<TensorType(int64, ())> [id BZ] -> [id U] (inner_in_seqs-1)

    Scan{scan_fn, while_loop=False, inplace=none} [id BE]
     ← Mul [id CA] (inner_out_sit_sot-0)
        ├─ *0-<TensorType(float64, (?,))> [id CB] -> [id BG] (inner_in_sit_sot-0)
        └─ *1-<TensorType(float64, (?,))> [id CC] -> [id BO] (inner_in_non_seqs-0)"""

    for truth, out in zip(expected_output.split("\n"), lines):
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
        │  │  │     └─ ScalarConstant{0} [id J]
        │  │  └─ ScalarConstant{0} [id K]
        │  └─ Subtensor{i} [id L] 3
        │     ├─ Shape [id M] 2
        │     │  └─ Subtensor{start:} [id N] 1
        │     │     ├─ ARange{dtype='int64'} [id O] 0
        │     │     │  ├─ TensorConstant{0} [id P]
        │     │     │  ├─ TensorConstant{10} [id Q]
        │     │     │  └─ TensorConstant{1} [id R]
        │     │     └─ ScalarConstant{0} [id S]
        │     └─ ScalarConstant{0} [id T]
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
     → *0-<TensorType(float64, ())> [id Y] -> [id U] (inner_in_seqs-0)
     → *1-<TensorType(int64, ())> [id Z] -> [id W] (inner_in_seqs-1)
     → *2-<TensorType(float64, (?,))> [id BA] -> [id C] (inner_in_non_seqs-0)
     → *3-<TensorType(int32, ())> [id BB] -> [id B] (inner_in_non_seqs-1)
     ← Mul [id BC] (inner_out_nit_sot-0)
        ├─ ExpandDims{axis=0} [id BD]
        │  └─ *0-<TensorType(float64, ())> [id Y] (inner_in_seqs-0)
        └─ Pow [id BE]
           ├─ Subtensor{i} [id BF]
           │  ├─ Subtensor{start:} [id BG]
           │  │  ├─ Scan{scan_fn, while_loop=False, inplace=none} [id BH] (outer_out_sit_sot-0)
           │  │  │  ├─ *3-<TensorType(int32, ())> [id BB] (inner_in_non_seqs-1) (n_steps)
           │  │  │  ├─ SetSubtensor{:stop} [id BI] (outer_in_sit_sot-0)
           │  │  │  │  ├─ AllocEmpty{dtype='float64'} [id BJ]
           │  │  │  │  │  ├─ Add [id BK]
           │  │  │  │  │  │  ├─ *3-<TensorType(int32, ())> [id BB] (inner_in_non_seqs-1)
           │  │  │  │  │  │  └─ Subtensor{i} [id BL]
           │  │  │  │  │  │     ├─ Shape [id BM]
           │  │  │  │  │  │     │  └─ Unbroadcast{0} [id BN]
           │  │  │  │  │  │     │     └─ ExpandDims{axis=0} [id BO]
           │  │  │  │  │  │     │        └─ Second [id BP]
           │  │  │  │  │  │     │           ├─ *2-<TensorType(float64, (?,))> [id BA] (inner_in_non_seqs-0)
           │  │  │  │  │  │     │           └─ ExpandDims{axis=0} [id BQ]
           │  │  │  │  │  │     │              └─ TensorConstant{1.0} [id BR]
           │  │  │  │  │  │     └─ ScalarConstant{0} [id BS]
           │  │  │  │  │  └─ Subtensor{i} [id BT]
           │  │  │  │  │     ├─ Shape [id BU]
           │  │  │  │  │     │  └─ Unbroadcast{0} [id BN]
           │  │  │  │  │     │     └─ ···
           │  │  │  │  │     └─ ScalarConstant{1} [id BV]
           │  │  │  │  ├─ Unbroadcast{0} [id BN]
           │  │  │  │  │  └─ ···
           │  │  │  │  └─ ScalarFromTensor [id BW]
           │  │  │  │     └─ Subtensor{i} [id BL]
           │  │  │  │        └─ ···
           │  │  │  └─ *2-<TensorType(float64, (?,))> [id BA] (inner_in_non_seqs-0) (outer_in_non_seqs-0)
           │  │  └─ ScalarConstant{1} [id BX]
           │  └─ ScalarConstant{-1} [id BY]
           └─ ExpandDims{axis=0} [id BZ]
              └─ *1-<TensorType(int64, ())> [id Z] (inner_in_seqs-1)

    Scan{scan_fn, while_loop=False, inplace=none} [id BH]
     → *0-<TensorType(float64, (?,))> [id CA] -> [id BI] (inner_in_sit_sot-0)
     → *1-<TensorType(float64, (?,))> [id CB] -> [id BA] (inner_in_non_seqs-0)
     ← Mul [id CC] (inner_out_sit_sot-0)
        ├─ *0-<TensorType(float64, (?,))> [id CA] (inner_in_sit_sot-0)
        └─ *1-<TensorType(float64, (?,))> [id CB] (inner_in_non_seqs-0)"""

    for truth, out in zip(expected_output.split("\n"), lines):
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
     │  │  ├─ TensorConstant{5} [id D] (n_steps)
     │  │  ├─ SetSubtensor{:stop} [id E] (outer_in_mit_sot-0)
     │  │  │  ├─ AllocEmpty{dtype='int64'} [id F]
     │  │  │  │  └─ Add [id G]
     │  │  │  │     ├─ TensorConstant{5} [id D]
     │  │  │  │     └─ Subtensor{i} [id H]
     │  │  │  │        ├─ Shape [id I]
     │  │  │  │        │  └─ Subtensor{:stop} [id J]
     │  │  │  │        │     ├─ <TensorType(int64, (?,))> [id K]
     │  │  │  │        │     └─ ScalarConstant{2} [id L]
     │  │  │  │        └─ ScalarConstant{0} [id M]
     │  │  │  ├─ Subtensor{:stop} [id J]
     │  │  │  │  └─ ···
     │  │  │  └─ ScalarFromTensor [id N]
     │  │  │     └─ Subtensor{i} [id H]
     │  │  │        └─ ···
     │  │  └─ SetSubtensor{:stop} [id O] (outer_in_mit_sot-1)
     │  │     ├─ AllocEmpty{dtype='int64'} [id P]
     │  │     │  └─ Add [id Q]
     │  │     │     ├─ TensorConstant{5} [id D]
     │  │     │     └─ Subtensor{i} [id R]
     │  │     │        ├─ Shape [id S]
     │  │     │        │  └─ Subtensor{:stop} [id T]
     │  │     │        │     ├─ <TensorType(int64, (?,))> [id U]
     │  │     │        │     └─ ScalarConstant{2} [id V]
     │  │     │        └─ ScalarConstant{0} [id W]
     │  │     ├─ Subtensor{:stop} [id T]
     │  │     │  └─ ···
     │  │     └─ ScalarFromTensor [id X]
     │  │        └─ Subtensor{i} [id R]
     │  │           └─ ···
     │  └─ ScalarConstant{2} [id Y]
     └─ Subtensor{start:} [id Z]
        ├─ Scan{scan_fn, while_loop=False, inplace=none}.1 [id C] (outer_out_mit_sot-1)
        │  └─ ···
        └─ ScalarConstant{2} [id BA]

    Inner graphs:

    Scan{scan_fn, while_loop=False, inplace=none} [id C]
     ← Add [id BB] (inner_out_mit_sot-0)
        ├─ *1-<TensorType(int64, ())> [id BC] -> [id E] (inner_in_mit_sot-0-1)
        └─ *0-<TensorType(int64, ())> [id BD] -> [id E] (inner_in_mit_sot-0-0)
     ← Add [id BE] (inner_out_mit_sot-1)
        ├─ *3-<TensorType(int64, ())> [id BF] -> [id O] (inner_in_mit_sot-1-1)
        └─ *2-<TensorType(int64, ())> [id BG] -> [id O] (inner_in_mit_sot-1-0)"""

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip()


@config.change_flags(floatX="float64")
def test_debugprint_mitmot():
    k = iscalar("k")
    A = dvector("A")

    # Symbolic description of the result
    result, updates = pytensor.scan(
        fn=lambda prior_result, A: prior_result * A,
        outputs_info=at.ones_like(A),
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
     │  │  │  │     │  │  │     │              └─ TensorConstant{1.0} [id R]
     │  │  │  │     │  │  │     └─ ScalarConstant{0} [id S]
     │  │  │  │     │  │  └─ Subtensor{i} [id T]
     │  │  │  │     │  │     ├─ Shape [id U]
     │  │  │  │     │  │     │  └─ Unbroadcast{0} [id M]
     │  │  │  │     │  │     │     └─ ···
     │  │  │  │     │  │     └─ ScalarConstant{1} [id V]
     │  │  │  │     │  ├─ Unbroadcast{0} [id M]
     │  │  │  │     │  │  └─ ···
     │  │  │  │     │  └─ ScalarFromTensor [id W]
     │  │  │  │     │     └─ Subtensor{i} [id K]
     │  │  │  │     │        └─ ···
     │  │  │  │     └─ A [id P] (outer_in_non_seqs-0)
     │  │  │  └─ ScalarConstant{0} [id X]
     │  │  └─ TensorConstant{1} [id Y]
     │  ├─ Subtensor{:stop} [id Z] (outer_in_seqs-0)
     │  │  ├─ Subtensor{::step} [id BA]
     │  │  │  ├─ Subtensor{:stop} [id BB]
     │  │  │  │  ├─ Scan{scan_fn, while_loop=False, inplace=none} [id F] (outer_out_sit_sot-0)
     │  │  │  │  │  └─ ···
     │  │  │  │  └─ ScalarConstant{-1} [id BC]
     │  │  │  └─ ScalarConstant{-1} [id BD]
     │  │  └─ ScalarFromTensor [id BE]
     │  │     └─ Sub [id C]
     │  │        └─ ···
     │  ├─ Subtensor{:stop} [id BF] (outer_in_seqs-1)
     │  │  ├─ Subtensor{:stop} [id BG]
     │  │  │  ├─ Subtensor{::step} [id BH]
     │  │  │  │  ├─ Scan{scan_fn, while_loop=False, inplace=none} [id F] (outer_out_sit_sot-0)
     │  │  │  │  │  └─ ···
     │  │  │  │  └─ ScalarConstant{-1} [id BI]
     │  │  │  └─ ScalarConstant{-1} [id BJ]
     │  │  └─ ScalarFromTensor [id BK]
     │  │     └─ Sub [id C]
     │  │        └─ ···
     │  ├─ Subtensor{::step} [id BL] (outer_in_mit_mot-0)
     │  │  ├─ IncSubtensor{start:} [id BM]
     │  │  │  ├─ Second [id BN]
     │  │  │  │  ├─ Scan{scan_fn, while_loop=False, inplace=none} [id F] (outer_out_sit_sot-0)
     │  │  │  │  │  └─ ···
     │  │  │  │  └─ ExpandDims{axes=[0, 1]} [id BO]
     │  │  │  │     └─ TensorConstant{0.0} [id BP]
     │  │  │  ├─ IncSubtensor{i} [id BQ]
     │  │  │  │  ├─ Second [id BR]
     │  │  │  │  │  ├─ Subtensor{start:} [id BS]
     │  │  │  │  │  │  ├─ Scan{scan_fn, while_loop=False, inplace=none} [id F] (outer_out_sit_sot-0)
     │  │  │  │  │  │  │  └─ ···
     │  │  │  │  │  │  └─ ScalarConstant{1} [id BT]
     │  │  │  │  │  └─ ExpandDims{axes=[0, 1]} [id BU]
     │  │  │  │  │     └─ TensorConstant{0.0} [id BV]
     │  │  │  │  ├─ Second [id BW]
     │  │  │  │  │  ├─ Subtensor{i} [id BX]
     │  │  │  │  │  │  ├─ Subtensor{start:} [id BS]
     │  │  │  │  │  │  │  └─ ···
     │  │  │  │  │  │  └─ ScalarConstant{-1} [id BY]
     │  │  │  │  │  └─ ExpandDims{axis=0} [id BZ]
     │  │  │  │  │     └─ Second [id CA]
     │  │  │  │  │        ├─ Sum{axes=None} [id CB]
     │  │  │  │  │        │  └─ Subtensor{i} [id BX]
     │  │  │  │  │        │     └─ ···
     │  │  │  │  │        └─ TensorConstant{1.0} [id CC]
     │  │  │  │  └─ ScalarConstant{-1} [id BY]
     │  │  │  └─ ScalarConstant{1} [id BT]
     │  │  └─ ScalarConstant{-1} [id CD]
     │  ├─ Alloc [id CE] (outer_in_sit_sot-0)
     │  │  ├─ TensorConstant{0.0} [id CF]
     │  │  ├─ Add [id CG]
     │  │  │  ├─ Sub [id C]
     │  │  │  │  └─ ···
     │  │  │  └─ TensorConstant{1} [id CH]
     │  │  └─ Subtensor{i} [id CI]
     │  │     ├─ Shape [id CJ]
     │  │     │  └─ A [id P]
     │  │     └─ ScalarConstant{0} [id CK]
     │  └─ A [id P] (outer_in_non_seqs-0)
     └─ ScalarConstant{-1} [id CL]

    Inner graphs:

    Scan{grad_of_scan_fn, while_loop=False, inplace=none} [id B]
     ← Add [id CM] (inner_out_mit_mot-0-0)
        ├─ Mul [id CN]
        │  ├─ *2-<TensorType(float64, (?,))> [id CO] -> [id BL] (inner_in_mit_mot-0-0)
        │  └─ *5-<TensorType(float64, (?,))> [id CP] -> [id P] (inner_in_non_seqs-0)
        └─ *3-<TensorType(float64, (?,))> [id CQ] -> [id BL] (inner_in_mit_mot-0-1)
     ← Add [id CR] (inner_out_sit_sot-0)
        ├─ Mul [id CS]
        │  ├─ *2-<TensorType(float64, (?,))> [id CO] -> [id BL] (inner_in_mit_mot-0-0)
        │  └─ *0-<TensorType(float64, (?,))> [id CT] -> [id Z] (inner_in_seqs-0)
        └─ *4-<TensorType(float64, (?,))> [id CU] -> [id CE] (inner_in_sit_sot-0)

    Scan{scan_fn, while_loop=False, inplace=none} [id F]
     ← Mul [id CV] (inner_out_sit_sot-0)
        ├─ *0-<TensorType(float64, (?,))> [id CT] -> [id H] (inner_in_sit_sot-0)
        └─ *1-<TensorType(float64, (?,))> [id CW] -> [id P] (inner_in_non_seqs-0)"""

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip()


def test_debugprint_compiled_fn():
    M = at.tensor(dtype=np.float64, shape=(20000, 2, 2))
    one = at.as_tensor(1, dtype=np.int64)
    zero = at.as_tensor(0, dtype=np.int64)

    def no_shared_fn(n, x_tm1, M):
        p = M[n, x_tm1]
        return at.switch(at.lt(zero, p[0]), one, zero)

    out, updates = pytensor.scan(
        no_shared_fn,
        outputs_info=[{"initial": zero, "taps": [-1]}],
        sequences=[at.arange(M.shape[0])],
        non_sequences=[M],
        allow_gc=False,
        mode="FAST_RUN",
    )

    # In this case, `debugprint` should print the compiled inner-graph
    # (i.e. from `Scan._fn`)
    out = pytensor.function([M], out, updates=updates, mode="FAST_RUN")

    expected_output = """Scan{scan_fn, while_loop=False, inplace=all} [id A] 2 (outer_out_sit_sot-0)
     ├─ TensorConstant{20000} [id B] (n_steps)
     ├─ TensorConstant{[    0    ..998 19999]} [id C] (outer_in_seqs-0)
     ├─ SetSubtensor{:stop} [id D] 1 (outer_in_sit_sot-0)
     │  ├─ AllocEmpty{dtype='int64'} [id E] 0
     │  │  └─ TensorConstant{20000} [id B]
     │  ├─ TensorConstant{(1,) of 0} [id F]
     │  └─ ScalarConstant{1} [id G]
     └─ <TensorType(float64, (20000, 2, 2))> [id H] (outer_in_non_seqs-0)

    Inner graphs:

    Scan{scan_fn, while_loop=False, inplace=all} [id A]
     ← Composite{switch(lt(i0, i1), i2, i0)} [id I] (inner_out_sit_sot-0)
        ├─ TensorConstant{0} [id J]
        ├─ Subtensor{i, j, k} [id K]
        │  ├─ *2-<TensorType(float64, (20000, 2, 2))> [id L] -> [id H] (inner_in_non_seqs-0)
        │  ├─ ScalarFromTensor [id M]
        │  │  └─ *0-<TensorType(int64, ())> [id N] -> [id C] (inner_in_seqs-0)
        │  ├─ ScalarFromTensor [id O]
        │  │  └─ *1-<TensorType(int64, ())> [id P] -> [id D] (inner_in_sit_sot-0)
        │  └─ ScalarConstant{0} [id Q]
        └─ TensorConstant{1} [id R]

    Composite{switch(lt(i0, i1), i2, i0)} [id I]
     ← Switch [id S] 'o0'
        ├─ LT [id T]
        │  ├─ i0 [id U]
        │  └─ i1 [id V]
        ├─ i2 [id W]
        └─ i0 [id U]
    """

    output_str = debugprint(out, file="str", print_op_info=True)
    lines = output_str.split("\n")

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip()


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
