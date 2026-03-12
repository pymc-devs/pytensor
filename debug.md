# no_more_expand_dims - Progress Summary

## Branch: `no_more_expand_dims` (based on `v3`)

The PR removes explicit `DimShuffle` (expand_dims) nodes that `Elemwise.make_node` previously inserted to left-pad shorter inputs. Instead, Elemwise now handles mixed-ndim inputs implicitly via numpy-style left-padding with 1s.

## Committed (in PR commit `1e641037d`)
- Core change in `pytensor/tensor/elemwise.py` - `Elemwise.make_node` no longer inserts DimShuffle
- Various rewrites updated to handle mixed-ndim inputs
- Tests updated

## Uncommitted fixes (working tree)

### 1. `pytensor/scan/rewriting.py` - scan_push_out_seq fix
**Problem**: When `scan_push_out_seq` pushes Elemwise ops outside scan, sequence-derived outer inputs (shape `(n_steps,)`) could collide with higher-ndim non-sequence outer inputs during broadcasting. E.g., a scalar inner seq + 1-d inner non-seq → outer shapes `(n_steps,)` and `(d,)` broadcast incorrectly.
**Fix**: Added trailing ExpandDims to sequence-derived outer inputs before the Elemwise make_node call, so the time dimension stays separate from data dimensions.
**Tests fixed**: ~16 scan tests including `test_grad_multiple_outs`, `test_multiple_inputs_multiple_outputs`

### 2. `pytensor/tensor/elemwise.py` - infer_shape fix
**Problem**: `Elemwise.infer_shape` passed shapes of different lengths to `broadcast_shape`, which requires equal-length tuples.
**Fix**: Left-pad shorter input shapes with 1s to match max ndim before calling `broadcast_shape`.
**Tests fixed**: SVD gradient tests (3 tests)

### 3. `tests/scan/test_printing.py` - expected output updates
**Problem**: Graph structure changed because `Second(A, 1.0)` no longer wraps `1.0` in ExpandDims.
**Fix**: Updated expected output strings for 4 tests: `test_debugprint_sitsot`, `test_debugprint_sitsot_no_extra_info`, `test_debugprint_nested_scans`, `test_debugprint_mitmot`.

### 4. `pytensor/tensor/slinalg.py` - ifelse import fix
**Problem**: `from pytensor import ifelse` imports the *module* `pytensor.ifelse`, not the function `pytensor.ifelse.ifelse`.
**Fix**: Changed to `from pytensor.ifelse import ifelse`.

### 5. `pytensor/tensor/rewriting/subtensor_lift.py` - subtensor_lift fix
**Problem**: `local_subtensor_of_batch_dims` didn't account for mixed-ndim Elemwise inputs.
**Fix**: Rewrote to handle the new mixed-ndim case.
**Tests fixed**: `test_grad_wrt_shared` and similar IndexError tests

### 6. `tests/test_raise_op.py` - assertion update
**Problem**: `test_vectorize` checked for DimShuffle node that no longer exists.
**Fix**: Updated assertion.

## Remaining failures (not yet fixed)

### QR gradient tests (2 tests)
- `test_qr_grad[real-shape=(3, 6), gradient_test_case=Q, mode=economic]`
- `test_qr_grad[real-shape=(3, 6), gradient_test_case=Q, mode=full]` (and possibly other (3,6) cases)
- The ifelse import fix was applied but `verify_grad` still reports GradientError
- **Key finding**: Direct gradient computation is correct (error ~2e-9), but `verify_grad` fails
- Need to investigate what differs in `verify_grad`'s specific approach (numeric_grad class, shared variable t_r updates)

### test_Subtensor_lift_restrictions (1 test)
- Not yet investigated

### Sparse Usmm tests (~8 tests)
- Node count assertions (`assert 2 == 3`) — not yet investigated
- Tests may be skipped locally if sparse dependencies aren't available

## Strategy / helpers to consider (user-requested)
- The user asked to think about helpers/strategies to make mixed-ndim errors less likely
- Not yet addressed
