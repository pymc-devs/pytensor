//! Dense linear algebra kernels on top of faer (sequential: `Par::Seq`).
//!
//! faer's work space comes from the context's scratch arena (`PtCtx::scratch`), so
//! steady-state calls don't allocate.

use crate::abi::{PtArray, PtCtx};
use faer::dyn_stack::MemStack;
use faer::linalg::cholesky::llt::factor::{cholesky_in_place, cholesky_in_place_scratch, LltRegularization};
use faer::linalg::triangular_solve as tri;
use faer::{MatMut, MatRef, Par};
use std::ffi::c_void;
use std::mem::MaybeUninit;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CholeskyParams {
    pub lower: i64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SolveTriangularParams {
    pub lower: i64,
    pub unit_diagonal: i64,
}

unsafe fn arg<'a>(args: *mut PtArray, i: usize) -> &'a PtArray {
    &*args.add(i)
}

/// (rows, cols, row stride, col stride) in elements; a vector is a column.
unsafe fn parts(x: &PtArray, isz: i64) -> (usize, usize, isize, isize) {
    let (sh, st) = (x.shape(), x.strides());
    if x.ndim == 2 {
        (sh[0] as usize, sh[1] as usize, (st[0] / isz) as isize, (st[1] / isz) as isize)
    } else {
        (sh[0] as usize, 1, (st[0] / isz) as isize, 0)
    }
}

unsafe fn stack<'a>(ctx: *mut PtCtx, bytes: usize) -> &'a mut MemStack {
    let p = PtCtx::scratch(ctx, bytes) as *mut MaybeUninit<u8>;
    MemStack::new(std::slice::from_raw_parts_mut(p, bytes))
}

macro_rules! linalg_kernels {
    ($t:ty, $chol:ident, $solve:ident) => {
        /// args = [a, out(provided)]. LAPACK potrf semantics as in PyTensor's Cholesky:
        /// reads the lower (upper) triangle, zeroes the other one, and fills the result
        /// with NaN if the matrix is not positive definite.
        pub unsafe extern "C" fn $chol(ctx: *mut PtCtx, args: *mut PtArray, params: *const c_void) -> i32 {
            let lower = (*(params as *const CholeskyParams)).lower != 0;
            let (a, out) = (arg(args, 0), arg(args, 1));
            let isz = std::mem::size_of::<$t>() as i64;
            let (n, n2, mut rsa, mut csa) = parts(a, isz);
            if n != n2 {
                return PtCtx::fail(ctx, 1, &format!("Cholesky: input must be square, got {n}x{n2}"));
            }
            let (_, _, mut rso, mut cso) = parts(out, isz);
            // upper: factor the transpose (A^T = A, its lower triangle is A's upper one)
            if !lower {
                std::mem::swap(&mut rsa, &mut csa);
                std::mem::swap(&mut rso, &mut cso);
            }
            let (src, dst) = (a.data as *const $t, out.data as *mut $t);
            let same = src as *const u8 == dst as *const u8 && rsa == rso && csa == cso;
            if !same {
                for j in 0..n as isize {
                    for i in j..n as isize {
                        *dst.offset(i * rso + j * cso) = *src.offset(i * rsa + j * csa);
                    }
                }
            }
            let m = MatMut::<$t>::from_raw_parts_mut(dst, n, n, rso, cso);
            let req = cholesky_in_place_scratch::<$t>(n, Par::Seq, Default::default());
            let st = stack(ctx, req.unaligned_bytes_required());
            let reg = LltRegularization { dynamic_regularization_delta: 0.0, dynamic_regularization_epsilon: 0.0 };
            let ok = cholesky_in_place(m, reg, Par::Seq, st, Default::default()).is_ok();
            // faer leaves the other triangle unspecified; LAPACK-clean output zeroes it
            for j in 0..n as isize {
                for i in 0..n as isize {
                    let d = dst.offset(i * rso + j * cso);
                    if !ok {
                        *d = <$t>::NAN;
                    } else if i < j {
                        *d = 0.0;
                    }
                }
            }
            0
        }

        /// args = [A, b, out(provided)], b 1-d or 2-d. Solves A x = b for triangular A
        /// (LAPACK trtrs semantics: NaN result if A has a zero on the diagonal).
        pub unsafe extern "C" fn $solve(ctx: *mut PtCtx, args: *mut PtArray, params: *const c_void) -> i32 {
            let prm = &*(params as *const SolveTriangularParams);
            let (lower, unit) = (prm.lower != 0, prm.unit_diagonal != 0);
            let (a, b, out) = (arg(args, 0), arg(args, 1), arg(args, 2));
            let isz = std::mem::size_of::<$t>() as i64;
            let (n, n2, rsa, csa) = parts(a, isz);
            let (nb, k, rsb, csb) = parts(b, isz);
            if n != n2 {
                return PtCtx::fail(ctx, 1, &format!("SolveTriangular: A must be square, got {n}x{n2}"));
            }
            if nb != n {
                return PtCtx::fail(ctx, 1, &format!("SolveTriangular: shapes of A ({n}x{n}) and b (first dim {nb}) are incompatible"));
            }
            let (_, _, rso, cso) = parts(out, isz);
            let (src, dst) = (b.data as *const $t, out.data as *mut $t);
            if !(src as *const u8 == dst as *const u8 && rsb == rso && csb == cso) {
                for j in 0..k as isize {
                    for i in 0..n as isize {
                        *dst.offset(i * rso + j * cso) = *src.offset(i * rsb + j * csb);
                    }
                }
            }
            let am = MatRef::<$t>::from_raw_parts(a.data as *const $t, n, n, rsa, csa);
            let singular = !unit && (0..n).any(|i| am[(i, i)] == 0.0);
            if singular {
                for j in 0..k as isize {
                    for i in 0..n as isize {
                        *dst.offset(i * rso + j * cso) = <$t>::NAN;
                    }
                }
                return 0;
            }
            let x = MatMut::<$t>::from_raw_parts_mut(dst, n, k, rso, cso);
            match (lower, unit) {
                (true, false) => tri::solve_lower_triangular_in_place(am, x, Par::Seq),
                (true, true) => tri::solve_unit_lower_triangular_in_place(am, x, Par::Seq),
                (false, false) => tri::solve_upper_triangular_in_place(am, x, Par::Seq),
                (false, true) => tri::solve_unit_upper_triangular_in_place(am, x, Par::Seq),
            }
            0
        }
    };
}

linalg_kernels!(f64, ptl_cholesky_f64, ptl_solve_triangular_f64);
linalg_kernels!(f32, ptl_cholesky_f32, ptl_solve_triangular_f32);
