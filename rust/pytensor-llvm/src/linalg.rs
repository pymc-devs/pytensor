//! Dense linear algebra kernels on top of faer (sequential: `Par::Seq`): Dot, Cholesky,
//! SolveTriangular.
//!
//! faer's work space is scratch memory of the context, so steady-state calls don't
//! allocate.

use crate::abi::OutputMode;
use crate::kernel::{scratch, Args, Array, ArrayMut, Element, Kernel};
use crate::runtime::PtCtx;
use faer::dyn_stack::MemStack;
use faer::linalg::cholesky::llt::factor::{cholesky_in_place, cholesky_in_place_scratch};
use faer::linalg::triangular_solve as tri;
use faer::prelude::ReborrowMut;
use faer::Par;
use std::marker::PhantomData;

/// float32 and float64.
pub trait Float: Element + faer::traits::RealField + Default + std::ops::Add<Output = Self> + std::ops::Mul<Output = Self> {
    const NAN: Self;
    const ZERO: Self;

    /// `self + a * b`, fused when the caller was compiled with FMA (see
    /// `gemv_short_rows`).
    fn madd<const FMA: bool>(self, a: Self, b: Self) -> Self;
}

macro_rules! float {
    ($($t:ty),*) => {$(
        impl Float for $t {
            const NAN: Self = <$t>::NAN;
            const ZERO: Self = 0.0;

            #[inline(always)]
            fn madd<const FMA: bool>(self, a: Self, b: Self) -> Self {
                if FMA { a.mul_add(b, self) } else { self + a * b }
            }
        }
    )*};
}
float!(f64, f32);

// ------------------------------------------------------------------ Dot

/// numpy's dot of vectors and matrices.
pub struct Dot<T> {
    pub mode: OutputMode,
    pub element: PhantomData<T>,
}

// SAFETY: two inputs and one output, arrays of `T` (`interp::py::Dot` picks `T` from
// the node's dtype).
unsafe impl<T: Float> Kernel for Dot<T> {
    fn arity(&self) -> (usize, usize) {
        (2, 1)
    }

    fn run(&self, ctx: &mut PtCtx, mut args: Args) -> Result<(), String> {
        let (a, b) = (args.inputs.array::<T>(0), args.inputs.array::<T>(1));
        let (sa, sb) = (a.shape(), b.shape());
        if !matches!(sa.len(), 1 | 2) || !matches!(sb.len(), 1 | 2) {
            return Err(format!("Dot: of arrays with {} and {} axes", sa.len(), sb.len()));
        }
        if sa.last() != sb.first() {
            return Err(format!("Dot: shapes {sa:?} and {sb:?} are not aligned"));
        }
        let mut shape = Vec::new();
        if sa.len() == 2 {
            shape.push(sa[0]);
        }
        if sb.len() == 2 {
            shape.push(sb[1]);
        }
        let mut out = args.outputs.array::<T>(ctx, 0, self.mode, &shape)?;
        // a vector on the left is a row, one on the right a column; the result of
        // vector . matrix is a row
        matmul(ctx, &a, &b, &mut out, sa.len() == 1);
        Ok(())
    }
}

/// Rows up to this length go through `gemv_short_rows`; longer ones through faer.
const SHORT_ROW: usize = 64;

/// `out = a b`, with `a` a matrix or a row, `b` a matrix or a column, `out` of the
/// product's shape (a vector result is a row if `out_as_row`).
fn matmul<T: Float>(ctx: &mut PtCtx, a: &Array<T>, b: &Array<T>, out: &mut ArrayMut<T>, out_as_row: bool) {
    let (lhs, rhs, mut dst) = (a.matrix(true), b.matrix(false), out.matrix_mut(out_as_row));
    let (m, n) = (lhs.nrows(), rhs.ncols());
    // matrix times vector: (the matrix, the vector, the result)
    let gemv = if n == 1 {
        Some((lhs, rhs.col(0), dst.rb_mut().col_mut(0)))
    } else if m == 1 {
        Some((rhs.transpose(), lhs.row(0).transpose(), dst.rb_mut().row_mut(0).transpose_mut()))
    } else {
        None
    };
    if let Some((matrix, x, y)) = gemv {
        let (rows, k) = (matrix.nrows(), matrix.ncols());
        // contiguous short rows (X @ beta with X row-major), where general kernels
        // do poorly
        if matrix.col_stride() == 1 && k <= SHORT_ROW && rows >= 4 {
            let x = if x.row_stride() == 1 {
                x.as_ptr()
            } else {
                let adjacent = scratch::<T>(ctx, k);
                for (j, value) in adjacent.iter_mut().enumerate() {
                    value.write(x[j]);
                }
                adjacent.as_ptr() as *const T
            };
            // SAFETY: `matrix` has `rows` rows of `k` adjacent elements, `row_stride`
            // apart; `x` points to `k` adjacent initialized elements (the vector's
            // own, or the copy just made); `y` has `rows` elements `row_stride`
            // apart, ours to write.
            unsafe { simd::gemv_short_rows(rows, k, matrix.as_ptr(), matrix.row_stride(), x, y.as_ptr_mut(), y.row_stride()) };
            return;
        }
    }
    faer::linalg::matmul::matmul(dst, faer::Accum::Replace, lhs, rhs, T::one_impl(), Par::Seq);
}

/// The hand-written loops, on raw pointers.
mod simd {
    // raw-pointer work is the whole body of these functions, so unsafe operations
    // aren't wrapped one by one; the one entry point documents what they need
    #![allow(unsafe_op_in_unsafe_fn)]

    use super::Float;

    /// y = A x for A with contiguous, short rows (k small), where general gemv kernels do
    /// poorly: 4 rows at a time, 4 independent lanes along each row (so LLVM vectorizes
    /// without reassociating), x contiguous.
    #[inline(always)]
    unsafe fn gemv_short_rows_impl<T: Float, const FMA: bool>(m: usize, k: usize, a: *const T, rsa: isize, x: *const T, y: *mut T, incy: isize) {
        const L: usize = 4;
        let kl = k - k % L;
        let mut i = 0;
        while i + 4 <= m {
            let rows = [0, 1, 2, 3].map(|r| a.offset((i + r) as isize * rsa));
            let mut acc = [[T::default(); L]; 4];
            let mut j = 0;
            while j < kl {
                for r in 0..4 {
                    for l in 0..L {
                        acc[r][l] = acc[r][l].madd::<FMA>(*rows[r].add(j + l), *x.add(j + l));
                    }
                }
                j += L;
            }
            for r in 0..4 {
                let mut s = (acc[r][0] + acc[r][1]) + (acc[r][2] + acc[r][3]);
                for j in kl..k {
                    s = s.madd::<FMA>(*rows[r].add(j), *x.add(j));
                }
                *y.offset((i + r) as isize * incy) = s;
            }
            i += 4;
        }
        while i < m {
            let row = a.offset(i as isize * rsa);
            let mut s = T::default();
            for j in 0..k {
                s = s.madd::<FMA>(*row.add(j), *x.add(j));
            }
            *y.offset(i as isize * incy) = s;
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn gemv_short_rows_avx2<T: Float>(m: usize, k: usize, a: *const T, rsa: isize, x: *const T, y: *mut T, incy: isize) {
        gemv_short_rows_impl::<T, true>(m, k, a, rsa, x, y, incy)
    }

    /// `y = A x` for `m` rows of `k` adjacent elements, `rsa` elements apart, `x` of `k`
    /// adjacent elements, and `y` of `m` elements `incy` apart. (The crate is built for
    /// baseline x86-64; this picks the AVX2+FMA version when the machine has it.)
    ///
    /// # Safety
    /// The pointers are valid for those elements, and `y` overlaps neither `A` nor `x`.
    pub unsafe fn gemv_short_rows<T: Float>(m: usize, k: usize, a: *const T, rsa: isize, x: *const T, y: *mut T, incy: isize) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                return gemv_short_rows_avx2(m, k, a, rsa, x, y, incy);
            }
        }
        gemv_short_rows_impl::<T, false>(m, k, a, rsa, x, y, incy)
    }
}

// ------------------------------------------------------------------ Cholesky

/// The Cholesky factor, with LAPACK potrf semantics as in PyTensor's Cholesky: reads
/// the lower (upper) triangle, zeroes the other one, and fills the result with NaN if
/// the matrix is not positive definite.
pub struct Cholesky<T> {
    pub mode: OutputMode,
    pub lower: bool,
    pub element: PhantomData<T>,
}

// SAFETY: one input and one output, arrays of `T` (`interp::py::Cholesky` picks `T`
// from the node's dtype).
unsafe impl<T: Float> Kernel for Cholesky<T> {
    fn arity(&self) -> (usize, usize) {
        (1, 1)
    }

    fn run(&self, ctx: &mut PtCtx, mut args: Args) -> Result<(), String> {
        let a = args.inputs.array::<T>(0);
        let &[n, columns] = a.shape() else {
            return Err(format!("Cholesky: input must be a matrix, got {} axes", a.ndim()));
        };
        if n != columns {
            return Err(format!("Cholesky: input must be square, got {n}x{columns}"));
        }
        let n = n as usize;
        let mut out = args.outputs.array::<T>(ctx, 0, self.mode, a.shape())?;
        let (mut source, mut factor) = (a.matrix(false), out.matrix_mut(false));
        // upper: factor the transpose (A^T = A, its lower triangle is A's upper one)
        if !self.lower {
            (source, factor) = (source.transpose(), factor.transpose_mut());
        }
        for j in 0..n {
            for i in j..n {
                factor[(i, j)] = source[(i, j)];
            }
        }
        let work = cholesky_in_place_scratch::<T>(n, Par::Seq, Default::default());
        let stack = MemStack::new(scratch::<u8>(ctx, work.unaligned_bytes_required()));
        let ok = cholesky_in_place(factor.rb_mut(), Default::default(), Par::Seq, stack, Default::default()).is_ok();
        // faer leaves the other triangle unspecified; LAPACK-clean output zeroes it
        for j in 0..n {
            for i in 0..n {
                if !ok {
                    factor[(i, j)] = T::NAN;
                } else if i < j {
                    factor[(i, j)] = T::ZERO;
                }
            }
        }
        Ok(())
    }
}

// ------------------------------------------------------------------ SolveTriangular

/// Solves A x = b for triangular A and a vector or matrix b, with LAPACK trtrs
/// semantics: the result is NaN if A has a zero on its diagonal.
pub struct SolveTriangular<T> {
    pub mode: OutputMode,
    pub lower: bool,
    pub unit_diagonal: bool,
    pub element: PhantomData<T>,
}

// SAFETY: two inputs and one output, arrays of `T` (`interp::py::SolveTriangular`
// picks `T` from the node's dtype).
unsafe impl<T: Float> Kernel for SolveTriangular<T> {
    fn arity(&self) -> (usize, usize) {
        (2, 1)
    }

    fn run(&self, ctx: &mut PtCtx, mut args: Args) -> Result<(), String> {
        let (a, b) = (args.inputs.array::<T>(0), args.inputs.array::<T>(1));
        let &[n, columns] = a.shape() else {
            return Err(format!("SolveTriangular: A must be a matrix, got {} axes", a.ndim()));
        };
        if n != columns {
            return Err(format!("SolveTriangular: A must be square, got {n}x{columns}"));
        }
        if !matches!(b.ndim(), 1 | 2) || b.shape()[0] != n {
            return Err(format!("SolveTriangular: shapes of A ({n}x{n}) and b ({:?}) are incompatible", b.shape()));
        }
        let mut out = args.outputs.array::<T>(ctx, 0, self.mode, b.shape())?;
        let matrix = a.matrix(false);
        if !self.unit_diagonal && (0..n as usize).any(|i| matrix[(i, i)] == T::ZERO) {
            out.matrix_mut(false).fill(T::NAN);
            return Ok(());
        }
        out.copy_from(&b)?;
        let x = out.matrix_mut(false);
        match (self.lower, self.unit_diagonal) {
            (true, false) => tri::solve_lower_triangular_in_place(matrix, x, Par::Seq),
            (true, true) => tri::solve_unit_lower_triangular_in_place(matrix, x, Par::Seq),
            (false, false) => tri::solve_upper_triangular_in_place(matrix, x, Par::Seq),
            (false, true) => tri::solve_unit_upper_triangular_in_place(matrix, x, Par::Seq),
        }
        Ok(())
    }
}
