//! Dense linear algebra kernels on top of faer (sequential, or parallel on the runtime's pool: `faer_kernels!`): Dot, Cholesky,
//! SolveTriangular, Solve, CholeskySolve, MatrixInverse, Det, SLogDet, LUFactor, LU,
//! PivotToPermutations, Eigh/Eigvalsh, SVD, QR and MatrixPinv.
//!
//! faer's work space is scratch memory of the context, so steady-state calls don't
//! allocate.

use crate::abi::OutputMode;
use crate::kernel::{Args, Array, ArrayMut, Element, Kernel, scratch};
use crate::layout::Layout;
use crate::runtime::{PtCtx, runtime};
use faer::Par;
use faer::dyn_stack::MemStack;
use faer::dyn_stack::StackReq;
use faer::linalg::cholesky::llt::factor::{cholesky_in_place, cholesky_in_place_scratch};
use faer::linalg::cholesky::llt::solve as llt_solve;
use faer::linalg::lu::partial_pivoting::{factor as lu, inverse as lu_inverse, solve as lu_solve};
use faer::linalg::qr::no_pivoting::factor as qr;
use faer::linalg::triangular_solve as tri;
use faer::linalg::{evd, householder, svd, temp_mat_scratch, temp_mat_zeroed};
use faer::mat::AsMatMut;
use faer::perm::PermRef;
use faer::prelude::{Reborrow, ReborrowMut};
use faer::{Conj, MatMut};
use std::marker::PhantomData;

/// float32 and float64.
pub trait Float:
    Element
    + faer::traits::RealField
    + Default
    + std::ops::Add<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + std::ops::AddAssign
    + std::ops::MulAssign
    + PartialOrd
{
    const NAN: Self;
    const ZERO: Self;
    const ONE: Self;
    const NEG_INFINITY: Self;

    fn abs(self) -> Self;
    fn ln(self) -> Self;
    fn from_f64(x: f64) -> Self;

    /// `self + a * b`, fused when the caller was compiled with FMA (see
    /// `gemv_short_rows`).
    fn madd<const FMA: bool>(self, a: Self, b: Self) -> Self;
}

macro_rules! float {
    ($($t:ty),*) => {$(
        impl Float for $t {
            const NAN: Self = <$t>::NAN;
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const NEG_INFINITY: Self = <$t>::NEG_INFINITY;

            #[inline(always)]
            fn abs(self) -> Self {
                <$t>::abs(self)
            }

            #[inline(always)]
            fn ln(self) -> Self {
                <$t>::ln(self)
            }

            #[inline(always)]
            fn from_f64(x: f64) -> Self {
                x as $t
            }

            #[inline(always)]
            fn madd<const FMA: bool>(self, a: Self, b: Self) -> Self {
                if FMA { a.mul_add(b, self) } else { self + a * b }
            }
        }
    )*};
}
float!(f64, f32);

// ------------------------------------------------------------------ parallelism

/// Kernels with fewer input elements than this run sequentially: the hop to the
/// thread pool would cost more than it saves.
const PARALLEL_MIN: i64 = 64 * 64;

/// A kernel computing with faer, with the parallelism `par`.
trait FaerKernel: Send + Sync + 'static {
    fn arity(&self) -> (usize, usize);
    fn run(&self, ctx: &mut PtCtx, args: Args, par: Par) -> Result<(), String>;
}

/// The `Kernel` of faer kernels. With `parallel` (the linker's `linalg_parallel`) they
/// run on the runtime's thread pool with faer parallelism, if the session allows
/// parallelism and an input has at least `PARALLEL_MIN` elements.
macro_rules! faer_kernels {
    ($($k:ident),*) => {$(
        impl<T: Float> Kernel for $k<T> {
            fn arity(&self) -> (usize, usize) {
                FaerKernel::arity(self)
            }

            fn output_layouts(&self) -> &[Layout] {
                &self.layouts
            }

            fn run(&self, ctx: &mut PtCtx, args: Args) -> Result<(), String> {
                run_faer(self, self.parallel, ctx, args)
            }
        }
    )*};
}

/// The layouts of the outputs of the faer kernels (of `out_ndims` axes each), for
/// inputs with `in_layouts`: column-major, faer's own order. (They are written through
/// faer matrices, which take any strides, but faer computes in column-major order.)
pub fn faer_layouts(out_ndims: &[usize], in_layouts: &[Option<Layout>]) -> Vec<Layout> {
    let _ = in_layouts;
    out_ndims.iter().map(|&ndim| Layout::column_major(ndim)).collect()
}

faer_kernels!(
    Dot,
    Cholesky,
    SolveTriangular,
    Solve,
    CholeskySolve,
    MatrixInverse,
    Det,
    SLogDet,
    LUFactor,
    LU,
    Eigh,
    Svd,
    Qr,
    MatrixPinv
);

fn run_faer<K: FaerKernel>(
    kernel: &K,
    parallel: bool,
    ctx: &mut PtCtx,
    args: Args,
) -> Result<(), String> {
    let large = (0..args.inputs.len()).any(|i| {
        args.inputs
            .get(i)
            .is_ok_and(|a| a.shape().iter().product::<i64>() >= PARALLEL_MIN)
    });
    let rt = runtime();
    if !(parallel && large && ctx.parallel && rt.threads() > 1) {
        return kernel.run(ctx, args, Par::Seq);
    }
    let par = Par::rayon(rt.threads());
    let call = CallerWaits((ctx, args));
    rt.pool().install(move || {
        let (ctx, args) = call.into_inner();
        kernel.run(ctx, args, par)
    })
}

/// The context and arguments of a call, run on the pool while the caller waits.
struct CallerWaits<T>(T);

// SAFETY: the calling thread blocks in `install` until the pool is done with them,
// and nothing else uses them meanwhile (the kernel contract).
unsafe impl<T> Send for CallerWaits<T> {}

impl<T> CallerWaits<T> {
    fn into_inner(self) -> T {
        self.0
    }
}

// ------------------------------------------------------------------ Dot

/// numpy's dot of vectors and matrices.
pub struct Dot<T> {
    pub parallel: bool,
    pub mode: OutputMode,
    pub layouts: Vec<Layout>,
    pub element: PhantomData<T>,
}

impl<T: Float> FaerKernel for Dot<T> {
    fn arity(&self) -> (usize, usize) {
        (2, 1)
    }

    fn run(&self, ctx: &mut PtCtx, mut args: Args, par: Par) -> Result<(), String> {
        let (a, b) = (args.inputs.array::<T>(0)?, args.inputs.array::<T>(1)?);
        let (sa, sb) = (a.shape(), b.shape());
        if !matches!(sa.len(), 1 | 2) || !matches!(sb.len(), 1 | 2) {
            return Err(format!(
                "Dot: of arrays with {} and {} axes",
                sa.len(),
                sb.len()
            ));
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
        matmul(ctx, &a, &b, &mut out, sa.len() == 1, par);
        Ok(())
    }
}

/// Rows up to this length go through `gemv_short_rows`; longer ones through faer.
const SHORT_ROW: usize = 64;

/// `out = a b`, with `a` a matrix or a row, `b` a matrix or a column, `out` of the
/// product's shape (a vector result is a row if `out_as_row`).
fn matmul<T: Float>(
    ctx: &mut PtCtx,
    a: &Array<T>,
    b: &Array<T>,
    out: &mut ArrayMut<T>,
    out_as_row: bool,
    par: Par,
) {
    let (lhs, rhs, mut dst) = (a.matrix(true), b.matrix(false), out.matrix_mut(out_as_row));
    let (m, n) = (lhs.nrows(), rhs.ncols());
    // matrix times vector: (the matrix, the vector, the result)
    let gemv = if n == 1 {
        Some((lhs, rhs.col(0), dst.rb_mut().col_mut(0)))
    } else if m == 1 {
        Some((
            rhs.transpose(),
            lhs.row(0).transpose(),
            dst.rb_mut().row_mut(0).transpose_mut(),
        ))
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
            unsafe {
                simd::gemv_short_rows(
                    rows,
                    k,
                    matrix.as_ptr(),
                    matrix.row_stride(),
                    x,
                    y.as_ptr_mut(),
                    y.row_stride(),
                )
            };
            return;
        }
    }
    faer::linalg::matmul::matmul(dst, faer::Accum::Replace, lhs, rhs, T::one_impl(), par);
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
    unsafe fn gemv_short_rows_impl<T: Float, const FMA: bool>(
        m: usize,
        k: usize,
        a: *const T,
        rsa: isize,
        x: *const T,
        y: *mut T,
        incy: isize,
    ) {
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
    unsafe fn gemv_short_rows_avx2<T: Float>(
        m: usize,
        k: usize,
        a: *const T,
        rsa: isize,
        x: *const T,
        y: *mut T,
        incy: isize,
    ) {
        gemv_short_rows_impl::<T, true>(m, k, a, rsa, x, y, incy)
    }

    /// `y = A x` for `m` rows of `k` adjacent elements, `rsa` elements apart, `x` of `k`
    /// adjacent elements, and `y` of `m` elements `incy` apart. (The crate is built for
    /// baseline x86-64; this picks the AVX2+FMA version when the machine has it.)
    ///
    /// # Safety
    /// The pointers are valid for those elements, and `y` overlaps neither `A` nor `x`.
    pub unsafe fn gemv_short_rows<T: Float>(
        m: usize,
        k: usize,
        a: *const T,
        rsa: isize,
        x: *const T,
        y: *mut T,
        incy: isize,
    ) {
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
    pub parallel: bool,
    pub mode: OutputMode,
    pub layouts: Vec<Layout>,
    pub lower: bool,
    pub element: PhantomData<T>,
}

impl<T: Float> FaerKernel for Cholesky<T> {
    fn arity(&self) -> (usize, usize) {
        (1, 1)
    }

    fn run(&self, ctx: &mut PtCtx, mut args: Args, par: Par) -> Result<(), String> {
        let a = args.inputs.array::<T>(0)?;
        let &[n, columns] = a.shape() else {
            return Err(format!(
                "Cholesky: input must be a matrix, got {} axes",
                a.ndim()
            ));
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
        // (all of it: faer's small-matrix code reads whole rows, the output may be fresh)
        for j in 0..n {
            for i in 0..n {
                factor[(i, j)] = if i >= j { source[(i, j)] } else { T::ZERO };
            }
        }
        let work = cholesky_in_place_scratch::<T>(n, par, Default::default());
        let stack = MemStack::new(scratch::<u8>(ctx, work.unaligned_bytes_required()));
        let ok = cholesky_in_place(
            factor.rb_mut(),
            Default::default(),
            par,
            stack,
            Default::default(),
        )
        .is_ok();
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
    pub parallel: bool,
    pub mode: OutputMode,
    pub layouts: Vec<Layout>,
    pub lower: bool,
    pub unit_diagonal: bool,
    pub element: PhantomData<T>,
}

impl<T: Float> FaerKernel for SolveTriangular<T> {
    fn arity(&self) -> (usize, usize) {
        (2, 1)
    }

    fn run(&self, ctx: &mut PtCtx, mut args: Args, par: Par) -> Result<(), String> {
        let (a, b) = (args.inputs.array::<T>(0)?, args.inputs.array::<T>(1)?);
        let &[n, columns] = a.shape() else {
            return Err(format!(
                "SolveTriangular: A must be a matrix, got {} axes",
                a.ndim()
            ));
        };
        if n != columns {
            return Err(format!(
                "SolveTriangular: A must be square, got {n}x{columns}"
            ));
        }
        if !matches!(b.ndim(), 1 | 2) || b.shape()[0] != n {
            return Err(format!(
                "SolveTriangular: shapes of A ({n}x{n}) and b ({:?}) are incompatible",
                b.shape()
            ));
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
            (true, false) => tri::solve_lower_triangular_in_place(matrix, x, par),
            (true, true) => tri::solve_unit_lower_triangular_in_place(matrix, x, par),
            (false, false) => tri::solve_upper_triangular_in_place(matrix, x, par),
            (false, true) => tri::solve_unit_upper_triangular_in_place(matrix, x, par),
        }
        Ok(())
    }
}

// ------------------------------------------------------------------ shared

/// Scratch memory for `req`, as a stack of temporaries.
fn stack_for(ctx: &mut PtCtx, req: StackReq) -> &mut MemStack {
    MemStack::new(scratch::<u8>(ctx, req.unaligned_bytes_required()))
}

/// The size of a square matrix.
fn square<T: Element>(a: &Array<T>) -> Result<usize, String> {
    match *a.shape() {
        [n, m] if n == m => Ok(n as usize),
        ref shape => Err(format!("expected a square matrix, got shape {shape:?}")),
    }
}

/// The shape of a matrix.
fn matrix_shape<T: Element>(a: &Array<T>) -> Result<(usize, usize), String> {
    match *a.shape() {
        [m, n] => Ok((m as usize, n as usize)),
        ref shape => Err(format!("expected a matrix, got shape {shape:?}")),
    }
}

/// Checks that `b` is a right-hand side (vector or matrix) for an `n` x `n` system.
fn check_rhs<T: Element>(n: usize, b: &Array<T>) -> Result<(), String> {
    if !matches!(b.ndim(), 1 | 2) || b.shape()[0] as usize != n {
        return Err(format!(
            "shapes of A ({n}x{n}) and b ({:?}) are incompatible",
            b.shape()
        ));
    }
    Ok(())
}

/// `n` indices from `stack`, each set to its position.
fn indices(stack: &mut MemStack, n: usize) -> (&mut [usize], &mut MemStack) {
    let (memory, stack) = stack.make_uninit::<usize>(n);
    for (i, x) in memory.iter_mut().enumerate() {
        x.write(i);
    }
    // SAFETY: all `n` elements were just initialized.
    let values = unsafe { &mut *(memory as *mut [std::mem::MaybeUninit<usize>] as *mut [usize]) };
    (values, stack)
}

/// Scratch for `lu_in_place` of an `n` x `n` matrix, its permutation included.
fn lu_scratch<T: Float>(n: usize, par: Par) -> StackReq {
    StackReq::all_of(&[
        StackReq::new::<usize>(n).array(2),
        lu::lu_in_place_scratch::<usize, T>(n, n, par, Default::default()),
    ])
}

/// LU factorization with partial pivoting (getrf) of the square `a`, in place: the
/// row permutation is returned (row i of PA is row `perm[i]` of A), with the number of
/// row swaps, or None for the swaps if a pivot is exactly zero (A is singular).
fn lu_in_place<'s, T: Float>(
    mut a: MatMut<T>,
    stack: &'s mut MemStack,
    par: Par,
) -> (PermRef<'s, usize>, Option<usize>, &'s mut MemStack) {
    let n = a.nrows();
    let (perm, stack) = indices(stack, n);
    let (perm_inv, stack) = indices(stack, n);
    let (info, _) = lu::lu_in_place(a.rb_mut(), perm, perm_inv, par, stack, Default::default());
    // a zero pivot stays on U's diagonal: the later steps don't change its row
    let swaps = (!(0..n).any(|i| a[(i, i)] == T::ZERO)).then_some(info.transposition_count);
    (PermRef::new_checked(perm, perm_inv, n), swaps, stack)
}

// ------------------------------------------------------------------ Solve

/// What Solve assumes about A.
#[derive(Clone, Copy, Debug)]
pub enum Assume {
    /// general (gesv): LU with partial pivoting
    General,
    /// symmetric (sysv): only the lower (upper) triangle is read
    Symmetric { lower: bool },
    /// positive definite (posv): Cholesky of the lower (upper) triangle
    PositiveDefinite { lower: bool },
}

/// Solves A x = b for square A and a vector or matrix b, with scipy.linalg.solve
/// semantics as in PyTensor's Solve: NaN if A is singular (not positive definite).
pub struct Solve<T> {
    pub parallel: bool,
    pub mode: OutputMode,
    pub layouts: Vec<Layout>,
    pub assume: Assume,
    pub element: PhantomData<T>,
}

impl<T: Float> FaerKernel for Solve<T> {
    fn arity(&self) -> (usize, usize) {
        (2, 1)
    }

    fn run(&self, ctx: &mut PtCtx, mut args: Args, par: Par) -> Result<(), String> {
        let (a, b) = (args.inputs.array::<T>(0)?, args.inputs.array::<T>(1)?);
        let n = square(&a)?;
        check_rhs(n, &b)?;
        let mut out = args.outputs.array::<T>(ctx, 0, self.mode, b.shape())?;
        out.copy_from(&b)?;
        let mut x = out.matrix_mut(false);
        let source = a.matrix(false);
        let k = x.ncols();
        let solved = match self.assume {
            Assume::PositiveDefinite { lower } => {
                let req = StackReq::all_of(&[
                    temp_mat_scratch::<T>(n, n),
                    cholesky_in_place_scratch::<T>(n, par, Default::default()),
                ]);
                let (mut l, stack) = temp_mat_zeroed::<T, _, _>(n, n, stack_for(ctx, req));
                let mut l = l.as_mat_mut();
                l.copy_from_triangular_lower(if lower { source } else { source.transpose() });
                let ok = cholesky_in_place(
                    l.rb_mut(),
                    Default::default(),
                    par,
                    stack,
                    Default::default(),
                )
                .is_ok();
                if ok {
                    llt_solve::solve_in_place(l.rb(), x.rb_mut(), par, stack);
                }
                ok
            }
            Assume::General | Assume::Symmetric { .. } => {
                let req = StackReq::all_of(&[
                    temp_mat_scratch::<T>(n, n),
                    lu_scratch::<T>(n, par),
                    lu_solve::solve_in_place_scratch::<usize, T>(n, k, par),
                ]);
                let (mut factor, stack) = temp_mat_zeroed::<T, _, _>(n, n, stack_for(ctx, req));
                let mut factor = factor.as_mat_mut();
                match self.assume {
                    Assume::Symmetric { lower } => {
                        // the full matrix, from the triangle that is read
                        let half = if lower { source } else { source.transpose() };
                        for j in 0..n {
                            for i in j..n {
                                factor[(i, j)] = half[(i, j)];
                                factor[(j, i)] = half[(i, j)];
                            }
                        }
                    }
                    _ => factor.copy_from(source),
                }
                let (perm, swaps, stack) = lu_in_place(factor.rb_mut(), stack, par);
                if swaps.is_some() {
                    lu_solve::solve_in_place(
                        factor.rb(),
                        factor.rb(),
                        perm,
                        x.rb_mut(),
                        par,
                        stack,
                    );
                }
                swaps.is_some()
            }
        };
        if !solved {
            x.fill(T::NAN);
        }
        Ok(())
    }
}

// ------------------------------------------------------------------ CholeskySolve

/// Solves A x = b given the Cholesky factor of A (lower: A = C C^T, else A = C^T C),
/// with LAPACK potrs semantics: only that triangle of C is read.
pub struct CholeskySolve<T> {
    pub parallel: bool,
    pub mode: OutputMode,
    pub layouts: Vec<Layout>,
    pub lower: bool,
    pub element: PhantomData<T>,
}

impl<T: Float> FaerKernel for CholeskySolve<T> {
    fn arity(&self) -> (usize, usize) {
        (2, 1)
    }

    fn run(&self, ctx: &mut PtCtx, mut args: Args, par: Par) -> Result<(), String> {
        let (c, b) = (args.inputs.array::<T>(0)?, args.inputs.array::<T>(1)?);
        let n = square(&c)?;
        check_rhs(n, &b)?;
        let mut out = args.outputs.array::<T>(ctx, 0, self.mode, b.shape())?;
        out.copy_from(&b)?;
        let factor = if self.lower {
            c.matrix(false)
        } else {
            c.matrix(false).transpose()
        };
        llt_solve::solve_in_place(factor, out.matrix_mut(false), par, MemStack::new(&mut []));
        Ok(())
    }
}

// ------------------------------------------------------------------ MatrixInverse

/// The inverse of a square matrix (numpy.linalg.inv): fails if it is singular.
pub struct MatrixInverse<T> {
    pub parallel: bool,
    pub mode: OutputMode,
    pub layouts: Vec<Layout>,
    pub element: PhantomData<T>,
}

impl<T: Float> FaerKernel for MatrixInverse<T> {
    fn arity(&self) -> (usize, usize) {
        (1, 1)
    }

    fn run(&self, ctx: &mut PtCtx, mut args: Args, par: Par) -> Result<(), String> {
        let a = args.inputs.array::<T>(0)?;
        let n = square(&a)?;
        let mut out = args.outputs.array::<T>(ctx, 0, self.mode, a.shape())?;
        let req = StackReq::all_of(&[
            temp_mat_scratch::<T>(n, n),
            lu_scratch::<T>(n, par),
            lu_inverse::inverse_scratch::<usize, T>(n, par),
        ]);
        let (mut factor, stack) = temp_mat_zeroed::<T, _, _>(n, n, stack_for(ctx, req));
        let mut factor = factor.as_mat_mut();
        factor.copy_from(a.matrix(false));
        let (perm, swaps, stack) = lu_in_place(factor.rb_mut(), stack, par);
        if swaps.is_none() {
            return Err("singular matrix".into());
        }
        lu_inverse::inverse(
            out.matrix_mut(false),
            factor.rb(),
            factor.rb(),
            perm,
            par,
            stack,
        );
        Ok(())
    }
}

// ------------------------------------------------------------------ Det, SLogDet

/// (determinant, sign, log of the absolute determinant) of a square matrix, by LU
/// (numpy.linalg.det and slogdet): (0, 0, -inf) if it is singular.
fn determinant<T: Float>(ctx: &mut PtCtx, a: &Array<T>, par: Par) -> Result<(T, T, T), String> {
    let n = square(a)?;
    let req = StackReq::all_of(&[temp_mat_scratch::<T>(n, n), lu_scratch::<T>(n, par)]);
    let (mut factor, stack) = temp_mat_zeroed::<T, _, _>(n, n, stack_for(ctx, req));
    let mut factor = factor.as_mat_mut();
    factor.copy_from(a.matrix(false));
    let (_, swaps, _) = lu_in_place(factor.rb_mut(), stack, par);
    let Some(swaps) = swaps else {
        return Ok((T::ZERO, T::ZERO, T::NEG_INFINITY));
    };
    let one = if swaps % 2 == 0 { T::ONE } else { -T::ONE };
    let (mut det, mut sign, mut log) = (one, one, T::ZERO);
    for i in 0..n {
        let u = factor[(i, i)];
        det *= u;
        if u < T::ZERO {
            sign = -sign;
        }
        log += u.abs().ln();
    }
    Ok((det, sign, log))
}

/// The determinant of a square matrix (numpy.linalg.det).
pub struct Det<T> {
    pub parallel: bool,
    pub mode: OutputMode,
    pub layouts: Vec<Layout>,
    pub element: PhantomData<T>,
}

impl<T: Float> FaerKernel for Det<T> {
    fn arity(&self) -> (usize, usize) {
        (1, 1)
    }

    fn run(&self, ctx: &mut PtCtx, mut args: Args, par: Par) -> Result<(), String> {
        let a = args.inputs.array::<T>(0)?;
        let (det, _, _) = determinant(ctx, &a, par)?;
        args.outputs
            .array::<T>(ctx, 0, self.mode, &[])?
            .matrix_mut(false)[(0, 0)] = det;
        Ok(())
    }
}

/// The sign and the log of the absolute value of the determinant (numpy.linalg.slogdet).
pub struct SLogDet<T> {
    pub parallel: bool,
    pub modes: [OutputMode; 2],
    pub layouts: Vec<Layout>,
    pub element: PhantomData<T>,
}

impl<T: Float> FaerKernel for SLogDet<T> {
    fn arity(&self) -> (usize, usize) {
        (1, 2)
    }

    fn run(&self, ctx: &mut PtCtx, mut args: Args, par: Par) -> Result<(), String> {
        let a = args.inputs.array::<T>(0)?;
        let (_, sign, log) = determinant(ctx, &a, par)?;
        let [s, l] = args.outputs.slots([0, 1])?;
        s.array::<T>(ctx, self.modes[0], &[])?.matrix_mut(false)[(0, 0)] = sign;
        l.array::<T>(ctx, self.modes[1], &[])?.matrix_mut(false)[(0, 0)] = log;
        Ok(())
    }
}

// ------------------------------------------------------------------ LUFactor, LU

/// The LU factorization of a square matrix as LAPACK getrf (scipy.linalg.lu_factor):
/// L and U packed in one matrix and the pivots (row i was swapped with row piv[i]),
/// the matrix NaN if A is singular, as PyTensor's LUFactor.
pub struct LUFactor<T> {
    pub parallel: bool,
    pub modes: [OutputMode; 2],
    pub layouts: Vec<Layout>,
    pub element: PhantomData<T>,
}

impl<T: Float> FaerKernel for LUFactor<T> {
    fn arity(&self) -> (usize, usize) {
        (1, 2)
    }

    fn run(&self, ctx: &mut PtCtx, mut args: Args, par: Par) -> Result<(), String> {
        let a = args.inputs.array::<T>(0)?;
        let n = square(&a)?;
        let [lu_out, piv_out] = args.outputs.slots([0, 1])?;
        let mut packed = lu_out.array::<T>(ctx, self.modes[0], a.shape())?;
        let mut piv = piv_out.array::<i32>(ctx, self.modes[1], &[n as i64])?;
        packed.copy_from(&a)?;
        let req = StackReq::all_of(&[lu_scratch::<T>(n, par), StackReq::new::<usize>(n).array(2)]);
        let mut factor = packed.matrix_mut(false);
        let (perm, swaps, stack) = lu_in_place(factor.rb_mut(), stack_for(ctx, req), par);
        // the row swaps that give the permutation: at step i, the row now at
        // position piv[i] comes to i
        let (row_at, stack) = indices(stack, n);
        let (position_of, _) = indices(stack, n);
        let (forward, _) = perm.arrays();
        let mut piv = piv.matrix_mut(false);
        for i in 0..n {
            let (row, j) = (forward[i], position_of[forward[i]]);
            piv[(i, 0)] = j as i32;
            let other = row_at[i];
            (row_at[i], row_at[j]) = (row, other);
            (position_of[row], position_of[other]) = (i, j);
        }
        if swaps.is_none() {
            factor.fill(T::NAN);
        }
        Ok(())
    }
}

/// What LU returns besides L and U.
#[derive(Clone, Copy, Debug)]
pub enum LuForm {
    /// (P, L, U) with A = P L U and P a matrix
    Matrix,
    /// (p, L, U) with p the row indices: A = L[p] U
    Indices,
    /// (P L, U)
    PermuteL,
}

/// The LU decomposition of a square matrix (scipy.linalg.lu).
pub struct LU<T> {
    pub parallel: bool,
    pub modes: Vec<OutputMode>,
    pub layouts: Vec<Layout>,
    pub form: LuForm,
    pub element: PhantomData<T>,
}

impl<T: Float> FaerKernel for LU<T> {
    fn arity(&self) -> (usize, usize) {
        (1, self.modes.len())
    }

    fn run(&self, ctx: &mut PtCtx, mut args: Args, par: Par) -> Result<(), String> {
        let a = args.inputs.array::<T>(0)?;
        let n = square(&a)?;
        let shape = a.shape();
        let (p_out, l_out, u_out, modes) = match self.form {
            LuForm::PermuteL => {
                let [l, u] = args.outputs.slots([0, 1])?;
                (None, l, u, [self.modes[0], self.modes[1]])
            }
            _ => {
                let [p, l, u] = args.outputs.slots([0, 1, 2])?;
                (Some(p), l, u, [self.modes[1], self.modes[2]])
            }
        };
        let p_out = match (p_out, self.form) {
            (Some(p), LuForm::Indices) => {
                Some(Err(p.array::<i32>(ctx, self.modes[0], &[n as i64])?))
            }
            (Some(p), _) => Some(Ok(p.array::<T>(ctx, self.modes[0], shape)?)),
            (None, _) => None,
        };
        let mut l_out = l_out.array::<T>(ctx, modes[0], shape)?;
        let mut u_out = u_out.array::<T>(ctx, modes[1], shape)?;
        let req = StackReq::all_of(&[temp_mat_scratch::<T>(n, n), lu_scratch::<T>(n, par)]);
        let (mut factor, stack) = temp_mat_zeroed::<T, _, _>(n, n, stack_for(ctx, req));
        let mut factor = factor.as_mat_mut();
        factor.copy_from(a.matrix(false));
        let (perm, _, _) = lu_in_place(factor.rb_mut(), stack, par);
        let (forward, inverse) = perm.arrays();
        let (mut l, mut u) = (l_out.matrix_mut(false), u_out.matrix_mut(false));
        let permute_l = matches!(self.form, LuForm::PermuteL);
        for i in 0..n {
            // P L: row forward[i] of A is row i of L U
            let row = if permute_l { forward[i] } else { i };
            for j in 0..n {
                l[(row, j)] = match i.cmp(&j) {
                    std::cmp::Ordering::Greater => factor[(i, j)],
                    std::cmp::Ordering::Equal => T::ONE,
                    std::cmp::Ordering::Less => T::ZERO,
                };
                u[(i, j)] = if i <= j { factor[(i, j)] } else { T::ZERO };
            }
        }
        match p_out {
            Some(Ok(mut p)) => {
                let mut p = p.matrix_mut(false);
                p.fill(T::ZERO);
                for (i, &row) in forward.iter().enumerate() {
                    p[(row, i)] = T::ONE;
                }
            }
            Some(Err(mut p)) => {
                let mut p = p.matrix_mut(false);
                for (row, &i) in inverse.iter().enumerate() {
                    p[(row, 0)] = i as i32;
                }
            }
            None => {}
        }
        Ok(())
    }
}

// ------------------------------------------------------------------ PivotToPermutations

/// The permutation of LAPACK pivots (row i swapped with row piv[i], in order): its
/// inverse (the rows of A in the order of P A), or the permutation itself.
pub struct PivotToPermutations<I> {
    pub mode: OutputMode,
    pub layouts: Vec<Layout>,
    pub inverse: bool,
    pub element: PhantomData<I>,
}

impl<I: Element + TryInto<usize>> Kernel for PivotToPermutations<I> {
    fn arity(&self) -> (usize, usize) {
        (1, 1)
    }

    fn output_layouts(&self) -> &[Layout] {
        &self.layouts
    }

    fn run(&self, ctx: &mut PtCtx, mut args: Args) -> Result<(), String> {
        let pivots = args.inputs.array::<I>(0)?;
        let &[n] = pivots.shape() else {
            return Err("expected a vector".into());
        };
        let n = n as usize;
        let mut out = args.outputs.array::<i64>(ctx, 0, self.mode, &[n as i64])?;
        let pivots = pivots.matrix(false);
        let p_inv = scratch::<i64>(ctx, n);
        for (i, p) in p_inv.iter_mut().enumerate() {
            p.write(i as i64);
        }
        // SAFETY: all `n` elements were just initialized.
        let p_inv = unsafe { &mut *(p_inv as *mut [std::mem::MaybeUninit<i64>] as *mut [i64]) };
        for i in 0..n {
            let j = pivots[(i, 0)]
                .try_into()
                .ok()
                .filter(|&j| j < n)
                .ok_or_else(|| format!("pivot {i} is out of range"))?;
            p_inv.swap(i, j);
        }
        let mut out = out.matrix_mut(false);
        for (i, &p) in p_inv.iter().enumerate() {
            if self.inverse {
                out[(i, 0)] = p;
            } else {
                out[(p as usize, 0)] = i as i64;
            }
        }
        Ok(())
    }
}

// ------------------------------------------------------------------ Eigh

/// Eigenvalues (ascending) and, if `vectors`, eigenvectors of a symmetric matrix of
/// which the lower (upper) triangle is read (scipy.linalg.eigh / eigvalsh without b).
pub struct Eigh<T> {
    pub parallel: bool,
    pub modes: Vec<OutputMode>,
    pub layouts: Vec<Layout>,
    pub lower: bool,
    pub vectors: bool,
    pub element: PhantomData<T>,
}

impl<T: Float> FaerKernel for Eigh<T> {
    fn arity(&self) -> (usize, usize) {
        (1, if self.vectors { 2 } else { 1 })
    }

    fn run(&self, ctx: &mut PtCtx, mut args: Args, par: Par) -> Result<(), String> {
        let a = args.inputs.array::<T>(0)?;
        let n = square(&a)?;
        let source = if self.lower {
            a.matrix(false)
        } else {
            a.matrix(false).transpose()
        };
        let compute = if self.vectors {
            evd::ComputeEigenvectors::Yes
        } else {
            evd::ComputeEigenvectors::No
        };
        // (the scratch of an empty matrix underflows in faer)
        let req = if n == 0 {
            StackReq::EMPTY
        } else {
            evd::self_adjoint_evd_scratch::<T>(n, compute, par, Default::default())
        };
        let result = if self.vectors {
            let [w, v] = args.outputs.slots([0, 1])?;
            let mut w = w.array::<T>(ctx, self.modes[0], &[n as i64])?;
            let mut v = v.array::<T>(ctx, self.modes[1], a.shape())?;
            evd::self_adjoint_evd(
                source,
                w.matrix_mut(false).col_mut(0).as_diagonal_mut(),
                Some(v.matrix_mut(false)),
                par,
                stack_for(ctx, req),
                Default::default(),
            )
        } else {
            let mut w = args
                .outputs
                .array::<T>(ctx, 0, self.modes[0], &[n as i64])?;
            evd::self_adjoint_evd(
                source,
                w.matrix_mut(false).col_mut(0).as_diagonal_mut(),
                None,
                par,
                stack_for(ctx, req),
                Default::default(),
            )
        };
        result.map_err(|_| "the eigenvalue algorithm did not converge".into())
    }
}

// ------------------------------------------------------------------ SVD

/// The singular value decomposition A = U diag(s) V^T (numpy.linalg.svd), singular
/// values in descending order.
pub struct Svd<T> {
    pub parallel: bool,
    pub modes: Vec<OutputMode>,
    pub layouts: Vec<Layout>,
    pub full_matrices: bool,
    pub compute_uv: bool,
    pub element: PhantomData<T>,
}

impl<T: Float> FaerKernel for Svd<T> {
    fn arity(&self) -> (usize, usize) {
        (1, if self.compute_uv { 3 } else { 1 })
    }

    fn run(&self, ctx: &mut PtCtx, mut args: Args, par: Par) -> Result<(), String> {
        let a = args.inputs.array::<T>(0)?;
        let (m, n) = matrix_shape(&a)?;
        let k = m.min(n);
        let failed = |_| "the algorithm did not converge".to_string();
        if !self.compute_uv {
            let mut s = args
                .outputs
                .array::<T>(ctx, 0, self.modes[0], &[k as i64])?;
            let no = svd::ComputeSvdVectors::No;
            let req = svd::svd_scratch::<T>(m, n, no, no, par, Default::default());
            return svd::svd(
                a.matrix(false),
                s.matrix_mut(false).col_mut(0).as_diagonal_mut(),
                None,
                None,
                par,
                stack_for(ctx, req),
                Default::default(),
            )
            .map_err(failed);
        }
        let (kind, u_cols, v_rows) = if self.full_matrices {
            (svd::ComputeSvdVectors::Full, m, n)
        } else {
            (svd::ComputeSvdVectors::Thin, k, k)
        };
        let [u, s, vt] = args.outputs.slots([0, 1, 2])?;
        let mut u = u.array::<T>(ctx, self.modes[0], &[m as i64, u_cols as i64])?;
        let mut s = s.array::<T>(ctx, self.modes[1], &[k as i64])?;
        let mut vt = vt.array::<T>(ctx, self.modes[2], &[v_rows as i64, n as i64])?;
        let (mut u, mut v) = (u.matrix_mut(false), vt.matrix_mut(false).transpose_mut());
        if k == 0 {
            // (faer leaves the vectors of an empty matrix alone)
            for (mut x, d) in [(u.rb_mut(), m), (v.rb_mut(), n)] {
                x.fill(T::ZERO);
                for i in 0..d.min(x.ncols()) {
                    x[(i, i)] = T::ONE;
                }
            }
            return Ok(());
        }
        let req = svd::svd_scratch::<T>(m, n, kind, kind, par, Default::default());
        svd::svd(
            a.matrix(false),
            s.matrix_mut(false).col_mut(0).as_diagonal_mut(),
            Some(u),
            Some(v),
            par,
            stack_for(ctx, req),
            Default::default(),
        )
        .map_err(failed)
    }
}

// ------------------------------------------------------------------ QR

/// What QR returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QrMode {
    /// Q (m x m), R (m x n)
    Full,
    /// Q (m x k), R (k x n), k = min(m, n)
    Economic,
    /// R (m x n)
    R,
}

/// The QR decomposition by Householder reflections (scipy.linalg.qr without pivoting).
pub struct Qr<T> {
    pub parallel: bool,
    pub modes: Vec<OutputMode>,
    pub layouts: Vec<Layout>,
    pub mode: QrMode,
    pub element: PhantomData<T>,
}

impl<T: Float> FaerKernel for Qr<T> {
    fn arity(&self) -> (usize, usize) {
        (1, if self.mode == QrMode::R { 1 } else { 2 })
    }

    fn run(&self, ctx: &mut PtCtx, mut args: Args, par: Par) -> Result<(), String> {
        let a = args.inputs.array::<T>(0)?;
        let (m, n) = matrix_shape(&a)?;
        let k = m.min(n);
        let (q_cols, r_rows) = match self.mode {
            QrMode::Economic => (k, k),
            _ => (m, m),
        };
        let (q, r) = if self.mode == QrMode::R {
            let r = args
                .outputs
                .array::<T>(ctx, 0, self.modes[0], &[m as i64, n as i64]);
            (None, r?)
        } else {
            let [q, r] = args.outputs.slots([0, 1])?;
            let q = q.array::<T>(ctx, self.modes[0], &[m as i64, q_cols as i64])?;
            let r = r.array::<T>(ctx, self.modes[1], &[r_rows as i64, n as i64])?;
            (Some(q), r)
        };
        let mut r = r;
        let bs = qr::recommended_block_size::<T>(m, n);
        let req = StackReq::all_of(&[
            temp_mat_scratch::<T>(m, n),
            temp_mat_scratch::<T>(bs, k),
            StackReq::any_of(&[
                qr::qr_in_place_scratch::<T>(m, n, bs, par, Default::default()),
                householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(
                    m, bs, q_cols,
                ),
            ]),
        ]);
        let (mut factor, stack) = temp_mat_zeroed::<T, _, _>(m, n, stack_for(ctx, req));
        let mut factor = factor.as_mat_mut();
        let (mut coeff, stack) = temp_mat_zeroed::<T, _, _>(bs, k, stack);
        let mut coeff = coeff.as_mat_mut();
        factor.copy_from(a.matrix(false));
        if k > 0 {
            qr::qr_in_place(
                factor.rb_mut(),
                coeff.rb_mut(),
                par,
                stack,
                Default::default(),
            );
        }
        let mut r = r.matrix_mut(false);
        for i in 0..r_rows {
            for j in 0..n {
                r[(i, j)] = if i <= j && i < m {
                    factor[(i, j)]
                } else {
                    T::ZERO
                };
            }
        }
        if let Some(mut q) = q {
            let mut q = q.matrix_mut(false);
            q.fill(T::ZERO);
            for i in 0..q_cols {
                q[(i, i)] = T::ONE;
            }
            if k > 0 {
                householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
                    factor.rb().get(.., ..k),
                    coeff.rb(),
                    Conj::No,
                    q,
                    par,
                    stack,
                );
            }
        }
        Ok(())
    }
}

// ------------------------------------------------------------------ MatrixPinv

/// numpy.linalg.pinv's default cutoff, relative to the largest singular value.
const PINV_RCOND: f64 = 1e-15;

/// The pseudo-inverse (numpy.linalg.pinv): from the SVD, or for `hermitian` from the
/// eigendecomposition of the lower triangle; singular values up to 1e-15 times the
/// largest count as zero.
pub struct MatrixPinv<T> {
    pub parallel: bool,
    pub mode: OutputMode,
    pub layouts: Vec<Layout>,
    pub hermitian: bool,
    pub element: PhantomData<T>,
}

impl<T: Float> FaerKernel for MatrixPinv<T> {
    fn arity(&self) -> (usize, usize) {
        (1, 1)
    }

    fn run(&self, ctx: &mut PtCtx, mut args: Args, par: Par) -> Result<(), String> {
        let a = args.inputs.array::<T>(0)?;
        let (m, n) = if self.hermitian {
            let n = square(&a)?;
            (n, n)
        } else {
            matrix_shape(&a)?
        };
        let mut out = args
            .outputs
            .array::<T>(ctx, 0, self.mode, &[n as i64, m as i64])?;
        let k = m.min(n);
        if k == 0 {
            return Ok(());
        }
        // pinv = V diag(1 / s) U^T, from (U, s, V) with the small s dropped; for a
        // symmetric matrix U = V the eigenvectors and s the eigenvalues
        let req = StackReq::all_of(&[
            temp_mat_scratch::<T>(m, k),
            temp_mat_scratch::<T>(n, k),
            temp_mat_scratch::<T>(k, 1),
            if self.hermitian {
                evd::self_adjoint_evd_scratch::<T>(
                    n,
                    evd::ComputeEigenvectors::Yes,
                    par,
                    Default::default(),
                )
            } else {
                let thin = svd::ComputeSvdVectors::Thin;
                svd::svd_scratch::<T>(m, n, thin, thin, par, Default::default())
            },
        ]);
        let (mut u, stack) = temp_mat_zeroed::<T, _, _>(m, k, stack_for(ctx, req));
        let mut u = u.as_mat_mut();
        let (mut v, stack) = temp_mat_zeroed::<T, _, _>(n, k, stack);
        let mut v = v.as_mat_mut();
        let (mut s, stack) = temp_mat_zeroed::<T, _, _>(k, 1, stack);
        let mut s = s.as_mat_mut();
        let diag = s.rb_mut().col_mut(0).as_diagonal_mut();
        let source = a.matrix(false);
        if self.hermitian {
            evd::self_adjoint_evd(
                source,
                diag,
                Some(u.rb_mut()),
                par,
                stack,
                Default::default(),
            )
            .map_err(|_| "the algorithm did not converge")?;
            v.copy_from(u.rb());
        } else {
            svd::svd(
                source,
                diag,
                Some(u.rb_mut()),
                Some(v.rb_mut()),
                par,
                stack,
                Default::default(),
            )
            .map_err(|_| "the algorithm did not converge")?;
        }
        let largest = (0..k)
            .map(|j| s[(j, 0)].abs())
            .fold(T::ZERO, |a, b| if b > a { b } else { a });
        let cutoff = T::from_f64(PINV_RCOND) * largest;
        for j in 0..k {
            let x = s[(j, 0)];
            let scale = if x.abs() > cutoff {
                T::ONE / x
            } else {
                T::ZERO
            };
            for i in 0..n {
                v[(i, j)] *= scale;
            }
        }
        faer::linalg::matmul::matmul(
            out.matrix_mut(false),
            faer::Accum::Replace,
            v.rb(),
            u.rb().transpose(),
            T::ONE,
            par,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests;
