//! Legacy Rust kernels: matmul and sort.
//!
//! They have the legacy kernel ABI (`abi::LegacyFn`: one provided output, a parameter
//! block) and are run through `interp::kernels::k_legacy`. `linalg.rs` has the others.
// Kernel module: every kernel here is called with the kernel contract (see the Safety
// section of DESIGN.md: arguments are valid arrays as planned, the config is the
// task's own). Raw-pointer work is the whole body, so unsafe ops aren't wrapped
// one by one; the preconditions are documented per kernel instead.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::abi::{PtArray, MAX_NDIM};
use crate::runtime::{PtCtx};
use crate::dtype::DType;
use std::cmp::Ordering;
use std::ffi::c_void;

// ------------------------------------------------------------------ params

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SortParams {
    pub axis: i64,
    pub stable: i64,
}

unsafe fn p<'a, T>(params: *const c_void) -> &'a T {
    &*(params as *const T)
}

unsafe fn arg<'a>(args: *mut PtArray, i: usize) -> &'a mut PtArray {
    &mut *args.add(i)
}

unsafe fn read_f64(ptr: *const u8, dt: DType) -> f64 {
    match dt {
        DType::Float64 => *(ptr as *const f64),
        DType::Float32 => *(ptr as *const f32) as f64,
        DType::Int64 => *(ptr as *const i64) as f64,
        DType::Int32 => *(ptr as *const i32) as f64,
        DType::Int16 => *(ptr as *const i16) as f64,
        DType::Int8 => *(ptr as *const i8) as f64,
        DType::Uint8 | DType::Bool => *ptr as f64,
    }
}

pub unsafe fn read_i64(ptr: *const u8, dt: DType) -> i64 {
    match dt {
        DType::Float64 | DType::Float32 => read_f64(ptr, dt) as i64,
        DType::Int64 => *(ptr as *const i64),
        DType::Int32 => *(ptr as *const i32) as i64,
        DType::Int16 => *(ptr as *const i16) as i64,
        DType::Int8 => *(ptr as *const i8) as i64,
        DType::Uint8 | DType::Bool => *ptr as i64,
    }
}

// ------------------------------------------------------------------ helpers

/// Visit all multi-indices of `shape` (except `skip_axis`), yielding byte offsets into two
/// arrays with the given strides.
unsafe fn for_each_lane(shape: &[i64], st_a: &[i64], st_b: &[i64], skip_axis: Option<usize>, mut f: impl FnMut(isize, isize)) {
    let nd = shape.len();
    if shape.iter().any(|&s| s == 0) {
        return;
    }
    let mut idx = [0i64; MAX_NDIM];
    let (mut oa, mut ob) = (0isize, 0isize);
    loop {
        f(oa, ob);
        let mut d = nd;
        loop {
            if d == 0 {
                return;
            }
            d -= 1;
            if Some(d) == skip_axis {
                continue;
            }
            idx[d] += 1;
            oa += st_a[d] as isize;
            ob += st_b[d] as isize;
            if idx[d] < shape[d] {
                break;
            }
            oa -= (st_a[d] * shape[d]) as isize;
            ob -= (st_b[d] * shape[d]) as isize;
            idx[d] = 0;
        }
    }
}

// ------------------------------------------------------------------ matmul

/// `rows x cols` view of a 1-d/2-d operand with element strides. A vector is a row
/// (`as_row`) or a column.
unsafe fn mat_parts(x: &PtArray, isz: i64, as_row: bool) -> (usize, usize, isize, isize) {
    let (sh, st) = (x.shape(), x.strides());
    match (x.ndim, as_row) {
        (2, _) => (sh[0] as usize, sh[1] as usize, (st[0] / isz) as isize, (st[1] / isz) as isize),
        (1, true) => (1, sh[0] as usize, 0, (st[0] / isz) as isize),
        (_, _) => (sh[0] as usize, 1, (st[0] / isz) as isize, 0),
    }
}

/// Scalars for the hand-written SIMD loops; `madd` is a fused multiply-add when the
/// caller was compiled with FMA (see `gemv_short_rows`).
pub trait Real: Copy + Default + std::ops::Add<Output = Self> + std::ops::Mul<Output = Self> {
    fn madd<const FMA: bool>(self, a: Self, b: Self) -> Self;
}

impl Real for f64 {
    #[inline(always)]
    fn madd<const FMA: bool>(self, a: f64, b: f64) -> f64 {
        if FMA { a.mul_add(b, self) } else { self + a * b }
    }
}

impl Real for f32 {
    #[inline(always)]
    fn madd<const FMA: bool>(self, a: f32, b: f32) -> f32 {
        if FMA { a.mul_add(b, self) } else { self + a * b }
    }
}

/// y = A x for A with contiguous, short rows (k small), where general gemv kernels do
/// poorly: 4 rows at a time, 4 independent lanes along each row (so LLVM vectorizes
/// without reassociating), x contiguous.
#[inline(always)]
unsafe fn gemv_short_rows_impl<T: Real, const FMA: bool>(m: usize, k: usize, a: *const T, rsa: isize, x: *const T, y: *mut T, incy: isize) {
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
unsafe fn gemv_short_rows_avx2<T: Real>(m: usize, k: usize, a: *const T, rsa: isize, x: *const T, y: *mut T, incy: isize) {
    gemv_short_rows_impl::<T, true>(m, k, a, rsa, x, y, incy)
}

/// The crate is built for baseline x86-64; pick the AVX2+FMA version at runtime.
unsafe fn gemv_short_rows<T: Real>(m: usize, k: usize, a: *const T, rsa: isize, x: *const T, y: *mut T, incy: isize) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return gemv_short_rows_avx2(m, k, a, rsa, x, y, incy);
        }
    }
    gemv_short_rows_impl::<T, false>(m, k, a, rsa, x, y, incy)
}

/// Rows up to this length go through `gemv_short_rows`; longer ones through faer.
const SHORT_ROW: usize = 64;

macro_rules! matmul_kernel {
    ($name:ident, $t:ty) => {
        /// args = [a, b, out(provided)]; 1-d/2-d operands (numpy dot semantics).
        /// faer, sequential.
        pub unsafe extern "C-unwind" fn $name(_ctx: *mut PtCtx, args: *mut PtArray, _params: *const c_void) -> i32 {
            let (a, b, c) = (*arg(args, 0), *arg(args, 1), *arg(args, 2));
            let isz = std::mem::size_of::<$t>() as i64;
            let (m, k, rsa, csa) = mat_parts(&a, isz, true);
            let (_, n, rsb, csb) = mat_parts(&b, isz, false);
            let (rsc, csc) = match (a.ndim, b.ndim) {
                (2, 2) => ((c.strides()[0] / isz) as isize, (c.strides()[1] / isz) as isize),
                (2, 1) => ((c.strides()[0] / isz) as isize, 0),
                (1, 2) => (0, (c.strides()[0] / isz) as isize),
                _ => (0, 0),
            };
            // matrix-vector with contiguous short rows (X @ beta with X row-major)
            let (gm, gk, ga, grs, gcs, gx, gincx, gincy) = if n == 1 {
                (m, k, a.data as *const $t, rsa, csa, b.data as *const $t, rsb, rsc)
            } else if m == 1 {
                (n, k, b.data as *const $t, csb, rsb, a.data as *const $t, csa, csc)
            } else {
                (0, usize::MAX, std::ptr::null(), 0, 0, std::ptr::null(), 0, 0)
            };
            if gcs == 1 && gk <= SHORT_ROW && gm >= 4 {
                let x = if gincx == 1 {
                    gx
                } else {
                    let tmp = PtCtx::scratch(_ctx, gk * std::mem::size_of::<$t>()) as *mut $t;
                    for j in 0..gk {
                        *tmp.add(j) = *gx.offset(j as isize * gincx);
                    }
                    tmp as *const $t
                };
                gemv_short_rows(gm, gk, ga, grs, x, c.data as *mut $t, gincy);
                return 0;
            }
            let lhs = faer::MatRef::<$t>::from_raw_parts(a.data as *const $t, m, k, rsa, csa);
            let rhs = faer::MatRef::<$t>::from_raw_parts(b.data as *const $t, k, n, rsb, csb);
            let dst = faer::MatMut::<$t>::from_raw_parts_mut(c.data as *mut $t, m, n, rsc, csc);
            faer::linalg::matmul::matmul(dst, faer::Accum::Replace, lhs, rhs, 1.0, faer::Par::Seq);
            0
        }
    };
}

matmul_kernel!(ptl_matmul_f64, f64);
matmul_kernel!(ptl_matmul_f32, f32);

// ------------------------------------------------------------------ sort

trait SortElem: Copy {
    fn cmp_nan_last(a: &Self, b: &Self) -> Ordering;
}

macro_rules! float_sort_elem {
    ($t:ty) => {
        impl SortElem for $t {
            fn cmp_nan_last(a: &Self, b: &Self) -> Ordering {
                match (a.is_nan(), b.is_nan()) {
                    (false, false) => a.partial_cmp(b).unwrap(),
                    (true, true) => Ordering::Equal,
                    (true, false) => Ordering::Greater,
                    (false, true) => Ordering::Less,
                }
            }
        }
    };
}
float_sort_elem!(f64);
float_sort_elem!(f32);
impl SortElem for i64 {
    fn cmp_nan_last(a: &Self, b: &Self) -> Ordering {
        a.cmp(b)
    }
}

unsafe fn sort_impl<T: SortElem>(args: *mut PtArray, params: *const c_void) -> i32 {
    let prm = p::<SortParams>(params);
    let (axis, stable) = (prm.axis as usize, prm.stable != 0);
    let (s, d) = (*arg(args, 0), *arg(args, 1));
    let (shape, ss, ds) = (s.shape(), s.strides(), d.strides());
    let n = shape[axis] as usize;
    let (sa, da) = (ss[axis] as isize, ds[axis] as isize);
    let mut buf: Vec<T> = Vec::with_capacity(n);
    for_each_lane(shape, ss, ds, Some(axis), |a, b| {
        buf.clear();
        for i in 0..n as isize {
            buf.push(*(s.data.offset(a + i * sa) as *const T));
        }
        if stable {
            buf.sort_by(T::cmp_nan_last);
        } else {
            buf.sort_unstable_by(T::cmp_nan_last);
        }
        for (i, v) in buf.iter().enumerate() {
            *(d.data.offset(b + i as isize * da) as *mut T) = *v;
        }
    });
    0
}

/// args = [src, dst(provided)].
pub unsafe extern "C-unwind" fn ptl_sort_f64(_c: *mut PtCtx, args: *mut PtArray, params: *const c_void) -> i32 {
    sort_impl::<f64>(args, params)
}
pub unsafe extern "C-unwind" fn ptl_sort_f32(_c: *mut PtCtx, args: *mut PtArray, params: *const c_void) -> i32 {
    sort_impl::<f32>(args, params)
}
pub unsafe extern "C-unwind" fn ptl_sort_i64(_c: *mut PtCtx, args: *mut PtArray, params: *const c_void) -> i32 {
    sort_impl::<i64>(args, params)
}
