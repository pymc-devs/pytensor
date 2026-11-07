//! Precompiled kernels written in Rust, using the same ABI as generated kernels.
//!
//! Scalar functions (`ptl_*`) are registered in the JIT as absolute symbols so generated
//! code can call them by name; cached object files stay position independent.

use crate::abi::{PtArray, PtCtx, MAX_NDIM};
use crate::dtype::DType;
use std::cmp::Ordering;
use std::ffi::c_void;

// ------------------------------------------------------------------ params

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CopyParams {
    pub itemsize: i64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SortParams {
    pub axis: i64,
    pub stable: i64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ShapeIParams {
    pub i: i64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MakeVectorParams {
    pub n: i64,
    pub itemsize: i64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AssertParams {
    pub n_conds: i64,
}

/// `order[k]` is the input dim for output dim k, or -1 for a new broadcastable dim.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DimShuffleParams {
    pub order: [i64; MAX_NDIM],
}

/// dtype codes of start, stop, step and the output.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ArangeParams {
    pub dtypes: [i32; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AllocParams {
    pub n_shape: i64,
    pub itemsize: i64,
    pub shape_dtypes: [i32; MAX_NDIM],
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

unsafe fn read_i64(ptr: *const u8, dt: DType) -> i64 {
    match dt {
        DType::Float64 | DType::Float32 => read_f64(ptr, dt) as i64,
        DType::Int64 => *(ptr as *const i64),
        DType::Int32 => *(ptr as *const i32) as i64,
        DType::Int16 => *(ptr as *const i16) as i64,
        DType::Int8 => *(ptr as *const i8) as i64,
        DType::Uint8 | DType::Bool => *ptr as i64,
    }
}

unsafe fn write_num(ptr: *mut u8, dt: DType, f: f64, i: i64, use_int: bool) {
    match dt {
        DType::Float64 => *(ptr as *mut f64) = if use_int { i as f64 } else { f },
        DType::Float32 => *(ptr as *mut f32) = if use_int { i as f32 } else { f as f32 },
        _ => {
            let v = if use_int { i } else { f as i64 };
            match dt {
                DType::Int64 => *(ptr as *mut i64) = v,
                DType::Int32 => *(ptr as *mut i32) = v as i32,
                DType::Int16 => *(ptr as *mut i16) = v as i16,
                DType::Int8 => *(ptr as *mut i8) = v as i8,
                DType::Uint8 => *ptr = v as u8,
                DType::Bool => *ptr = (v != 0) as u8,
                _ => unreachable!(),
            }
        }
    }
}

fn c_strides(shape: &[i64], itemsize: i64, out: &mut [i64]) {
    let mut acc = itemsize;
    for d in (0..shape.len()).rev() {
        out[d] = acc;
        acc *= shape[d].max(1);
    }
}

// ------------------------------------------------------------------ scalar functions

fn digamma(mut x: f64) -> f64 {
    use std::f64::consts::PI;
    if x.is_nan() || x == f64::NEG_INFINITY {
        return f64::NAN;
    }
    if x == 0.0 {
        return if x.is_sign_negative() { f64::INFINITY } else { f64::NEG_INFINITY };
    }
    if x < 0.0 {
        if x == x.floor() {
            return f64::NAN;
        }
        return digamma(1.0 - x) - PI / (PI * x).tan();
    }
    let mut result = 0.0;
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let f = 1.0 / (x * x);
    let t = f * (-1.0 / 12.0 + f * (1.0 / 120.0 + f * (-1.0 / 252.0 + f * (1.0 / 240.0 + f * (-1.0 / 132.0)))));
    result + x.ln() - 0.5 / x + t
}

macro_rules! scalar_fns {
    ($($name:ident, $name32:ident => $f:expr;)*) => {
        $(
            #[unsafe(no_mangle)]
            pub extern "C" fn $name(x: f64) -> f64 { ($f)(x) }
            #[unsafe(no_mangle)]
            pub extern "C" fn $name32(x: f32) -> f32 { ($f)(x as f64) as f32 }
        )*
        pub fn scalar_symbols() -> Vec<(&'static str, usize)> {
            vec![$(
                (stringify!($name), $name as extern "C" fn(f64) -> f64 as usize),
                (stringify!($name32), $name32 as extern "C" fn(f32) -> f32 as usize),
            )*]
        }
    };
}

scalar_fns! {
    ptl_erf_f64, ptl_erf_f32 => libm::erf;
    ptl_erfc_f64, ptl_erfc_f32 => libm::erfc;
    ptl_gammaln_f64, ptl_gammaln_f32 => libm::lgamma;
    ptl_gamma_f64, ptl_gamma_f32 => libm::tgamma;
    ptl_psi_f64, ptl_psi_f32 => digamma;
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

// ------------------------------------------------------------------ copy

/// args = [src, dst(provided)].
pub unsafe extern "C" fn ptl_copy(_ctx: *mut PtCtx, args: *mut PtArray, params: *const c_void) -> i32 {
    let isz = p::<CopyParams>(params).itemsize as usize;
    let (s, d) = (arg(args, 0), arg(args, 1));
    for_each_lane(s.shape(), s.strides(), d.strides(), None, |a, b| {
        std::ptr::copy_nonoverlapping(s.data.offset(a), d.data.offset(b), isz);
    });
    0
}

// ------------------------------------------------------------------ views

/// args = [x, out(view of x)].
pub unsafe extern "C" fn ptl_dimshuffle(_ctx: *mut PtCtx, args: *mut PtArray, params: *const c_void) -> i32 {
    let order = &p::<DimShuffleParams>(params).order;
    let x = *arg(args, 0);
    let o = arg(args, 1);
    o.data = x.data;
    let (xs, xst) = (x.shape(), x.strides());
    for k in 0..o.ndim as usize {
        let src = order[k];
        let (sh, st) = if src < 0 { (1, 0) } else { (xs[src as usize], xst[src as usize]) };
        *o.shape.add(k) = sh;
        *o.strides.add(k) = st;
    }
    0
}

/// args = [x, conds..., out(view of x)]. Fails if any (0-d) condition is false.
pub unsafe extern "C" fn ptl_assert(_ctx: *mut PtCtx, args: *mut PtArray, params: *const c_void) -> i32 {
    let n = p::<AssertParams>(params).n_conds as usize;
    for i in 0..n {
        if *arg(args, 1 + i).data == 0 {
            return 1; // the driver maps this to the Op's message
        }
    }
    let x = *arg(args, 0);
    let o = arg(args, 1 + n);
    o.data = x.data;
    o.shape_mut().copy_from_slice(x.shape());
    o.strides_mut().copy_from_slice(x.strides());
    0
}

// ------------------------------------------------------------------ kernel-allocated outputs

/// args = [start, stop, step, out(kernel-allocated, 1-d)]. numpy.arange semantics.
pub unsafe extern "C" fn ptl_arange(ctx: *mut PtCtx, args: *mut PtArray, params: *const c_void) -> i32 {
    let dts = p::<ArangeParams>(params).dtypes;
    let dt = |i: usize| DType::from_code(dts[i]).unwrap();
    let all_int = (0..3).all(|i| !dt(i).is_float());
    let out_dt = dt(3);
    let (n, fstart, fstep, istart, istep) = if all_int {
        let (a, b, s) = (read_i64(arg(args, 0).data, dt(0)), read_i64(arg(args, 1).data, dt(1)), read_i64(arg(args, 2).data, dt(2)));
        if s == 0 {
            return PtCtx::fail(ctx, 2, "arange: step must not be zero");
        }
        let n = if s > 0 { (b - a + s - 1).div_euclid(s) } else { (a - b - s - 1).div_euclid(-s) };
        (n.max(0), 0.0, 0.0, a, s)
    } else {
        let (a, b, s) = (read_f64(arg(args, 0).data, dt(0)), read_f64(arg(args, 1).data, dt(1)), read_f64(arg(args, 2).data, dt(2)));
        if s == 0.0 {
            return PtCtx::fail(ctx, 2, "arange: step must not be zero");
        }
        (((b - a) / s).ceil().max(0.0) as i64, a, s, 0, 0)
    };
    let isz = out_dt.itemsize();
    let (data, owner) = PtCtx::alloc(ctx, n as usize * isz);
    let o = arg(args, 3);
    o.data = data;
    o.owner = owner;
    *o.shape = n;
    *o.strides = isz as i64;
    for i in 0..n {
        let ptr = data.add(i as usize * isz);
        write_num(ptr, out_dt, fstart + i as f64 * fstep, istart + i * istep, all_int);
    }
    0
}

/// args = [value, shape_0 .. shape_{n-1} (0-d ints), out(kernel-allocated)].
/// Broadcasts `value` (numpy rules, right-aligned) to the requested shape.
pub unsafe extern "C" fn ptl_alloc(ctx: *mut PtCtx, args: *mut PtArray, params: *const c_void) -> i32 {
    let prm = p::<AllocParams>(params);
    let n = prm.n_shape as usize;
    let isz = prm.itemsize;
    let v = *arg(args, 0);
    let mut shape = [0i64; MAX_NDIM];
    for i in 0..n {
        let s = read_i64(arg(args, 1 + i).data, DType::from_code(prm.shape_dtypes[i]).unwrap());
        if s < 0 {
            return PtCtx::fail(ctx, 2, "alloc: negative dimension");
        }
        shape[i] = s;
    }
    let shape = &shape[..n];
    // effective source strides (0 where broadcast)
    let vn = v.ndim as usize;
    if vn > n {
        return PtCtx::fail(ctx, 2, "alloc: value has more dims than the requested shape");
    }
    let mut src_st = [0i64; MAX_NDIM];
    for j in 0..vn {
        let d = n - vn + j;
        let vs = v.shape()[j];
        if vs == shape[d] {
            src_st[d] = v.strides()[j];
        } else if vs != 1 {
            return PtCtx::fail(ctx, 2, &format!("alloc: cannot broadcast value of shape {:?} to {:?}", v.shape(), shape));
        }
    }
    let total: i64 = shape.iter().product();
    let (data, owner) = PtCtx::alloc(ctx, total as usize * isz as usize);
    let o = arg(args, 1 + n);
    o.data = data;
    o.owner = owner;
    o.shape_mut().copy_from_slice(shape);
    c_strides(shape, isz, o.strides_mut());
    let dst_st = o.strides().to_vec();
    for_each_lane(shape, &src_st[..n], &dst_st, None, |a, b| {
        std::ptr::copy_nonoverlapping(v.data.offset(a), data.offset(b), isz as usize);
    });
    0
}

// ------------------------------------------------------------------ shape ops

/// args = [x, out(0-d int64)].
pub unsafe extern "C" fn ptl_shape_i(_ctx: *mut PtCtx, args: *mut PtArray, params: *const c_void) -> i32 {
    let i = p::<ShapeIParams>(params).i as usize;
    *(arg(args, 1).data as *mut i64) = arg(args, 0).shape()[i];
    0
}

/// args = [x, out(1-d int64)].
pub unsafe extern "C" fn ptl_shape(_ctx: *mut PtCtx, args: *mut PtArray, _params: *const c_void) -> i32 {
    let x = *arg(args, 0);
    let o = arg(args, 1);
    let os = o.strides()[0] as isize;
    for (d, s) in x.shape().iter().enumerate() {
        *(o.data.offset(d as isize * os) as *mut i64) = *s;
    }
    0
}

/// args = n 0-d inputs, then the 1-d output.
pub unsafe extern "C" fn ptl_make_vector(_ctx: *mut PtCtx, args: *mut PtArray, params: *const c_void) -> i32 {
    let prm = p::<MakeVectorParams>(params);
    let (n, isz) = (prm.n as usize, prm.itemsize as usize);
    let o = *arg(args, n);
    let os = o.strides()[0] as isize;
    for i in 0..n {
        std::ptr::copy_nonoverlapping(arg(args, i).data, o.data.offset(i as isize * os), isz);
    }
    0
}

// ------------------------------------------------------------------ matmul

/// y[i] = sum_j a[i, j] * x[j]   (strides in elements). Picks the loop order that walks
/// `a` contiguously.
#[allow(clippy::too_many_arguments)]
unsafe fn gemv<T>(m: usize, k: usize, a: *const T, rsa: isize, csa: isize, x: *const T, incx: isize, y: *mut T, incy: isize)
where
    T: Copy + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    if csa.abs() <= rsa.abs() {
        for i in 0..m as isize {
            let row = a.offset(i * rsa);
            let mut acc = T::default();
            if csa == 1 && incx == 1 {
                for j in 0..k {
                    acc = acc + *row.add(j) * *x.add(j);
                }
            } else {
                for j in 0..k as isize {
                    acc = acc + *row.offset(j * csa) * *x.offset(j * incx);
                }
            }
            *y.offset(i * incy) = acc;
        }
    } else if m <= 64 {
        let mut acc = [T::default(); 64];
        let acc = &mut acc[..m];
        for j in 0..k as isize {
            let xj = *x.offset(j * incx);
            let col = a.offset(j * csa);
            for (i, ac) in acc.iter_mut().enumerate() {
                *ac = *ac + *col.offset(i as isize * rsa) * xj;
            }
        }
        for (i, ac) in acc.iter().enumerate() {
            *y.offset(i as isize * incy) = *ac;
        }
    } else {
        for i in 0..m as isize {
            *y.offset(i * incy) = T::default();
        }
        for j in 0..k as isize {
            let xj = *x.offset(j * incx);
            let col = a.offset(j * csa);
            for i in 0..m as isize {
                *y.offset(i * incy) = *y.offset(i * incy) + *col.offset(i * rsa) * xj;
            }
        }
    }
}

macro_rules! matmul_kernel {
    ($name:ident, $t:ty, $gemm:path) => {
        /// args = [a, b, out(provided)]. 1-d/2-d operands.
        pub unsafe extern "C" fn $name(_ctx: *mut PtCtx, args: *mut PtArray, _params: *const c_void) -> i32 {
            let (a, b, c) = (*arg(args, 0), *arg(args, 1), *arg(args, 2));
            let isz = std::mem::size_of::<$t>() as i64;
            let (na, nb) = (a.ndim, b.ndim);
            let (ash, ast) = (a.shape(), a.strides());
            let (bsh, bst) = (b.shape(), b.strides());
            let (m, k, rsa, csa) = if na == 2 { (ash[0], ash[1], ast[0] / isz, ast[1] / isz) } else { (1, ash[0], 0, ast[0] / isz) };
            let (n, rsb, csb) = if nb == 2 { (bsh[1], bst[0] / isz, bst[1] / isz) } else { (1, bst[0] / isz, 0) };
            let cst = c.strides();
            let (rsc, csc) = match (na, nb) {
                (2, 2) => (cst[0] / isz, cst[1] / isz),
                (2, 1) => (cst[0] / isz, 0),
                (1, 2) => (0, cst[0] / isz),
                _ => (0, 0),
            };
            let (ap, bp, cp) = (a.data as *const $t, b.data as *const $t, c.data as *mut $t);
            if n == 1 {
                gemv(m as usize, k as usize, ap, rsa as isize, csa as isize, bp, rsb as isize, cp, rsc as isize);
            } else if m == 1 {
                gemv(n as usize, k as usize, bp, csb as isize, rsb as isize, ap, csa as isize, cp, csc as isize);
            } else {
                $gemm(m as usize, k as usize, n as usize, 1.0, ap, rsa as isize, csa as isize, bp, rsb as isize, csb as isize, 0.0, cp, rsc as isize, csc as isize);
            }
            0
        }
    };
}

matmul_kernel!(ptl_matmul_f64, f64, matrixmultiply::dgemm);
matmul_kernel!(ptl_matmul_f32, f32, matrixmultiply::sgemm);

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
pub unsafe extern "C" fn ptl_sort_f64(_c: *mut PtCtx, args: *mut PtArray, params: *const c_void) -> i32 {
    sort_impl::<f64>(args, params)
}
pub unsafe extern "C" fn ptl_sort_f32(_c: *mut PtCtx, args: *mut PtArray, params: *const c_void) -> i32 {
    sort_impl::<f32>(args, params)
}
pub unsafe extern "C" fn ptl_sort_i64(_c: *mut PtCtx, args: *mut PtArray, params: *const c_void) -> i32 {
    sort_impl::<i64>(args, params)
}

// ------------------------------------------------------------------ indexing

/// One basic-index entry. `kind`: 1 = integer index, 2 = slice. `a` is the integer's
/// argument position; for slices `a`, `b`, `c` are the positions of start/stop/step,
/// -1 when absent.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct IndexSpec {
    pub kind: i64,
    pub a: i64,
    pub b: i64,
    pub c: i64,
}

pub const MAX_ARGS: usize = 4 * MAX_NDIM;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SubtensorParams {
    pub n: i64,
    pub entries: [IndexSpec; MAX_NDIM],
    /// dtype code of every argument (index variables can be any integer type)
    pub dtypes: [i32; MAX_ARGS],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct IncSubtensorParams {
    pub sub: SubtensorParams,
    pub set: i64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TakeParams {
    pub axis: i64,
    pub idx_dtype: i32,
    pub dtype: i32,
    pub set: i64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct JoinParams {
    pub axis: i64,
    pub n: i64,
    pub itemsize: i64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReshapeParams {
    pub shape_dtype: i32,
    pub itemsize: i32,
    /// 1: input is known C-contiguous and the output is a declared view
    pub view: i64,
}

struct Resolved {
    offset: isize,
    ndim: usize,
    shape: [i64; MAX_NDIM],
    strides: [i64; MAX_NDIM],
}

/// Apply a basic index to `base` (Python/numpy semantics).
unsafe fn resolve_index(base: &PtArray, args: *mut PtArray, prm: &SubtensorParams) -> Result<Resolved, String> {
    let val = |pos: i64| -> i64 { read_i64(arg(args, pos as usize).data, DType::from_code(prm.dtypes[pos as usize]).unwrap()) };
    let (bs, bst) = (base.shape(), base.strides());
    let mut r = Resolved { offset: 0, ndim: 0, shape: [0; MAX_NDIM], strides: [0; MAX_NDIM] };
    let n = prm.n as usize;
    for d in 0..bs.len() {
        let len = bs[d];
        if d >= n {
            r.shape[r.ndim] = len;
            r.strides[r.ndim] = bst[d];
            r.ndim += 1;
            continue;
        }
        let e = prm.entries[d];
        if e.kind == 1 {
            let mut i = val(e.a);
            if i < 0 {
                i += len;
            }
            if i < 0 || i >= len {
                return Err(format!("index {} is out of bounds for axis {d} with size {len}", val(e.a)));
            }
            r.offset += (i * bst[d]) as isize;
            continue;
        }
        let step = if e.c >= 0 { val(e.c) } else { 1 };
        if step == 0 {
            return Err("slice step cannot be zero".into());
        }
        let norm = |v: i64, lo: i64, hi: i64| -> i64 {
            let v = if v < 0 { v + len } else { v };
            v.clamp(lo, hi)
        };
        let (start, stop) = if step > 0 {
            (if e.a >= 0 { norm(val(e.a), 0, len) } else { 0 }, if e.b >= 0 { norm(val(e.b), 0, len) } else { len })
        } else {
            (if e.a >= 0 { norm(val(e.a), -1, len - 1) } else { len - 1 }, if e.b >= 0 { norm(val(e.b), -1, len - 1) } else { -1 })
        };
        let count = if step > 0 { (stop - start + step - 1).div_euclid(step) } else { (start - stop - step - 1).div_euclid(-step) };
        let count = count.max(0);
        if count > 0 {
            r.offset += (start * bst[d]) as isize;
        }
        r.shape[r.ndim] = count;
        r.strides[r.ndim] = bst[d] * step;
        r.ndim += 1;
    }
    Ok(r)
}

/// args = [x, index vars..., out(view of x)].
pub unsafe extern "C" fn ptl_subtensor(ctx: *mut PtCtx, args: *mut PtArray, params: *const c_void) -> i32 {
    let prm = p::<SubtensorParams>(params);
    let x = *arg(args, 0);
    let r = match resolve_index(&x, args, prm) {
        Ok(r) => r,
        Err(m) => return PtCtx::fail(ctx, 2, &m),
    };
    let o = arg(args, n_args_before_out(prm, 1));
    o.data = x.data.offset(r.offset);
    o.shape_mut().copy_from_slice(&r.shape[..r.ndim]);
    o.strides_mut().copy_from_slice(&r.strides[..r.ndim]);
    0
}

/// position of the output argument: after the fixed inputs and every index variable
fn n_args_before_out(prm: &SubtensorParams, n_fixed: usize) -> usize {
    let mut m = n_fixed as i64 - 1;
    for e in &prm.entries[..prm.n as usize] {
        m = m.max(e.a).max(e.b).max(e.c);
    }
    (m + 1) as usize
}

// typed element updates
macro_rules! add_fns {
    ($($name:ident: $t:ty => $op:expr;)*) => {
        $(unsafe fn $name(d: *mut u8, s: *const u8) {
            let (a, b) = (*(d as *const $t), *(s as *const $t));
            *(d as *mut $t) = ($op)(a, b);
        })*
    };
}
add_fns! {
    add_f64: f64 => |a: f64, b: f64| a + b;
    add_f32: f32 => |a: f32, b: f32| a + b;
    add_i64: i64 => |a: i64, b: i64| a.wrapping_add(b);
    add_i32: i32 => |a: i32, b: i32| a.wrapping_add(b);
    add_i16: i16 => |a: i16, b: i16| a.wrapping_add(b);
    add_i8: i8 => |a: i8, b: i8| a.wrapping_add(b);
    add_u8: u8 => |a: u8, b: u8| a.wrapping_add(b);
    or_bool: u8 => |a: u8, b: u8| ((a | b) != 0) as u8;
}

type Update = unsafe fn(*mut u8, *const u8);

fn updater(dt: DType, set: bool) -> Update {
    unsafe fn set8(d: *mut u8, s: *const u8) { *(d as *mut u64) = *(s as *const u64) }
    unsafe fn set4(d: *mut u8, s: *const u8) { *(d as *mut u32) = *(s as *const u32) }
    unsafe fn set2(d: *mut u8, s: *const u8) { *(d as *mut u16) = *(s as *const u16) }
    unsafe fn set1(d: *mut u8, s: *const u8) { *d = *s }
    if set {
        return match dt.itemsize() { 8 => set8, 4 => set4, 2 => set2, _ => set1 };
    }
    match dt {
        DType::Float64 => add_f64,
        DType::Float32 => add_f32,
        DType::Int64 => add_i64,
        DType::Int32 => add_i32,
        DType::Int16 => add_i16,
        DType::Int8 => add_i8,
        DType::Uint8 => add_u8,
        DType::Bool => or_bool,
    }
}

/// Copy `src` into `dst` (same shape) unless they are the same memory (in place).
unsafe fn copy_unless_same(src: &PtArray, dst: &PtArray, itemsize: usize) {
    if src.data == dst.data {
        return;
    }
    for_each_lane(src.shape(), src.strides(), dst.strides(), None, |a, b| {
        std::ptr::copy_nonoverlapping(src.data.offset(a), dst.data.offset(b), itemsize);
    });
}

/// Strides of `y` broadcast (numpy, right-aligned) to `shape`; 0 where y has length 1.
fn broadcast_strides(y: &PtArray, shape: &[i64]) -> Result<[i64; MAX_NDIM], String> {
    let (ys, yst) = unsafe { (y.shape(), y.strides()) };
    let n = shape.len();
    if ys.len() > n {
        return Err(format!("cannot broadcast value of shape {ys:?} to {shape:?}"));
    }
    let mut st = [0i64; MAX_NDIM];
    for j in 0..ys.len() {
        let d = n - ys.len() + j;
        if ys[j] == shape[d] {
            st[d] = yst[j];
        } else if ys[j] != 1 {
            return Err(format!("cannot broadcast value of shape {ys:?} to {shape:?}"));
        }
    }
    Ok(st)
}

/// args = [x, y, index vars..., out(provided, shape of x)].
pub unsafe extern "C" fn ptl_inc_subtensor(ctx: *mut PtCtx, args: *mut PtArray, params: *const c_void) -> i32 {
    let prm = p::<IncSubtensorParams>(params);
    let (x, y) = (*arg(args, 0), *arg(args, 1));
    let out = *arg(args, n_args_before_out(&prm.sub, 2));
    let dt = DType::from_code(prm.sub.dtypes[0]).unwrap();
    copy_unless_same(&x, &out, dt.itemsize());
    let r = match resolve_index(&out, args, &prm.sub) {
        Ok(r) => r,
        Err(m) => return PtCtx::fail(ctx, 2, &m),
    };
    let shape = &r.shape[..r.ndim];
    let yst = match broadcast_strides(&y, shape) {
        Ok(s) => s,
        Err(m) => return PtCtx::fail(ctx, 2, &m),
    };
    let up = updater(dt, prm.set != 0);
    let base = out.data.offset(r.offset);
    for_each_lane(shape, &r.strides[..r.ndim], &yst[..r.ndim], None, |a, b| up(base.offset(a), y.data.offset(b)));
    0
}

// ---- fast paths for take / inc_take

trait Index: Copy {
    fn get(self) -> i64;
}
impl Index for i64 {
    fn get(self) -> i64 {
        self
    }
}
impl Index for i32 {
    fn get(self) -> i64 {
        self as i64
    }
}

/// out[j] = x[idx[j]] for 1-d x and idx (strides in bytes). Err(raw index) if out of bounds.
#[allow(clippy::too_many_arguments)]
unsafe fn take1d<T: Copy, I: Index>(x: *const u8, xs: isize, len: i64, idx: *const u8, is: isize, out: *mut u8, os: isize, n: usize) -> Result<(), i64> {
    for j in 0..n as isize {
        let raw = (*(idx.offset(j * is) as *const I)).get();
        let i = if raw < 0 { raw + len } else { raw };
        if (i as u64) >= len as u64 {
            return Err(raw);
        }
        *(out.offset(j * os) as *mut T) = *(x.offset(i as isize * xs) as *const T);
    }
    Ok(())
}

trait Acc: Copy {
    fn acc(a: Self, b: Self) -> Self;
}
macro_rules! acc_impl {
    ($($t:ty => $f:expr;)*) => { $(impl Acc for $t { fn acc(a: Self, b: Self) -> Self { ($f)(a, b) } })* };
}
acc_impl! {
    f64 => |a: f64, b: f64| a + b;
    f32 => |a: f32, b: f32| a + b;
    i64 => |a: i64, b: i64| a.wrapping_add(b);
    i32 => |a: i32, b: i32| a.wrapping_add(b);
}

/// out[idx[j]] (+)= y[j] for 1-d out, idx, y (y stride 0 = broadcast scalar).
#[allow(clippy::too_many_arguments)]
unsafe fn inc1d<T: Acc, I: Index>(out: *mut u8, os: isize, len: i64, idx: *const u8, is: isize, y: *const u8, ys: isize, n: usize, set: bool) -> Result<(), i64> {
    for j in 0..n as isize {
        let raw = (*(idx.offset(j * is) as *const I)).get();
        let i = if raw < 0 { raw + len } else { raw };
        if (i as u64) >= len as u64 {
            return Err(raw);
        }
        let d = out.offset(i as isize * os) as *mut T;
        let v = *(y.offset(j * ys) as *const T);
        *d = if set { v } else { T::acc(*d, v) };
    }
    Ok(())
}

/// Is an array with this shape/strides C-contiguous? (size-1 dims ignored)
fn is_c_contig(shape: &[i64], strides: &[i64], itemsize: i64) -> bool {
    let mut expect = itemsize;
    for d in (0..shape.len()).rev() {
        if shape[d] != 1 && strides[d] != expect {
            return false;
        }
        expect *= shape[d];
    }
    true
}

/// args = [x, idx, out(provided)].
pub unsafe extern "C" fn ptl_take(ctx: *mut PtCtx, args: *mut PtArray, params: *const c_void) -> i32 {
    let prm = p::<TakeParams>(params);
    let (x, idx, out) = (*arg(args, 0), *arg(args, 1), *arg(args, 2));
    let axis = prm.axis as usize;
    let k = idx.ndim as usize;
    let idt = DType::from_code(prm.idx_dtype).unwrap();
    let isz = DType::from_code(prm.dtype).unwrap().itemsize();
    let (xs, xst, ost) = (x.shape(), x.strides(), out.strides());
    let len = xs[axis];
    let oob = |raw: i64| PtCtx::fail(ctx, 2, &format!("index {raw} is out of bounds for axis {axis} with size {len}"));
    // fast path: 1-d gather
    if xs.len() == 1 && k == 1 {
        let n = idx.shape()[0] as usize;
        let (xsd, isd, osd) = (xst[0] as isize, idx.strides()[0] as isize, ost[0] as isize);
        let r = match (isz, idt) {
            (8, DType::Int64) => take1d::<u64, i64>(x.data, xsd, len, idx.data, isd, out.data, osd, n),
            (8, DType::Int32) => take1d::<u64, i32>(x.data, xsd, len, idx.data, isd, out.data, osd, n),
            (4, DType::Int64) => take1d::<u32, i64>(x.data, xsd, len, idx.data, isd, out.data, osd, n),
            (4, DType::Int32) => take1d::<u32, i32>(x.data, xsd, len, idx.data, isd, out.data, osd, n),
            _ => Err(i64::MIN),
        };
        match r {
            Ok(()) => return 0,
            Err(i64::MIN) => {} // other dtypes: generic path below
            Err(raw) => return oob(raw),
        }
    }
    // the sub-array copied per index: x without `axis`, out without the idx dims
    let mut sub_shape = [0i64; MAX_NDIM];
    let mut sx = [0i64; MAX_NDIM];
    let mut so = [0i64; MAX_NDIM];
    let mut m = 0;
    for d in 0..xs.len() {
        if d == axis {
            continue;
        }
        sub_shape[m] = xs[d];
        sx[m] = xst[d];
        so[m] = if d < axis { ost[d] } else { ost[d + k - 1] };
        m += 1;
    }
    // contiguous sub-arrays in both (e.g. gathering rows): one memcpy per index
    let block = if is_c_contig(&sub_shape[..m], &sx[..m], isz as i64) && is_c_contig(&sub_shape[..m], &so[..m], isz as i64) {
        Some(sub_shape[..m].iter().product::<i64>() as usize * isz)
    } else {
        None
    };
    let mut bad: Option<i64> = None;
    for_each_lane(idx.shape(), idx.strides(), &ost[axis..axis + k], None, |io, oo| {
        if bad.is_some() {
            return;
        }
        let raw = read_i64(idx.data.offset(io), idt);
        let i = if raw < 0 { raw + len } else { raw };
        if i < 0 || i >= len {
            bad = Some(raw);
            return;
        }
        let (xb, ob) = (x.data.offset((i * xst[axis]) as isize), out.data.offset(oo));
        if let Some(bytes) = block {
            std::ptr::copy_nonoverlapping(xb, ob, bytes);
            return;
        }
        for_each_lane(&sub_shape[..m], &sx[..m], &so[..m], None, |a, b| {
            std::ptr::copy_nonoverlapping(xb.offset(a), ob.offset(b), isz);
        });
    });
    if let Some(raw) = bad {
        return oob(raw);
    }
    0
}

/// args = [x, y, idx, out(provided, shape of x)]. Duplicate indices accumulate.
pub unsafe extern "C" fn ptl_inc_take(ctx: *mut PtCtx, args: *mut PtArray, params: *const c_void) -> i32 {
    let prm = p::<TakeParams>(params);
    let (x, y, idx, out) = (*arg(args, 0), *arg(args, 1), *arg(args, 2), *arg(args, 3));
    let axis = prm.axis as usize;
    let k = idx.ndim as usize;
    let dt = DType::from_code(prm.dtype).unwrap();
    let idt = DType::from_code(prm.idx_dtype).unwrap();
    copy_unless_same(&x, &out, dt.itemsize());
    let (xs, ost) = (out.shape(), out.strides());
    let len = xs[axis];
    // fast path: 1-d scatter with a 1-d or scalar y
    if xs.len() == 1 && k == 1 && y.ndim <= 1 {
        let n = idx.shape()[0] as usize;
        let ys = if y.ndim == 1 && y.shape()[0] != 1 { y.strides()[0] as isize } else { 0 };
        if y.ndim == 1 && y.shape()[0] != 1 && y.shape()[0] != n as i64 {
            return PtCtx::fail(ctx, 2, &format!("cannot broadcast value of shape {:?} to {:?}", y.shape(), idx.shape()));
        }
        let (osd, isd, set) = (ost[0] as isize, idx.strides()[0] as isize, prm.set != 0);
        let r = match (dt, idt) {
            (DType::Float64, DType::Int64) => inc1d::<f64, i64>(out.data, osd, len, idx.data, isd, y.data, ys, n, set),
            (DType::Float64, DType::Int32) => inc1d::<f64, i32>(out.data, osd, len, idx.data, isd, y.data, ys, n, set),
            (DType::Float32, DType::Int64) => inc1d::<f32, i64>(out.data, osd, len, idx.data, isd, y.data, ys, n, set),
            (DType::Float32, DType::Int32) => inc1d::<f32, i32>(out.data, osd, len, idx.data, isd, y.data, ys, n, set),
            (DType::Int64, DType::Int64) => inc1d::<i64, i64>(out.data, osd, len, idx.data, isd, y.data, ys, n, set),
            (DType::Int32, DType::Int64) => inc1d::<i32, i64>(out.data, osd, len, idx.data, isd, y.data, ys, n, set),
            _ => Err(i64::MIN),
        };
        match r {
            Ok(()) => return 0,
            Err(i64::MIN) => {}
            Err(raw) => return PtCtx::fail(ctx, 2, &format!("index {raw} is out of bounds for axis {axis} with size {len}")),
        }
    }
    // y is broadcast against x.shape[:axis] + idx.shape + x.shape[axis+1:]
    let mut full = [0i64; MAX_NDIM];
    let mut nf = 0;
    for d in 0..xs.len() {
        if d == axis {
            for s in idx.shape() {
                full[nf] = *s;
                nf += 1;
            }
        } else {
            full[nf] = xs[d];
            nf += 1;
        }
    }
    if nf > MAX_NDIM {
        return PtCtx::fail(ctx, 2, "too many dimensions");
    }
    let yst = match broadcast_strides(&y, &full[..nf]) {
        Ok(s) => s,
        Err(m) => return PtCtx::fail(ctx, 2, &m),
    };
    let mut sub_shape = [0i64; MAX_NDIM];
    let mut so = [0i64; MAX_NDIM];
    let mut sy = [0i64; MAX_NDIM];
    let mut m = 0;
    for d in 0..xs.len() {
        if d == axis {
            continue;
        }
        sub_shape[m] = xs[d];
        so[m] = ost[d];
        sy[m] = if d < axis { yst[d] } else { yst[d + k - 1] };
        m += 1;
    }
    let up = updater(dt, prm.set != 0);
    let mut bad: Option<i64> = None;
    for_each_lane(idx.shape(), idx.strides(), &yst[axis..axis + k], None, |io, yo| {
        if bad.is_some() {
            return;
        }
        let raw = read_i64(idx.data.offset(io), idt);
        let i = if raw < 0 { raw + len } else { raw };
        if i < 0 || i >= len {
            bad = Some(raw);
            return;
        }
        let (ob, yb) = (out.data.offset((i * ost[axis]) as isize), y.data.offset(yo));
        for_each_lane(&sub_shape[..m], &so[..m], &sy[..m], None, |a, b| up(ob.offset(a), yb.offset(b)));
    });
    if let Some(raw) = bad {
        return PtCtx::fail(ctx, 2, &format!("index {raw} is out of bounds for axis {axis} with size {len}"));
    }
    0
}

/// args = [x_0 .. x_{n-1}, out(provided)].
pub unsafe extern "C" fn ptl_join(_ctx: *mut PtCtx, args: *mut PtArray, params: *const c_void) -> i32 {
    let prm = p::<JoinParams>(params);
    let (axis, n, isz) = (prm.axis as usize, prm.n as usize, prm.itemsize as usize);
    let out = *arg(args, n);
    let ost = out.strides();
    let mut off = 0i64;
    for i in 0..n {
        let x = *arg(args, i);
        let base = out.data.offset((off * ost[axis]) as isize);
        for_each_lane(x.shape(), x.strides(), ost, None, |a, b| {
            std::ptr::copy_nonoverlapping(x.data.offset(a), base.offset(b), isz);
        });
        off += x.shape()[axis];
    }
    0
}

/// args = [x, shape(1-d ints), out]. Output is a declared view (params.view) or allocated.
pub unsafe extern "C" fn ptl_reshape(ctx: *mut PtCtx, args: *mut PtArray, params: *const c_void) -> i32 {
    let prm = p::<ReshapeParams>(params);
    let (x, shp) = (*arg(args, 0), *arg(args, 1));
    let o = arg(args, 2);
    let nd = o.ndim as usize;
    let sdt = DType::from_code(prm.shape_dtype).unwrap();
    let total: i64 = x.shape().iter().product();
    let ss = shp.strides()[0] as isize;
    let mut shape = [0i64; MAX_NDIM];
    let mut infer: Option<usize> = None;
    let mut known = 1i64;
    for d in 0..nd {
        let v = read_i64(shp.data.offset(d as isize * ss), sdt);
        if v == -1 {
            if infer.is_some() {
                return PtCtx::fail(ctx, 2, "reshape: can only specify one unknown dimension");
            }
            infer = Some(d);
        } else if v < 0 {
            return PtCtx::fail(ctx, 2, "reshape: negative dimension");
        } else {
            known *= v;
        }
        shape[d] = v;
    }
    if let Some(d) = infer {
        if known == 0 || total % known != 0 {
            return PtCtx::fail(ctx, 2, &format!("cannot reshape array of size {total} into shape {:?}", &shape[..nd]));
        }
        shape[d] = total / known;
    } else if known != total {
        return PtCtx::fail(ctx, 2, &format!("cannot reshape array of size {total} into shape {:?}", &shape[..nd]));
    }
    let isz = prm.itemsize as i64;
    o.shape_mut().copy_from_slice(&shape[..nd]);
    c_strides(&shape[..nd], isz, o.strides_mut());
    if prm.view != 0 {
        o.data = x.data;
        return 0;
    }
    let (data, owner) = PtCtx::alloc(ctx, total as usize * isz as usize);
    o.data = data;
    o.owner = owner;
    // copy x in C order into the new buffer
    let mut cst = [0i64; MAX_NDIM];
    c_strides(x.shape(), isz, &mut cst[..x.ndim as usize]);
    for_each_lane(x.shape(), x.strides(), &cst[..x.ndim as usize], None, |a, b| {
        std::ptr::copy_nonoverlapping(x.data.offset(a), data.offset(b), isz as usize);
    });
    0
}
