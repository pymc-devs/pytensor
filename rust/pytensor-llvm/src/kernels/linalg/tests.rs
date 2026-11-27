//! The linear algebra kernels, called through their ABI on small matrices in several
//! layouts (also under miri). Results are checked by what they must satisfy
//! (`A x = b`, `P L U = A`, ...), not against reference values.

use super::*;
use crate::abi::OutputMode::{Allocated, Provided};
use crate::testing::{In, Out, call};

/// The faer kernels' layouts of outputs with `ndims` axes each.
fn layouts(ndims: &[usize]) -> Vec<Layout> {
    faer_layouts(ndims, &[])
}

/// A general matrix whose LU factorization pivots; its determinant is -16.
const A: [f64; 9] = [2.0, 1.0, 1.0, 4.0, -6.0, 0.0, -2.0, 7.0, 2.0];
/// A Cholesky factor, and S = L L^T, symmetric positive definite.
const L: [f64; 9] = [2.0, 0.0, 0.0, 1.0, 3.0, 0.0, -1.0, 2.0, 1.0];
const S: [f64; 9] = [4.0, 2.0, -2.0, 2.0, 10.0, 5.0, -2.0, 5.0, 6.0];
/// 3 x 2, of full column rank.
const TALL: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0];
const SINGULAR: [f64; 4] = [1.0, 2.0, 2.0, 4.0];
const B: [f64; 3] = [1.0, -2.0, 3.0];

/// `a` (m x k) times `b` (k x n), C order.
fn mm(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    assert_eq!((a.len(), b.len()), (m * k, k * n));
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = (0..k).map(|l| a[i * k + l] * b[l * n + j]).sum();
        }
    }
    c
}

/// The transpose of `a` (m x n).
fn t(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    (0..n * m).map(|k| a[(k % m) * n + k / m]).collect()
}

fn eye(n: usize) -> Vec<f64> {
    (0..n * n)
        .map(|k| if k % (n + 1) == 0 { 1.0 } else { 0.0 })
        .collect()
}

#[track_caller]
fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} != {expected:?}");
    for (x, y) in actual.iter().zip(expected) {
        assert!(
            x == y || (x - y).abs() <= 1e-10 * (1.0 + y.abs()),
            "{actual:?} != {expected:?}"
        );
    }
}

/// `a` (m x n) with what is not in the lower (`lower`) or upper triangle set to 99.
fn half(a: &[f64], n: usize, lower: bool) -> Vec<f64> {
    (0..n * n)
        .map(|k| {
            let (i, j) = (k / n, k % n);
            if (lower && i < j) || (!lower && i > j) {
                99.0
            } else {
                a[k]
            }
        })
        .collect()
}

/// The matrix `a` (m x n, C order) as inputs of several layouts, each with provided
/// outputs: C, Fortran, with gaps, read backwards; and C again with allocated outputs.
fn cases(a: &[f64], m: usize, n: usize) -> Vec<(In, OutputMode)> {
    let backwards: Vec<f64> = (0..m)
        .rev()
        .flat_map(|i| a[i * n..(i + 1) * n].to_vec())
        .collect();
    vec![
        (In::new(&[m, n], a), Provided),
        (In::new(&[n, m], &t(a, m, n)).t(), Provided),
        (In::spread(&[m, n], a, 2), Provided),
        (In::new(&[m, n], &backwards).flip(0), Provided),
        (In::new(&[m, n], a), Allocated),
    ]
}

fn all_nan(values: &[f64]) -> bool {
    values.iter().all(|x| x.is_nan())
}

#[test]
fn dot_of_vectors_and_matrices() {
    // 5 x 3: rows of 3 adjacent elements take the short-row gemv
    let a: Vec<f64> = (0..15).map(|v| v as f64 * 0.5 - 2.0).collect();
    let v = [1.0, -1.0, 2.0];
    for (x, mode) in cases(&a, 5, 3) {
        // (a vector with gaps is copied to scratch first)
        let v = if mode == Provided {
            In::spread(&[3], &v, 2)
        } else {
            In::new(&[3], &v)
        };
        let kernel = Dot::<f64> {
            parallel: false,
            mode,
            layouts: layouts(&[1]),
            element: PhantomData,
        };
        let r = call(&kernel, &[x, v], vec![Out::new::<f64>(mode, &[5])]).unwrap();
        assert_close(&r[0].values::<f64>(), &mm(&a, &[1.0, -1.0, 2.0], 5, 3, 1));
    }
    let w = [1.0, 2.0, 3.0, 4.0, 5.0];
    let kernel = Dot::<f64> {
        parallel: false,
        mode: Provided,
        layouts: layouts(&[1]),
        element: PhantomData,
    };
    let r = call(
        &kernel,
        &[In::new(&[5], &w), In::new(&[5, 3], &a)],
        vec![Out::provided::<f64>(&[3])],
    );
    assert_close(&r.unwrap()[0].values::<f64>(), &mm(&w, &a, 1, 5, 3));
    let kernel = Dot::<f64> {
        parallel: false,
        mode: Allocated,
        layouts: layouts(&[2]),
        element: PhantomData,
    };
    let at = t(&a, 5, 3);
    let r = call(
        &kernel,
        &[In::new(&[5, 3], &a), In::new(&[3, 5], &at)],
        vec![Out::allocated::<f64>(2)],
    );
    assert_close(&r.unwrap()[0].values::<f64>(), &mm(&a, &at, 5, 3, 5));
    let err = call(
        &kernel,
        &[In::new(&[5, 3], &a), In::new(&[5], &w)],
        vec![Out::allocated::<f64>(1)],
    );
    assert!(err.unwrap_err().contains("not aligned"));
}

#[test]
fn cholesky_and_solve_triangular() {
    for (x, mode) in cases(&S, 3, 3) {
        for lower in [true, false] {
            let kernel = Cholesky::<f64> {
                parallel: false,
                mode,
                layouts: layouts(&[2]),
                lower,
                element: PhantomData,
            };
            let r = call(
                &kernel,
                std::slice::from_ref(&x),
                vec![Out::new::<f64>(mode, &[3, 3])],
            )
            .unwrap();
            assert_close(
                &r[0].values::<f64>(),
                &if lower { L.to_vec() } else { t(&L, 3, 3) },
            );
        }
    }
    let kernel = Cholesky::<f64> {
        parallel: false,
        mode: Provided,
        layouts: layouts(&[2]),
        lower: true,
        element: PhantomData,
    };
    let r = call(
        &kernel,
        &[In::new(&[2, 2], &SINGULAR)],
        vec![Out::provided::<f64>(&[2, 2])],
    );
    assert!(all_nan(&r.unwrap()[0].values::<f64>()));

    // L with a unit diagonal
    let unit: Vec<f64> = (0..9)
        .map(|k| if k % 4 == 0 { 1.0 } else { L[k] })
        .collect();
    for (x, mode) in cases(&L, 3, 3) {
        for (lower, unit_diagonal, x, matrix) in [
            (true, false, x.clone(), L.to_vec()),
            (false, false, x.clone().t(), t(&L, 3, 3)),
            (true, true, x, unit.clone()),
        ] {
            let kernel = SolveTriangular::<f64> {
                parallel: false,
                mode,
                layouts: layouts(&[1]),
                lower,
                unit_diagonal,
                element: PhantomData,
            };
            let r = call(
                &kernel,
                &[x, In::new(&[3], &B)],
                vec![Out::new::<f64>(mode, &[3])],
            )
            .unwrap();
            assert_close(&mm(&matrix, &r[0].values::<f64>(), 3, 3, 1), &B);
        }
    }
    let kernel = SolveTriangular::<f64> {
        parallel: false,
        mode: Provided,
        layouts: layouts(&[1]),
        lower: true,
        unit_diagonal: false,
        element: PhantomData,
    };
    let zero_diagonal = [1.0, 0.0, 1.0, 0.0];
    let r = call(
        &kernel,
        &[In::new(&[2, 2], &zero_diagonal), In::new(&[2], &[1.0, 1.0])],
        vec![Out::provided::<f64>(&[2])],
    );
    assert!(all_nan(&r.unwrap()[0].values::<f64>()));
}

#[test]
fn solve() {
    let rhs = [1.0, 0.0, -2.0, 1.0, 3.0, 2.0]; // 3 x 2
    for (assume, a) in [
        (Assume::General, A),
        (Assume::Symmetric { lower: true }, S),
        (Assume::PositiveDefinite { lower: true }, S),
    ] {
        for (x, mode) in cases(&a, 3, 3) {
            let kernel = Solve::<f64> {
                parallel: false,
                mode,
                layouts: layouts(&[1]),
                assume,
                element: PhantomData,
            };
            let r = call(
                &kernel,
                &[x, In::new(&[3], &B)],
                vec![Out::new::<f64>(mode, &[3])],
            )
            .unwrap();
            assert_close(&mm(&a, &r[0].values::<f64>(), 3, 3, 1), &B);
        }
        // a matrix right-hand side with gaps, into Fortran order
        let kernel = Solve::<f64> {
            parallel: false,
            mode: Provided,
            layouts: layouts(&[2]),
            assume,
            element: PhantomData,
        };
        let inputs = [In::new(&[3, 3], &a), In::spread(&[3, 2], &rhs, 3)];
        let r = call(
            &kernel,
            &inputs,
            vec![Out::provided_fortran::<f64>(&[3, 2])],
        )
        .unwrap();
        assert_close(&mm(&a, &r[0].values::<f64>(), 3, 3, 2), &rhs);
        // singular, or not positive definite: NaN
        let inputs = [In::new(&[2, 2], &SINGULAR), In::new(&[2], &[1.0, 1.0])];
        let r = call(&kernel, &inputs, vec![Out::provided::<f64>(&[2])]).unwrap();
        assert!(all_nan(&r[0].values::<f64>()));
    }
    // only the triangle named is read
    for lower in [true, false] {
        for assume in [
            Assume::Symmetric { lower },
            Assume::PositiveDefinite { lower },
        ] {
            let kernel = Solve::<f64> {
                parallel: false,
                mode: Provided,
                layouts: layouts(&[1]),
                assume,
                element: PhantomData,
            };
            let inputs = [In::new(&[3, 3], &half(&S, 3, lower)), In::new(&[3], &B)];
            let r = call(&kernel, &inputs, vec![Out::provided::<f64>(&[3])]).unwrap();
            assert_close(&mm(&S, &r[0].values::<f64>(), 3, 3, 1), &B);
        }
    }
    // float32
    let kernel = Solve::<f32> {
        parallel: false,
        mode: Allocated,
        layouts: layouts(&[1]),
        assume: Assume::General,
        element: PhantomData,
    };
    let a32: Vec<f32> = A.iter().map(|&v| v as f32).collect();
    let b32: Vec<f32> = B.iter().map(|&v| v as f32).collect();
    let r = call(
        &kernel,
        &[In::new(&[3, 3], &a32), In::new(&[3], &b32)],
        vec![Out::allocated::<f32>(1)],
    )
    .unwrap();
    let x: Vec<f64> = r[0].values::<f32>().into_iter().map(f64::from).collect();
    for (ax, b) in mm(&A, &x, 3, 3, 1).iter().zip(B) {
        assert!((ax - b).abs() < 1e-5);
    }
    let err = call(
        &kernel,
        &[In::new(&[3, 3], &a32), In::new(&[2], &b32[..2])],
        vec![Out::allocated::<f32>(1)],
    );
    assert!(err.unwrap_err().contains("incompatible"));
}

#[test]
fn cholesky_solve_reads_one_triangle() {
    for (lower, factor) in [
        (true, half(&L, 3, true)),
        (false, t(&half(&L, 3, true), 3, 3)),
    ] {
        for (x, mode) in cases(&factor, 3, 3) {
            let kernel = CholeskySolve::<f64> {
                parallel: false,
                mode,
                layouts: layouts(&[1]),
                lower,
                element: PhantomData,
            };
            let r = call(
                &kernel,
                &[x, In::new(&[3], &B)],
                vec![Out::new::<f64>(mode, &[3])],
            )
            .unwrap();
            assert_close(&mm(&S, &r[0].values::<f64>(), 3, 3, 1), &B);
        }
    }
}

#[test]
fn inverse() {
    for (x, mode) in cases(&A, 3, 3) {
        let kernel = MatrixInverse::<f64> {
            parallel: false,
            mode,
            layouts: layouts(&[2]),
            element: PhantomData,
        };
        let r = call(&kernel, &[x], vec![Out::new::<f64>(mode, &[3, 3])]).unwrap();
        assert_close(&mm(&A, &r[0].values::<f64>(), 3, 3, 3), &eye(3));
    }
    let kernel = MatrixInverse::<f64> {
        parallel: false,
        mode: Provided,
        layouts: layouts(&[2]),
        element: PhantomData,
    };
    let r = call(
        &kernel,
        &[In::new(&[3, 3], &A)],
        vec![Out::provided_fortran::<f64>(&[3, 3])],
    );
    assert_close(&mm(&A, &r.unwrap()[0].values::<f64>(), 3, 3, 3), &eye(3));
    let err = call(
        &kernel,
        &[In::new(&[2, 2], &SINGULAR)],
        vec![Out::provided::<f64>(&[2, 2])],
    );
    assert!(err.unwrap_err().contains("singular"));
    let err = call(
        &kernel,
        &[In::new(&[2, 3], &TALL)],
        vec![Out::provided::<f64>(&[2, 3])],
    );
    assert!(err.unwrap_err().contains("square"));
}

#[test]
fn determinants() {
    for (x, mode) in cases(&A, 3, 3) {
        let kernel = Det::<f64> {
            parallel: false,
            mode,
            layouts: layouts(&[0]),
            element: PhantomData,
        };
        let r = call(&kernel, &[x], vec![Out::new::<f64>(mode, &[])]).unwrap();
        assert_close(&r[0].values::<f64>(), &[-16.0]);
    }
    let kernel = SLogDet::<f64> {
        parallel: false,
        modes: [Provided, Allocated],
        layouts: layouts(&[0, 0]),
        element: PhantomData,
    };
    let outputs = || vec![Out::provided::<f64>(&[]), Out::allocated::<f64>(0)];
    let r = call(&kernel, &[In::new(&[3, 3], &A).t()], outputs()).unwrap();
    assert_close(
        &[r[0].values::<f64>()[0], r[1].values::<f64>()[0]],
        &[-1.0, 16f64.ln()],
    );
    let r = call(&kernel, &[In::new(&[2, 2], &SINGULAR)], outputs()).unwrap();
    assert_eq!(
        [r[0].values::<f64>()[0], r[1].values::<f64>()[0]],
        [0.0, f64::NEG_INFINITY]
    );
    let kernel = Det::<f64> {
        parallel: false,
        mode: Allocated,
        layouts: layouts(&[0]),
        element: PhantomData,
    };
    let r = call(
        &kernel,
        &[In::new(&[2, 2], &SINGULAR)],
        vec![Out::allocated::<f64>(0)],
    );
    assert_eq!(r.unwrap()[0].values::<f64>(), [0.0]);
}

/// The rows of `a` (n columns), swapped as the pivots say: row i with row piv[i].
fn pivoted(a: &[f64], n: usize, piv: &[i32]) -> Vec<f64> {
    let mut rows: Vec<Vec<f64>> = a.chunks(n).map(<[f64]>::to_vec).collect();
    for (i, &p) in piv.iter().enumerate() {
        rows.swap(i, p as usize);
    }
    rows.concat()
}

/// L (unit lower) and U from a packed LU factorization (n x n).
fn unpack(packed: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let l = (0..n * n)
        .map(|k| match (k / n).cmp(&(k % n)) {
            std::cmp::Ordering::Greater => packed[k],
            std::cmp::Ordering::Equal => 1.0,
            std::cmp::Ordering::Less => 0.0,
        })
        .collect();
    let u = (0..n * n)
        .map(|k| if k / n <= k % n { packed[k] } else { 0.0 })
        .collect();
    (l, u)
}

#[test]
fn lu_factor() {
    for (x, mode) in cases(&A, 3, 3) {
        let kernel = LUFactor::<f64> {
            parallel: false,
            modes: [mode, mode],
            layouts: layouts(&[2, 1]),
            element: PhantomData,
        };
        let outputs = vec![Out::new::<f64>(mode, &[3, 3]), Out::new::<i32>(mode, &[3])];
        let r = call(&kernel, &[x], outputs).unwrap();
        let piv = r[1].values::<i32>();
        // the largest of the first column is in row 1
        assert_eq!(piv[0], 1);
        let (l, u) = unpack(&r[0].values::<f64>(), 3);
        assert_close(&mm(&l, &u, 3, 3, 3), &pivoted(&A, 3, &piv));
    }
    let kernel = LUFactor::<f64> {
        parallel: false,
        modes: [Provided, Provided],
        layouts: layouts(&[2, 1]),
        element: PhantomData,
    };
    let outputs = vec![Out::provided::<f64>(&[2, 2]), Out::provided::<i32>(&[2])];
    let r = call(&kernel, &[In::new(&[2, 2], &SINGULAR)], outputs).unwrap();
    assert!(all_nan(&r[0].values::<f64>()));
    assert!(r[1].values::<i32>().iter().all(|&p| (0..2).contains(&p)));
}

#[test]
fn lu_in_its_three_forms() {
    for (x, mode) in cases(&A, 3, 3) {
        let kernel = LU::<f64> {
            parallel: false,
            modes: vec![mode; 3],
            layouts: layouts(&[2, 2, 2]),
            form: LuForm::Matrix,
            element: PhantomData,
        };
        let r = call(&kernel, &[x], vec![Out::new::<f64>(mode, &[3, 3]); 3]).unwrap();
        let (p, l, u) = (
            r[0].values::<f64>(),
            r[1].values::<f64>(),
            r[2].values::<f64>(),
        );
        assert_close(&mm(&p, &mm(&l, &u, 3, 3, 3), 3, 3, 3), &A);
        // L is unit lower triangular, U upper triangular
        let packed: Vec<f64> = (0..9)
            .map(|e| if e / 3 > e % 3 { l[e] } else { u[e] })
            .collect();
        assert_eq!(unpack(&packed, 3), (l, u));
    }
    // p: A = L[p] U
    let kernel = LU::<f64> {
        parallel: false,
        modes: vec![Provided, Allocated, Provided],
        layouts: layouts(&[1, 2, 2]),
        form: LuForm::Indices,
        element: PhantomData,
    };
    let outputs = vec![
        Out::provided::<i32>(&[3]),
        Out::allocated::<f64>(2),
        Out::provided::<f64>(&[3, 3]),
    ];
    let r = call(&kernel, &[In::new(&[3, 3], &A)], outputs).unwrap();
    let lu = mm(&r[1].values::<f64>(), &r[2].values::<f64>(), 3, 3, 3);
    let rows: Vec<f64> = r[0]
        .values::<i32>()
        .iter()
        .flat_map(|&i| lu[i as usize * 3..i as usize * 3 + 3].to_vec())
        .collect();
    assert_close(&rows, &A);
    // (P L, U)
    let kernel = LU::<f64> {
        parallel: false,
        modes: vec![Allocated; 2],
        layouts: layouts(&[2, 2]),
        form: LuForm::PermuteL,
        element: PhantomData,
    };
    let r = call(
        &kernel,
        &[In::new(&[3, 3], &A)],
        vec![Out::allocated::<f64>(2); 2],
    )
    .unwrap();
    assert_close(
        &mm(&r[0].values::<f64>(), &r[1].values::<f64>(), 3, 3, 3),
        &A,
    );
}

#[test]
fn pivots_to_permutations() {
    // pivots [1, 2, 2]: rows [0, 1, 2] become [1, 0, 2], then [1, 2, 0]
    for (inverse, expected) in [(true, [1i64, 2, 0]), (false, [2, 0, 1])] {
        let kernel = PivotToPermutations::<i32> {
            mode: Provided,
            layouts: layouts(&[1]),
            inverse,
            element: PhantomData,
        };
        let r = call(
            &kernel,
            &[In::new(&[3], &[1i32, 2, 2])],
            vec![Out::provided::<i64>(&[3])],
        );
        assert_eq!(r.unwrap()[0].values::<i64>(), expected);
        let kernel = PivotToPermutations::<i64> {
            mode: Allocated,
            layouts: layouts(&[1]),
            inverse,
            element: PhantomData,
        };
        let pivots = In::new(&[3], &[2i64, 2, 1]).flip(0);
        let r = call(&kernel, &[pivots], vec![Out::allocated::<i64>(1)]);
        assert_eq!(r.unwrap()[0].values::<i64>(), expected);
    }
    let kernel = PivotToPermutations::<i32> {
        mode: Provided,
        layouts: layouts(&[1]),
        inverse: true,
        element: PhantomData,
    };
    for pivots in [[0i32, 2], [-1, 1]] {
        let r = call(
            &kernel,
            &[In::new(&[2], &pivots)],
            vec![Out::provided::<i64>(&[2])],
        );
        assert!(r.unwrap_err().contains("out of range"));
    }
}

#[test]
fn eigh() {
    let mut eigenvalues = None;
    for (x, mode) in cases(&S, 3, 3) {
        let kernel = Eigh::<f64> {
            parallel: false,
            modes: vec![mode; 2],
            layouts: layouts(&[1, 2]),
            lower: true,
            vectors: true,
            element: PhantomData,
        };
        let outputs = vec![Out::new::<f64>(mode, &[3]), Out::new::<f64>(mode, &[3, 3])];
        let r = call(&kernel, &[x], outputs).unwrap();
        let (w, v) = (r[0].values::<f64>(), r[1].values::<f64>());
        assert!(w[0] <= w[1] && w[1] <= w[2]);
        let scaled: Vec<f64> = (0..9).map(|k| v[k] * w[k % 3]).collect();
        assert_close(&mm(&scaled, &t(&v, 3, 3), 3, 3, 3), &S);
        assert_close(&mm(&t(&v, 3, 3), &v, 3, 3, 3), &eye(3));
        eigenvalues = Some(w);
    }
    // the upper triangle, eigenvalues alone
    let kernel = Eigh::<f64> {
        parallel: false,
        modes: vec![Allocated],
        layouts: layouts(&[1]),
        lower: false,
        vectors: false,
        element: PhantomData,
    };
    let r = call(
        &kernel,
        &[In::new(&[3, 3], &half(&S, 3, false))],
        vec![Out::allocated::<f64>(1)],
    );
    assert_close(&r.unwrap()[0].values::<f64>(), &eigenvalues.unwrap());
}

#[test]
fn svd() {
    for (m, n, a) in [(3, 2, TALL.to_vec()), (2, 3, t(&TALL, 3, 2))] {
        let k = m.min(n);
        let mut singular_values = None;
        for (x, mode) in cases(&a, m, n) {
            let kernel = Svd::<f64> {
                parallel: false,
                modes: vec![mode; 3],
                layouts: layouts(&[2, 1, 2]),
                full_matrices: true,
                compute_uv: true,
                element: PhantomData,
            };
            let outputs = vec![
                Out::new::<f64>(mode, &[m, m]),
                Out::new::<f64>(mode, &[k]),
                Out::new::<f64>(mode, &[n, n]),
            ];
            let r = call(&kernel, &[x], outputs).unwrap();
            let (u, s, vt) = (
                r[0].values::<f64>(),
                r[1].values::<f64>(),
                r[2].values::<f64>(),
            );
            assert!(s.windows(2).all(|w| w[0] >= w[1]));
            assert_close(&mm(&t(&u, m, m), &u, m, m, m), &eye(m));
            assert_close(&mm(&vt, &t(&vt, n, n), n, n, n), &eye(n));
            // A = U[:, :k] diag(s) Vt[:k]
            let us: Vec<f64> = (0..m * k)
                .map(|e| u[e / k * m + e % k] * s[e % k])
                .collect();
            assert_close(&mm(&us, &vt[..k * n], m, k, n), &a);
            singular_values = Some(s);
        }
        let s = singular_values.unwrap();
        let kernel = Svd::<f64> {
            parallel: false,
            modes: vec![Allocated; 3],
            layouts: layouts(&[2, 1, 2]),
            full_matrices: false,
            compute_uv: true,
            element: PhantomData,
        };
        let r = call(
            &kernel,
            &[In::new(&[m, n], &a)],
            vec![
                Out::allocated::<f64>(2),
                Out::allocated::<f64>(1),
                Out::allocated::<f64>(2),
            ],
        )
        .unwrap();
        assert_eq!(
            (r[0].shape.clone(), r[2].shape.clone()),
            (vec![m as i64, k as i64], vec![k as i64, n as i64])
        );
        let u = r[0].values::<f64>();
        let us: Vec<f64> = (0..m * k).map(|e| u[e] * s[e % k]).collect();
        assert_close(&mm(&us, &r[2].values::<f64>(), m, k, n), &a);
        let kernel = Svd::<f64> {
            parallel: false,
            modes: vec![Provided],
            layouts: layouts(&[1]),
            full_matrices: true,
            compute_uv: false,
            element: PhantomData,
        };
        let r = call(
            &kernel,
            &[In::new(&[m, n], &a)],
            vec![Out::provided::<f64>(&[k])],
        );
        assert_close(&r.unwrap()[0].values::<f64>(), &s);
    }
}

#[test]
fn qr() {
    for (m, n, a) in [(3, 2, TALL.to_vec()), (2, 3, t(&TALL, 3, 2))] {
        let k = m.min(n);
        let mut full_r = None;
        for (x, mode) in cases(&a, m, n) {
            let kernel = Qr::<f64> {
                parallel: false,
                modes: vec![mode; 2],
                layouts: layouts(&[2, 2]),
                mode: QrMode::Full,
                element: PhantomData,
            };
            let r = call(
                &kernel,
                &[x],
                vec![
                    Out::new::<f64>(mode, &[m, m]),
                    Out::new::<f64>(mode, &[m, n]),
                ],
            )
            .unwrap();
            let (q, upper) = (r[0].values::<f64>(), r[1].values::<f64>());
            assert_close(&mm(&q, &upper, m, m, n), &a);
            assert_close(&mm(&t(&q, m, m), &q, m, m, m), &eye(m));
            assert!((0..m * n).all(|e| e / n <= e % n || upper[e] == 0.0));
            full_r = Some(upper);
        }
        let full_r = full_r.unwrap();
        let kernel = Qr::<f64> {
            parallel: false,
            modes: vec![Allocated; 2],
            layouts: layouts(&[2, 2]),
            mode: QrMode::Economic,
            element: PhantomData,
        };
        let r = call(
            &kernel,
            &[In::new(&[m, n], &a)],
            vec![Out::allocated::<f64>(2); 2],
        )
        .unwrap();
        assert_close(
            &mm(&r[0].values::<f64>(), &r[1].values::<f64>(), m, k, n),
            &a,
        );
        assert_close(&r[1].values::<f64>(), &full_r[..k * n]);
        let kernel = Qr::<f64> {
            parallel: false,
            modes: vec![Provided],
            layouts: layouts(&[2]),
            mode: QrMode::R,
            element: PhantomData,
        };
        let r = call(
            &kernel,
            &[In::new(&[m, n], &a)],
            vec![Out::provided::<f64>(&[m, n])],
        );
        assert_close(&r.unwrap()[0].values::<f64>(), &full_r);
    }
}

#[test]
fn pinv() {
    for (x, mode) in cases(&TALL, 3, 2) {
        let kernel = MatrixPinv::<f64> {
            parallel: false,
            mode,
            layouts: layouts(&[2]),
            hermitian: false,
            element: PhantomData,
        };
        let r = call(&kernel, &[x], vec![Out::new::<f64>(mode, &[2, 3])]).unwrap();
        let p = r[0].values::<f64>();
        assert_close(&mm(&mm(&TALL, &p, 3, 2, 3), &TALL, 3, 3, 2), &TALL);
        assert_close(&mm(&p, &TALL, 2, 3, 2), &eye(2));
    }
    let kernel = MatrixPinv::<f64> {
        parallel: false,
        mode: Provided,
        layouts: layouts(&[2]),
        hermitian: true,
        element: PhantomData,
    };
    let r = call(
        &kernel,
        &[In::new(&[3, 3], &half(&S, 3, true))],
        vec![Out::provided::<f64>(&[3, 3])],
    );
    assert_close(&mm(&S, &r.unwrap()[0].values::<f64>(), 3, 3, 3), &eye(3));
}

#[test]
fn empty_matrices() {
    let empty = |m: usize, n: usize| In::new::<f64>(&[m, n], &[]);
    let r = call(
        &Det::<f64> {
            parallel: false,
            mode: Provided,
            layouts: layouts(&[0]),
            element: PhantomData,
        },
        &[empty(0, 0)],
        vec![Out::provided::<f64>(&[])],
    );
    assert_eq!(r.unwrap()[0].values::<f64>(), [1.0]);
    let r = call(
        &MatrixInverse::<f64> {
            parallel: false,
            mode: Allocated,
            layouts: layouts(&[2]),
            element: PhantomData,
        },
        &[empty(0, 0)],
        vec![Out::allocated::<f64>(2)],
    );
    assert_eq!(r.unwrap()[0].shape, [0, 0]);
    let kernel = Eigh::<f64> {
        parallel: false,
        modes: vec![Allocated; 2],
        layouts: layouts(&[1, 2]),
        lower: true,
        vectors: true,
        element: PhantomData,
    };
    let r = call(
        &kernel,
        &[empty(0, 0)],
        vec![Out::allocated::<f64>(1), Out::allocated::<f64>(2)],
    );
    assert_eq!(r.unwrap()[1].shape, [0, 0]);
    let kernel = Svd::<f64> {
        parallel: false,
        modes: vec![Allocated; 3],
        layouts: layouts(&[2, 1, 2]),
        full_matrices: true,
        compute_uv: true,
        element: PhantomData,
    };
    let outputs = vec![
        Out::allocated::<f64>(2),
        Out::allocated::<f64>(1),
        Out::allocated::<f64>(2),
    ];
    let r = call(&kernel, &[empty(3, 0)], outputs).unwrap();
    assert_eq!(
        (r[0].values::<f64>(), r[2].shape.clone()),
        (eye(3), vec![0, 0])
    );
    let kernel = Qr::<f64> {
        parallel: false,
        modes: vec![Allocated; 2],
        layouts: layouts(&[2, 2]),
        mode: QrMode::Full,
        element: PhantomData,
    };
    let r = call(&kernel, &[empty(3, 0)], vec![Out::allocated::<f64>(2); 2]).unwrap();
    assert_eq!(
        (r[0].values::<f64>(), r[1].shape.clone()),
        (eye(3), vec![3, 0])
    );
    let kernel = MatrixPinv::<f64> {
        parallel: false,
        mode: Allocated,
        layouts: layouts(&[2]),
        hermitian: false,
        element: PhantomData,
    };
    let r = call(&kernel, &[empty(0, 3)], vec![Out::allocated::<f64>(2)]);
    assert_eq!(r.unwrap()[0].shape, [3, 0]);
}
