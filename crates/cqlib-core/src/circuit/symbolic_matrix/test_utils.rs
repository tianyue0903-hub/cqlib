// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Shared test utilities for the `symbolic_matrix` submodule.

use ndarray::Array2;
use num_complex::Complex64;

/// Asserts that two complex matrices are element-wise approximately equal.
///
/// Panics if the shapes differ or if any element differs by more than `eps`.
pub fn assert_matrix_approx_eq(actual: &Array2<Complex64>, expected: &Array2<Complex64>, eps: f64) {
    assert_eq!(actual.shape(), expected.shape());
    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (*a - *e).norm();
        assert!(
            diff < eps,
            "matrix element {idx} differs: got {a}, expected {e}, diff {diff}"
        );
    }
}
