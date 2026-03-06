// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025-2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use ndarray::{Array2, arr2};
use num_complex::Complex64;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pauli {
    X,
    Y,
    Z,
    I,
}

impl fmt::Display for Pauli {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            Pauli::I => "I",
            Pauli::X => "X",
            Pauli::Y => "Y",
            Pauli::Z => "Z",
        };
        write!(f, "{}", s)
    }
}

impl Pauli {
    pub fn to_symplectic(&self) -> (bool, bool) {
        match self {
            Pauli::I => (false, false),
            Pauli::X => (true, false),
            Pauli::Y => (true, true),
            Pauli::Z => (false, true),
        }
    }

    // 获取矩阵形式 (用于调试或小系统)
    // 返回 2x2 复数矩阵
    pub fn to_matrix(&self) -> Array2<Complex64> {
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let i = Complex64::new(0.0, 1.0);
        let neg_one = Complex64::new(-1.0, 0.0);
        let neg_i = Complex64::new(0.0, -1.0);

        match self {
            Pauli::I => arr2(&[[one, zero], [zero, one]]),
            Pauli::X => arr2(&[[zero, one], [one, zero]]),
            Pauli::Y => arr2(&[[zero, neg_i], [i, zero]]),
            Pauli::Z => arr2(&[[one, zero], [zero, neg_one]]),
        }
    }

    // 4. 核心功能：乘法逻辑 (单比特)
    // 返回 (结果Pauli, 相位因子)
    // 例如: X * Z = Y (相位 -i, 或者根据你的约定是 i)
    pub fn multiply(&self, other: Pauli) -> (Pauli, Complex64) {
        // 这里可以硬编码查表，速度最快
        let i = Complex64::new(0.0, 1.0);
        let one = Complex64::new(1.0, 0.0);
        let neg_i = Complex64::new(0.0, -1.0);

        match (self, other) {
            (Pauli::I, p) => (p, one),
            (p, Pauli::I) => (*p, one),
            (Pauli::X, Pauli::X) => (Pauli::I, one),
            (Pauli::Y, Pauli::Y) => (Pauli::I, one),
            (Pauli::Z, Pauli::Z) => (Pauli::I, one),

            (Pauli::X, Pauli::Y) => (Pauli::Z, i), // XY = iZ
            (Pauli::X, Pauli::Z) => (Pauli::Y, neg_i), // XZ = -iY

            (Pauli::Y, Pauli::X) => (Pauli::Z, neg_i), // YX = -iZ
            (Pauli::Y, Pauli::Z) => (Pauli::X, i),     // YZ = iX

            (Pauli::Z, Pauli::X) => (Pauli::Y, i), // ZX = iY
            (Pauli::Z, Pauli::Y) => (Pauli::X, neg_i), // ZY = -iX
        }
    }
}
