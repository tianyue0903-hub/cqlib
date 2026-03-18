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

use crate::circuit::Circuit;
use crate::qis::Hamiltonian;

pub mod error;
pub mod zne_mitigation;
pub mod virtual_distillation;

pub type Estimator<'a> =
    dyn Fn(&Circuit, Option<&Hamiltonian>, Option<usize>) -> (f64, f64) + 'a;

pub use error::ErrorMitigationError;
pub use virtual_distillation::VirtualDistillation;
pub use zne_mitigation::{ExtrapolateMethod, ZNEMitigation};
