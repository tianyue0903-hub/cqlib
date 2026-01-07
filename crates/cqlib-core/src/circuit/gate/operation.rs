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

use std::fmt;

#[repr(u8)]
#[derive(Eq, Hash, PartialEq, Debug, Clone, Copy)]
pub enum Operation {
    Barrier,
    Measure,
    Reset,
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Barrier => write!(f, "Barrier"),
            Self::Measure => write!(f, "Measure"),
            Self::Reset => write!(f, "Reset"),
        }
    }
}
