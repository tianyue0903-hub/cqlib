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

//! Quantum Delay Operation
//!
//! This module provides the [`DelayOp`] type, representing a time delay
//! in quantum circuit execution. Used primarily for timing synchronization
//! in hardware-aware circuit representations (e.g., QCIS format).

/// A time delay operation for quantum circuits.
///
/// The delay duration is specified as an integer `t`, where the actual time
/// is calculated as `t * 0.5 nanoseconds`. This granularity aligns with
/// common quantum control hardware timing resolutions.
///
/// # Examples
///
/// ```
/// use cqlib_core::circuit::gate::DelayOp;
///
/// // Create a 0.5ns delay (t=1)
/// let short_delay = DelayOp {};
///
/// // For a 10ns delay, t would be 20
/// // delay_time = 20 * 0.5ns = 10ns
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DelayOp {}
