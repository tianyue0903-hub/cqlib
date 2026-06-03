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

//! SABRE layout refinement and routing core.
//!
//! SABRE is a SWAP-based bidirectional heuristic search for mapping logical
//! qubits onto a device with limited two-qubit connectivity. The algorithm
//! incrementally routes executable two-qubit operations, scores candidate SWAPs
//! with current and lookahead interaction distances, and uses forward/backward
//! trial routing to improve the initial layout before final routing.
//!
//! This implementation follows the original SABRE structure and incorporates
//! selected LightSABRE/Qiskit-style production enhancements: deterministic
//! multi-trial selection, relative/delta layer scoring, release-valve fallback,
//! trial-level parallelism, control-flow body restoration, and routing-quality
//! tie-breakers. It is not a complete implementation of every LightSABRE
//! heuristic; depth and critical-path scoring are intentionally not used as
//! default swap-selection terms without benchmark coverage.
//!
//! This module is intentionally independent from compiler workflow selection.
//! It exposes reusable SABRE building blocks, but it does not decide whether a
//! workflow should prefer trivial, greedy, VF2, SABRE, or another layout and
//! routing strategy.
//!
//! # Reference
//!
//! Gushu Li, Yufei Ding, and Yuan Xie, "Tackling the Qubit Mapping Problem for
//! NISQ-Era Quantum Devices", ASPLOS 2019. DOI: 10.1145/3297858.3304023.
//! arXiv: 1809.02573.
//!
//! Shaohan Zou, Matthew Treinish, Kevin Hartman, Davide Ivrii, and John Lishman,
//! "LightSABRE: A Lightweight and Enhanced SABRE Algorithm", arXiv: 2409.08368,
//! 2024.
//!
//! # Entry Points
//!
//! - [`sabre_refine_layout`] evaluates deterministic and randomized initial
//!   layout candidates with forward/backward refinement and returns the best
//!   layout according to final-route SWAP count plus the supplied layout
//!   objective as a tie-breaker.
//! - [`sabre_route`] routes a circuit from a supplied initial layout and returns
//!   a physical circuit with inserted SWAP operations, the final layout, and
//!   diagnostics.
//! - [`sabre_layout_and_route`] combines layout refinement and final routing for
//!   callers that want the complete SABRE path.
//!
//! # Example
//!
//! ```
//! use cqlib_core::circuit::{Circuit, Qubit};
//! use cqlib_core::compiler::sabre::{SabreConfig, sabre_route};
//! use cqlib_core::device::{Device, Layout, LogicalQubit, PhysicalQubit, Topology};
//! use std::collections::{BTreeMap, HashSet};
//!
//! let physical = vec![
//!     PhysicalQubit::new(0),
//!     PhysicalQubit::new(1),
//!     PhysicalQubit::new(2),
//! ];
//! let topology = Topology::new(
//!     physical.clone(),
//!     vec![
//!         (PhysicalQubit::new(0), PhysicalQubit::new(1), "cx".to_string()),
//!         (PhysicalQubit::new(1), PhysicalQubit::new(2), "cx".to_string()),
//!     ],
//! )?;
//! let device = Device::new(
//!     "line3",
//!     physical.iter().copied().collect::<HashSet<_>>(),
//!     topology,
//! )?;
//!
//! let logical = vec![LogicalQubit::new(0), LogicalQubit::new(1)];
//! let mapping = BTreeMap::from([
//!     (LogicalQubit::new(0), PhysicalQubit::new(0)),
//!     (LogicalQubit::new(1), PhysicalQubit::new(2)),
//! ]);
//! let layout = Layout::new(logical, physical, Some(mapping))?;
//!
//! let mut circuit = Circuit::new(2);
//! circuit.cx(Qubit::new(0), Qubit::new(1))?;
//!
//! let config = SabreConfig {
//!     routing_trials: 1,
//!     seed: Some(7),
//!     ..SabreConfig::default()
//! };
//! let routed = sabre_route(&circuit, &device, &layout, &config)?;
//!
//! assert_eq!(routed.diagnostics.trials_evaluated, 1);
//! assert!(routed.swap_count <= 1);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod dag;
mod heuristic;
mod layer;
mod refine;
mod routing;

pub use heuristic::{SabreConfig, SabreHeuristicConfig, SabreTrialObjective};
pub use refine::{
    SabreCompileResult, sabre_layout_and_route, sabre_refine_layout, sabre_refine_layout_prepared,
};
pub use routing::{SabreRoutingDiagnostics, SabreRoutingResult, sabre_route};

#[cfg(test)]
#[path = "./sabre_test.rs"]
mod sabre_test;
