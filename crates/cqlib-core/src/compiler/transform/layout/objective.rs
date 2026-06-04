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

//! Layout scoring objective shared by layout algorithms.

use super::analysis::CircuitLayoutAnalysis;
use super::physical::PhysicalLayoutGraph;
use crate::compiler::CompilerError;
use crate::device::Layout;
use std::collections::BTreeMap;

/// Weighted scoring objective used to rank candidate layouts.
#[derive(Debug, Clone, PartialEq)]
pub struct LayoutObjective {
    /// Weight for weighted logical interaction distance.
    pub distance_weight: f64,
    /// Weight for directly adjacent interactions whose observed operation
    /// direction is unsupported by the directed coupling graph.
    pub direction_weight: f64,
    /// Weight for known two-qubit gate error rates.
    pub two_qubit_error_weight: f64,
    /// Weight for known readout error rates.
    pub readout_error_weight: f64,
}

impl LayoutObjective {
    /// Creates a topology-only objective.
    pub fn topology_only() -> Self {
        Self {
            distance_weight: 1.0,
            direction_weight: 1.0,
            two_qubit_error_weight: 0.0,
            readout_error_weight: 0.0,
        }
    }

    /// Creates a fidelity-aware objective when the physical graph has useful
    /// calibration data, otherwise falls back to topology-only scoring.
    pub fn auto_from_physical(physical: &PhysicalLayoutGraph) -> Self {
        if physical.has_fidelity_data() {
            Self::fidelity_aware()
        } else {
            Self::topology_only()
        }
    }

    /// Creates a fidelity-aware objective and rejects physical graphs that do
    /// not contain usable calibration data.
    pub fn fidelity_required(physical: &PhysicalLayoutGraph) -> Result<Self, CompilerError> {
        if physical.has_fidelity_data() {
            Ok(Self::fidelity_aware())
        } else {
            Err(CompilerError::InvalidInput(
                "layout fidelity scoring was requested but the device has no usable fidelity data"
                    .to_string(),
            ))
        }
    }

    /// Creates the default fidelity-aware objective.
    pub fn fidelity_aware() -> Self {
        Self {
            distance_weight: 1.0,
            direction_weight: 1.0,
            two_qubit_error_weight: 10.0,
            readout_error_weight: 1.0,
        }
    }

    /// Returns whether any fidelity/error term can affect scoring.
    pub fn uses_fidelity(&self) -> bool {
        self.two_qubit_error_weight != 0.0 || self.readout_error_weight != 0.0
    }

    /// Scores a candidate layout against circuit and physical-graph analysis.
    pub fn score_layout(
        &self,
        analysis: &CircuitLayoutAnalysis,
        physical: &PhysicalLayoutGraph,
        layout: &Layout,
    ) -> Result<LayoutScore, CompilerError> {
        validate_objective_weight(self.distance_weight, "distance_weight")?;
        validate_objective_weight(self.direction_weight, "direction_weight")?;
        validate_objective_weight(self.two_qubit_error_weight, "two_qubit_error_weight")?;
        validate_objective_weight(self.readout_error_weight, "readout_error_weight")?;

        let mut distance = 0.0;
        let mut direction = 0.0;
        let mut two_qubit_error = 0.0;

        for interaction in analysis.interactions.interactions() {
            let left_physical = layout.get_physical(interaction.left).ok_or_else(|| {
                CompilerError::InvariantViolation(format!(
                    "layout does not map logical qubit {}",
                    interaction.left
                ))
            })?;
            let right_physical = layout.get_physical(interaction.right).ok_or_else(|| {
                CompilerError::InvariantViolation(format!(
                    "layout does not map logical qubit {}",
                    interaction.right
                ))
            })?;
            let Some(pair_distance) = physical.distance(left_physical, right_physical) else {
                return Err(CompilerError::InvalidInput(format!(
                    "physical qubits {} and {} are disconnected in the usable topology",
                    left_physical, right_physical
                )));
            };

            distance += interaction.weight * f64::from(pair_distance);

            if pair_distance == 1 {
                if interaction.directed_weight_left_to_right > 0.0
                    && !physical.supports_directed_coupling(left_physical, right_physical)
                {
                    direction += interaction.directed_weight_left_to_right;
                }
                if interaction.directed_weight_right_to_left > 0.0
                    && !physical.supports_directed_coupling(right_physical, left_physical)
                {
                    direction += interaction.directed_weight_right_to_left;
                }
                if self.two_qubit_error_weight != 0.0 {
                    if let Some(error) =
                        physical.two_qubit_error_undirected(left_physical, right_physical)
                    {
                        two_qubit_error += interaction.weight * error;
                    }
                }
            }
        }

        let readout_error = if self.readout_error_weight == 0.0 {
            0.0
        } else {
            score_readout_error(analysis, physical, layout)?
        };

        let total = self.distance_weight * distance
            + self.direction_weight * direction
            + self.two_qubit_error_weight * two_qubit_error
            + self.readout_error_weight * readout_error;

        Ok(LayoutScore {
            total,
            distance,
            direction,
            two_qubit_error,
            readout_error,
            used_fidelity: self.uses_fidelity(),
        })
    }
}

impl Default for LayoutObjective {
    fn default() -> Self {
        Self::topology_only()
    }
}

/// Breakdown of one layout score.
#[derive(Debug, Clone, PartialEq)]
pub struct LayoutScore {
    /// Weighted sum according to [`LayoutObjective`].
    pub total: f64,
    /// Raw weighted-distance component.
    pub distance: f64,
    /// Raw direction-mismatch component.
    pub direction: f64,
    /// Raw two-qubit error component.
    pub two_qubit_error: f64,
    /// Raw readout error component.
    pub readout_error: f64,
    /// Whether the objective was configured to use fidelity terms.
    pub used_fidelity: bool,
}

fn score_readout_error(
    analysis: &CircuitLayoutAnalysis,
    physical: &PhysicalLayoutGraph,
    layout: &Layout,
) -> Result<f64, CompilerError> {
    let mut activity = analysis.interactions.logical_activity();
    if activity.is_empty() {
        activity = analysis
            .logical_qubits
            .iter()
            .copied()
            .map(|logical| (logical, 1.0))
            .collect::<BTreeMap<_, _>>();
    }

    let mut readout_error = 0.0;
    for logical in &analysis.logical_qubits {
        let physical_qubit = layout.get_physical(*logical).ok_or_else(|| {
            CompilerError::InvariantViolation(format!(
                "layout does not map logical qubit {logical}"
            ))
        })?;
        if let Some(error) = physical.readout_error(physical_qubit) {
            readout_error += activity.get(logical).copied().unwrap_or(1.0) * error;
        }
    }
    Ok(readout_error)
}

fn validate_objective_weight(value: f64, name: &str) -> Result<(), CompilerError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(CompilerError::InvalidInput(format!(
            "layout objective {name} must be finite and non-negative"
        )))
    }
}
