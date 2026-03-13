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

use super::canonical::{CanonicalGate, CanonicalOp, approx_zero};
use super::dag::SegmentDag;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ParityKey(Vec<usize>);

impl ParityKey {
    fn basis(bit: usize) -> Self {
        Self(vec![bit])
    }

    fn xor(&self, other: &Self) -> Self {
        let mut out = Vec::with_capacity(self.0.len() + other.0.len());
        let mut i = 0usize;
        let mut j = 0usize;
        while i < self.0.len() || j < other.0.len() {
            match (self.0.get(i), other.0.get(j)) {
                (Some(&lhs), Some(&rhs)) if lhs == rhs => {
                    i += 1;
                    j += 1;
                }
                (Some(&lhs), Some(&rhs)) if lhs < rhs => {
                    out.push(lhs);
                    i += 1;
                }
                (Some(_), Some(&rhs)) => {
                    out.push(rhs);
                    j += 1;
                }
                (Some(&lhs), None) => {
                    out.push(lhs);
                    i += 1;
                }
                (None, Some(&rhs)) => {
                    out.push(rhs);
                    j += 1;
                }
                (None, None) => break,
            }
        }
        Self(out)
    }
}

#[derive(Debug, Clone)]
struct AffineParity {
    key: ParityKey,
    constant: bool,
}

impl AffineParity {
    fn basis(bit: usize) -> Self {
        Self {
            key: ParityKey::basis(bit),
            constant: false,
        }
    }

    fn xor_assign(&mut self, other: &Self) {
        self.key = self.key.xor(&other.key);
        self.constant ^= other.constant;
    }

    fn sign(&self) -> f64 {
        if self.constant { -1.0 } else { 1.0 }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PhasePolynomial {
    active_qubits: Vec<usize>,
    phases: HashMap<ParityKey, f64>,
    gates: Vec<CanonicalOp>,
}

impl PhasePolynomial {
    pub(crate) fn optimize_component(
        dag: &SegmentDag,
        component: &[usize],
        tol: f64,
    ) -> Option<Vec<CanonicalOp>> {
        let original: Vec<CanonicalOp> = component
            .iter()
            .map(|&node_id| dag.node(node_id).op.clone())
            .collect();
        if original.len() < 2 {
            return None;
        }

        let poly = Self::build(&original, tol);
        let rewritten = poly.synthesize(tol);
        if rewritten.len() < original.len() {
            Some(rewritten)
        } else {
            None
        }
    }

    fn build(component: &[CanonicalOp], tol: f64) -> Self {
        let mut active_qubits: Vec<usize> = component
            .iter()
            .flat_map(|op| op.logical_qubits.iter().copied())
            .collect();
        active_qubits.sort_unstable();
        active_qubits.dedup();

        let local_index: HashMap<usize, usize> = active_qubits
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, logical)| (logical, idx))
            .collect();
        let mut parities = initial_parities(&active_qubits, &local_index);
        let mut phases = HashMap::<ParityKey, f64>::new();
        let mut gates = Vec::new();

        for op in component {
            match op.gate {
                CanonicalGate::X => {
                    let target = op.logical_qubits[0];
                    if let Some(parity) = parities.get_mut(&target) {
                        parity.constant = !parity.constant;
                    }
                    gates.push(op.clone());
                }
                CanonicalGate::CX => {
                    let control = op.logical_qubits[0];
                    let target = op.logical_qubits[1];
                    let Some(control_parity) = parities.get(&control).cloned() else {
                        continue;
                    };
                    if let Some(target_parity) = parities.get_mut(&target) {
                        target_parity.xor_assign(&control_parity);
                    }
                    gates.push(op.clone());
                }
                CanonicalGate::RZ => {
                    let target = op.logical_qubits[0];
                    let Some(parity) = parities.get(&target) else {
                        continue;
                    };
                    *phases.entry(parity.key.clone()).or_insert(0.0) +=
                        parity.sign() * op.theta_value();
                }
                CanonicalGate::H => {}
            }
        }

        phases.retain(|_, phase| !approx_zero(*phase, tol));
        Self {
            active_qubits,
            phases,
            gates,
        }
    }

    fn synthesize(&self, tol: f64) -> Vec<CanonicalOp> {
        if self.phases.is_empty() {
            return self.gates.clone();
        }

        let local_index: HashMap<usize, usize> = self
            .active_qubits
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, logical)| (logical, idx))
            .collect();
        let mut parities = initial_parities(&self.active_qubits, &local_index);
        let mut emitted = HashSet::<ParityKey>::new();
        let mut out = Vec::new();

        for &logical in &self.active_qubits {
            emit_phase_if_ready(
                &mut out,
                &mut emitted,
                &self.phases,
                &parities,
                logical,
                tol,
            );
        }

        for gate in &self.gates {
            out.push(gate.clone());
            match gate.gate {
                CanonicalGate::X => {
                    let target = gate.logical_qubits[0];
                    if let Some(parity) = parities.get_mut(&target) {
                        parity.constant = !parity.constant;
                    }
                    emit_phase_if_ready(
                        &mut out,
                        &mut emitted,
                        &self.phases,
                        &parities,
                        target,
                        tol,
                    );
                }
                CanonicalGate::CX => {
                    let control = gate.logical_qubits[0];
                    let target = gate.logical_qubits[1];
                    let Some(control_parity) = parities.get(&control).cloned() else {
                        continue;
                    };
                    if let Some(target_parity) = parities.get_mut(&target) {
                        target_parity.xor_assign(&control_parity);
                    }
                    emit_phase_if_ready(
                        &mut out,
                        &mut emitted,
                        &self.phases,
                        &parities,
                        target,
                        tol,
                    );
                }
                CanonicalGate::RZ | CanonicalGate::H => {}
            }
        }

        out
    }
}

fn initial_parities(
    active_qubits: &[usize],
    local_index: &HashMap<usize, usize>,
) -> HashMap<usize, AffineParity> {
    active_qubits
        .iter()
        .copied()
        .map(|logical| {
            (
                logical,
                AffineParity::basis(
                    *local_index
                        .get(&logical)
                        .expect("active qubit missing local parity index"),
                ),
            )
        })
        .collect()
}

fn emit_phase_if_ready(
    out: &mut Vec<CanonicalOp>,
    emitted: &mut HashSet<ParityKey>,
    phases: &HashMap<ParityKey, f64>,
    parities: &HashMap<usize, AffineParity>,
    logical: usize,
    tol: f64,
) {
    let Some(parity) = parities.get(&logical) else {
        return;
    };
    let Some(phase) = phases.get(&parity.key) else {
        return;
    };
    if emitted.contains(&parity.key) || approx_zero(*phase, tol) {
        return;
    }
    emitted.insert(parity.key.clone());
    out.push(CanonicalOp::rz(logical, parity.sign() * phase));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::optimization::clifford_rz::dag::SegmentDag;

    #[test]
    fn test_phase_polynomial_merges_across_x() {
        let ops = vec![
            CanonicalOp::x(0),
            CanonicalOp::rz(0, 0.3),
            CanonicalOp::x(0),
            CanonicalOp::rz(0, 0.3),
        ];
        let dag = SegmentDag::from_ops(&ops);
        let rewritten = PhasePolynomial::optimize_component(&dag, &[0, 1, 2, 3], 1e-10).unwrap();
        assert_eq!(rewritten, vec![CanonicalOp::x(0), CanonicalOp::x(0)]);
    }

    #[test]
    fn test_phase_polynomial_merges_across_cx_network() {
        let ops = vec![
            CanonicalOp::rz(1, 0.2),
            CanonicalOp::cx(0, 1),
            CanonicalOp::cx(1, 0),
            CanonicalOp::cx(0, 1),
            CanonicalOp::rz(0, 0.4),
        ];
        let dag = SegmentDag::from_ops(&ops);
        let rewritten = PhasePolynomial::optimize_component(&dag, &[0, 1, 2, 3, 4], 1e-10).unwrap();
        assert!(rewritten.len() < ops.len());
    }
}
