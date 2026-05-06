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

//! Equivalence validation for knowledge-base rewrite rules.
//!
//! Rule definitions live in [`super::rule`].  This module validates a rule by
//! comparing the symbolic unitary matrices of its left- and right-hand sides,
//! with numerical sampling as a fallback for identities the symbolic
//! simplifier cannot prove directly.

use crate::circuit::error::{CircuitError, ParameterError};
use crate::circuit::symbolic_matrix::{
    SymbolicComplex, SymbolicMatrix, apply_gate_to_matrix_num, apply_standard_gate_to_matrix,
    evaluate_symbolic_matrix, symbolic_eye, symbolic_matrices_equivalent,
};
use crate::circuit::{Instruction, Parameter, ParameterValue, StandardGate};
use crate::compiler::knowledge::rule::{Condition, Rule, RuleItem};
use ndarray::Array2;
use ndarray::parallel::prelude::*;
use num_complex::Complex64;
use rand::Rng;
use smallvec::SmallVec;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub enum RuleEquivalenceMode {
    StrictMatrix,
    UpToGlobalPhase,
}

#[derive(Debug, Clone, Copy)]
pub struct RuleEquivalenceOptions {
    pub num_bindings: usize,
    pub tolerance: f64,
    pub mode: RuleEquivalenceMode,
}

impl RuleEquivalenceOptions {
    pub fn up_to_global_phase(num_bindings: usize, tolerance: f64) -> Self {
        Self {
            num_bindings,
            tolerance,
            mode: RuleEquivalenceMode::UpToGlobalPhase,
        }
    }

    pub fn strict_matrix(num_bindings: usize, tolerance: f64) -> Self {
        Self {
            num_bindings,
            tolerance,
            mode: RuleEquivalenceMode::StrictMatrix,
        }
    }
}

/// Result of verifying a rule via symbolic matrix comparison.
#[derive(Debug)]
pub enum VerifyResult {
    /// Symbolic matrices are equal after simplification under the selected mode.
    SymbolicEqual,
    /// Symbolic matrices differ structurally but are numerically equal at all
    /// checked parameter bindings under the selected comparison mode.
    NumericallyEqual { num_bindings: usize },
    /// The rule failed verification.
    Fail(VerifyFailure),
    /// Could not verify (e.g., cannot generate satisfying bindings).
    Inconclusive { reason: String },
}

/// Details of a verification failure.
#[derive(Debug)]
pub struct VerifyFailure {
    /// Maximum element-wise difference found.
    pub max_diff: f64,
    /// Parameter bindings that caused the failure.
    pub bindings: HashMap<String, f64>,
}

/// Errors during verification setup (not verification failure).
#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    #[error("unsupported instruction pattern: {0}")]
    UnsupportedPattern(String),
    #[error("parameter error: {0}")]
    Parameter(#[from] ParameterError),
    #[error("circuit error: {0}")]
    Circuit(#[from] CircuitError),
}

impl Rule {
    /// Verify this rule by comparing the symbolic unitary matrices of the LHS
    /// and RHS up to global phase.
    pub fn verify(&self, num_bindings: usize, tolerance: f64) -> Result<VerifyResult, VerifyError> {
        verify_rule_equivalence(
            self,
            RuleEquivalenceOptions::up_to_global_phase(num_bindings, tolerance),
        )
    }

    /// Verify this rule with strict element-wise matrix equality.
    ///
    /// Unlike [`Rule::verify`], this method does not quotient out a global
    /// phase. Use it when compiler rewrites must preserve the exact unitary
    /// matrix, including global phase.
    pub fn verify_strict_matrix(
        &self,
        num_bindings: usize,
        tolerance: f64,
    ) -> Result<VerifyResult, VerifyError> {
        verify_rule_equivalence(
            self,
            RuleEquivalenceOptions::strict_matrix(num_bindings, tolerance),
        )
    }
}

pub fn verify_rule_equivalence(
    rule: &Rule,
    options: RuleEquivalenceOptions,
) -> Result<VerifyResult, VerifyError> {
    let num_qubits = rule.num_qubits();

    let (lhs, rhs) = rayon::join(
        || build_rule_item_matrix(&rule.operations, num_qubits),
        || build_rule_item_matrix(&rule.target, num_qubits),
    );
    let lhs = lhs?;
    let rhs = rhs?;

    let (lhs, rhs) = rayon::join(|| simplify_matrix(&lhs), || simplify_matrix(&rhs));
    let lhs = lhs?;
    let rhs = rhs?;

    if symbolic_stage_passes(&lhs, &rhs, options.mode)? {
        return Ok(VerifyResult::SymbolicEqual);
    }

    verify_numerically(rule, &lhs, &rhs, options)
}

fn symbolic_stage_passes(
    lhs: &SymbolicMatrix,
    rhs: &SymbolicMatrix,
    mode: RuleEquivalenceMode,
) -> Result<bool, VerifyError> {
    match mode {
        RuleEquivalenceMode::StrictMatrix => Ok(lhs == rhs),
        RuleEquivalenceMode::UpToGlobalPhase => Ok(symbolic_matrices_equivalent(lhs, rhs)?),
    }
}

fn verify_numerically(
    rule: &Rule,
    lhs: &SymbolicMatrix,
    rhs: &SymbolicMatrix,
    options: RuleEquivalenceOptions,
) -> Result<VerifyResult, VerifyError> {
    let free_symbols = rule.collect_free_symbols();
    if free_symbols.is_empty() {
        let (lhs_num, rhs_num) = rayon::join(
            || evaluate_symbolic_matrix(lhs, &None),
            || evaluate_symbolic_matrix(rhs, &None),
        );
        let lhs_num = lhs_num?;
        let rhs_num = rhs_num?;
        let diff = matrix_diff(&lhs_num, &rhs_num, options.mode);
        if diff < options.tolerance {
            return Ok(VerifyResult::NumericallyEqual { num_bindings: 1 });
        }
        return Ok(VerifyResult::Fail(VerifyFailure {
            max_diff: diff,
            bindings: HashMap::new(),
        }));
    }

    let symbol_names: Vec<&str> = free_symbols.iter().map(|s| s.as_str()).collect();
    let bindings = generate_satisfying_bindings(
        &symbol_names,
        rule.conditions.as_deref(),
        options.num_bindings,
    );

    if bindings.is_empty() {
        return Ok(VerifyResult::Inconclusive {
            reason: "could not generate parameter bindings satisfying conditions".to_string(),
        });
    }

    let mut overall_max_diff = 0.0_f64;
    let mut failing_bindings = HashMap::new();

    for binding in &bindings {
        let bindings_ref: Option<HashMap<&str, f64>> =
            Some(binding.iter().map(|(k, &v)| (k.as_str(), v)).collect());
        let (lhs_num, rhs_num) = rayon::join(
            || evaluate_symbolic_matrix(lhs, &bindings_ref),
            || evaluate_symbolic_matrix(rhs, &bindings_ref),
        );
        let lhs_num = lhs_num?;
        let rhs_num = rhs_num?;
        let diff = matrix_diff(&lhs_num, &rhs_num, options.mode);
        if diff >= options.tolerance {
            failing_bindings = binding.clone();
            overall_max_diff = diff;
            break;
        }
        overall_max_diff = overall_max_diff.max(diff);
    }

    if failing_bindings.is_empty() {
        Ok(VerifyResult::NumericallyEqual {
            num_bindings: bindings.len(),
        })
    } else {
        Ok(VerifyResult::Fail(VerifyFailure {
            max_diff: overall_max_diff,
            bindings: failing_bindings,
        }))
    }
}

/// Build the symbolic unitary matrix for a sequence of PatternOps (LHS).
fn build_rule_item_matrix(
    ops: &[RuleItem],
    num_qubits: usize,
) -> Result<SymbolicMatrix, VerifyError> {
    let dim = 1usize << num_qubits;
    let mut matrix = symbolic_eye(dim);
    let mut numeric_block = NumericOpBlock::new(num_qubits);

    for op in ops {
        let gate = match &op.instruction {
            Instruction::Standard(g) => *g,
            other => {
                return Err(VerifyError::UnsupportedPattern(format!("{other:?}")));
            }
        };

        let params: SmallVec<[Parameter; 3]> = op
            .params
            .as_deref()
            .map(|ps| {
                ps.iter()
                    .map(|p| match p {
                        ParameterValue::Param(p) => p.clone(),
                        ParameterValue::Fixed(v) => Parameter::from(*v),
                    })
                    .collect()
            })
            .unwrap_or_default();

        apply_rule_gate(
            &mut matrix,
            &mut numeric_block,
            gate,
            &op.qubits,
            &params,
            num_qubits,
        )?;
    }

    numeric_block.flush_into(&mut matrix)?;
    Ok(matrix)
}

fn apply_rule_gate(
    matrix: &mut SymbolicMatrix,
    numeric_block: &mut NumericOpBlock,
    gate: StandardGate,
    qubits: &[u32],
    params: &[Parameter],
    num_qubits: usize,
) -> Result<(), VerifyError> {
    if num_qubits <= 2 && params.iter().all(Parameter::is_constant) {
        numeric_block.apply_standard_gate(gate, qubits, params)?;
    } else if gate.num_qubits() == 0 {
        numeric_block.flush_into(matrix)?;
        apply_global_phase(matrix, &params[0]);
    } else {
        numeric_block.flush_into(matrix)?;
        let reversed_bits: SmallVec<[usize; 3]> =
            qubits.iter().map(|&q| q as usize).rev().collect();
        apply_standard_gate_to_matrix(matrix, gate, &reversed_bits, params)?;
    }
    Ok(())
}

/// Accumulates consecutive constant gates as one dense numeric full-system
/// operator before it is applied to the symbolic matrix.
struct NumericOpBlock {
    num_qubits: usize,
    matrix: Option<Array2<Complex64>>,
}

impl NumericOpBlock {
    fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            matrix: None,
        }
    }

    fn apply_standard_gate(
        &mut self,
        gate: StandardGate,
        qubits: &[u32],
        params: &[Parameter],
    ) -> Result<(), VerifyError> {
        let numeric_params: SmallVec<[f64; 3]> = params
            .iter()
            .map(|p| {
                p.evaluate(&None)
                    .map_err(|_| CircuitError::SymbolicParameterError)
            })
            .collect::<Result<_, _>>()?;

        let block = self
            .matrix
            .get_or_insert_with(|| Array2::eye(1usize << self.num_qubits));

        if gate.num_qubits() == 0 {
            let phase = Complex64::from_polar(1.0, numeric_params[0]);
            block.par_mapv_inplace(|value| phase * value);
            return Ok(());
        }

        let gate_matrix = gate
            .matrix(numeric_params.as_slice())
            .map_err(|_| CircuitError::NoMatrixRepresentation)?;
        let reversed_bits: SmallVec<[usize; 3]> =
            qubits.iter().map(|&q| q as usize).rev().collect();
        apply_dense_numeric_gate_to_numeric_matrix(block, gate_matrix.as_ref(), &reversed_bits)?;
        Ok(())
    }

    fn flush_into(&mut self, matrix: &mut SymbolicMatrix) -> Result<(), VerifyError> {
        let Some(block) = self.matrix.take() else {
            return Ok(());
        };
        let bits: SmallVec<[usize; 8]> = (0..self.num_qubits).collect();
        apply_gate_to_matrix_num(matrix, &block, &bits)?;
        Ok(())
    }
}

fn apply_dense_numeric_gate_to_numeric_matrix(
    matrix: &mut Array2<Complex64>,
    gate: &Array2<Complex64>,
    bits: &[usize],
) -> Result<(), CircuitError> {
    let expected_dim = 1usize << bits.len();
    if gate.nrows() != expected_dim || gate.ncols() != expected_dim {
        return Err(CircuitError::QubitCountMismatch {
            expected: gate.nrows().trailing_zeros() as usize,
            actual: bits.len(),
        });
    }

    let expanded = expand_numeric_gate(gate, bits, matrix.nrows());
    *matrix = expanded.dot(matrix);
    Ok(())
}

fn expand_numeric_gate(gate: &Array2<Complex64>, bits: &[usize], dim: usize) -> Array2<Complex64> {
    let mut expanded = Array2::from_elem((dim, dim), Complex64::new(0.0, 0.0));

    for row in 0..dim {
        for col in 0..dim {
            let mut local_row = 0usize;
            let mut local_col = 0usize;
            let mut unaffected_equal = true;

            for bit in 0..dim.trailing_zeros() as usize {
                let row_bit = (row >> bit) & 1;
                let col_bit = (col >> bit) & 1;
                if let Some(local_bit) = bits.iter().position(|&target| target == bit) {
                    local_row |= row_bit << local_bit;
                    local_col |= col_bit << local_bit;
                } else if row_bit != col_bit {
                    unaffected_equal = false;
                    break;
                }
            }

            if unaffected_equal {
                expanded[[row, col]] = gate[[local_row, local_col]];
            }
        }
    }

    expanded
}

/// Apply a global phase `e^{i*theta}` to the entire matrix.
fn apply_global_phase(matrix: &mut SymbolicMatrix, theta: &Parameter) {
    if theta.is_constant() {
        let val = theta.evaluate(&None).expect("constant parameter evaluates");
        let phase = Complex64::new(val.cos(), val.sin());
        matrix
            .as_slice_mut()
            .expect("symbolic matrix must be contiguous")
            .par_iter_mut()
            .for_each(|elem| {
                let old = std::mem::take(elem);
                *elem = phase * old;
            });
    } else {
        let phase = SymbolicComplex::exp_i(theta.clone());
        matrix
            .as_slice_mut()
            .expect("symbolic matrix must be contiguous")
            .par_iter_mut()
            .for_each(|elem| {
                let old = std::mem::take(elem);
                *elem = &phase * old;
            });
    }
}

/// Simplify all elements of a symbolic matrix.
fn simplify_matrix(m: &SymbolicMatrix) -> Result<SymbolicMatrix, VerifyError> {
    let mut out = m.clone();
    out.as_slice_mut()
        .expect("symbolic matrix must be contiguous")
        .par_iter_mut()
        .try_for_each(|elem| -> Result<(), VerifyError> {
            *elem = elem.simplify()?;
            Ok(())
        })?;
    Ok(out)
}

fn matrix_diff(
    lhs: &ndarray::Array2<Complex64>,
    rhs: &ndarray::Array2<Complex64>,
    mode: RuleEquivalenceMode,
) -> f64 {
    match mode {
        RuleEquivalenceMode::StrictMatrix => max_diff_strict(lhs, rhs),
        RuleEquivalenceMode::UpToGlobalPhase => max_diff_up_to_global_phase(lhs, rhs),
    }
}

/// Compute the maximum element-wise difference between two numerical matrices.
fn max_diff_strict(lhs: &ndarray::Array2<Complex64>, rhs: &ndarray::Array2<Complex64>) -> f64 {
    (0..lhs.len())
        .into_par_iter()
        .map(|i| {
            let l = lhs.as_slice().unwrap()[i];
            let r = rhs.as_slice().unwrap()[i];
            (l - r).norm()
        })
        .fold(|| 0.0_f64, f64::max)
        .reduce(|| 0.0_f64, f64::max)
}

/// Compute the maximum element-wise difference between two numerical matrices.
fn max_diff_up_to_global_phase(
    lhs: &ndarray::Array2<Complex64>,
    rhs: &ndarray::Array2<Complex64>,
) -> f64 {
    const ZERO_EPS: f64 = 1e-14;
    const PHASE_EPS: f64 = 1e-8;

    let mut phase_ratio = None;

    for (&l, &r) in lhs.iter().zip(rhs.iter()) {
        let l_zero = l.norm() <= ZERO_EPS;
        let r_zero = r.norm() <= ZERO_EPS;

        if l_zero != r_zero {
            return max_diff_strict(lhs, rhs);
        }

        if !l_zero {
            let ratio = r / l;
            let ratio_norm = ratio.norm();

            if !ratio_norm.is_finite() || (ratio_norm - 1.0).abs() > PHASE_EPS {
                return max_diff_strict(lhs, rhs);
            }

            phase_ratio = Some(ratio);
            break;
        }
    }

    let Some(phase_ratio) = phase_ratio else {
        return max_diff_strict(lhs, rhs);
    };

    let lhs_slice = lhs.as_slice().expect("lhs matrix must be contiguous");
    let rhs_slice = rhs.as_slice().expect("rhs matrix must be contiguous");

    (0..lhs_slice.len())
        .into_par_iter()
        .map(|i| {
            let l = lhs_slice[i];
            let r = rhs_slice[i];

            // phase_ratio = rhs / lhs, so rhs / phase_ratio aligns with lhs.
            let normalized_r = r / phase_ratio;
            (l - normalized_r).norm()
        })
        .fold(|| 0.0_f64, f64::max)
        .reduce(|| 0.0_f64, f64::max)
}

/// Generate parameter bindings that satisfy the rule's conditions.
fn generate_satisfying_bindings(
    symbols: &[&str],
    conditions: Option<&[Condition]>,
    num_bindings: usize,
) -> Vec<HashMap<String, f64>> {
    let mut result = Vec::new();
    let mut rng = rand::rng();
    let two_pi = 2.0 * std::f64::consts::PI;

    for _ in 0..num_bindings {
        let mut bindings: HashMap<String, f64> = HashMap::new();
        for &sym in symbols {
            let val = rng.random_range(-two_pi..two_pi);
            bindings.insert(sym.to_string(), val);
        }

        if let Some(conds) = conditions {
            for cond in conds {
                adjust_binding_for_condition(&mut bindings, cond, symbols);
            }
        }

        if let Some(conds) = conditions {
            let bindings_ref: Option<HashMap<&str, f64>> =
                Some(bindings.iter().map(|(k, &v)| (k.as_str(), v)).collect());
            let all_satisfied = conds.iter().all(|c| match c {
                Condition::Eq(a, b) => {
                    if let (Ok(va), Ok(vb)) = (a.evaluate(&bindings_ref), b.evaluate(&bindings_ref))
                    {
                        (va - vb).abs() < 1e-8
                    } else {
                        false
                    }
                }
                Condition::EqMod(a, b, m) => {
                    if let (Ok(va), Ok(vb), Ok(vm)) = (
                        a.evaluate(&bindings_ref),
                        b.evaluate(&bindings_ref),
                        m.evaluate(&bindings_ref),
                    ) {
                        if vm.abs() < 1e-14 {
                            false
                        } else {
                            let remainder = (va - vb) % vm;
                            remainder.abs() < 1e-8 || (vm - remainder).abs() < 1e-8
                        }
                    } else {
                        false
                    }
                }
            });
            if !all_satisfied {
                continue;
            }
        }

        result.push(bindings);
    }

    result
}

/// Adjust one binding to satisfy a condition.
fn adjust_binding_for_condition(
    bindings: &mut HashMap<String, f64>,
    condition: &Condition,
    symbols: &[&str],
) {
    match condition {
        Condition::Eq(lhs, rhs) => {
            adjust_for_equality(bindings, lhs, rhs, None, symbols);
        }
        Condition::EqMod(lhs, rhs, modulus) => {
            if let Ok(mod_val) = modulus.evaluate(&Some(
                bindings.iter().map(|(k, &v)| (k.as_str(), v)).collect(),
            )) && mod_val.abs() > 1e-14
            {
                adjust_for_equality(bindings, lhs, rhs, Some(mod_val), symbols);
            }
        }
    }
}

/// Adjust one binding so that `lhs == rhs`, or `lhs == rhs mod modulus`.
fn adjust_for_equality(
    bindings: &mut HashMap<String, f64>,
    lhs: &Parameter,
    rhs: &Parameter,
    modulus: Option<f64>,
    symbols: &[&str],
) {
    if try_adjust_symbol(bindings, rhs, lhs, modulus, symbols) {
        return;
    }
    try_adjust_symbol(bindings, lhs, rhs, modulus, symbols);
}

/// Try to adjust one symbol in `expr` so that `expr == target mod modulus`.
fn try_adjust_symbol(
    bindings: &mut HashMap<String, f64>,
    expr: &Parameter,
    target: &Parameter,
    modulus: Option<f64>,
    symbols: &[&str],
) -> bool {
    let expr_syms = expr.get_symbols();
    let target_sym = symbols.iter().find(|&&s| expr_syms.contains(s));

    let Some(&sym) = target_sym else {
        return false;
    };

    let bindings_ref: Option<HashMap<&str, f64>> =
        Some(bindings.iter().map(|(k, &v)| (k.as_str(), v)).collect());

    let Ok(target_val) = target.evaluate(&bindings_ref) else {
        return false;
    };

    let bindings_without: Option<HashMap<&str, f64>> = Some(
        bindings
            .iter()
            .map(|(k, &v)| {
                if k == sym {
                    (k.as_str(), 0.0)
                } else {
                    (k.as_str(), v)
                }
            })
            .collect(),
    );
    let bindings_one: Option<HashMap<&str, f64>> = Some(
        bindings
            .iter()
            .map(|(k, &v)| {
                if k == sym {
                    (k.as_str(), 1.0)
                } else {
                    (k.as_str(), v)
                }
            })
            .collect(),
    );
    let (Ok(expr_zero), Ok(expr_one)) = (
        expr.evaluate(&bindings_without),
        expr.evaluate(&bindings_one),
    ) else {
        return false;
    };

    let coeff = expr_one - expr_zero;
    if coeff.abs() < 1e-14 {
        return false;
    }

    let current_sym_val = bindings.get(sym).copied().unwrap_or(0.0);

    if let Some(mod_val) = modulus {
        let sym_for_k0 = (target_val - expr_zero) / coeff;
        let k = ((current_sym_val - sym_for_k0) / mod_val).round();
        let adjusted = sym_for_k0 + k * mod_val;
        bindings.insert(sym.to_string(), adjusted);
    } else {
        let new_val = (target_val - expr_zero) / coeff;
        bindings.insert(sym.to_string(), new_val);
    }

    true
}

#[cfg(test)]
#[path = "rule_equivalence_test.rs"]
mod rule_equivalence_test;
