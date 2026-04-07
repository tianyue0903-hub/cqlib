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

//! Optimize the given Circuit by merging the adjacent gates with
//! the commutative relation between gates in consideration.
//!
//! During the process, several parameterization and deparameterization process could be included, as listed
//! `'x'`: Rx <--> X, SX, SX_dagger
//! `'y'`: Ry <--> Y, SY, SY_dagger
//! `'z'`: Rz <--> Z, S, T, S_dagger, T_dagger
//! Whether to parameterize or deparameterize certain kinds of gates could be specified
//! by listing them in `para` and `depara`.

use crate::circuit::dag::CircuitDag;
use crate::circuit::gate::{Instruction, StandardGate};
use crate::circuit::{Circuit, CircuitParam, Operation, Qubit};
use alloc::borrow::Cow;
use lazy_static::lazy_static;
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use smallvec::smallvec;
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

lazy_static! {
    static ref PARA_RULER: HashMap<char, HashMap<StandardGate, (StandardGate, f64, f64)>> = {
        let mut para_ruler: HashMap<char, HashMap<StandardGate, (StandardGate, f64, f64)>> =
            HashMap::new();
        let mut x_para_ruler: HashMap<StandardGate, (StandardGate, f64, f64)> = HashMap::new();
        x_para_ruler.insert(StandardGate::X, (StandardGate::RX, PI, PI / 2.0_f64));
        let mut y_para_ruler: HashMap<StandardGate, (StandardGate, f64, f64)> = HashMap::new();
        y_para_ruler.insert(StandardGate::Y, (StandardGate::RY, PI, PI / 2.0_f64));
        y_para_ruler.insert(StandardGate::Y2P, (StandardGate::RY, PI / 2.0_f64, 0_f64));
        y_para_ruler.insert(StandardGate::Y2M, (StandardGate::RY, PI / -2.0_f64, 0_f64));
        let mut z_para_ruler: HashMap<StandardGate, (StandardGate, f64, f64)> = HashMap::new();
        z_para_ruler.insert(StandardGate::Z, (StandardGate::RZ, PI, PI / 2.0_f64));
        z_para_ruler.insert(
            StandardGate::S,
            (StandardGate::RZ, PI / 2.0_f64, PI / 4.0_f64),
        );
        z_para_ruler.insert(
            StandardGate::SDG,
            (StandardGate::RZ, PI / -2.0_f64, PI / -4.0_f64),
        );
        z_para_ruler.insert(
            StandardGate::T,
            (StandardGate::RZ, PI / 4.0_f64, PI / 8.0_f64),
        );
        z_para_ruler.insert(
            StandardGate::TDG,
            (StandardGate::RZ, PI / -4.0_f64, PI / -8.0_f64),
        );
        para_ruler.insert('x', x_para_ruler);
        para_ruler.insert('y', y_para_ruler);
        para_ruler.insert('z', z_para_ruler);
        para_ruler
    };
    static ref DEPARA_RULER: HashMap<char, HashMap<i32, (Vec<StandardGate>, f64)>> = {
        let mut depara_ruler: HashMap<char, HashMap<i32, (Vec<StandardGate>, f64)>> =
            HashMap::new();
        let mut x_depara_ruler: HashMap<i32, (Vec<StandardGate>, f64)> = HashMap::new();
        x_depara_ruler.insert(0, (Vec::from([StandardGate::I]), 0_f64));
        x_depara_ruler.insert(4, (Vec::from([StandardGate::X]), PI / -2.0_f64));
        x_depara_ruler.insert(8, (Vec::from([StandardGate::I]), PI));
        x_depara_ruler.insert(12, (Vec::from([StandardGate::X]), PI / 2.0_f64));

        let mut y_depara_ruler: HashMap<i32, (Vec<StandardGate>, f64)> = HashMap::new();
        y_depara_ruler.insert(0, (Vec::from([StandardGate::I]), 0_f64));
        y_depara_ruler.insert(2, (Vec::from([StandardGate::Y2P]), 0_f64));
        y_depara_ruler.insert(4, (Vec::from([StandardGate::Y]), PI / -2.0_f64));
        y_depara_ruler.insert(
            6,
            (
                Vec::from([StandardGate::Y, StandardGate::Y2P]),
                PI / -2.0_f64,
            ),
        );
        y_depara_ruler.insert(8, (Vec::from([StandardGate::I]), PI));
        y_depara_ruler.insert(10, (Vec::from([StandardGate::Y2P]), PI));
        y_depara_ruler.insert(12, (Vec::from([StandardGate::Y]), PI / 2.0_f64));
        y_depara_ruler.insert(
            14,
            (
                Vec::from([StandardGate::Y, StandardGate::Y2P]),
                PI / 2.0_f64,
            ),
        );

        let mut z_depara_ruler: HashMap<i32, (Vec<StandardGate>, f64)> = HashMap::new();
        z_depara_ruler.insert(0, (Vec::from([StandardGate::I]), 0_f64));
        z_depara_ruler.insert(1, (Vec::from([StandardGate::T]), PI / -8.0_f64));
        z_depara_ruler.insert(2, (Vec::from([StandardGate::S]), PI / -4.0_f64));
        z_depara_ruler.insert(
            3,
            (
                Vec::from([StandardGate::S, StandardGate::T]),
                3.0_f64 * PI / -8.0_f64,
            ),
        );
        z_depara_ruler.insert(4, (Vec::from([StandardGate::Z]), PI / -2.0_f64));
        z_depara_ruler.insert(
            5,
            (
                Vec::from([StandardGate::S, StandardGate::T]),
                5.0 * PI / -8.0_f64,
            ),
        );
        z_depara_ruler.insert(6, (Vec::from([StandardGate::SDG]), 5.0 * PI / 4.0_f64));
        z_depara_ruler.insert(7, (Vec::from([StandardGate::TDG]), 9.0 * PI / 8.0_f64));
        z_depara_ruler.insert(8, (Vec::from([StandardGate::I]), PI));
        z_depara_ruler.insert(9, (Vec::from([StandardGate::T]), 7.0 * PI / 8.0_f64));
        z_depara_ruler.insert(10, (Vec::from([StandardGate::S]), 3.0 * PI / 4.0_f64));
        z_depara_ruler.insert(
            11,
            (
                Vec::from([StandardGate::S, StandardGate::T]),
                5.0 * PI / 8.0_f64,
            ),
        );
        z_depara_ruler.insert(12, (Vec::from([StandardGate::Z]), PI / 2.0_f64));
        z_depara_ruler.insert(
            13,
            (
                Vec::from([StandardGate::Z, StandardGate::T]),
                3.0 * PI / 8.0_f64,
            ),
        );
        z_depara_ruler.insert(14, (Vec::from([StandardGate::SDG]), PI / 4.0_f64));
        z_depara_ruler.insert(15, (Vec::from([StandardGate::TDG]), PI / 8.0_f64));

        depara_ruler.insert('x', x_depara_ruler);
        depara_ruler.insert('y', y_depara_ruler);
        depara_ruler.insert('z', z_depara_ruler);
        depara_ruler
    };
}

const ELIMINATION_GATE: [StandardGate; 9] = [
    StandardGate::H,
    StandardGate::SWAP,
    StandardGate::X,
    StandardGate::CX,
    StandardGate::CCX,
    StandardGate::CY,
    StandardGate::Y,
    StandardGate::CZ,
    StandardGate::Z,
];

const ADDITION_GATE: [StandardGate; 7] = [
    StandardGate::RX,
    StandardGate::CRX,
    StandardGate::RY,
    StandardGate::CRY,
    StandardGate::RZ,
    StandardGate::CRZ,
    StandardGate::Phase,
];

// The comparison threshold
const EPS: f64 = 1e-6;

// Implementation of Directed Acyclic Graph (DAG) used in this code
pub struct GateNode {
    // Gate represented by the node
    gate: Operation,

    // Whether the gate is identity (upon a global phase)
    identity: bool,

    // Predecessors of the node
    predecessor: HashSet<usize>,

    // Whether this node needs to be compared with the new node
    reachable: bool,
}

impl GateNode {
    pub fn new(gate: Operation) -> Self {
        GateNode {
            gate,
            identity: false,
            predecessor: HashSet::new(),
            reachable: false,
        }
    }
}

const FMT_PARA_SYMBOL: [char; 3] = ['x', 'y', 'z'];

const FMT_DEPARA_GTYPE: [StandardGate; 3] = [StandardGate::RX, StandardGate::RY, StandardGate::RZ];

pub struct CommutativeOptimization {
    para: Vec<char>,
    depara: Vec<char>,
    keep_phase: bool,
    keep_order: bool,
    gphase: f64,
}

impl CommutativeOptimization {
    pub fn new(
        // Parameters to be parameterized, default to ['x', 'y', 'z']
        para: Option<Vec<char>>,

        // Parameters to be deparameterized, default to []
        depara: Option<Vec<char>>,

        // Whether to keep the global phase of the circuit
        keep_phase: bool,

        // Force to ignore parameterization and deparameterization
        keep_order: bool,
    ) -> Self {
        let mut fmt_para: Vec<char> = Vec::new();
        if let Some(vec_strs) = para {
            for vec_str in vec_strs {
                if !FMT_PARA_SYMBOL.contains(&vec_str) {
                    panic!("Invalid para, should be a subset of ['x', 'y', 'z']");
                }
                if !fmt_para.contains(&vec_str) {
                    fmt_para.push(vec_str);
                }
            }
        } else {
            fmt_para.push('x');
            fmt_para.push('y');
            fmt_para.push('z');
        }

        let mut fmt_depara: Vec<char> = Vec::new();
        if let Some(depara_strs) = depara {
            for depara_str in depara_strs {
                if !FMT_PARA_SYMBOL.contains(&depara_str) {
                    panic!("Invalid para, should be a subset of ['x', 'y', 'z']");
                }
                if !fmt_depara.contains(&depara_str) {
                    fmt_depara.push(depara_str);
                }
            }
        }

        CommutativeOptimization {
            para: fmt_para,
            depara: fmt_depara,
            keep_phase,
            keep_order,
            gphase: 0_f64,
        }
    }

    fn gate_expand_rust(
        gate_mat: Cow<'_, Array2<Complex64>>,
        gate_qubits: Vec<i32>,
        expand_qubits: i32,
    ) -> Array2<Complex64> {
        // assert!(gate.control_num + gate.target_num == 1_i32, "Only support expand single qubit gates into two qubit gates.");
        let expand_mat_shape = 1 << expand_qubits;
        let mut xor_value = expand_mat_shape - 1;
        let mut expand_mat: Array2<Complex64> =
            Array2::<Complex64>::zeros((expand_mat_shape, expand_mat_shape));

        let gq_len: i32 = gate_qubits.len() as i32;
        for gq in &gate_qubits {
            assert!(
                *gq >= 0 && *gq < expand_qubits,
                "The given gate_qubits must be positive and less than expand qubits."
            );
            xor_value ^= 1 << (expand_qubits - 1 - gq);
        }

        let mut expand_vec: Array1<usize> = Array1::zeros(expand_mat_shape);
        for i in 0..expand_mat_shape {
            let mut nowi: usize = 0;
            for (gq_idx, gq) in gate_qubits.iter().enumerate() {
                let k: i32 = expand_qubits - 1 - gq;
                if (1 << k) & i != 0 {
                    nowi += 1 << (gq_len - 1 - gq_idx as i32);
                }
            }
            expand_vec[i] = nowi;
        }

        for ii in 0..expand_mat_shape {
            for jj in 0..expand_mat_shape {
                if ii & xor_value == jj & xor_value {
                    expand_mat[[ii, jj]] = gate_mat[[expand_vec[ii], expand_vec[jj]]];
                }
            }
        }

        expand_mat
    }

    // Check whether two operations are commutative
    pub fn is_commutative(a: &Operation, b: &Operation) -> bool {
        // Only unitary gates can have commutation relationships
        let is_valid_instruction = |instr: &Instruction| {
            matches!(
                instr,
                Instruction::Standard(_)
                    | Instruction::McGate(_)
                    | Instruction::UnitaryGate(_)
                    | Instruction::CircuitGate(_)
            )
        };
        if !is_valid_instruction(&a.instruction) || !is_valid_instruction(&b.instruction) {
            return false;
        }

        // Identity gate detect
        if matches!(a.instruction, Instruction::Standard(StandardGate::I))
            || matches!(b.instruction, Instruction::Standard(StandardGate::I))
        {
            return true;
        }

        // Same gate detect
        if let Instruction::Standard(ga) = a.instruction
            && let Instruction::Standard(gb) = b.instruction
            && ga == gb
            && a.qubits.len() == 1
        {
            return true;
        }

        // Qubit interaction detect
        let a_set: HashSet<Qubit> = a.qubits.clone().into_iter().collect();
        let b_set: HashSet<Qubit> = b.qubits.clone().into_iter().collect();
        let inter_qubits: HashSet<Qubit> = a_set.intersection(&b_set).cloned().collect();
        if inter_qubits.is_empty() {
            return true;
        }

        // Commutative detect
        let mut comb_qubits: Vec<&Qubit> = a_set.union(&b_set).collect::<Vec<&Qubit>>();
        comb_qubits.sort();

        let mut a_order_qubits: Vec<i32> = Vec::new();
        for q in &a.qubits {
            a_order_qubits.push(comb_qubits.iter().position(|r| *r == q).unwrap() as i32);
        }
        let mut b_order_qubits: Vec<i32> = Vec::new();
        for oq in &b.qubits {
            b_order_qubits.push(comb_qubits.iter().position(|r| *r == oq).unwrap() as i32);
        }

        let a_matrix = a.matrix();
        // If the operation contains symbolic parameters that cannot be resolved, we consider it non-commutative for safety.
        if a_matrix.is_err() {
            return false;
        }
        let b_matrix = b.matrix();
        if b_matrix.is_err() {
            return false;
        }

        let expand_a_mat =
            Self::gate_expand_rust(a_matrix.unwrap(), a_order_qubits, comb_qubits.len() as i32);
        let expand_b_mat =
            Self::gate_expand_rust(b_matrix.unwrap(), b_order_qubits, comb_qubits.len() as i32);

        let a_b_dot: Array2<Complex64> = expand_a_mat.dot(&expand_b_mat);
        let b_a_dot: Array2<Complex64> = expand_b_mat.dot(&expand_a_mat);

        a_b_dot
            .iter()
            .zip(b_a_dot.iter())
            .all(|(a, b)| (*a - *b).norm() <= EPS)
    }

    fn _parameterize(&mut self, op: &Operation) -> Option<Operation> {
        for pa_str in &self.para {
            match &op.instruction {
                Instruction::Standard(std_gate) => {
                    let target_para_ruler = PARA_RULER.get(pa_str).unwrap().get(std_gate);
                    if let Some((para_gtype, para_parg, para_gphase)) = target_para_ruler {
                        let para_op = Operation {
                            instruction: Instruction::Standard(*para_gtype),
                            qubits: op.qubits.clone(),
                            params: smallvec![CircuitParam::Fixed(*para_parg)],
                            label: None,
                        };
                        self.gphase += *para_gphase;
                        return Some(para_op);
                    }
                }
                _ => {
                    return None;
                }
            }
        }
        None
    }

    fn _deparameterize(&mut self, op: &Operation) -> Option<Vec<Operation>> {
        let mut depara_gates: Vec<Operation> = Vec::new();
        if let Instruction::Standard(std_gate) = op.instruction
            && FMT_DEPARA_GTYPE.contains(&std_gate)
        {
            let fmt_gtype_str = FMT_PARA_SYMBOL
                .get(
                    FMT_DEPARA_GTYPE
                        .iter()
                        .position(|x| *x == std_gate)
                        .unwrap(),
                )
                .unwrap();
            if self.depara.contains(fmt_gtype_str)
            // Not yet support symbolic parameter
            && let CircuitParam::Fixed(parg) = op.params[0]
            {
                let num_multiple_quarter = parg % (4.0_f64 * PI) / (PI / 4.0_f64);
                let int_parg = num_multiple_quarter as i32;

                // Early return if the parameter is not a multiple of pi/4, which cannot be deparameterized
                if (num_multiple_quarter - int_parg as f64).abs() > f64::EPSILON {
                    return None;
                }

                // Math Calculate
                let target_depara_ruler = DEPARA_RULER.get(fmt_gtype_str).unwrap().get(&int_parg);
                if let Some((depara_gtypes, depara_gphase)) = target_depara_ruler {
                    for dgt in depara_gtypes {
                        if dgt != &StandardGate::I {
                            let temp_gate = Operation {
                                instruction: Instruction::Standard(*dgt),
                                qubits: op.qubits.clone(),
                                params: smallvec![],
                                label: None,
                            };
                            depara_gates.push(temp_gate);
                        }
                    }
                    self.gphase += *depara_gphase;
                    return Some(depara_gates);
                }
            }
        }
        None
    }

    pub fn optimize_operations(&mut self, operations: Vec<Operation>) -> Vec<Operation> {
        let mut nodes: Vec<GateNode> = Vec::new();
        self.gphase = 0_f64;

        for op in operations {
            if let Instruction::Standard(gate) = op.instruction {
                if gate == StandardGate::I {
                    continue;
                } else if gate == StandardGate::GPhase
                    && let CircuitParam::Fixed(parg) = op.params[0]
                {
                    self.gphase += parg;
                    continue;
                }
            }

            // Parameterize
            let para_gate = self._parameterize(&op);
            let mut new_gnode: GateNode = match para_gate.is_some() && !self.keep_order {
                false => GateNode::new(op.clone()),
                true => GateNode::new(para_gate.unwrap()),
            };

            // Procedure
            let node_length = nodes.len();
            for node in nodes.iter_mut().take(node_length) {
                node.reachable = !node.identity;
            }

            let mut enable_comb: bool = false;
            for prev_id in (0..node_length).rev() {
                let prev_node: &mut GateNode = nodes.get_mut(prev_id).unwrap();
                if !prev_node.reachable || new_gnode.predecessor.contains(&prev_id) {
                    continue;
                }

                let pgate = &prev_node.gate;
                if pgate.qubits == op.qubits
                    && let Instruction::Standard(gate) = new_gnode.gate.instruction
                    && matches!(&pgate.instruction, Instruction::Standard(pnode_gate) if gate == *pnode_gate)
                {
                    if ELIMINATION_GATE.contains(&gate) {
                        enable_comb = true;
                        prev_node.gate = Operation {
                            instruction: Instruction::Standard(StandardGate::I),
                            qubits: op.qubits.clone(),
                            params: smallvec![],
                            label: None,
                        };
                        prev_node.identity = true;
                        break;
                    } else if ADDITION_GATE.contains(&gate)
                        && let CircuitParam::Fixed(arg) = new_gnode.gate.params[0]
                        && let CircuitParam::Fixed(parg) = pgate.params[0]
                    {
                        // Combine two gates of the same type
                        enable_comb = true;
                        let angle = (arg + parg) % (PI * 4.0_f64);
                        prev_node.gate = Operation {
                            instruction: Instruction::Standard(gate),
                            qubits: op.qubits.clone(),
                            params: smallvec![CircuitParam::Fixed(angle)],
                            label: None,
                        };
                        prev_node.identity = angle == 0_f64;
                        break;
                    }
                }

                if !Self::is_commutative(&op, pgate) {
                    new_gnode.predecessor.insert(prev_id);
                    new_gnode.predecessor.extend(&prev_node.predecessor);
                }
            }

            if !enable_comb {
                nodes.push(new_gnode);
            }
        }

        let mut optimized: Vec<Operation> = Vec::new();
        for node in nodes {
            let ngate = node.gate;
            if matches!(ngate.instruction, Instruction::Standard(StandardGate::I)) {
                continue;
            } else if !self.keep_order {
                let depara_gates = self._deparameterize(&ngate);
                if let Some(gates) = depara_gates {
                    optimized.extend(gates);
                } else {
                    optimized.push(ngate);
                }
            } else {
                optimized.push(ngate);
            }
        }
        let phase_angle = self.gphase % (PI * 2.0_f64);
        if self.keep_phase && phase_angle != 0_f64 && phase_angle != (PI * 2.0_f64) {
            optimized.push(Operation {
                instruction: Instruction::Standard(StandardGate::GPhase),
                qubits: smallvec![],
                params: smallvec![CircuitParam::Fixed(phase_angle)],
                label: None,
            });
        }
        optimized
    }

    // Executes the commutative optimization on the given circuit
    pub fn execute(&mut self, cir: &Circuit) -> Circuit {
        let mut dag = CircuitDag::from_circuit(cir).expect("Failed to create CircuitDag");

        let block_indices: Vec<_> = dag.blocks().map(|(idx, _)| idx).collect();
        for idx in block_indices {
            let operations = dag.data[idx].operations.clone();
            let transformed = self.optimize_operations(operations);
            dag.data[idx].operations = transformed;
        }

        dag.to_circuit()
            .expect("Failed to convert CircuitDag to Circuit")
    }
}

#[cfg(test)]
#[path = "./commutative_test.rs"]
mod commutative_test;
