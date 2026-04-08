use crate::circuit::Parameter;
use crate::circuit::gate::StandardGate;
use smallvec::{SmallVec, smallvec};

#[derive(Debug, Clone)]
pub struct DecomposedOp {
    pub gate: StandardGate,
    pub params: SmallVec<[Parameter; 3]>,
    pub qubits: SmallVec<[i32; 2]>,
}

/// Shared decomposition container for transform rules with explicit qubit indices.
#[derive(Debug, Clone)]
pub struct DecomposedGate {
    pub ops: Vec<DecomposedOp>,
}

impl Default for DecomposedGate {
    fn default() -> Self {
        Self::new()
    }
}

impl DecomposedGate {
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    pub fn push_single(
        &mut self,
        gate: StandardGate,
        params: SmallVec<[Parameter; 3]>,
        qubit: i32,
    ) {
        self.ops.push(DecomposedOp {
            gate,
            params,
            qubits: smallvec![qubit],
        });
    }

    pub fn push_two(
        &mut self,
        gate: StandardGate,
        params: SmallVec<[Parameter; 3]>,
        qubit0: i32,
        qubit1: i32,
    ) {
        self.ops.push(DecomposedOp {
            gate,
            params,
            qubits: smallvec![qubit0, qubit1],
        });
    }
}
