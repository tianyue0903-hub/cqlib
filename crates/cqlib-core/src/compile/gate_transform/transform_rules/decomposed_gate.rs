use crate::circuit::Parameter;
use crate::circuit::gate::StandardGate;
use smallvec::SmallVec;

/// Shared decomposition container for transform rules with explicit qubit indices.
#[derive(Debug, Clone)]
pub struct DecomposedGate {
    pub gates: Vec<(StandardGate, SmallVec<[Parameter; 3]>)>,
    pub qubits: Vec<Vec<i32>>,
}

impl DecomposedGate {
    pub fn new() -> Self {
        Self {
            gates: Vec::new(),
            qubits: Vec::new(),
        }
    }

    pub fn push_single(
        &mut self,
        gate: StandardGate,
        params: SmallVec<[Parameter; 3]>,
        qubit: i32,
    ) {
        self.gates.push((gate, params));
        self.qubits.push(vec![qubit]);
    }

    pub fn push_two(
        &mut self,
        gate: StandardGate,
        params: SmallVec<[Parameter; 3]>,
        qubit0: i32,
        qubit1: i32,
    ) {
        self.gates.push((gate, params));
        self.qubits.push(vec![qubit0, qubit1]);
    }
}
