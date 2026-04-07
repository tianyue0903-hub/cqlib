use crate::circuit::gate::StandardGate;
use crate::compile::gate_transform::transform_rules::rule_registry::{
    SingleQubitParamTransformRule, TransformRuleKind, TwoQubitTransformRule,
};
use std::collections::HashMap;

/// Represents a step in a gate transformation chain.
#[derive(Debug, Clone)]
pub struct TransformStep {
    pub source_gate: StandardGate,
    pub rule: TransformRuleKind,
}

impl TransformStep {
    pub fn new(source_gate: StandardGate, rule: TransformRuleKind) -> Self {
        Self { source_gate, rule }
    }
}

/// Two-qubit gate categories.
/// Gates in the same category are equivalent under 1-qubit gates.
/// In the Weyl chamber viewpoint, they share the same point or trajectory.
///
/// Categories:
/// - CX category (key: CX): CX, CY, CZ
/// - RZZ category (key: RZZ): RXX, RYY, RZX, RZZ
/// - FSIM category (key: FSIM): FSIM
fn get_two_qubit_categories() -> HashMap<StandardGate, Vec<StandardGate>> {
    let mut categories = HashMap::new();
    categories.insert(
        StandardGate::CX,
        vec![StandardGate::CX, StandardGate::CY, StandardGate::CZ],
    );
    categories.insert(
        StandardGate::RZZ,
        vec![
            StandardGate::RXX,
            StandardGate::RYY,
            StandardGate::RZX,
            StandardGate::RZZ,
            StandardGate::CRX,
            StandardGate::CRY,
            StandardGate::CRZ,
        ],
    );
    categories.insert(StandardGate::FSIM, vec![StandardGate::FSIM]);
    categories
}

/// Single-qubit parameterized gate categories.
///
/// Categories:
/// - RX category (key: RX): RX, RY, RZ
/// - U category (key: U): U
/// - RXY category (key: RXY): RXY, XY, XY2P, XY2M
fn get_single_qubit_param_categories() -> HashMap<StandardGate, Vec<StandardGate>> {
    let mut categories = HashMap::new();
    categories.insert(
        StandardGate::RX,
        vec![StandardGate::RX, StandardGate::RY, StandardGate::RZ],
    );
    categories.insert(StandardGate::U, vec![StandardGate::U]);
    categories.insert(
        StandardGate::RXY,
        vec![
            StandardGate::RXY,
            StandardGate::XY,
            StandardGate::XY2P,
            StandardGate::XY2M,
        ],
    );
    categories
}

/// Get the key gate for a given gate type, if it belongs to a known category.
fn get_category_key(gate: &StandardGate) -> Option<StandardGate> {
    let categories = get_two_qubit_categories();
    for (key, members) in categories.iter() {
        if members.contains(gate) {
            return Some(*key);
        }
    }
    None
}

/// Get the key gate for a single-qubit parameterized gate.
fn get_single_qubit_param_category_key(gate: &StandardGate) -> Option<StandardGate> {
    let categories = get_single_qubit_param_categories();
    for (key, members) in categories.iter() {
        if members.contains(gate) {
            return Some(*key);
        }
    }
    None
}

fn gate_to_string(gate: &StandardGate) -> String {
    let name = match *gate {
        StandardGate::I => "id",
        StandardGate::X => "x",
        StandardGate::Y => "y",
        StandardGate::Z => "z",
        StandardGate::H => "h",
        StandardGate::S => "s",
        StandardGate::SDG => "sdg",
        StandardGate::T => "t",
        StandardGate::TDG => "tdg",
        StandardGate::RX => "rx",
        StandardGate::RY => "ry",
        StandardGate::RZ => "rz",
        StandardGate::Phase => "u1",
        StandardGate::SWAP => "swap",
        StandardGate::CCX => "ccx",
        StandardGate::U => "u3",
        StandardGate::XY => "xy",
        StandardGate::FSIM => "fsim",
        StandardGate::RXY => "rxy",
        StandardGate::CX => "cx",
        StandardGate::CY => "cy",
        StandardGate::CZ => "cz",
        StandardGate::CRX => "crx",
        StandardGate::CRY => "cry",
        StandardGate::CRZ => "crz",
        StandardGate::RXX => "rxx",
        StandardGate::RYY => "ryy",
        StandardGate::RZZ => "rzz",
        StandardGate::RZX => "rzx",
        StandardGate::X2P => "x2p",
        StandardGate::X2M => "x2m",
        StandardGate::Y2P => "y2p",
        StandardGate::Y2M => "y2m",
        StandardGate::XY2P => "xy2p",
        StandardGate::XY2M => "xy2m",
        StandardGate::GPhase => "gphase",
    };
    name.to_string()
    // gate.to_string("qasm".to_string())
}

fn make_two_qubit_rule(source: &StandardGate, target: &StandardGate) -> TransformRuleKind {
    TransformRuleKind::TwoQubit(
        TwoQubitTransformRule::from_gates(source, target).unwrap_or_else(|| {
            panic!(
                "No typed two-qubit transform rule registered for {:?} -> {:?}",
                source, target
            )
        }),
    )
}

fn make_single_qubit_param_rule(source: &StandardGate, target: &StandardGate) -> TransformRuleKind {
    TransformRuleKind::SingleQubitParam(
        SingleQubitParamTransformRule::from_gates(source, target).unwrap_or_else(|| {
            panic!(
                "No typed single-qubit param transform rule registered for {:?} -> {:?}",
                source, target
            )
        }),
    )
}

#[derive(Debug, Default, Clone)]
pub struct InstructionSet {
    pub single_qubit_gates: Vec<StandardGate>,
    pub double_qubit_gate: Vec<StandardGate>,
    single_qubit_decomposition_rule: String,
    /// Cache of two-qubit transform rules: source gate -> list of transform steps
    two_qubit_rule_map: HashMap<StandardGate, Vec<TransformStep>>,
    /// Cache of symbolic single-qubit transform rules: source gate -> list of transform steps
    single_qubit_rule_map: HashMap<StandardGate, Vec<TransformStep>>,
}

impl InstructionSet {
    pub fn new(
        single_qubit_gates: Vec<StandardGate>,
        double_qubit_gate: Vec<StandardGate>,
        single_qubit_decomposition_ruler: Option<String>,
    ) -> Self {
        let sqdr: String = match single_qubit_decomposition_ruler {
            None => {
                let contains_u3: bool = single_qubit_gates.contains(&StandardGate::U);
                let contains_rx: bool = single_qubit_gates.contains(&StandardGate::RX);
                let contains_ry: bool = single_qubit_gates.contains(&StandardGate::RY);
                let contains_rz: bool = single_qubit_gates.contains(&StandardGate::RZ);
                let contains_h: bool = single_qubit_gates.contains(&StandardGate::H);
                let contains_sx: bool = single_qubit_gates.contains(&StandardGate::X2P);
                let contains_sxdg: bool = single_qubit_gates.contains(&StandardGate::X2M);
                let contains_x: bool = single_qubit_gates.contains(&StandardGate::X);
                let contains_sy: bool = single_qubit_gates.contains(&StandardGate::Y2P);
                let contains_sydg: bool = single_qubit_gates.contains(&StandardGate::Y2M);
                if contains_u3 {
                    String::from("u3_rule")
                } else if contains_rx & contains_rz {
                    String::from("zxz_rule")
                } else if contains_rz & contains_ry {
                    String::from("zyz_rule")
                } else if contains_rx & contains_ry {
                    String::from("xyx_rule")
                } else if contains_h & contains_rz {
                    String::from("hrz_rule")
                } else if contains_sx & contains_x & contains_rz {
                    String::from("xsxrz_rule")
                } else if contains_sx & contains_sy & contains_sxdg & contains_sydg & contains_rz {
                    String::from("sxypmrz_rule")
                } else {
                    String::from("qcis_rule")
                    // panic!(
                    //     "Cannot find a suitable gate decomposition rule for single qubits in this instruction."
                    // )
                }
            }
            Some(s) => s,
        };
        InstructionSet {
            single_qubit_gates,
            double_qubit_gate,
            single_qubit_decomposition_rule: sqdr,
            two_qubit_rule_map: HashMap::new(),
            single_qubit_rule_map: HashMap::new(),
        }
    }

    pub fn get_gatetype(&self) -> Vec<String> {
        let mut gatetypes: Vec<String> = Vec::new();
        for sg in &self.single_qubit_gates {
            gatetypes.push(gate_to_string(sg));
        }
        for dg in &self.double_qubit_gate {
            gatetypes.push(gate_to_string(dg));
        }
        gatetypes
    }
}

impl InstructionSet {
    /// Get the single qubit decomposition rule name.
    pub fn get_single_qubit_decomposition_rule(&self) -> &str {
        &self.single_qubit_decomposition_rule
    }

    /// Select the transform rule chain to convert from source gate to the instruction set's
    /// double_qubit_gate.
    ///
    /// This function determines the sequence of transformation rules needed based on the
    /// category structure:
    /// - If source and target are in the same category:
    ///   - Direct rule if either is the key gate
    ///   - Otherwise: source -> key -> target
    /// - If source and target are in different categories:
    ///   - source -> source_key -> target_key -> target
    ///
    /// Returns a vector of TransformStep, each containing the source gate and typed rule.
    pub fn select_transform_rule(
        &mut self,
        source: StandardGate,
    ) -> Result<Vec<TransformStep>, String> {
        // If already cached, return the cached result
        if let Some(rules) = self.two_qubit_rule_map.get(&source) {
            return Ok(rules.clone());
        }

        // If source is already the target gate, no transformation needed
        if self.double_qubit_gate.contains(&source) {
            let empty_rules = Vec::new();
            self.two_qubit_rule_map.insert(source, empty_rules.clone());
            return Ok(empty_rules);
        }

        // Find categories for source and target
        let source_cate = get_category_key(&source);
        let source_cate = source_cate.ok_or_else(|| {
            format!(
                "Transform rule not found: source gate {:?} is not in any known category",
                source
            )
        })?;

        // Find shortest path for source to target
        let mut rules: Vec<TransformStep> = Vec::new();
        for dg in &self.double_qubit_gate {
            let target_cate = get_category_key(dg);
            let target_cate = target_cate.ok_or_else(|| {
                format!(
                    "Transform rule not found: target gate {:?} is not in any known category",
                    dg
                )
            })?;

            let mut curr_rules: Vec<TransformStep> = Vec::new();
            if source_cate == target_cate {
                // Source and target are in the same category
                if source == source_cate || *dg == target_cate {
                    // Direct transform if either is the key gate
                    curr_rules.push(TransformStep::new(source, make_two_qubit_rule(&source, dg)));
                } else {
                    // Need to go through the key gate
                    // source -> key -> target
                    curr_rules.push(TransformStep::new(
                        source,
                        make_two_qubit_rule(&source, &source_cate),
                    ));
                    curr_rules.push(TransformStep::new(
                        target_cate,
                        make_two_qubit_rule(&target_cate, dg),
                    ));
                }
            } else {
                // Source and target are in different categories
                // Need to use key gates as transfer points

                // Step 1: source -> source_key (if source is not already the key)
                if source != source_cate {
                    curr_rules.push(TransformStep::new(
                        source,
                        make_two_qubit_rule(&source, &source_cate),
                    ));
                }

                // Step 2: source_key -> target_key
                curr_rules.push(TransformStep::new(
                    source_cate,
                    make_two_qubit_rule(&source_cate, &target_cate),
                ));

                // Step 3: target_key -> target (if target is not the key)
                if *dg != target_cate {
                    curr_rules.push(TransformStep::new(
                        target_cate,
                        make_two_qubit_rule(&target_cate, dg),
                    ));
                }
            }

            if rules.is_empty() || curr_rules.len() < rules.len() {
                rules = curr_rules;
            }
        }

        // Cache the result
        self.two_qubit_rule_map.insert(source, rules.clone());

        Ok(rules)
    }

    /// Get the cached two-qubit rule map.
    pub fn get_two_qubit_rule_map(&self) -> &HashMap<StandardGate, Vec<TransformStep>> {
        &self.two_qubit_rule_map
    }

    /// Select the transform rule chain to convert a symbolic single-qubit parameterized
    /// source gate into one of the instruction set's supported single-qubit parameterized gates.
    pub fn select_single_qubit_param_transform_rule(
        &mut self,
        source: StandardGate,
    ) -> Result<Vec<TransformStep>, String> {
        if let Some(rules) = self.single_qubit_rule_map.get(&source) {
            return Ok(rules.clone());
        }

        if self.single_qubit_gates.contains(&source) {
            let empty_rules = Vec::new();
            self.single_qubit_rule_map
                .insert(source, empty_rules.clone());
            return Ok(empty_rules);
        }

        let source_cate = get_single_qubit_param_category_key(&source).ok_or_else(|| {
            format!(
                "Transform rule not found: source gate {:?} is not in any known single-qubit parameterized category",
                source
            )
        })?;

        let target_gates: Vec<StandardGate> = self
            .single_qubit_gates
            .iter()
            .copied()
            .filter(|gate| get_single_qubit_param_category_key(gate).is_some())
            .collect();

        if target_gates.is_empty() {
            return Err(
                "Transform rule not found: instruction set does not contain any supported single-qubit parameterized target gate"
                    .to_string(),
            );
        }

        let mut rules: Vec<TransformStep> = Vec::new();
        for target in &target_gates {
            let target_cate = get_single_qubit_param_category_key(target).ok_or_else(|| {
                format!(
                    "Transform rule not found: target gate {:?} is not in any known single-qubit parameterized category",
                    target
                )
            })?;

            let mut curr_rules: Vec<TransformStep> = Vec::new();
            if source_cate == target_cate {
                if source == source_cate || *target == target_cate {
                    curr_rules.push(TransformStep::new(
                        source,
                        make_single_qubit_param_rule(&source, target),
                    ));
                } else {
                    curr_rules.push(TransformStep::new(
                        source,
                        make_single_qubit_param_rule(&source, &source_cate),
                    ));
                    curr_rules.push(TransformStep::new(
                        target_cate,
                        make_single_qubit_param_rule(&target_cate, target),
                    ));
                }
            } else {
                if source != source_cate {
                    curr_rules.push(TransformStep::new(
                        source,
                        make_single_qubit_param_rule(&source, &source_cate),
                    ));
                }

                curr_rules.push(TransformStep::new(
                    source_cate,
                    make_single_qubit_param_rule(&source_cate, &target_cate),
                ));

                if *target != target_cate {
                    curr_rules.push(TransformStep::new(
                        target_cate,
                        make_single_qubit_param_rule(&target_cate, target),
                    ));
                }
            }

            if rules.is_empty() || curr_rules.len() < rules.len() {
                rules = curr_rules;
            }
        }

        self.single_qubit_rule_map.insert(source, rules.clone());
        Ok(rules)
    }

    /// Get the cached symbolic single-qubit transform rule map.
    pub fn get_single_qubit_rule_map(&self) -> &HashMap<StandardGate, Vec<TransformStep>> {
        &self.single_qubit_rule_map
    }
}

#[cfg(test)]
#[path = "./instruction_set_test.rs"]
mod instruction_set_test;
