use crate::circuit::gate::StandardGate;
use std::collections::HashMap;
use rand::Rng;

/// Represents a step in the two-qubit gate transformation chain.
/// Each step transforms from `source_gate` using `rule_name`.
#[derive(Debug, Clone)]
pub struct TransformStep {
    pub source_gate: StandardGate,
    pub rule_name: String,
}

impl TransformStep {
    pub fn new(source_gate: StandardGate, rule_name: String) -> Self {
        Self {
            source_gate,
            rule_name,
        }
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
    categories.insert(StandardGate::CX, vec![StandardGate::CX, StandardGate::CY, StandardGate::CZ]);
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

/// Get the key gate for a given gate type, if it belongs to a known category.
fn get_category_key(gate: &StandardGate) -> Option<StandardGate> {
    let categories = get_two_qubit_categories();
    for (key, members) in categories.iter() {
        if members.contains(gate) {
            return Some(key.clone());
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
        _ => "Unknown Gate",
    };
    name.to_string()
    // gate.to_string("qasm".to_string())
}

/// Generate the rule name for transforming from source to target gate.
fn make_rule_name(source: &StandardGate, target: &StandardGate) -> String {
    format!(
        "{}2{}_rule",
        gate_to_string(source),
        gate_to_string(target)
    )
}

#[derive(Debug, Default, Clone)]
pub struct InstructionSet {
    pub single_qubit_gates: Vec<StandardGate>,
    pub double_qubit_gate: Vec<StandardGate>,
    single_qubit_decomposition_rule: String,
    /// Cache of two-qubit transform rules: source gate -> list of transform steps
    two_qubit_rule_map: HashMap<StandardGate, Vec<TransformStep>>,
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
    /// Returns a vector of TransformStep, each containing the source gate and rule name.
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
            self.two_qubit_rule_map
                .insert(source.clone(), empty_rules.clone());
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
                    curr_rules.push(TransformStep::new(
                        source.clone(),
                        make_rule_name(&source, dg),
                    ));
                } else {
                    // Need to go through the key gate
                    // source -> key -> target
                    curr_rules.push(TransformStep::new(
                        source.clone(),
                        make_rule_name(&source, &source_cate),
                    ));
                    curr_rules.push(TransformStep::new(
                        target_cate.clone(),
                        make_rule_name(&target_cate, dg),
                    ));
                }
            } else {
                // Source and target are in different categories
                // Need to use key gates as transfer points

                // Step 1: source -> source_key (if source is not already the key)
                if source != source_cate {
                    curr_rules.push(TransformStep::new(
                        source.clone(),
                        make_rule_name(&source, &source_cate),
                    ));
                }

                // Step 2: source_key -> target_key
                curr_rules.push(TransformStep::new(
                    source_cate.clone(),
                    make_rule_name(&source_cate, &target_cate),
                ));

                // Step 3: target_key -> target (if target is not the key)
                if *dg != target_cate {
                    curr_rules.push(TransformStep::new(
                        target_cate.clone(),
                        make_rule_name(&target_cate, dg),
                    ));
                }
            }
        
            if rules.is_empty() || curr_rules.len() < rules.len() {
                rules = curr_rules;
            }
        }

        // Cache the result
        self.two_qubit_rule_map
            .insert(source.clone(), rules.clone());

        Ok(rules)
    }

    /// Get the cached two-qubit rule map.
    pub fn get_two_qubit_rule_map(&self) -> &HashMap<StandardGate, Vec<TransformStep>> {
        &self.two_qubit_rule_map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_transform_rule_same_category_key_to_member() {
        // CX -> CY (both in CX category, CX is the key)
        let mut iset = InstructionSet::new(
            vec![StandardGate::RZ, StandardGate::RX], 
            vec![StandardGate::CY], 
            None
        );
        let rules = iset.select_transform_rule(StandardGate::CX).unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].source_gate, StandardGate::CX);
        assert_eq!(rules[0].rule_name, "cx2cy_rule");
    }

    #[test]
    fn test_select_transform_rule_same_category_member_to_key() {
        // CY -> CX (both in CX category, CX is the key)
        let mut iset = InstructionSet::new(
            vec![StandardGate::RZ, StandardGate::RX], 
            vec![StandardGate::CX], 
            None
        );
        let rules = iset.select_transform_rule(StandardGate::CY).unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].source_gate, StandardGate::CY);
        assert_eq!(rules[0].rule_name, "cy2cx_rule");
    }

    #[test]
    fn test_select_transform_rule_same_category_member_to_member() {
        // CY -> CZ (both in CX category, neither is the key)
        let mut iset = InstructionSet::new(
            vec![StandardGate::RZ, StandardGate::RX], 
            vec![StandardGate::CZ], 
            None
        );
        let rules = iset.select_transform_rule(StandardGate::CY).unwrap();
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].source_gate, StandardGate::CY);
        assert_eq!(rules[0].rule_name, "cy2cx_rule");
        assert_eq!(rules[1].source_gate, StandardGate::CX);
        assert_eq!(rules[1].rule_name, "cx2cz_rule");
    }

    #[test]
    fn test_select_transform_rule_different_category_key_to_key() {
        // CX -> RZZ (different categories, both are keys)
        let mut iset = InstructionSet::new(
            vec![StandardGate::RZ, StandardGate::RX], 
            vec![StandardGate::RZZ], 
            None
        );
        let rules = iset.select_transform_rule(StandardGate::CX).unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].source_gate, StandardGate::CX);
        assert_eq!(rules[0].rule_name, "cx2rzz_rule");
    }

    #[test]
    fn test_select_transform_rule_cx_to_fsim() {
        // CX -> FSIM (different categories, both are keys)
        let mut iset = InstructionSet::new(
            vec![StandardGate::RZ, StandardGate::RX],
            vec![StandardGate::FSIM],
            None
        );
        let rules = iset.select_transform_rule(StandardGate::CX).unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].source_gate, StandardGate::CX);
        assert_eq!(rules[0].rule_name, "cx2fsim_rule");
    }

    #[test]
    fn test_select_transform_rule_fsim_to_cx() {
        // FSIM -> CX (different categories, both are keys)
        let mut iset = InstructionSet::new(
            vec![StandardGate::RZ, StandardGate::RX],
            vec![StandardGate::CX],
            None
        );
        let rules = iset.select_transform_rule(StandardGate::FSIM).unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].source_gate, StandardGate::FSIM);
        assert_eq!(rules[0].rule_name, "fsim2cx_rule");
    }

    #[test]
    fn test_select_transform_rule_fsim_to_rxx() {
        // FSIM -> RXX (different categories, FSIM key to RZZ member)
        let mut iset = InstructionSet::new(
            vec![StandardGate::RZ, StandardGate::RX],
            vec![StandardGate::RXX],
            None
        );
        let rules = iset.select_transform_rule(StandardGate::FSIM).unwrap();
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].source_gate, StandardGate::FSIM);
        assert_eq!(rules[0].rule_name, "fsim2rzz_rule");
        assert_eq!(rules[1].source_gate, StandardGate::RZZ);
        assert_eq!(rules[1].rule_name, "rzz2rxx_rule");
    }

    #[test]
    fn test_select_transform_rule_different_category_member_to_key() {
        // CY -> RZZ (different categories, CY is member, RZZ is key)
        let mut iset = InstructionSet::new(
            vec![StandardGate::RZ, StandardGate::RX], 
            vec![StandardGate::RZZ], 
            None
        );
        let rules = iset.select_transform_rule(StandardGate::CY).unwrap();
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].source_gate, StandardGate::CY);
        assert_eq!(rules[0].rule_name, "cy2cx_rule");
        assert_eq!(rules[1].source_gate, StandardGate::CX);
        assert_eq!(rules[1].rule_name, "cx2rzz_rule");
    }

    #[test]
    fn test_select_transform_rule_different_category_key_to_member() {
        // CX -> RXX (different categories, CX is key, RXX is member)
        let mut iset = InstructionSet::new(
            vec![StandardGate::RZ, StandardGate::RX], 
            vec![StandardGate::RXX], 
            None
        );
        let rules = iset.select_transform_rule(StandardGate::CX).unwrap();
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].source_gate, StandardGate::CX);
        assert_eq!(rules[0].rule_name, "cx2rzz_rule");
        assert_eq!(rules[1].source_gate, StandardGate::RZZ);
        assert_eq!(rules[1].rule_name, "rzz2rxx_rule");
    }

    #[test]
    fn test_select_transform_rule_different_category_member_to_member() {
        // CY -> RXX (different categories, both are members)
        let mut iset = InstructionSet::new(
            vec![StandardGate::RZ, StandardGate::RX], 
            vec![StandardGate::RXX], 
            None
        );
        let rules = iset.select_transform_rule(StandardGate::CY).unwrap();
        assert_eq!(rules.len(), 3);
        assert_eq!(rules[0].source_gate, StandardGate::CY);
        assert_eq!(rules[0].rule_name, "cy2cx_rule");
        assert_eq!(rules[1].source_gate, StandardGate::CX);
        assert_eq!(rules[1].rule_name, "cx2rzz_rule");
        assert_eq!(rules[2].source_gate, StandardGate::RZZ);
        assert_eq!(rules[2].rule_name, "rzz2rxx_rule");
    }

    #[test]
    fn test_select_transform_rule_same_gate() {
        // CX -> CX (no transformation needed)
        let mut iset = InstructionSet::new(
            vec![StandardGate::RZ, StandardGate::RX], 
            vec![StandardGate::CX], 
            None
        );
        let rules = iset.select_transform_rule(StandardGate::CX).unwrap();
        assert_eq!(rules.len(), 0);
    }

    #[test]
    fn test_select_transform_rule_caching() {
        let mut iset = InstructionSet::new(
            vec![StandardGate::RZ, StandardGate::RX], 
            vec![StandardGate::CY], 
            None
        );

        // First call calculates and caches
        let rules1 = iset.select_transform_rule(StandardGate::CX).unwrap();
        assert!(iset.get_two_qubit_rule_map().contains_key(&StandardGate::CX));

        // Second call returns cached result
        let rules2 = iset.select_transform_rule(StandardGate::CX).unwrap();
        assert_eq!(rules1.len(), rules2.len());
        assert_eq!(rules1[0].rule_name, rules2[0].rule_name);
    }

    #[test]
    fn test_unknown_gate_returns_error() {
        // SWAP is not in any category, should return error
        let mut iset = InstructionSet::new(
            vec![StandardGate::RZ, StandardGate::RX], 
            vec![StandardGate::CX], 
            None
        );
        let result = iset.select_transform_rule(StandardGate::SWAP);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_double_qubit_gate_support() {
        // Test with multiple double qubit gates in instruction set
        let mut iset = InstructionSet::new(
            vec![StandardGate::RZ, StandardGate::RX], 
            vec![StandardGate::CZ, StandardGate::RZZ], 
            None
        );

        // Test that transformation rules can be generated for various source gates
        let source_gates = vec![
            StandardGate::CX,
            StandardGate::CY,
            StandardGate::CZ,
            StandardGate::RXX,
            StandardGate::RYY,
            StandardGate::RZZ,
            StandardGate::RZX
        ];

        for source in &source_gates {
            let result = iset.select_transform_rule(*source);
            assert!(result.is_ok(), "Failed for source gate: {:?}", source);
            let rules = result.unwrap();
            // Rules should be generated for all source gates
            assert!(rules.len() <= 2);
        }

        // Test caching for multiple source gates
        assert!(iset.get_two_qubit_rule_map().len() >= source_gates.len());
    }

    #[test]
    fn test_multi_double_qubit_gate_random_selection() {
        // Test that different target gates can be selected randomly
        let mut iset = InstructionSet::new(
            vec![StandardGate::RZ, StandardGate::RX], 
            vec![StandardGate::CX, StandardGate::RZZ, StandardGate::CZ], 
            None
        );

        // Test multiple times to ensure random selection works
        let mut selected_targets = std::collections::HashSet::new();
        for _ in 0..10 {
            let result = iset.select_transform_rule(StandardGate::CY);
            assert!(result.is_ok());
            let rules = result.unwrap();
            if !rules.is_empty() {
                // Get the last rule's target from the rule name
                let last_rule = &rules[rules.len() - 1];
                selected_targets.insert(last_rule.rule_name.clone());
            }
        }

        // Should have selected multiple different target gates
        assert!(selected_targets.len() > 0);
    }

    #[test]
    fn test_multi_double_qubit_gate_category_handling() {
        // Test with multiple gates from different categories
        let mut iset = InstructionSet::new(
            vec![StandardGate::RZ, StandardGate::RX], 
            vec![StandardGate::CX, StandardGate::RZZ], // From different categories
            None
        );

        // Test transformation from CX category to any target
        let result = iset.select_transform_rule(StandardGate::CY);
        assert!(result.is_ok());
        let rules = result.unwrap();
        // Rules should be generated (length depends on random target selection)
        assert!(rules.len() == 1);

        // Test transformation from RZZ category to any target
        let result = iset.select_transform_rule(StandardGate::RXX);
        assert!(result.is_ok());
        let rules = result.unwrap();
        // Rules should be generated (length depends on random target selection)
        assert!(rules.len() == 1);
    }
}
