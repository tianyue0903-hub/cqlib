use super::{ClassicalExpr, simplify};
use crate::circuit::{CircuitId, ClassicalType, ClassicalValue, ClassicalVar};
use std::sync::OnceLock;

fn test_circuit_id() -> CircuitId {
    static ID: OnceLock<CircuitId> = OnceLock::new();
    *ID.get_or_init(CircuitId::new)
}

fn mk_bool_var(idx: u32) -> ClassicalExpr {
    ClassicalExpr::var(ClassicalVar::new(
        test_circuit_id(),
        idx,
        ClassicalType::Bool,
    ))
}

fn mk_bit_var(idx: u32) -> ClassicalExpr {
    ClassicalExpr::var(ClassicalVar::new(
        test_circuit_id(),
        idx,
        ClassicalType::Bit,
    ))
}

fn mk_uint_var(idx: u32, width: u32) -> ClassicalExpr {
    ClassicalExpr::var(ClassicalVar::new(
        test_circuit_id(),
        idx,
        ClassicalType::uint(width).unwrap(),
    ))
}

fn mk_bitvec_var(idx: u32, width: u32) -> ClassicalExpr {
    ClassicalExpr::var(ClassicalVar::new(
        test_circuit_id(),
        idx,
        ClassicalType::bit_vec(width).unwrap(),
    ))
}

fn mk_bool_value(idx: u32) -> ClassicalExpr {
    ClassicalExpr::value(ClassicalValue::new(
        test_circuit_id(),
        idx,
        ClassicalType::Bool,
    ))
}

#[test]
fn leaf_var_is_unchanged() {
    let expr = mk_bool_var(0);
    assert_eq!(simplify(&expr), expr);
}

#[test]
fn leaf_value_is_unchanged() {
    let expr = mk_bool_value(0);
    assert_eq!(simplify(&expr), expr);
}

#[test]
fn leaf_bool_literal_is_unchanged() {
    let expr = ClassicalExpr::bool_literal(true);
    assert_eq!(simplify(&expr), expr);
}

#[test]
fn leaf_bit_literal_is_unchanged() {
    let expr = ClassicalExpr::bit_literal(false);
    assert_eq!(simplify(&expr), expr);
}

#[test]
fn leaf_uint_literal_is_unchanged() {
    let expr = ClassicalExpr::uint_literal(8, 42).unwrap();
    assert_eq!(simplify(&expr), expr);
}

#[test]
fn leaf_bitvec_literal_is_unchanged() {
    let expr = ClassicalExpr::bit_vec_literal(4, 0b1010).unwrap();
    assert_eq!(simplify(&expr), expr);
}

#[test]
fn double_not_bool() {
    let x = mk_bool_var(0);
    let expr = ClassicalExpr::try_not(ClassicalExpr::try_not(x.clone()).unwrap()).unwrap();
    assert_eq!(simplify(&expr), x);
}

#[test]
fn double_not_bit() {
    let x = mk_bit_var(0);
    let expr = ClassicalExpr::try_not(ClassicalExpr::try_not(x.clone()).unwrap()).unwrap();
    assert_eq!(simplify(&expr), x);
}

#[test]
fn single_not_is_unchanged() {
    let x = mk_bool_var(0);
    let expr = ClassicalExpr::try_not(x.clone()).unwrap();
    assert_eq!(simplify(&expr), expr);
}

#[test]
fn and_with_bool_true_rhs() {
    let x = mk_bool_var(0);
    let expr = ClassicalExpr::try_and(x.clone(), ClassicalExpr::bool_literal(true)).unwrap();
    assert_eq!(simplify(&expr), x);
}

#[test]
fn and_with_bool_true_lhs() {
    let x = mk_bool_var(0);
    let expr = ClassicalExpr::try_and(ClassicalExpr::bool_literal(true), x.clone()).unwrap();
    assert_eq!(simplify(&expr), x);
}

#[test]
fn and_with_bit_true_rhs() {
    let x = mk_bit_var(0);
    let expr = ClassicalExpr::try_and(x.clone(), ClassicalExpr::bit_literal(true)).unwrap();
    assert_eq!(simplify(&expr), x);
}

#[test]
fn and_with_bit_true_lhs() {
    let x = mk_bit_var(0);
    let expr = ClassicalExpr::try_and(ClassicalExpr::bit_literal(true), x.clone()).unwrap();
    assert_eq!(simplify(&expr), x);
}

#[test]
fn and_with_false_is_not_identity() {
    let x = mk_bool_var(0);
    let expr = ClassicalExpr::try_and(x.clone(), ClassicalExpr::bool_literal(false)).unwrap();
    assert_eq!(simplify(&expr), expr);
}

#[test]
fn or_with_bool_false_rhs() {
    let x = mk_bool_var(0);
    let expr = ClassicalExpr::try_or(x.clone(), ClassicalExpr::bool_literal(false)).unwrap();
    assert_eq!(simplify(&expr), x);
}

#[test]
fn or_with_bool_false_lhs() {
    let x = mk_bool_var(0);
    let expr = ClassicalExpr::try_or(ClassicalExpr::bool_literal(false), x.clone()).unwrap();
    assert_eq!(simplify(&expr), x);
}

#[test]
fn or_with_true_is_not_identity() {
    let x = mk_bool_var(0);
    let expr = ClassicalExpr::try_or(x.clone(), ClassicalExpr::bool_literal(true)).unwrap();
    assert_eq!(simplify(&expr), expr);
}

#[test]
fn xor_with_bool_false_rhs() {
    let x = mk_bool_var(0);
    let expr = ClassicalExpr::try_xor(x.clone(), ClassicalExpr::bool_literal(false)).unwrap();
    assert_eq!(simplify(&expr), x);
}

#[test]
fn xor_with_bool_false_lhs() {
    let x = mk_bool_var(0);
    let expr = ClassicalExpr::try_xor(ClassicalExpr::bool_literal(false), x.clone()).unwrap();
    assert_eq!(simplify(&expr), x);
}

#[test]
fn xor_with_bit_false_rhs() {
    let x = mk_bit_var(0);
    let expr = ClassicalExpr::try_xor(x.clone(), ClassicalExpr::bit_literal(false)).unwrap();
    assert_eq!(simplify(&expr), x);
}

#[test]
fn and_same_var_bool() {
    let x = mk_bool_var(0);
    let expr = ClassicalExpr::try_and(x.clone(), x.clone()).unwrap();
    assert_eq!(simplify(&expr), x);
}

#[test]
fn or_same_var_bool() {
    let x = mk_bool_var(0);
    let expr = ClassicalExpr::try_or(x.clone(), x.clone()).unwrap();
    assert_eq!(simplify(&expr), x);
}

#[test]
fn and_same_var_bit() {
    let x = mk_bit_var(0);
    let expr = ClassicalExpr::try_and(x.clone(), x.clone()).unwrap();
    assert_eq!(simplify(&expr), x);
}

#[test]
fn and_different_vars_not_simplified() {
    let x = mk_bool_var(0);
    let y = mk_bool_var(1);
    let expr = ClassicalExpr::try_and(x.clone(), y.clone()).unwrap();
    assert_eq!(simplify(&expr), expr);
}

#[test]
fn xor_same_var_bool() {
    let x = mk_bool_var(0);
    let expr = ClassicalExpr::try_xor(x.clone(), x).unwrap();
    assert_eq!(simplify(&expr), ClassicalExpr::bool_literal(false));
}

#[test]
fn xor_same_var_bit() {
    let x = mk_bit_var(0);
    let expr = ClassicalExpr::try_xor(x.clone(), x).unwrap();
    assert_eq!(simplify(&expr), ClassicalExpr::bit_literal(false));
}

#[test]
fn and_with_complement_rhs() {
    let x = mk_bool_var(0);
    let not_x = ClassicalExpr::try_not(x.clone()).unwrap();
    let expr = ClassicalExpr::try_and(x, not_x).unwrap();
    assert_eq!(simplify(&expr), ClassicalExpr::bool_literal(false));
}

#[test]
fn and_with_complement_lhs() {
    let x = mk_bool_var(0);
    let not_x = ClassicalExpr::try_not(x.clone()).unwrap();
    let expr = ClassicalExpr::try_and(not_x, x).unwrap();
    assert_eq!(simplify(&expr), ClassicalExpr::bool_literal(false));
}

#[test]
fn or_with_complement() {
    let x = mk_bool_var(0);
    let not_x = ClassicalExpr::try_not(x.clone()).unwrap();
    let expr = ClassicalExpr::try_or(x, not_x).unwrap();
    assert_eq!(simplify(&expr), ClassicalExpr::bool_literal(true));
}

#[test]
fn xor_with_complement() {
    let x = mk_bool_var(0);
    let not_x = ClassicalExpr::try_not(x.clone()).unwrap();
    let expr = ClassicalExpr::try_xor(x, not_x).unwrap();
    assert_eq!(simplify(&expr), ClassicalExpr::bool_literal(true));
}

#[test]
fn complement_bit_type() {
    let x = mk_bit_var(0);
    let not_x = ClassicalExpr::try_not(x.clone()).unwrap();
    let expr = ClassicalExpr::try_and(x, not_x).unwrap();
    assert_eq!(simplify(&expr), ClassicalExpr::bit_literal(false));
}

#[test]
fn non_complement_not_simplified() {
    // and(not(a), not(b)) — neither side is the complement of the other
    let a = mk_bool_var(0);
    let b = mk_bool_var(1);
    let not_a = ClassicalExpr::try_not(a).unwrap();
    let not_b = ClassicalExpr::try_not(b).unwrap();
    let expr = ClassicalExpr::try_and(not_a, not_b).unwrap();
    assert_eq!(simplify(&expr), expr);
}

#[test]
fn eq_same_uint_is_true() {
    let x = mk_uint_var(0, 8);
    let expr = ClassicalExpr::eq(x.clone(), x).unwrap();
    assert_eq!(simplify(&expr), ClassicalExpr::bool_literal(true));
}

#[test]
fn ne_same_uint_is_false() {
    let x = mk_uint_var(0, 8);
    let expr = ClassicalExpr::ne(x.clone(), x).unwrap();
    assert_eq!(simplify(&expr), ClassicalExpr::bool_literal(false));
}

#[test]
fn lt_same_uint_is_false() {
    let x = mk_uint_var(0, 8);
    let expr = ClassicalExpr::lt(x.clone(), x).unwrap();
    assert_eq!(simplify(&expr), ClassicalExpr::bool_literal(false));
}

#[test]
fn le_same_uint_is_true() {
    let x = mk_uint_var(0, 8);
    let expr = ClassicalExpr::le(x.clone(), x).unwrap();
    assert_eq!(simplify(&expr), ClassicalExpr::bool_literal(true));
}

#[test]
fn gt_same_uint_is_false() {
    let x = mk_uint_var(0, 8);
    let expr = ClassicalExpr::gt(x.clone(), x).unwrap();
    assert_eq!(simplify(&expr), ClassicalExpr::bool_literal(false));
}

#[test]
fn ge_same_uint_is_true() {
    let x = mk_uint_var(0, 8);
    let expr = ClassicalExpr::ge(x.clone(), x).unwrap();
    assert_eq!(simplify(&expr), ClassicalExpr::bool_literal(true));
}

#[test]
fn eq_same_bitvec_is_true() {
    let x = mk_bitvec_var(0, 4);
    let expr = ClassicalExpr::eq(x.clone(), x).unwrap();
    assert_eq!(simplify(&expr), ClassicalExpr::bool_literal(true));
}

#[test]
fn eq_different_vars_is_unchanged() {
    let x = mk_uint_var(0, 8);
    let y = mk_uint_var(1, 8);
    let expr = ClassicalExpr::eq(x.clone(), y.clone()).unwrap();
    assert_eq!(simplify(&expr), expr);
}

#[test]
fn select_true_picks_then() {
    let a = mk_bool_var(0);
    let b = mk_bool_var(1);
    let expr =
        ClassicalExpr::select(ClassicalExpr::bool_literal(true), a.clone(), b.clone()).unwrap();
    assert_eq!(simplify(&expr), a);
}

#[test]
fn select_false_picks_else() {
    let a = mk_bool_var(0);
    let b = mk_bool_var(1);
    let expr =
        ClassicalExpr::select(ClassicalExpr::bool_literal(false), a.clone(), b.clone()).unwrap();
    assert_eq!(simplify(&expr), b);
}

#[test]
fn select_runtime_condition_unchanged() {
    let cond = mk_bool_var(0);
    let a = mk_bool_var(1);
    let b = mk_bool_var(2);
    let expr = ClassicalExpr::select(cond, a, b).unwrap();
    assert_eq!(simplify(&expr), expr);
}

#[test]
fn cast_bit_literal_true_to_bool() {
    let expr = ClassicalExpr::bit_to_bool(ClassicalExpr::bit_literal(true)).unwrap();
    assert_eq!(simplify(&expr), ClassicalExpr::bool_literal(true));
}

#[test]
fn cast_bit_literal_false_to_bool() {
    let expr = ClassicalExpr::bit_to_bool(ClassicalExpr::bit_literal(false)).unwrap();
    assert_eq!(simplify(&expr), ClassicalExpr::bool_literal(false));
}

#[test]
fn cast_bitvec_literal_to_uint() {
    let expr =
        ClassicalExpr::bit_vec_to_uint(ClassicalExpr::bit_vec_literal(4, 0b1010).unwrap()).unwrap();
    assert_eq!(
        simplify(&expr),
        ClassicalExpr::uint_literal(4, 0b1010).unwrap()
    );
}

#[test]
fn cast_runtime_bit_unchanged() {
    let x = mk_bit_var(0);
    let expr = ClassicalExpr::bit_to_bool(x.clone()).unwrap();
    assert_eq!(simplify(&expr), expr);
}

#[test]
fn nested_double_negation() {
    // not(not(not(not(x)))) → x
    let x = mk_bool_var(0);
    let expr = ClassicalExpr::try_not(
        ClassicalExpr::try_not(
            ClassicalExpr::try_not(ClassicalExpr::try_not(x.clone()).unwrap()).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(simplify(&expr), x);
}

#[test]
fn and_of_simplified_children() {
    // and(not(not(x)), true) → x
    let x = mk_bool_var(0);
    let not_not_x = ClassicalExpr::try_not(ClassicalExpr::try_not(x.clone()).unwrap()).unwrap();
    let expr = ClassicalExpr::try_and(not_not_x, ClassicalExpr::bool_literal(true)).unwrap();
    assert_eq!(simplify(&expr), x);
}

#[test]
fn select_with_simplified_condition() {
    // select(eq(x, x), a, b) → a
    let x = mk_uint_var(0, 8);
    let a = mk_bool_var(1);
    let b = mk_bool_var(2);
    let eq_cond = ClassicalExpr::eq(x.clone(), x).unwrap();
    let expr = ClassicalExpr::select(eq_cond, a.clone(), b).unwrap();
    assert_eq!(simplify(&expr), a);
}

#[test]
fn deep_tree_all_rules_combined() {
    // or(and(not(not(x)), true), false) ∧ select(eq(x,x), y, z) ... — target: x ∧ y
    // Actually keep it simple: eq(and(not(not(a)), true), a) → true
    let a = mk_bool_var(0);
    let not_not_a = ClassicalExpr::try_not(ClassicalExpr::try_not(a.clone()).unwrap()).unwrap();
    let and_with_true =
        ClassicalExpr::try_and(not_not_a, ClassicalExpr::bool_literal(true)).unwrap();
    let expr = ClassicalExpr::eq(and_with_true, a).unwrap();
    assert_eq!(simplify(&expr), ClassicalExpr::bool_literal(true));
}

#[test]
fn extract_bit_recurses_into_child() {
    // extract_bit(select(true, uint_a, uint_b), 0) → extract_bit(uint_a, 0)
    let a = mk_uint_var(0, 8);
    let b = mk_uint_var(1, 8);
    let source = ClassicalExpr::select(ClassicalExpr::bool_literal(true), a.clone(), b).unwrap();
    let expr = ClassicalExpr::extract_bit(source, 3).unwrap();
    let expected = ClassicalExpr::extract_bit(a, 3).unwrap();
    assert_eq!(simplify(&expr), expected);
}

#[test]
fn concat_recurses_into_parts() {
    // concat([and(bit, true), bit]) → concat([bit, bit])
    let bx = mk_bit_var(2);
    let by = mk_bit_var(3);
    let simplified_child =
        ClassicalExpr::try_and(bx.clone(), ClassicalExpr::bit_literal(true)).unwrap();
    let concat_expr = ClassicalExpr::concat([simplified_child, by.clone()]).unwrap();
    let expected = ClassicalExpr::concat([bx, by]).unwrap();
    assert_eq!(simplify(&concat_expr), expected);
}

#[test]
fn pack_bits_recurses_into_bits() {
    // pack_bits([and(bit, true), and(true, bit)]) → pack_bits([bit, bit])
    let b0 = mk_bit_var(0);
    let b1 = mk_bit_var(1);
    let child0 = ClassicalExpr::try_and(b0.clone(), ClassicalExpr::bit_literal(true)).unwrap();
    let child1 = ClassicalExpr::try_and(ClassicalExpr::bit_literal(true), b1.clone()).unwrap();
    let expr = ClassicalExpr::pack_bits([child0, child1]).unwrap();
    let expected = ClassicalExpr::pack_bits([b0, b1]).unwrap();
    assert_eq!(simplify(&expr), expected);
}

#[test]
fn simplify_is_idempotent() {
    // Build a moderately complex expression and verify s(s(expr)) == s(expr)
    let x = mk_bool_var(0);
    let y = mk_bool_var(1);
    let not_x = ClassicalExpr::try_not(x.clone()).unwrap();
    let inner = ClassicalExpr::try_and(
        ClassicalExpr::try_not(not_x).unwrap(), // not(not(x))
        ClassicalExpr::bool_literal(true),
    )
    .unwrap();
    let expr = ClassicalExpr::try_or(inner, y).unwrap();

    let once = simplify(&expr);
    let twice = simplify(&once);
    assert_eq!(once, twice);
}

#[test]
fn simplify_preserves_type() {
    let cases: Vec<ClassicalExpr> = vec![
        ClassicalExpr::bool_literal(true),
        ClassicalExpr::bit_literal(false),
        ClassicalExpr::uint_literal(8, 42).unwrap(),
        ClassicalExpr::bit_vec_literal(4, 0b1010).unwrap(),
        ClassicalExpr::try_not(mk_bool_var(0)).unwrap(),
        ClassicalExpr::try_and(mk_bool_var(0), mk_bool_var(1)).unwrap(),
        ClassicalExpr::try_or(mk_bool_var(0), mk_bool_var(1)).unwrap(),
        ClassicalExpr::try_xor(mk_bit_var(0), mk_bit_var(1)).unwrap(),
        ClassicalExpr::eq(mk_uint_var(0, 4), mk_uint_var(1, 4)).unwrap(),
        ClassicalExpr::bit_to_bool(mk_bit_var(0)).unwrap(),
        ClassicalExpr::select(mk_bool_var(0), mk_bool_var(1), mk_bool_var(2)).unwrap(),
        ClassicalExpr::extract_bit(mk_uint_var(0, 8), 3).unwrap(),
        ClassicalExpr::extract_bits(mk_bitvec_var(0, 8), 2, 3).unwrap(),
        ClassicalExpr::concat([mk_bit_var(0), mk_bitvec_var(1, 3)]).unwrap(),
        ClassicalExpr::pack_bits([mk_bit_var(0), mk_bit_var(1)]).unwrap(),
    ];

    for expr in &cases {
        let simplified = simplify(expr);
        assert_eq!(
            simplified.ty(),
            expr.ty(),
            "type changed for {:?}",
            expr.kind()
        );
    }
}

#[test]
fn simplify_vars_are_subset() {
    let a = mk_bool_var(0);
    let b = mk_bool_var(1);
    // and(a, not(a)) → false — removes all vars
    let not_a = ClassicalExpr::try_not(a.clone()).unwrap();
    let expr = ClassicalExpr::try_and(a.clone(), not_a).unwrap();
    let simplified = simplify(&expr);
    let sv = simplified.vars();
    assert!(
        sv.is_empty(),
        "complement simplification should yield no vars, got {:?}",
        sv
    );

    // and(a, b) and a ≠ b — vars preserved
    let expr2 = ClassicalExpr::try_and(a.clone(), b.clone()).unwrap();
    let s2 = simplify(&expr2);
    assert!(s2.vars().len() <= expr2.vars().len());
    for v in s2.vars() {
        assert!(expr2.vars().contains(&v));
    }
}

#[test]
fn simplify_values_are_subset() {
    let v0 = mk_bool_value(0);
    let v1 = mk_bool_value(1);
    // or(v0, v0) → v0 (idempotence) — vars subset preserved
    let expr = ClassicalExpr::try_or(v0.clone(), v0.clone()).unwrap();
    let simplified = simplify(&expr);
    assert!(simplified.values().len() <= expr.values().len());
    for val in simplified.values() {
        assert!(expr.values().contains(&val));
    }

    // and(v0, v1) with v0 ≠ v1 — unchanged
    let expr2 = ClassicalExpr::try_and(v0, v1).unwrap();
    let s2 = simplify(&expr2);
    assert_eq!(s2, expr2);
}
