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

//! Typed, side-effect-free runtime classical expressions.
//!
//! Expressions read [`ClassicalVar`] handles and combine them into typed
//! runtime values. They do not measure qubits, write classical storage, or
//! transfer control. Control-flow statements consume these expressions in a
//! later layer.

use crate::circuit::classical::{ClassicalType, ClassicalVar};
use crate::circuit::error::CircuitError;
use std::collections::BTreeSet;
use std::num::NonZeroU32;
use std::sync::Arc;

/// Unary operators for classical expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClassicalUnaryOp {
    /// Logical negation for `Bool`, or bit inversion for `Bit`.
    Not,
}

/// Binary operators for boolean and bit expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClassicalBinaryOp {
    /// Logical or bitwise conjunction.
    And,
    /// Logical or bitwise disjunction.
    Or,
    /// Logical or bitwise exclusive-or.
    Xor,
}

/// Comparison operators for classical expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClassicalCompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Explicit casts between classical expression types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClassicalCast {
    /// Interprets a runtime bit as a runtime boolean.
    BitToBool,
    /// Interprets an ordered bit-vector as a little-endian unsigned integer.
    BitVecToUInt,
}

/// A typed, side-effect-free runtime classical expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ClassicalExpr {
    node: Arc<ClassicalExprNode>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ClassicalExprNode {
    ty: ClassicalType,
    kind: ClassicalExprKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum ClassicalExprKind {
    Var(ClassicalVar),
    BoolLiteral(bool),
    BitLiteral(bool),
    UIntLiteral {
        width: NonZeroU32,
        value: u128,
    },
    BitVecLiteral {
        width: NonZeroU32,
        value: u128,
    },
    Unary {
        op: ClassicalUnaryOp,
        expr: ClassicalExpr,
    },
    Binary {
        op: ClassicalBinaryOp,
        lhs: ClassicalExpr,
        rhs: ClassicalExpr,
    },
    Compare {
        op: ClassicalCompareOp,
        lhs: ClassicalExpr,
        rhs: ClassicalExpr,
    },
    Cast {
        cast: ClassicalCast,
        expr: ClassicalExpr,
    },
    Select {
        condition: ClassicalExpr,
        then_expr: ClassicalExpr,
        else_expr: ClassicalExpr,
    },
    ExtractBit {
        value: ClassicalExpr,
        index: u32,
    },
    ExtractBits {
        value: ClassicalExpr,
        offset: u32,
        width: NonZeroU32,
    },
    Concat {
        parts: Box<[ClassicalExpr]>,
    },
    PackBits {
        bits: Box<[ClassicalExpr]>,
    },
}

impl ClassicalExpr {
    /// Creates an expression that reads the current runtime value of `var`.
    pub fn var(var: ClassicalVar) -> Self {
        Self {
            node: Arc::new(ClassicalExprNode {
                ty: var.ty(),
                kind: ClassicalExprKind::Var(var),
            }),
        }
    }

    /// Creates a boolean literal expression.
    pub fn bool_literal(value: bool) -> Self {
        Self {
            node: Arc::new(ClassicalExprNode {
                ty: ClassicalType::Bool,
                kind: ClassicalExprKind::BoolLiteral(value),
            }),
        }
    }

    /// Creates a bit literal expression.
    pub fn bit_literal(value: bool) -> Self {
        Self {
            node: Arc::new(ClassicalExprNode {
                ty: ClassicalType::Bit,
                kind: ClassicalExprKind::BitLiteral(value),
            }),
        }
    }

    /// Creates an unsigned integer literal with a non-zero width.
    ///
    /// Literal construction currently supports constants up to 128 bits wide.
    pub fn uint_literal(width: u32, value: u128) -> Result<Self, CircuitError> {
        let Some(width) = NonZeroU32::new(width) else {
            return Err(CircuitError::InvalidOperation(
                "UInt literal width must be non-zero".to_string(),
            ));
        };
        if width.get() > 128 {
            return Err(CircuitError::InvalidOperation(format!(
                "UInt literal width {} exceeds the 128-bit literal limit",
                width.get()
            )));
        }
        if width.get() < 128 && value >= (1u128 << width.get()) {
            return Err(CircuitError::InvalidOperation(format!(
                "UInt literal value {value} does not fit in width {}",
                width.get()
            )));
        }
        Ok(Self {
            node: Arc::new(ClassicalExprNode {
                ty: ClassicalType::UInt(width),
                kind: ClassicalExprKind::UIntLiteral { width, value },
            }),
        })
    }

    /// Creates a bit-vector literal with a non-zero width.
    ///
    /// Literal construction currently supports constants up to 128 bits wide.
    pub fn bit_vec_literal(width: u32, value: u128) -> Result<Self, CircuitError> {
        let Some(width) = NonZeroU32::new(width) else {
            return Err(CircuitError::InvalidOperation(
                "BitVec literal width must be non-zero".to_string(),
            ));
        };
        if width.get() > 128 {
            return Err(CircuitError::InvalidOperation(format!(
                "BitVec literal width {} exceeds the 128-bit literal limit",
                width.get()
            )));
        }
        if width.get() < 128 && value >= (1u128 << width.get()) {
            return Err(CircuitError::InvalidOperation(format!(
                "BitVec literal value {value} does not fit in width {}",
                width.get()
            )));
        }
        Ok(Self {
            node: Arc::new(ClassicalExprNode {
                ty: ClassicalType::BitVec(width),
                kind: ClassicalExprKind::BitVecLiteral { width, value },
            }),
        })
    }

    /// Returns this expression's static type.
    pub fn ty(&self) -> ClassicalType {
        self.node.ty
    }

    /// Returns all runtime classical variables read by this expression.
    pub fn vars(&self) -> BTreeSet<ClassicalVar> {
        let mut vars = BTreeSet::new();
        self.collect_vars(&mut vars);
        vars
    }

    /// Adds every runtime classical variable read by this expression to `out`.
    pub fn collect_vars(&self, out: &mut BTreeSet<ClassicalVar>) {
        match &self.node.kind {
            ClassicalExprKind::Var(var) => {
                out.insert(*var);
            }
            ClassicalExprKind::BoolLiteral(_)
            | ClassicalExprKind::BitLiteral(_)
            | ClassicalExprKind::UIntLiteral { .. }
            | ClassicalExprKind::BitVecLiteral { .. } => {}
            ClassicalExprKind::Unary { expr, .. } | ClassicalExprKind::Cast { expr, .. } => {
                expr.collect_vars(out);
            }
            ClassicalExprKind::Binary { lhs, rhs, .. }
            | ClassicalExprKind::Compare { lhs, rhs, .. } => {
                lhs.collect_vars(out);
                rhs.collect_vars(out);
            }
            ClassicalExprKind::Select {
                condition,
                then_expr,
                else_expr,
            } => {
                condition.collect_vars(out);
                then_expr.collect_vars(out);
                else_expr.collect_vars(out);
            }
            ClassicalExprKind::ExtractBit { value, .. }
            | ClassicalExprKind::ExtractBits { value, .. } => {
                value.collect_vars(out);
            }
            ClassicalExprKind::Concat { parts } => {
                for part in parts.iter() {
                    part.collect_vars(out);
                }
            }
            ClassicalExprKind::PackBits { bits } => {
                for bit in bits.iter() {
                    bit.collect_vars(out);
                }
            }
        }
    }

    /// Creates a `not` expression for `Bool` or `Bit` values.
    pub fn not(expr: Self) -> Result<Self, CircuitError> {
        match expr.ty() {
            ClassicalType::Bool | ClassicalType::Bit => Ok(Self {
                node: Arc::new(ClassicalExprNode {
                    ty: expr.ty(),
                    kind: ClassicalExprKind::Unary {
                        op: ClassicalUnaryOp::Not,
                        expr,
                    },
                }),
            }),
            ty => Err(CircuitError::InvalidOperation(format!(
                "not expects Bool or Bit, got {ty:?}"
            ))),
        }
    }

    /// Creates an `and` expression for matching `Bool` or matching `Bit` values.
    pub fn and(lhs: Self, rhs: Self) -> Result<Self, CircuitError> {
        let ty = lhs.ty();
        if ty != rhs.ty() {
            return Err(CircuitError::InvalidOperation(format!(
                "binary operands must have the same type, got {:?} and {:?}",
                lhs.ty(),
                rhs.ty()
            )));
        }
        match ty {
            ClassicalType::Bool | ClassicalType::Bit => Ok(Self {
                node: Arc::new(ClassicalExprNode {
                    ty,
                    kind: ClassicalExprKind::Binary {
                        op: ClassicalBinaryOp::And,
                        lhs,
                        rhs,
                    },
                }),
            }),
            ty => Err(CircuitError::InvalidOperation(format!(
                "binary And expects Bool or Bit operands, got {ty:?}"
            ))),
        }
    }

    /// Creates an `or` expression for matching `Bool` or matching `Bit` values.
    pub fn or(lhs: Self, rhs: Self) -> Result<Self, CircuitError> {
        let ty = lhs.ty();
        if ty != rhs.ty() {
            return Err(CircuitError::InvalidOperation(format!(
                "binary operands must have the same type, got {:?} and {:?}",
                lhs.ty(),
                rhs.ty()
            )));
        }
        match ty {
            ClassicalType::Bool | ClassicalType::Bit => Ok(Self {
                node: Arc::new(ClassicalExprNode {
                    ty,
                    kind: ClassicalExprKind::Binary {
                        op: ClassicalBinaryOp::Or,
                        lhs,
                        rhs,
                    },
                }),
            }),
            ty => Err(CircuitError::InvalidOperation(format!(
                "binary Or expects Bool or Bit operands, got {ty:?}"
            ))),
        }
    }

    /// Creates an `xor` expression for matching `Bool` or matching `Bit` values.
    pub fn xor(lhs: Self, rhs: Self) -> Result<Self, CircuitError> {
        let ty = lhs.ty();
        if ty != rhs.ty() {
            return Err(CircuitError::InvalidOperation(format!(
                "binary operands must have the same type, got {:?} and {:?}",
                lhs.ty(),
                rhs.ty()
            )));
        }
        match ty {
            ClassicalType::Bool | ClassicalType::Bit => Ok(Self {
                node: Arc::new(ClassicalExprNode {
                    ty,
                    kind: ClassicalExprKind::Binary {
                        op: ClassicalBinaryOp::Xor,
                        lhs,
                        rhs,
                    },
                }),
            }),
            ty => Err(CircuitError::InvalidOperation(format!(
                "binary Xor expects Bool or Bit operands, got {ty:?}"
            ))),
        }
    }

    /// Creates an equality comparison. Operands must have the same type.
    pub fn eq(lhs: Self, rhs: Self) -> Result<Self, CircuitError> {
        if lhs.ty() != rhs.ty() {
            return Err(CircuitError::InvalidOperation(format!(
                "comparison operands must have the same type, got {:?} and {:?}",
                lhs.ty(),
                rhs.ty()
            )));
        }
        Ok(Self {
            node: Arc::new(ClassicalExprNode {
                ty: ClassicalType::Bool,
                kind: ClassicalExprKind::Compare {
                    op: ClassicalCompareOp::Eq,
                    lhs,
                    rhs,
                },
            }),
        })
    }

    /// Creates an inequality comparison. Operands must have the same type.
    pub fn ne(lhs: Self, rhs: Self) -> Result<Self, CircuitError> {
        if lhs.ty() != rhs.ty() {
            return Err(CircuitError::InvalidOperation(format!(
                "comparison operands must have the same type, got {:?} and {:?}",
                lhs.ty(),
                rhs.ty()
            )));
        }
        Ok(Self {
            node: Arc::new(ClassicalExprNode {
                ty: ClassicalType::Bool,
                kind: ClassicalExprKind::Compare {
                    op: ClassicalCompareOp::Ne,
                    lhs,
                    rhs,
                },
            }),
        })
    }

    /// Creates an unsigned less-than comparison.
    pub fn lt(lhs: Self, rhs: Self) -> Result<Self, CircuitError> {
        if lhs.ty() != rhs.ty() {
            return Err(CircuitError::InvalidOperation(format!(
                "comparison operands must have the same type, got {:?} and {:?}",
                lhs.ty(),
                rhs.ty()
            )));
        }
        if !matches!(lhs.ty(), ClassicalType::UInt(_)) {
            return Err(CircuitError::InvalidOperation(format!(
                "ordered comparisons require UInt operands, got {:?}",
                lhs.ty()
            )));
        }
        Ok(Self {
            node: Arc::new(ClassicalExprNode {
                ty: ClassicalType::Bool,
                kind: ClassicalExprKind::Compare {
                    op: ClassicalCompareOp::Lt,
                    lhs,
                    rhs,
                },
            }),
        })
    }

    /// Creates an unsigned less-than-or-equal comparison.
    pub fn le(lhs: Self, rhs: Self) -> Result<Self, CircuitError> {
        if lhs.ty() != rhs.ty() {
            return Err(CircuitError::InvalidOperation(format!(
                "comparison operands must have the same type, got {:?} and {:?}",
                lhs.ty(),
                rhs.ty()
            )));
        }
        if !matches!(lhs.ty(), ClassicalType::UInt(_)) {
            return Err(CircuitError::InvalidOperation(format!(
                "ordered comparisons require UInt operands, got {:?}",
                lhs.ty()
            )));
        }
        Ok(Self {
            node: Arc::new(ClassicalExprNode {
                ty: ClassicalType::Bool,
                kind: ClassicalExprKind::Compare {
                    op: ClassicalCompareOp::Le,
                    lhs,
                    rhs,
                },
            }),
        })
    }

    /// Creates an unsigned greater-than comparison.
    pub fn gt(lhs: Self, rhs: Self) -> Result<Self, CircuitError> {
        if lhs.ty() != rhs.ty() {
            return Err(CircuitError::InvalidOperation(format!(
                "comparison operands must have the same type, got {:?} and {:?}",
                lhs.ty(),
                rhs.ty()
            )));
        }
        if !matches!(lhs.ty(), ClassicalType::UInt(_)) {
            return Err(CircuitError::InvalidOperation(format!(
                "ordered comparisons require UInt operands, got {:?}",
                lhs.ty()
            )));
        }
        Ok(Self {
            node: Arc::new(ClassicalExprNode {
                ty: ClassicalType::Bool,
                kind: ClassicalExprKind::Compare {
                    op: ClassicalCompareOp::Gt,
                    lhs,
                    rhs,
                },
            }),
        })
    }

    /// Creates an unsigned greater-than-or-equal comparison.
    pub fn ge(lhs: Self, rhs: Self) -> Result<Self, CircuitError> {
        if lhs.ty() != rhs.ty() {
            return Err(CircuitError::InvalidOperation(format!(
                "comparison operands must have the same type, got {:?} and {:?}",
                lhs.ty(),
                rhs.ty()
            )));
        }
        if !matches!(lhs.ty(), ClassicalType::UInt(_)) {
            return Err(CircuitError::InvalidOperation(format!(
                "ordered comparisons require UInt operands, got {:?}",
                lhs.ty()
            )));
        }
        Ok(Self {
            node: Arc::new(ClassicalExprNode {
                ty: ClassicalType::Bool,
                kind: ClassicalExprKind::Compare {
                    op: ClassicalCompareOp::Ge,
                    lhs,
                    rhs,
                },
            }),
        })
    }

    /// Explicitly casts a `Bit` expression to `Bool`.
    pub fn bit_to_bool(expr: Self) -> Result<Self, CircuitError> {
        if expr.ty() != ClassicalType::Bit {
            return Err(CircuitError::InvalidOperation(format!(
                "bit_to_bool expects Bit, got {:?}",
                expr.ty()
            )));
        }
        Ok(Self {
            node: Arc::new(ClassicalExprNode {
                ty: ClassicalType::Bool,
                kind: ClassicalExprKind::Cast {
                    cast: ClassicalCast::BitToBool,
                    expr,
                },
            }),
        })
    }

    /// Explicitly casts a `BitVec` expression to a little-endian `UInt`.
    pub fn bit_vec_to_uint(expr: Self) -> Result<Self, CircuitError> {
        match expr.ty() {
            ClassicalType::BitVec(width) => Ok(Self {
                node: Arc::new(ClassicalExprNode {
                    ty: ClassicalType::UInt(width),
                    kind: ClassicalExprKind::Cast {
                        cast: ClassicalCast::BitVecToUInt,
                        expr,
                    },
                }),
            }),
            ty => Err(CircuitError::InvalidOperation(format!(
                "bit_vec_to_uint expects BitVec, got {ty:?}"
            ))),
        }
    }

    /// Creates an expression that chooses between two same-typed values.
    pub fn select(condition: Self, then_expr: Self, else_expr: Self) -> Result<Self, CircuitError> {
        if condition.ty() != ClassicalType::Bool {
            return Err(CircuitError::InvalidOperation(format!(
                "select condition expects Bool, got {:?}",
                condition.ty()
            )));
        }
        if then_expr.ty() != else_expr.ty() {
            return Err(CircuitError::InvalidOperation(format!(
                "select branches must have the same type, got {:?} and {:?}",
                then_expr.ty(),
                else_expr.ty()
            )));
        }
        Ok(Self {
            node: Arc::new(ClassicalExprNode {
                ty: then_expr.ty(),
                kind: ClassicalExprKind::Select {
                    condition,
                    then_expr,
                    else_expr,
                },
            }),
        })
    }

    /// Extracts a single bit from a `BitVec` or `UInt`.
    ///
    /// Index `0` is the least-significant bit.
    pub fn extract_bit(value: Self, index: u32) -> Result<Self, CircuitError> {
        let width = match value.ty() {
            ClassicalType::UInt(width) | ClassicalType::BitVec(width) => width.get(),
            ty => {
                return Err(CircuitError::InvalidOperation(format!(
                    "extract_bit expects UInt or BitVec, got {ty:?}"
                )));
            }
        };
        if index >= width {
            return Err(CircuitError::InvalidOperation(format!(
                "extract_bit index {index} out of bounds for width {width}"
            )));
        }
        Ok(Self {
            node: Arc::new(ClassicalExprNode {
                ty: ClassicalType::Bit,
                kind: ClassicalExprKind::ExtractBit { value, index },
            }),
        })
    }

    /// Extracts a bit range from a `BitVec` or `UInt`.
    ///
    /// `offset` is little-endian: offset `0` starts at the least-significant bit.
    pub fn extract_bits(value: Self, offset: u32, width: u32) -> Result<Self, CircuitError> {
        let source_width = match value.ty() {
            ClassicalType::UInt(width) | ClassicalType::BitVec(width) => width.get(),
            ty => {
                return Err(CircuitError::InvalidOperation(format!(
                    "extract_bits expects UInt or BitVec, got {ty:?}"
                )));
            }
        };
        let Some(width) = NonZeroU32::new(width) else {
            return Err(CircuitError::InvalidOperation(
                "extract_bits width must be non-zero".to_string(),
            ));
        };
        let Some(end) = offset.checked_add(width.get()) else {
            return Err(CircuitError::InvalidOperation(
                "extract_bits range overflows u32".to_string(),
            ));
        };
        if end > source_width {
            return Err(CircuitError::InvalidOperation(format!(
                "extract_bits range [{offset}, {end}) out of bounds for width {source_width}"
            )));
        }
        Ok(Self {
            node: Arc::new(ClassicalExprNode {
                ty: ClassicalType::BitVec(width),
                kind: ClassicalExprKind::ExtractBits {
                    value,
                    offset,
                    width,
                },
            }),
        })
    }

    /// Concatenates bit-vector values into a larger bit-vector.
    ///
    /// The first part occupies the least-significant output bits.
    pub fn concat(parts: impl IntoIterator<Item = Self>) -> Result<Self, CircuitError> {
        let parts: Vec<Self> = parts.into_iter().collect();
        if parts.is_empty() {
            return Err(CircuitError::InvalidOperation(
                "concat requires at least one part".to_string(),
            ));
        }

        let mut total_width = 0u32;
        for part in &parts {
            match part.ty() {
                ClassicalType::Bit => {
                    let Some(next_width) = total_width.checked_add(1) else {
                        return Err(CircuitError::InvalidOperation(
                            "concat width overflows u32".to_string(),
                        ));
                    };
                    total_width = next_width;
                }
                ClassicalType::BitVec(width) => {
                    let Some(next_width) = total_width.checked_add(width.get()) else {
                        return Err(CircuitError::InvalidOperation(
                            "concat width overflows u32".to_string(),
                        ));
                    };
                    total_width = next_width;
                }
                ty => {
                    return Err(CircuitError::InvalidOperation(format!(
                        "concat expects Bit or BitVec parts, got {ty:?}"
                    )));
                }
            }
        }

        let Some(width) = NonZeroU32::new(total_width) else {
            return Err(CircuitError::InvalidOperation(
                "concat width must be non-zero".to_string(),
            ));
        };
        Ok(Self {
            node: Arc::new(ClassicalExprNode {
                ty: ClassicalType::BitVec(width),
                kind: ClassicalExprKind::Concat {
                    parts: parts.into_boxed_slice(),
                },
            }),
        })
    }

    /// Packs single-bit expressions into a `BitVec`.
    ///
    /// The first bit becomes output index `0`, the least-significant bit.
    pub fn pack_bits(bits: impl IntoIterator<Item = Self>) -> Result<Self, CircuitError> {
        let bits: Vec<Self> = bits.into_iter().collect();
        if bits.is_empty() {
            return Err(CircuitError::InvalidOperation(
                "pack_bits requires at least one bit".to_string(),
            ));
        }
        for bit in &bits {
            if bit.ty() != ClassicalType::Bit {
                return Err(CircuitError::InvalidOperation(format!(
                    "pack_bits expects Bit expressions, got {:?}",
                    bit.ty()
                )));
            }
        }

        let Some(width) = NonZeroU32::new(bits.len() as u32) else {
            return Err(CircuitError::InvalidOperation(
                "pack_bits width must be non-zero".to_string(),
            ));
        };
        Ok(Self {
            node: Arc::new(ClassicalExprNode {
                ty: ClassicalType::BitVec(width),
                kind: ClassicalExprKind::PackBits {
                    bits: bits.into_boxed_slice(),
                },
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::ClassicalExpr;
    use crate::circuit::{ClassicalType, ClassicalVar};

    #[test]
    fn literals_have_static_types_and_validate_widths() {
        assert_eq!(ClassicalExpr::bool_literal(true).ty(), ClassicalType::Bool);
        assert_eq!(ClassicalExpr::bit_literal(false).ty(), ClassicalType::Bit);
        assert_eq!(
            ClassicalExpr::uint_literal(8, 255).unwrap().ty(),
            ClassicalType::uint(8).unwrap()
        );
        assert_eq!(
            ClassicalExpr::bit_vec_literal(3, 0b101).unwrap().ty(),
            ClassicalType::bit_vec(3).unwrap()
        );

        assert!(ClassicalExpr::uint_literal(0, 0).is_err());
        assert!(ClassicalExpr::uint_literal(129, 0).is_err());
        assert!(ClassicalExpr::uint_literal(3, 8).is_err());
    }

    #[test]
    fn boolean_and_bit_operations_are_typed_separately() {
        let b0 = ClassicalExpr::var(ClassicalVar::new(0, ClassicalType::Bool));
        let b1 = ClassicalExpr::var(ClassicalVar::new(1, ClassicalType::Bool));
        assert_eq!(
            ClassicalExpr::and(b0.clone(), b1.clone()).unwrap().ty(),
            ClassicalType::Bool
        );
        assert_eq!(ClassicalExpr::not(b0).unwrap().ty(), ClassicalType::Bool);

        let bit0 = ClassicalExpr::var(ClassicalVar::new(2, ClassicalType::Bit));
        let bit1 = ClassicalExpr::var(ClassicalVar::new(3, ClassicalType::Bit));
        assert_eq!(
            ClassicalExpr::xor(bit0.clone(), bit1).unwrap().ty(),
            ClassicalType::Bit
        );

        assert!(ClassicalExpr::and(b1, bit0).is_err());
    }

    #[test]
    fn comparisons_return_bool_and_enforce_ordered_uints() {
        let bit0 = ClassicalExpr::var(ClassicalVar::new(0, ClassicalType::Bit));
        let bit1 = ClassicalExpr::var(ClassicalVar::new(1, ClassicalType::Bit));
        assert_eq!(
            ClassicalExpr::eq(bit0.clone(), bit1.clone()).unwrap().ty(),
            ClassicalType::Bool
        );
        assert!(ClassicalExpr::lt(bit0, bit1).is_err());

        let u0 = ClassicalExpr::var(ClassicalVar::new(2, ClassicalType::uint(4).unwrap()));
        let u1 = ClassicalExpr::var(ClassicalVar::new(3, ClassicalType::uint(4).unwrap()));
        assert_eq!(ClassicalExpr::ge(u0, u1).unwrap().ty(), ClassicalType::Bool);
    }

    #[test]
    fn casts_are_explicit() {
        let bit = ClassicalExpr::var(ClassicalVar::new(0, ClassicalType::Bit));
        assert_eq!(
            ClassicalExpr::bit_to_bool(bit).unwrap().ty(),
            ClassicalType::Bool
        );

        let bits = ClassicalExpr::var(ClassicalVar::new(1, ClassicalType::bit_vec(5).unwrap()));
        assert_eq!(
            ClassicalExpr::bit_vec_to_uint(bits).unwrap().ty(),
            ClassicalType::uint(5).unwrap()
        );

        assert!(ClassicalExpr::bit_to_bool(ClassicalExpr::bool_literal(true)).is_err());
        assert!(
            ClassicalExpr::bit_vec_to_uint(ClassicalExpr::var(ClassicalVar::new(
                2,
                ClassicalType::uint(5).unwrap()
            )))
            .is_err()
        );
    }

    #[test]
    fn select_requires_bool_condition_and_matching_branch_types() {
        let condition = ClassicalExpr::bool_literal(true);
        let then_expr = ClassicalExpr::var(ClassicalVar::new(0, ClassicalType::Bit));
        let else_expr = ClassicalExpr::var(ClassicalVar::new(1, ClassicalType::Bit));

        assert_eq!(
            ClassicalExpr::select(condition, then_expr, else_expr)
                .unwrap()
                .ty(),
            ClassicalType::Bit
        );

        assert!(
            ClassicalExpr::select(
                ClassicalExpr::bit_literal(true),
                ClassicalExpr::var(ClassicalVar::new(2, ClassicalType::Bit)),
                ClassicalExpr::var(ClassicalVar::new(3, ClassicalType::Bit)),
            )
            .is_err()
        );
        assert!(
            ClassicalExpr::select(
                ClassicalExpr::bool_literal(true),
                ClassicalExpr::var(ClassicalVar::new(4, ClassicalType::Bit)),
                ClassicalExpr::var(ClassicalVar::new(5, ClassicalType::Bool)),
            )
            .is_err()
        );
    }

    #[test]
    fn extraction_uses_little_endian_indices() {
        let value = ClassicalExpr::var(ClassicalVar::new(0, ClassicalType::bit_vec(8).unwrap()));
        assert_eq!(
            ClassicalExpr::extract_bit(value.clone(), 0).unwrap().ty(),
            ClassicalType::Bit
        );
        assert_eq!(
            ClassicalExpr::extract_bits(value.clone(), 2, 3)
                .unwrap()
                .ty(),
            ClassicalType::bit_vec(3).unwrap()
        );

        assert!(ClassicalExpr::extract_bit(value.clone(), 8).is_err());
        assert!(ClassicalExpr::extract_bits(value, 7, 2).is_err());
    }

    #[test]
    fn pack_bits_and_concat_build_bit_vectors() {
        let bit0 = ClassicalExpr::var(ClassicalVar::new(0, ClassicalType::Bit));
        let bit1 = ClassicalExpr::var(ClassicalVar::new(1, ClassicalType::Bit));
        let packed = ClassicalExpr::pack_bits([bit0.clone(), bit1.clone()]).unwrap();
        assert_eq!(packed.ty(), ClassicalType::bit_vec(2).unwrap());

        let vec3 = ClassicalExpr::var(ClassicalVar::new(2, ClassicalType::bit_vec(3).unwrap()));
        let concat = ClassicalExpr::concat([bit0, vec3]).unwrap();
        assert_eq!(concat.ty(), ClassicalType::bit_vec(4).unwrap());

        assert!(ClassicalExpr::pack_bits([ClassicalExpr::bool_literal(true)]).is_err());
        assert!(ClassicalExpr::concat([ClassicalExpr::bool_literal(true)]).is_err());
        assert!(ClassicalExpr::concat(std::iter::empty()).is_err());
    }

    #[test]
    fn variables_are_collected_recursively() {
        let bit0 = ClassicalExpr::var(ClassicalVar::new(0, ClassicalType::Bit));
        let bit1 = ClassicalExpr::var(ClassicalVar::new(1, ClassicalType::Bit));
        let condition = ClassicalExpr::bit_to_bool(bit0.clone()).unwrap();
        let expr = ClassicalExpr::select(condition, bit0, bit1).unwrap();

        let vars = expr.vars();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains(&ClassicalVar::new(0, ClassicalType::Bit)));
        assert!(vars.contains(&ClassicalVar::new(1, ClassicalType::Bit)));
    }
}
