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

//! Recursive-descent parser for the rule DSL.
//!
//! Expression parsing is delegated to [`Parameter::try_from`] by extracting
//! the raw text span from the input string. This avoids building a custom
//! expression AST and reuses the full symbolic math support in
//! [`Parameter`].

use crate::circuit::Parameter;
use crate::compile::knowledge::rule::Condition;
use crate::compile::knowledge::rule_dsl::ast::{GatePattern, GateSpec, RuleDef};
use crate::compile::knowledge::rule_dsl::lexer::{Lexer, Token, TokenKind};

/// Errors that can occur while parsing a rule DSL source.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ParseError {
    /// An error originating from the lexer.
    #[error("lexer error at byte {pos}: {msg}")]
    Lexer {
        /// Error message from the lexer.
        msg: String,
        /// Byte offset where the error occurred.
        pos: usize,
    },
    /// The parser encountered a token other than the one it expected.
    #[error("unexpected token at byte {pos}: expected {expected}, found {found}")]
    UnexpectedToken {
        /// Human-readable description of the expected token kind.
        expected: String,
        /// Description of the token that was actually found.
        found: String,
        /// Byte offset of the unexpected token.
        pos: usize,
    },
    /// An expression span was empty (e.g. two commas with nothing between them).
    #[error("empty expression at byte {pos}")]
    EmptyExpr {
        /// Byte offset where the empty expression started.
        pos: usize,
    },
    /// A captured expression string could not be parsed by `Parameter::try_from`.
    #[error("invalid expression {expr:?}: {reason}")]
    InvalidExpr {
        /// The raw expression text that failed to parse.
        expr: String,
        /// Underlying error message.
        reason: String,
    },
    /// A bracketed multi-control gate name did not use the `MC<Gate>[n]` form.
    #[error("invalid multi-controlled gate name at byte {pos}: {gate_name:?}")]
    InvalidMultiControlledGateName {
        /// Gate name that appeared before `[n]`.
        gate_name: String,
        /// Byte offset where the gate name began.
        pos: usize,
    },
}

/// Recursive-descent parser for the rule DSL.
pub struct Parser<'a> {
    /// The full input string; used to extract raw expression spans.
    input: &'a str,
    /// The underlying lexer.
    lexer: Lexer<'a>,
    /// The current lookahead token.
    current: Token<'a>,
}

impl<'a> Parser<'a> {
    /// Creates a new parser for the given input.
    ///
    /// Returns an error if the lexer fails on the very first token.
    pub fn new(input: &'a str) -> Result<Self, ParseError> {
        let mut lexer = Lexer::new(input);
        let current = lexer.next_token().map_err(|e| ParseError::Lexer {
            msg: e.msg,
            pos: e.pos,
        })?;
        Ok(Self {
            input,
            lexer,
            current,
        })
    }

    /// Advances the lexer and stores the next token in `self.current`.
    fn advance(&mut self) -> Result<(), ParseError> {
        self.current = self.lexer.next_token().map_err(|e| ParseError::Lexer {
            msg: e.msg,
            pos: e.pos,
        })?;
        Ok(())
    }

    /// Consumes the current token if it matches `kind`, otherwise returns an
    /// [`ParseError::UnexpectedToken`].
    fn expect(&mut self, kind: TokenKind) -> Result<Token<'a>, ParseError> {
        if self.current.kind == kind {
            let token = Token {
                kind: self.current.kind.clone(),
                text: self.current.text,
                pos: self.current.pos,
            };
            self.advance()?;
            Ok(token)
        } else {
            Err(ParseError::UnexpectedToken {
                expected: format!("{:?}", kind),
                found: format!("{:?}({})", self.current.kind, self.current.text),
                pos: self.current.pos,
            })
        }
    }

    /// Skips line breaks where the grammar permits layout whitespace.
    fn skip_newlines(&mut self) -> Result<(), ParseError> {
        while self.current.kind == TokenKind::Newline {
            self.advance()?;
        }
        Ok(())
    }

    /// Parses an entire rule file (zero or more rules).
    pub fn parse_rule_file(&mut self) -> Result<Vec<RuleDef>, ParseError> {
        let mut rules = Vec::new();
        self.skip_newlines()?;
        while self.current.kind != TokenKind::Eof {
            rules.push(self.parse_rule()?);
            self.skip_newlines()?;
        }
        Ok(rules)
    }

    /// Parses a single `rule ident { ... }` definition.
    fn parse_rule(&mut self) -> Result<RuleDef, ParseError> {
        self.expect(TokenKind::Rule)?;
        let name = self.expect(TokenKind::Ident)?.text.to_string();
        self.expect(TokenKind::LBrace)?;
        self.skip_newlines()?;

        self.expect(TokenKind::Match)?;
        self.expect(TokenKind::LBrace)?;
        let match_ops = self.parse_gate_patterns()?;
        self.expect(TokenKind::RBrace)?;
        self.skip_newlines()?;

        let mut conditions = Vec::new();
        if self.current.kind == TokenKind::Require {
            self.advance()?;
            self.expect(TokenKind::LBrace)?;
            conditions = self.parse_conditions()?;
            self.expect(TokenKind::RBrace)?;
            self.skip_newlines()?;
        }

        self.expect(TokenKind::Rewrite)?;
        self.expect(TokenKind::LBrace)?;
        let rewrite_ops = self.parse_gate_patterns()?;
        self.expect(TokenKind::RBrace)?;
        self.skip_newlines()?;

        self.expect(TokenKind::RBrace)?;

        Ok(RuleDef {
            name,
            match_ops,
            conditions,
            rewrite_ops,
        })
    }

    /// Parses a list of gate patterns separated by optional commas.
    ///
    /// Returns an empty vector if the list is immediately closed by `}`.
    /// Commas are required only when multiple patterns appear on the same line;
    /// they may be omitted between line breaks.
    fn parse_gate_patterns(&mut self) -> Result<Vec<GatePattern>, ParseError> {
        let mut patterns = Vec::new();
        self.skip_newlines()?;
        while self.current.kind != TokenKind::RBrace && self.current.kind != TokenKind::Eof {
            patterns.push(self.parse_gate_pattern()?);
            match self.current.kind {
                TokenKind::Comma => {
                    self.advance()?;
                    self.skip_newlines()?;
                }
                TokenKind::Newline => {
                    self.skip_newlines()?;
                }
                TokenKind::RBrace | TokenKind::Eof => {}
                _ => {
                    return Err(ParseError::UnexpectedToken {
                        expected: "Comma, Newline, or RBrace".to_string(),
                        found: format!("{:?}({})", self.current.kind, self.current.text),
                        pos: self.current.pos,
                    });
                }
            }
        }
        Ok(patterns)
    }

    /// Parses a single gate pattern: `gate_spec [ "(" params ")" ] qubits`.
    fn parse_gate_pattern(&mut self) -> Result<GatePattern, ParseError> {
        let gate_token = self.expect(TokenKind::Ident)?;
        let gate_name = gate_token.text.to_string();
        let gate = if self.current.kind == TokenKind::LBracket {
            self.advance()?;
            let control_token = self.expect(TokenKind::Number)?;
            let added_controls =
                control_token
                    .text
                    .parse::<u8>()
                    .map_err(|_| ParseError::UnexpectedToken {
                        expected: "u8 multi-control count".to_string(),
                        found: control_token.text.to_string(),
                        pos: control_token.pos,
                    })?;
            self.expect(TokenKind::RBracket)?;
            let base_gate_name = gate_name.strip_prefix("MC").ok_or_else(|| {
                ParseError::InvalidMultiControlledGateName {
                    gate_name: gate_name.clone(),
                    pos: gate_token.pos,
                }
            })?;
            if base_gate_name.is_empty() {
                return Err(ParseError::InvalidMultiControlledGateName {
                    gate_name,
                    pos: gate_token.pos,
                });
            }
            GateSpec::MultiControlled {
                base_gate_name: base_gate_name.to_string(),
                added_controls,
            }
        } else {
            GateSpec::Standard { gate_name }
        };

        let mut params = Vec::new();
        if self.current.kind == TokenKind::LParen {
            self.advance()?;
            if self.current.kind != TokenKind::RParen {
                params.push(self.parse_param()?);
                while self.current.kind == TokenKind::Comma {
                    self.advance()?;
                    params.push(self.parse_param()?);
                }
            }
            self.expect(TokenKind::RParen)?;
        }

        let mut qubits = Vec::new();
        while self.current.kind == TokenKind::Number {
            let text = self.current.text;
            let q = text
                .parse::<u32>()
                .map_err(|_| ParseError::UnexpectedToken {
                    expected: "qubit index".to_string(),
                    found: text.to_string(),
                    pos: self.current.pos,
                })?;
            qubits.push(q);
            self.advance()?;
        }

        Ok(GatePattern {
            gate,
            params,
            qubits,
        })
    }

    /// Parses a single parameter expression up to `,` or `)`.
    fn parse_param(&mut self) -> Result<Parameter, ParseError> {
        self.parse_expr(|kind| *kind == TokenKind::Comma || *kind == TokenKind::RParen)
    }

    /// Parses a list of conditions separated by optional commas.
    ///
    /// Returns an empty vector if the list is immediately closed by `}`.
    /// Commas are required only when multiple conditions appear on the same line;
    /// they may be omitted between line breaks.
    fn parse_conditions(&mut self) -> Result<Vec<Condition>, ParseError> {
        let mut conditions = Vec::new();
        self.skip_newlines()?;
        while self.current.kind != TokenKind::RBrace && self.current.kind != TokenKind::Eof {
            conditions.push(self.parse_condition()?);
            match self.current.kind {
                TokenKind::Comma => {
                    self.advance()?;
                    self.skip_newlines()?;
                }
                TokenKind::Newline => {
                    self.skip_newlines()?;
                }
                TokenKind::RBrace | TokenKind::Eof => {}
                _ => {
                    return Err(ParseError::UnexpectedToken {
                        expected: "Comma, Newline, or RBrace".to_string(),
                        found: format!("{:?}({})", self.current.kind, self.current.text),
                        pos: self.current.pos,
                    });
                }
            }
        }
        Ok(conditions)
    }

    /// Parses a single condition: `expr == expr [mod expr]`.
    fn parse_condition(&mut self) -> Result<Condition, ParseError> {
        let lhs = self.parse_expr(|kind| *kind == TokenKind::EqEq)?;
        self.expect(TokenKind::EqEq)?;
        let rhs = self.parse_expr(|kind| {
            *kind == TokenKind::Mod
                || *kind == TokenKind::Comma
                || *kind == TokenKind::Newline
                || *kind == TokenKind::RBrace
        })?;

        if self.current.kind == TokenKind::Mod {
            self.advance()?;
            let modulus = self.parse_expr(|kind| {
                *kind == TokenKind::Comma
                    || *kind == TokenKind::Newline
                    || *kind == TokenKind::RBrace
            })?;
            Ok(Condition::EqMod(lhs, rhs, modulus))
        } else {
            Ok(Condition::Eq(lhs, rhs))
        }
    }

    /// Parses an expression by consuming tokens until `stop` returns true.
    ///
    /// The raw text spanning from the first token to the boundary token is
    /// trimmed and passed to [`Parameter::try_from`]. This avoids building a
    /// custom expression AST and reuses the existing symbolic math parser.
    fn parse_expr(&mut self, stop: impl Fn(&TokenKind) -> bool) -> Result<Parameter, ParseError> {
        let start = self.current.pos;
        let mut paren_depth = 0usize;
        while self.current.kind != TokenKind::Eof {
            if paren_depth == 0 && stop(&self.current.kind) {
                break;
            }
            match self.current.kind {
                TokenKind::LParen => paren_depth += 1,
                TokenKind::RParen if paren_depth > 0 => paren_depth -= 1,
                _ => {}
            }
            self.advance()?;
        }
        let end = self.current.pos;
        let expr_str = self.input[start..end].trim();
        if expr_str.is_empty() {
            return Err(ParseError::EmptyExpr { pos: start });
        }
        Parameter::try_from(expr_str).map_err(|e| ParseError::InvalidExpr {
            expr: expr_str.to_string(),
            reason: format!("{}", e),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::ParameterValue;

    #[test]
    fn parse_cancel_rule() {
        let input = r#"
            rule cancel_rx_inverse {
                match {
                    RX(a) 0
                    RX(b) 0
                }
                require {
                    a + b == 0 mod 2*π
                }
                rewrite {
                }
            }

            rule cancel_h {
                match {
                    H 0
                    H 0
                }
                rewrite {
                }
            }
        "#;
        let mut parser = Parser::new(input).unwrap();
        let rules = parser.parse_rule_file().unwrap();
        assert_eq!(rules.len(), 2);

        // cancel_rx_inverse
        let r0 = rules[0].clone().into_rule().unwrap();
        assert_eq!(r0.name, "cancel_rx_inverse");
        assert_eq!(r0.operations.len(), 2);
        assert_eq!(r0.conditions.as_ref().unwrap().len(), 1);
        assert!(r0.target.is_empty());

        // cancel_h
        let r1 = rules[1].clone().into_rule().unwrap();
        assert_eq!(r1.name, "cancel_h");
        assert_eq!(r1.operations.len(), 2);
        assert!(r1.conditions.is_none());
        assert!(r1.target.is_empty());
    }

    #[test]
    fn parse_merge_rule() {
        let input = r#"
            rule merge_rz {
                match {
                    RZ(a) 0
                    RZ(b) 0
                }
                rewrite {
                    RZ(a + b) 0
                }
            }
        "#;
        let mut parser = Parser::new(input).unwrap();
        let rules = parser.parse_rule_file().unwrap();
        assert_eq!(rules.len(), 1);

        let r = rules[0].clone().into_rule().unwrap();
        assert_eq!(r.name, "merge_rz");
        assert_eq!(r.operations.len(), 2);
        assert_eq!(r.target.len(), 1);

        let target = &r.target[0];
        assert_eq!(target.qubits.as_slice(), &[0]);
        let params = target.params.as_ref().unwrap();
        assert_eq!(params.len(), 1);
        match &params[0] {
            ParameterValue::Param(p) => {
                assert_eq!(p.to_string(), "a + b");
            }
            _ => panic!("expected Expr pattern"),
        }
    }

    #[test]
    fn parse_newline_separated_patterns() {
        let input = r#"rule test {
            match { H 0
            H 0
            H 0 }
            rewrite {}
        }"#;
        let mut parser = Parser::new(input).unwrap();
        let rules = parser.parse_rule_file().unwrap();
        let r = rules[0].clone().into_rule().unwrap();
        assert_eq!(r.operations.len(), 3);
    }

    #[test]
    fn parse_comma_separated_patterns() {
        let input = r#"rule test {
            match { H 0, H 0, H 0 }
            rewrite {}
        }"#;
        let mut parser = Parser::new(input).unwrap();
        let rules = parser.parse_rule_file().unwrap();
        let r = rules[0].clone().into_rule().unwrap();
        assert_eq!(r.operations.len(), 3);
    }

    #[test]
    fn reject_same_line_patterns_without_comma() {
        let input = r#"rule test {
            match { H 0 H 0 }
            rewrite {}
        }"#;
        let mut parser = Parser::new(input).unwrap();
        assert!(parser.parse_rule_file().is_err());
    }

    #[test]
    fn parse_nested_parenthesized_param_expr() {
        let input = r#"rule test {
            match { RZ(a) 0 }
            rewrite { RZ((a + 1)) 0 }
        }"#;
        let mut parser = Parser::new(input).unwrap();
        let rules = parser.parse_rule_file().unwrap();
        let r = rules[0].clone().into_rule().unwrap();
        assert_eq!(r.target.len(), 1);
    }

    #[test]
    fn parse_gphase_rule() {
        let input = r#"
            rule merge_gphase {
                match { GPhase(a), GPhase(b) }
                rewrite { GPhase(a + b) }
            }
            rule cancel_gphase_inverse {
                match { GPhase(a), GPhase(b) }
                require { a + b == 0 mod 2*π }
                rewrite {}
            }
        "#;
        let mut parser = Parser::new(input).unwrap();
        let rules = parser.parse_rule_file().unwrap();
        assert_eq!(rules.len(), 2);

        let r0 = rules[0].clone().into_rule().unwrap();
        assert_eq!(r0.name, "merge_gphase");
        assert_eq!(r0.operations.len(), 2);
        assert!(r0.operations[0].qubits.is_empty());
        assert_eq!(r0.target.len(), 1);
        assert!(r0.target[0].qubits.is_empty());

        let r1 = rules[1].clone().into_rule().unwrap();
        assert_eq!(r1.name, "cancel_gphase_inverse");
        assert_eq!(r1.operations.len(), 2);
        assert!(r1.operations[0].qubits.is_empty());
        assert_eq!(r1.conditions.as_ref().unwrap().len(), 1);
        assert!(r1.target.is_empty());
    }

    #[test]
    fn parse_multi_controlled_gate_pattern() {
        let input = r#"rule decompose_m3cx {
            match { MCX[3] 0 1 2 3 }
            rewrite { CCX 0 1 2 }
        }"#;
        let mut parser = Parser::new(input).unwrap();
        let rules = parser.parse_rule_file().unwrap();
        assert_eq!(rules.len(), 1);
        let pattern = &rules[0].match_ops[0];
        assert_eq!(
            pattern.gate,
            GateSpec::MultiControlled {
                base_gate_name: "X".to_string(),
                added_controls: 3
            }
        );
        assert_eq!(pattern.qubits, vec![0, 1, 2, 3]);
    }

    #[test]
    fn parse_parameterized_multi_controlled_gate_pattern() {
        let input = r#"rule decompose_m2rz {
            match { MCRZ[2](theta) 0 1 2 }
            rewrite { CRZ(theta) 1 2 }
        }"#;
        let mut parser = Parser::new(input).unwrap();
        let rules = parser.parse_rule_file().unwrap();
        assert_eq!(rules[0].match_ops[0].params.len(), 1);
        assert_eq!(
            rules[0].match_ops[0].gate,
            GateSpec::MultiControlled {
                base_gate_name: "RZ".to_string(),
                added_controls: 2
            }
        );
    }

    #[test]
    fn reject_bracketed_non_multi_controlled_gate_name() {
        let input = r#"rule bad {
            match { X[3] 0 1 2 3 }
            rewrite {}
        }"#;
        let mut parser = Parser::new(input).unwrap();
        assert!(matches!(
            parser.parse_rule_file(),
            Err(ParseError::InvalidMultiControlledGateName { .. })
        ));
    }

    #[test]
    fn reject_fractional_multi_control_count() {
        let input = r#"rule bad {
            match { MCX[1.5] 0 1 }
            rewrite {}
        }"#;
        let mut parser = Parser::new(input).unwrap();
        assert!(matches!(
            parser.parse_rule_file(),
            Err(ParseError::UnexpectedToken { .. })
        ));
    }
}
