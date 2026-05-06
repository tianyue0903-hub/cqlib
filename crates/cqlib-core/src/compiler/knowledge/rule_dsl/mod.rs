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

//! Rule DSL parser, loader, and dumper for the cqlib compiler.
//!
//! This module provides a small domain-specific language (DSL) for describing
//! peephole optimization rules. A rule consists of:
//! - A **match** block: a sequence of gate patterns to match in the circuit.
//! - An optional **require** block: algebraic conditions that must hold for the
//!   variables appearing in the match.
//! - A **rewrite** block: the replacement gates (may be empty, meaning deletion).
//!
//! # Grammar (EBNF)
//!
//! ```ebnf
//! rule_file     ::= rule*
//!
//! rule          ::= "rule" ident "{"
//!                     "match" "{" gate_patterns "}"
//!                     ("require" "{" conditions "}")?
//!                     "rewrite" "{" gate_patterns "}"
//!                   "}"
//!
//! gate_patterns ::= [gate_pattern (","? gate_pattern)*]
//!
//! gate_pattern  ::= ident ["(" param_list ")"] qubit_list
//! param_list    ::= expr ("," expr)*
//! qubit_list    ::= number+
//!
//! conditions    ::= [condition (","? condition)*]
//! condition     ::= expr "==" expr ["mod" expr]
//! expr          ::= mathematical expression (parsed by `Parameter::try_from`)
//! ```
//!
//! # Key constraints
//!
//! - **Qubits must appear outside parentheses**: write `RX(a) 0`, not `RX(0, a)`.
//! - **Qubit numbers are rule-local logical labels**, not fixed physical qubit
//!   indices. The same label refers to the same matched qubit within one rule.
//!   Rewrite operations may only reference labels that appeared in the match block.
//! - **Multiple patterns on the same line must be separated by commas**;
//!   commas are optional between line breaks.
//! - **Line comments** starting with `//` are supported.
//! - The mathematical constant `π` is automatically resolved to `f64::consts::PI`
//!   when expressions are evaluated.
//!
//! # Example
//!
//! ```text
//! rule merge_rz {
//!     match {
//!         RZ(a) 0
//!         RZ(b) 0
//!     }
//!     rewrite {
//!         RZ(a + b) 0
//!     }
//! }
//! ```
//!
//! # Module overview
//!
//! - [`lexer`](crate::compiler::knowledge::rule_dsl::lexer) – Tokenizer that
//!   produces span-based tokens borrowing from the input string.
//! - [`parser`](crate::compiler::knowledge::rule_dsl::parser) – Recursive-descent
//!   parser that extracts raw expression text and delegates evaluation to
//!   [`Parameter::try_from`](crate::circuit::Parameter).
//! - [`ast`](crate::compiler::knowledge::rule_dsl::ast) – AST definitions (`RuleDef`, `GatePattern`).
//! - [`lower`](crate::compiler::knowledge::rule_dsl::lower) – Lowering logic from
//!   AST to the runtime `Rule` type.
//! - [`load`](crate::compiler::knowledge::rule_dsl::load) – Convenience helpers
//!   for loading rules from files or strings.
//! - [`dump`](crate::compiler::knowledge::rule_dsl::dump) – Serialization helpers
//!   for writing `Rule` or `RuleDef` back to the DSL text format.

pub mod ast;
pub mod dump;
pub mod lexer;
pub mod load;
pub mod lower;
pub mod parser;

pub use ast::{GatePattern, RuleDef};
pub use dump::{dump_rule_to_file, dump_rule_to_string, dump_rules_to_file};
pub use lexer::{LexError, Lexer, Token, TokenKind};
pub use load::{
    LoadError, load_rule_defs_from_file, load_rule_defs_from_str, load_rules_from_file,
    load_rules_from_str,
};
pub use lower::LowerError;
pub use parser::{ParseError, Parser};
