// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Symbolic parameter system for quantum circuits.
//!
//! This module provides a complete symbolic mathematics framework for quantum computing
//! applications, supporting parameterized quantum circuits (PQC) and variational quantum
//! algorithms (VQA).
//!
//! # Module Structure
//!
//! - [`impls`]: Core [`Parameter`] type and its implementations
//! - [`expr_node`]: Abstract syntax tree (AST) nodes for expressions
//! - [`derivative`]: Symbolic differentiation engine
//! - [`simplify`]: Expression simplification and optimization
//! - [`parse`]: Mathematical expression parser
//!
//! # Quick Start
//!
//! ```rust
//! use cqlib_core::circuit::parameter::{Parameter, parse_parameter};
//! use std::collections::HashMap;
//!
//! // Create symbolic parameters
//! let theta = Parameter::try_from("theta").unwrap();
//! let phi = Parameter::try_from("phi").unwrap();
//!
//! // Build expression: sin(theta) + cos(phi)
//! let expr = theta.sin() + phi.cos();
//!
//! // Evaluate with concrete values
//! let mut bindings = HashMap::new();
//! bindings.insert("theta".to_string(), std::f64::consts::PI / 2.0); // sin(π/2) = 1
//! bindings.insert("phi".to_string(), 0.0); // cos(0) = 1
//!
//! let result = expr.evaluate(&Some(bindings)).unwrap();
//! assert!((result - 2.0).abs() < 1e-10); // 1 + 1 = 2
//! ```

pub mod derivative;
pub mod expr_node;
pub mod impls;
pub mod parse;
pub mod simplify;

pub use impls::Parameter;
pub use parse::{ParseError, parse_parameter};
