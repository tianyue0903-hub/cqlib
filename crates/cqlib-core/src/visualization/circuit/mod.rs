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

//! # Circuit Visualization Module
//!
//! This module provides a complete visualization pipeline for
//! [`Circuit`](crate::circuit::Circuit): from backend-agnostic IR construction
//! to concrete text/figure rendering.
//!
//! ## Core Components
//!
//! - **IR builder**: [`build_visual_circuit`] converts circuit operations into layered
//!   [`VisualCircuit`] IR.
//! - **Text drawer**: [`circuit_to_text`] renders Unicode box-drawing circuit diagrams.
//! - **Figure drawer**: [`circuit_to_figure`] and [`render_figure_to_file`]
//!   generate SVG/PNG outputs.

pub mod builder;
pub mod error;
pub mod figure;
pub mod ir_utils;
pub mod model;
pub mod parameter_formatter;
pub mod style;
pub mod text;
pub use builder::{VisualBuildOptions, build_visual_circuit};
pub use error::VisualizationError;
pub use figure::{FigureDrawerOptions, FigureDrawStyle, circuit_to_figure, render_figure_to_file};
pub use model::{
    VisualChildren, VisualCircuit, VisualCondition, VisualControlFlowKind, VisualOpStyle,
    VisualOperation,
};
pub use parameter_formatter::{ParameterDisplayMode, ParameterFormatOptions, ParameterFormatter};
pub use style::GateStyle;
pub use text::{TextDrawerOptions, circuit_to_text};

#[cfg(test)]
#[path = "builder_tests.rs"]
mod builder_tests;

#[cfg(test)]
#[path = "error_tests.rs"]
mod error_tests;

#[cfg(test)]
#[path = "figure_tests.rs"]
mod figure_tests;

#[cfg(test)]
#[path = "model_tests.rs"]
mod model_tests;

#[cfg(test)]
#[path = "parameter_formatter_tests.rs"]
mod parameter_formatter_tests;

#[cfg(test)]
#[path = "style_tests.rs"]
mod style_tests;

#[cfg(test)]
#[path = "text_tests.rs"]
mod text_tests;
