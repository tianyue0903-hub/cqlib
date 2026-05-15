//! Gate and circuit decomposition transformers.
//!
//! This module owns compiler-level decomposition and target-basis lowering.
//! The production [`Decomposer`] is intentionally small: it lowers standard
//! gates toward an explicitly configured basis, or falls back to the current
//! [`CompilerContext`] device native standard gates when no explicit basis is
//! configured.
//!
//! The remaining submodules provide decomposition primitives used by this
//! compiler layer. They are crate-visible building blocks rather than public
//! user APIs: clean-ancilla MCX expansion, MCX-like multi-controlled gate
//! lowering, numeric single-qubit unitary synthesis, and the numeric two-qubit
//! KAK primitive.
//!
//! [`CompilerContext`]: crate::compiler::CompilerContext

mod config;
mod decomposer;
mod mc_gate;
pub mod mcx;
pub mod one_qubit_unitary;
pub mod two_qubit_kak;

pub use config::DecomposeConfig;
pub use decomposer::Decomposer;
pub use mc_gate::decompose::{AncillaMode, McGateDecomposeConfig};
pub use mc_gate::transformer::McGateDecomposer;

#[cfg(test)]
#[path = "./decompose_test.rs"]
mod decompose_test;

#[cfg(test)]
#[path = "./mcx_test.rs"]
mod mcx_test;

#[cfg(test)]
#[path = "./one_qubit_unitary_test.rs"]
mod one_qubit_unitary_test;

#[cfg(test)]
#[path = "./two_qubit_kak_test.rs"]
mod two_qubit_kak_test;
