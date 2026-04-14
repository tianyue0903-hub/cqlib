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

use crate::compiler::context::CompilerContext;
use crate::compiler::error::CompilerError;
use crate::compiler::transform::descriptor::TransformDescriptor;

/// Lightweight execution result returned by a transformer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TransformOutcome {
    pub changed: bool,
    pub notes: Vec<String>,
}

impl TransformOutcome {
    /// Creates an outcome that reports no state change.
    pub fn unchanged() -> Self {
        Self::default()
    }

    /// Creates an outcome that reports a state change.
    pub fn changed() -> Self {
        Self {
            changed: true,
            notes: Vec::new(),
        }
    }

    /// Appends a diagnostic note to the outcome.
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }
}

/// Smallest executable unit in the compiler transform layer.
pub trait Transformer {
    fn descriptor(&self) -> &'static TransformDescriptor;

    fn transform(&self, ctx: &mut CompilerContext) -> Result<TransformOutcome, CompilerError>;

    fn run(&self, ctx: &mut CompilerContext) -> Result<TransformOutcome, CompilerError> {
        self.descriptor().validate(ctx)?;
        self.transform(ctx)
    }
}

/// Sequential composition of multiple transformers.
pub struct CompositeTransformer {
    descriptor: &'static TransformDescriptor,
    transformers: Vec<Box<dyn Transformer>>,
}

impl CompositeTransformer {
    pub fn new(
        descriptor: &'static TransformDescriptor,
        transformers: Vec<Box<dyn Transformer>>,
    ) -> Self {
        Self {
            descriptor,
            transformers,
        }
    }
}

impl Transformer for CompositeTransformer {
    fn descriptor(&self) -> &'static TransformDescriptor {
        self.descriptor
    }

    fn transform(&self, ctx: &mut CompilerContext) -> Result<TransformOutcome, CompilerError> {
        let mut aggregate = TransformOutcome::unchanged();

        for transformer in &self.transformers {
            let outcome = transformer.run(ctx)?;
            aggregate.changed |= outcome.changed;
            aggregate.notes.extend(outcome.notes);
        }

        Ok(aggregate)
    }
}

#[cfg(test)]
mod tests {
    use super::{CompositeTransformer, TransformOutcome, Transformer};
    use crate::circuit::Circuit;
    use crate::compiler::context::CompilerContext;
    use crate::compiler::error::CompilerError;
    use crate::compiler::transform::descriptor::TransformDescriptor;

    struct MetadataTagger;

    static TAGGER_DESCRIPTOR: TransformDescriptor =
        TransformDescriptor::new("test.metadata", "Annotate workflow metadata");

    impl Transformer for MetadataTagger {
        fn descriptor(&self) -> &'static TransformDescriptor {
            &TAGGER_DESCRIPTOR
        }

        fn transform(&self, ctx: &mut CompilerContext) -> Result<TransformOutcome, CompilerError> {
            ctx.metadata_mut().tags.push("tagged".to_string());
            Ok(TransformOutcome::changed().with_note("metadata updated"))
        }
    }

    struct NoOp;

    static NOOP_DESCRIPTOR: TransformDescriptor =
        TransformDescriptor::new("test.noop", "No-op transformer");

    impl Transformer for NoOp {
        fn descriptor(&self) -> &'static TransformDescriptor {
            &NOOP_DESCRIPTOR
        }

        fn transform(&self, _ctx: &mut CompilerContext) -> Result<TransformOutcome, CompilerError> {
            Ok(TransformOutcome::unchanged())
        }
    }

    struct DeviceOnly;

    static DEVICE_ONLY_DESCRIPTOR: TransformDescriptor =
        TransformDescriptor::new("test.device", "Needs a device").requires_device();

    impl Transformer for DeviceOnly {
        fn descriptor(&self) -> &'static TransformDescriptor {
            &DEVICE_ONLY_DESCRIPTOR
        }

        fn transform(&self, _ctx: &mut CompilerContext) -> Result<TransformOutcome, CompilerError> {
            Ok(TransformOutcome::changed())
        }
    }

    #[test]
    fn unchanged_outcome_is_default() {
        let outcome = TransformOutcome::unchanged();

        assert!(!outcome.changed);
        assert!(outcome.notes.is_empty());
    }

    #[test]
    fn changed_outcome_records_note() {
        let outcome = TransformOutcome::changed().with_note("rewritten");

        assert!(outcome.changed);
        assert_eq!(outcome.notes, vec!["rewritten"]);
    }

    #[test]
    fn composite_transformer_aggregates_changes_and_notes() {
        static COMPOSITE_DESCRIPTOR: TransformDescriptor =
            TransformDescriptor::new("test.composite", "Composite transformer");

        let composite = CompositeTransformer::new(
            &COMPOSITE_DESCRIPTOR,
            vec![Box::new(NoOp), Box::new(MetadataTagger)],
        );
        let mut ctx = CompilerContext::new(Circuit::new(1));

        let outcome = composite.run(&mut ctx).unwrap();

        assert!(outcome.changed);
        assert_eq!(outcome.notes, vec!["metadata updated"]);
        assert_eq!(ctx.metadata().tags, vec!["tagged"]);
    }

    #[test]
    fn run_validates_descriptor_requirements_before_transform() {
        let transformer = DeviceOnly;
        let mut ctx = CompilerContext::new(Circuit::new(1));

        let err = transformer.run(&mut ctx).unwrap_err();
        assert!(matches!(err, CompilerError::MissingDevice));
    }
}
