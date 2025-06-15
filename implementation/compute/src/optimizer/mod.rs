use crate::{error::SyntaxError, stmt::Code};

/// The optimizer does a series of transformations on the relational algebra IR
/// which retain the semantics of the program but intend to improve performance.
#[derive(Clone, Debug, Default)]
pub struct Optimizer {}

impl Optimizer {
    /// For now, the optimizer does nothing but simply returns the IR as is.
    /// This is a placeholder for future optimization logic which could perform
    /// various transformations on the relational algebra IR such as:
    /// - predicate pushdown
    /// - expression simplification
    pub fn optimize(self, code: Code) -> Result<Code, SyntaxError> {
        Ok(code)
    }
}
