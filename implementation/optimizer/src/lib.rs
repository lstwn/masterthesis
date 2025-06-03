#![allow(dead_code, unused_variables)]

use compute::{error::SyntaxError, stmt::Code};

#[derive(Clone, Debug, Default)]
pub struct Optimizer {}

// TODO:
// Remove redundant operators:
// - [ ] Projections on relations which already have the requested schema.
//       Examples: \pi_{attrs} -> \pi_{attrs}
//                 \setminus -> \pi_{attrs}
// - [ ] As many equi joins as possible.
// - [ ] Predicate pushdown.
// - [ ] Merge Join -> Projection.

impl Optimizer {
    pub fn optimize(self, code: Code) -> Result<Code, SyntaxError> {
        Ok(code)
    }
}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn it_works() {}
}
