#![allow(dead_code, unused_variables)]

use compute::{error::SyntaxError, stmt::Code};

#[derive(Clone, Debug, Default)]
pub struct Optimizer {}

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
