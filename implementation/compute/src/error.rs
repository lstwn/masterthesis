use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
/// Public error type for any Incremental Datalog error.
pub enum IncLogError {
    #[error(transparent)]
    Syntax(#[from] SyntaxError),
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
}

#[derive(Error, Debug, Clone, PartialEq, Eq)]
#[error("{message}")]
/// An error that occurs during parsing or static analysis at compile time.
pub struct SyntaxError {
    // TODO: source location
    pub message: String,
}

impl SyntaxError {
    pub fn new<T: Into<String>>(message: T) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[derive(Error, Debug, Clone, PartialEq, Eq)]
#[error("{message}")]
/// An error that occurs during run time.
pub struct RuntimeError {
    message: String,
}

impl RuntimeError {
    pub fn new<T: Into<String>>(message: T) -> Self {
        Self {
            message: message.into(),
        }
    }
}
