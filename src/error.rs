use self::Error::*;
use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// This error occurs when converting between vectors and distribution types
    /// Can probably be removed with constant generics https://github.com/rust-lang/rust/issues/44580
    IncompatibleDistributionParameterCount(usize, usize),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            IncompatibleDistributionParameterCount(vec, dist) =>
                write!(f, "Error converting between vector of {} elements and distribution with {} parameters", vec, dist),
        }
    }
}

impl std::error::Error for Error {}
