#[cfg(doctest)]
use doc_comment::doctest;

#[cfg(test)]
#[macro_use]
mod tests {
    #[macro_export]
    macro_rules! assert_diff_within_tolerance {
        ($actual: expr, $expected: expr, $tolerance: expr) => {
            for diff in ($actual - $expected).iter() {
                assert!(diff.abs() < $tolerance);
            }
        };
    }
}

pub mod distribution;
pub mod error;
pub mod sample;
mod utils;

pub use error::Error;
pub use sample::fitter::{BaseFitter, Fitter};
pub use sample::{IntervalCensored, LeftCensored, PartiallyObserved, RightCensored, Weighted};

#[cfg(doctest)]
doctest!("../README.md");
