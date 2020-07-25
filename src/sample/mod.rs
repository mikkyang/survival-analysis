use ndarray::{ArrayBase, RawData};

pub mod fitter;
pub mod univariate;

/// The log likelihood of a data based on a distribution.
pub trait LogLikelihood<Distribution, F> {
    fn log_likelihood(&self, distribution: &Distribution) -> F;
}

/// Data that is partially observed and partially censored.
pub struct PartiallyObserved<T: RawData, D, C> {
    pub observed: Uncensored<T, D>,
    pub censored: C,
}

/// Data that is completely uncensored.
pub struct Uncensored<T: RawData, D>(pub ArrayBase<T, D>);

/// Data that is right censored.
pub struct RightCensored<T: RawData, D>(pub ArrayBase<T, D>);

/// Data that is left censored.
pub struct LeftCensored<T: RawData, D>(pub ArrayBase<T, D>);

/// Data that is interval censored.
pub struct IntervalCensored<T: RawData, D> {
    pub start: ArrayBase<T, D>,
    pub stop: ArrayBase<T, D>,
}

/// A wrapper around data to apply weights to log likelihoods.
pub struct Weighted<T, W: RawData, D> {
    pub time: T,
    pub weight: ArrayBase<W, D>,
}

/// The entry times for left truncated data.
pub struct LeftTruncation<T: RawData, D>(ArrayBase<T, D>);
