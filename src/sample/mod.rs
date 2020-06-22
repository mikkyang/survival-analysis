use ndarray::{ArrayBase, RawData};

pub mod fitter;
pub mod univariate;

pub trait LogLikelihood<Distribution, F> {
    fn log_likelihood(&self, distribution: &Distribution) -> F;
}

pub struct PartiallyObserved<T: RawData, D, C> {
    pub observed: Uncensored<T, D>,
    pub censored: C,
}

pub struct Uncensored<T: RawData, D>(pub ArrayBase<T, D>);

pub struct RightCensored<T: RawData, D>(pub ArrayBase<T, D>);

pub struct LeftCensored<T: RawData, D>(pub ArrayBase<T, D>);

pub struct IntervalCensored<T: RawData, D> {
    pub start: ArrayBase<T, D>,
    pub stop: ArrayBase<T, D>,
}

pub struct Weighted<T, W: RawData, D> {
    pub time: T,
    pub weight: ArrayBase<W, D>,
}

pub struct LeftTruncation<T: RawData, D>(ArrayBase<T, D>);
