use ndarray::prelude::*;
use ndarray::Data;
use std::iter::FromIterator;
pub trait SafeLogExp {
    fn safe_ln(self) -> Self;

    fn safe_exp(self) -> Self;
}

const F32_SAFE_LN_MIN: f32 = 1e-25;
const F64_SAFE_LN_MIN: f64 = 1e-25;

impl SafeLogExp for f32 {
    fn safe_ln(self) -> Self {
        f32::max(self, F32_SAFE_LN_MIN).ln()
    }

    fn safe_exp(self) -> Self {
        f32::min(self, f32::MAX.ln() - 75.).exp()
    }
}

impl SafeLogExp for f64 {
    fn safe_ln(self) -> Self {
        f64::max(self, F64_SAFE_LN_MIN).ln()
    }

    fn safe_exp(self) -> Self {
        f64::min(self, f64::MAX.ln() - 75.).exp()
    }
}

pub fn filter<A, S, B, D>(target: &ArrayBase<S, D>, should_keep: &ArrayBase<B, D>) -> Array1<A>
where
    A: Copy,
    S: Data<Elem = A>,
    B: Data<Elem = bool>,
    D: Dimension,
{
    let iter = target
        .iter()
        .zip(should_keep.iter())
        .filter_map(|(x, keep)| if *keep { Some(*x) } else { None });
    Array::from_iter(iter)
}

pub fn partition<A, S, B, D>(
    target: &ArrayBase<S, D>,
    predicate: &ArrayBase<B, D>,
) -> (Array1<A>, Array1<A>)
where
    A: Copy,
    S: Data<Elem = A>,
    B: Data<Elem = bool>,
    D: Dimension,
{
    let half_capacity = target.len() / 2;
    let mut is_true = Vec::with_capacity(half_capacity);
    let mut is_false = Vec::with_capacity(half_capacity);

    for (x, pred) in target.iter().zip(predicate.iter()) {
        if *pred {
            is_true.push(*x)
        } else {
            is_false.push(*x)
        }
    }

    (Array::from(is_true), Array::from(is_false))
}
