use super::LogHazard;
use ndarray::{Array, ArrayBase, Data, Dimension, ScalarOperand};
use num_traits::Float;

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct WeibullDistribution<F> {
    pub rho: F,
    pub lambda: F,
}

impl<S, D, F> LogHazard<ArrayBase<S, D>, Array<F, D>> for WeibullDistribution<F>
where
    S: Data<Elem = F>,
    D: Dimension,
    F: Float + ScalarOperand,
{
    fn log_hazard(&self, input: ArrayBase<S, D>) -> Array<F, D> {
        let WeibullDistribution { rho, lambda } = *self;

        // calculate scalars first to avoid applying to entire array
        let scalar: F = rho.ln() - (rho * lambda.ln());
        let array = input.mapv(F::ln) * (rho - F::one());

        array + scalar
    }
}
