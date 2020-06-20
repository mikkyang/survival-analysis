use super::{CumulativeHazard, LogHazard};
use crate::utils::SafeLogExp;
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

impl<S, D, F> CumulativeHazard<ArrayBase<S, D>, Array<F, D>> for WeibullDistribution<F>
where
    S: Data<Elem = F>,
    D: Dimension,
    F: Float + SafeLogExp + ScalarOperand,
{
    fn cumulative_hazard(&self, input: ArrayBase<S, D>) -> Array<F, D> {
        let WeibullDistribution { rho, lambda } = *self;

        let log = (input.mapv(SafeLogExp::safe_ln) - lambda.ln()) * rho;
        log.mapv_into(SafeLogExp::safe_exp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    const TOLERANCE_F32: f32 = 1e-5;
    const TOLERANCE_F64: f64 = 1e-5;

    #[test]
    fn log_hazard_f32() {
        let distribution = WeibullDistribution {
            rho: 0.5f32,
            lambda: 1.,
        };

        let actual = distribution.log_hazard(array![1., 2., 3., 4.]);
        let expected = array![-0.69314718, -1.03972077, -1.24245332, -1.38629436];
        assert_diff_within_tolerance!(&actual, &expected, TOLERANCE_F32);
    }

    #[test]
    fn log_hazard_f64() {
        let distribution = WeibullDistribution {
            rho: 1.5f64,
            lambda: 2.,
        };

        let actual = distribution.log_hazard(array![5., 6., 7., 8.]);
        let expected = array![0.17046329, 0.26162407, 0.33869941, 0.40546511];
        assert_diff_within_tolerance!(&actual, &expected, TOLERANCE_F64);
    }

    #[test]
    fn cumulative_hazard() {
        let distribution = WeibullDistribution {
            rho: 2.0f64,
            lambda: 1.4,
        };

        let actual = distribution.cumulative_hazard(array![[5., 6.], [7., 8.]]);
        let expected = array![[12.75510204, 18.36734694,], [25., 32.65306122,]];
        assert_diff_within_tolerance!(&actual, &expected, TOLERANCE_F64);
    }
}
