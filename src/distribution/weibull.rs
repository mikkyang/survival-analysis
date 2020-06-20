use super::{CumulativeHazard, LogCumulativeDensity, LogHazard, Survival};
use crate::sample::univariate::{
    IntervalCensoredDuration, LeftCensoredDuration, RightCensoredDuration,
};
use crate::sample::InitialSolvePoint;
use crate::utils::SafeLogExp;
use ndarray::{Array, ArrayBase, Data, Dimension, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::ops::{Neg, Sub};

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct WeibullDistribution<F> {
    pub rho: F,
    pub lambda: F,
}

impl<F> From<WeibullDistribution<F>> for Vec<F> {
    fn from(distribution: WeibullDistribution<F>) -> Self {
        vec![distribution.lambda, distribution.rho]
    }
}

impl<'a, F> From<&'a Vec<F>> for WeibullDistribution<F>
where
    F: Copy,
{
    fn from(array: &'a Vec<F>) -> Self {
        WeibullDistribution {
            lambda: array[0],
            rho: array[1],
        }
    }
}

impl<S, D, F> LogHazard<ArrayBase<S, D>, Array<F, D>> for WeibullDistribution<F>
where
    S: Data<Elem = F>,
    D: Dimension,
    F: Float + ScalarOperand,
{
    fn log_hazard(&self, input: &ArrayBase<S, D>) -> Array<F, D> {
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
    fn cumulative_hazard(&self, input: &ArrayBase<S, D>) -> Array<F, D> {
        let WeibullDistribution { rho, lambda } = *self;

        let log = (input.mapv(SafeLogExp::safe_ln) - lambda.ln()) * rho;
        log.mapv_into(SafeLogExp::safe_exp)
    }
}

impl<S, D, F> Survival<ArrayBase<S, D>, Array<F, D>> for WeibullDistribution<F>
where
    S: Data<Elem = F>,
    D: Dimension,
    F: Float + SafeLogExp + ScalarOperand + Neg,
{
    fn survival(&self, input: &ArrayBase<S, D>) -> Array<F, D> {
        self.cumulative_hazard(input).mapv_into(|x| (-x).exp())
    }
}

impl<S, D, F> LogCumulativeDensity<ArrayBase<S, D>, Array<F, D>> for WeibullDistribution<F>
where
    S: Data<Elem = F>,
    D: Dimension,
    F: Float + SafeLogExp + ScalarOperand + Neg + Sub<Array<F, D>, Output = Array<F, D>>,
{
    fn log_cumulative_density(&self, input: &ArrayBase<S, D>) -> Array<F, D> {
        (F::one() - self.survival(input)).mapv_into(|x| x.ln())
    }
}

impl<S, F> InitialSolvePoint<WeibullDistribution<F>> for RightCensoredDuration<S, F>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive,
{
    fn initial_solve_point(&self) -> WeibullDistribution<F> {
        let lambda = self.duration.mean().unwrap_or_else(F::zero);
        WeibullDistribution {
            rho: F::one(),
            lambda,
        }
    }
}

impl<S, F> InitialSolvePoint<WeibullDistribution<F>> for LeftCensoredDuration<S, F>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive,
{
    fn initial_solve_point(&self) -> WeibullDistribution<F> {
        let lambda = self.duration.mean().unwrap_or_else(F::zero);
        WeibullDistribution {
            rho: F::one(),
            lambda,
        }
    }
}

impl<S, F> InitialSolvePoint<WeibullDistribution<F>> for IntervalCensoredDuration<S, F>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive,
{
    fn initial_solve_point(&self) -> WeibullDistribution<F> {
        let lambda = self.start_time.mean().unwrap_or_else(F::zero);
        WeibullDistribution {
            rho: F::one(),
            lambda,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::univariate::*;
    use crate::sample::LogLikelihood;
    use ndarray::prelude::*;

    const TOLERANCE_F32: f32 = 1e-5;
    const TOLERANCE_F64: f64 = 1e-5;

    #[test]
    fn log_hazard_f32() {
        let distribution = WeibullDistribution {
            rho: 0.5f32,
            lambda: 1.,
        };

        let actual = distribution.log_hazard(&array![1., 2., 3., 4.]);
        let expected = array![-0.69314718, -1.03972077, -1.24245332, -1.38629436];
        assert_diff_within_tolerance!(&actual, &expected, TOLERANCE_F32);
    }

    #[test]
    fn log_hazard_f64() {
        let distribution = WeibullDistribution {
            rho: 1.5f64,
            lambda: 2.,
        };

        let actual = distribution.log_hazard(&array![5., 6., 7., 8.]);
        let expected = array![0.17046329, 0.26162407, 0.33869941, 0.40546511];
        assert_diff_within_tolerance!(&actual, &expected, TOLERANCE_F64);
    }

    #[test]
    fn cumulative_hazard() {
        let distribution = WeibullDistribution {
            rho: 2.0f64,
            lambda: 1.4,
        };

        let actual = distribution.cumulative_hazard(&array![[5., 6.], [7., 8.]]);
        let expected = array![[12.75510204, 18.36734694,], [25., 32.65306122,]];
        assert_diff_within_tolerance!(&actual, &expected, TOLERANCE_F64);
    }

    #[test]
    fn log_likelihood_right() {
        let distribution = WeibullDistribution {
            rho: 1.3f64,
            lambda: 2.3,
        };

        let durations = array![1., 2., 3., 4.];

        let events = Events {
            time: RightCensoredDuration {
                duration: durations.view(),
            },
            observed: array![true, false, true, false],
            weight: (),
            truncation: (),
        };

        let actual = events.log_likelihood(&distribution);
        let expected = -1.487385086209172;

        assert!((actual - expected).abs() < TOLERANCE_F64);
    }

    #[test]
    fn log_likelihood_left() {
        let distribution = WeibullDistribution {
            rho: 0.7f64,
            lambda: 0.5,
        };

        let events = Events {
            time: LeftCensoredDuration {
                duration: array![1., 2., 3., 4.],
            },
            observed: array![true, false, true, false],
            weight: (),
            truncation: (),
        };

        let actual = events.log_likelihood(&distribution);
        let expected = -1.322531928066164;

        assert!((actual - expected).abs() < TOLERANCE_F64);
    }

    #[test]
    fn log_likelihood_interval() {
        let distribution = WeibullDistribution {
            rho: 1.7f64,
            lambda: 0.5,
        };

        let events = Events {
            time: IntervalCensoredDuration {
                start_time: array![1., 2., 3., 4., 5.],
                stop_time: array![5., 6., 7., 8., 9.],
            },
            observed: array![true, false, true, false, true],
            weight: (),
            truncation: (),
        };

        let actual = events.log_likelihood(&distribution);
        let expected = -62.15034242023675;

        assert!((actual - expected).abs() < TOLERANCE_F64);
    }

    #[test]
    fn log_likelihood_interval_weights() {
        let distribution = WeibullDistribution {
            rho: 1.0170410407859767f64,
            lambda: 2.0410538960706726,
        };

        let events = Events {
            time: IntervalCensoredDuration {
                start_time: array![0., 2., 5., 10.],
                stop_time: array![2., 5., 10., 1e10f64],
            },
            observed: Array::from_elem((4,), false),
            weight: array![1000. - 376., 376. - 82., 82. - 7., 7.],
            truncation: (),
        };

        let actual = events.log_likelihood(&distribution);
        let expected = -0.8832316840607934;

        assert!((actual - expected).abs() < TOLERANCE_F64);
    }
}
