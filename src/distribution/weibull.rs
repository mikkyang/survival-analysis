use super::{CumulativeHazard, LogCumulativeDensity, LogHazard, Survival};
use crate::error::Error;
use crate::sample::fitter::InitialSolvePoint;
use crate::sample::{IntervalCensored, LeftCensored, RightCensored};
use crate::utils::SafeLogExp;
use ndarray::{Array, ArrayBase, Data, Dimension, Ix1, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::convert::TryFrom;
use std::ops::{Neg, Sub};

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct WeibullDistribution<F> {
    pub shape: F,
    pub scale: F,
}

impl<F> From<WeibullDistribution<F>> for Vec<F> {
    fn from(distribution: WeibullDistribution<F>) -> Self {
        vec![distribution.scale, distribution.shape]
    }
}

impl<'a, F> TryFrom<&'a [F]> for WeibullDistribution<F>
where
    F: Copy,
{
    type Error = Error;

    fn try_from(array: &'a [F]) -> Result<Self, Self::Error> {
        if array.len() != 2 {
            return Err(Error::IncompatibleDistributionParameterCount(
                array.len(),
                2,
            ));
        }

        Ok(WeibullDistribution {
            scale: array[0],
            shape: array[1],
        })
    }
}

impl<S, D, F> LogHazard<ArrayBase<S, D>, Array<F, D>> for WeibullDistribution<F>
where
    S: Data<Elem = F>,
    D: Dimension,
    F: Float + ScalarOperand,
{
    fn log_hazard(&self, input: &ArrayBase<S, D>) -> Array<F, D> {
        let rho = self.shape;
        let lambda = self.scale;

        // calculate scalars first to avoid applying to entire array
        let scalar: F = rho.ln() - (rho * lambda.ln());
        let array = input.mapv(F::ln) * (rho - F::one());

        array + scalar
    }
}

impl<S, D, F> LogHazard<ArrayBase<S, D>, F> for WeibullDistribution<F>
where
    S: Data<Elem = F>,
    D: Dimension,
    F: Float + FromPrimitive + ScalarOperand,
{
    fn log_hazard(&self, input: &ArrayBase<S, D>) -> F {
        let rho = self.shape;
        let lambda = self.scale;

        let n = F::from_usize(input.len()).unwrap();
        n * (rho.ln() - (rho * lambda.ln())) + (rho - F::one()) * input.mapv(F::ln).sum()
    }
}

impl<S, D, F> CumulativeHazard<ArrayBase<S, D>, Array<F, D>> for WeibullDistribution<F>
where
    S: Data<Elem = F>,
    D: Dimension,
    F: Float + SafeLogExp + ScalarOperand,
{
    fn cumulative_hazard(&self, input: &ArrayBase<S, D>) -> Array<F, D> {
        let rho = self.shape;
        let lambda = self.scale;

        let log_cumulative_hazard = (input.mapv(SafeLogExp::safe_ln) - lambda.ln()) * rho;
        log_cumulative_hazard.mapv_into(SafeLogExp::safe_exp)
    }
}

impl<S, D, F> CumulativeHazard<ArrayBase<S, D>, F> for WeibullDistribution<F>
where
    S: Data<Elem = F>,
    D: Dimension,
    F: Float + SafeLogExp + ScalarOperand,
{
    fn cumulative_hazard(&self, input: &ArrayBase<S, D>) -> F {
        CumulativeHazard::<_, Array<F, D>>::cumulative_hazard(self, input).sum()
    }
}

impl<S, D, F> Survival<ArrayBase<S, D>, Array<F, D>> for WeibullDistribution<F>
where
    S: Data<Elem = F>,
    D: Dimension,
    F: Float + SafeLogExp + ScalarOperand + Neg,
{
    fn survival(&self, input: &ArrayBase<S, D>) -> Array<F, D> {
        let cumulative_hazard: Array<F, D> = self.cumulative_hazard(input);
        cumulative_hazard.mapv_into(|x| (-x).exp())
    }
}

impl<S, D, F> Survival<ArrayBase<S, D>, F> for WeibullDistribution<F>
where
    S: Data<Elem = F>,
    D: Dimension,
    F: Float + SafeLogExp + ScalarOperand + Neg,
{
    fn survival(&self, input: &ArrayBase<S, D>) -> F {
        Survival::<_, Array<F, D>>::survival(self, input).sum()
    }
}

impl<S, D, F> LogCumulativeDensity<ArrayBase<S, D>, Array<F, D>> for WeibullDistribution<F>
where
    S: Data<Elem = F>,
    D: Dimension,
    F: Float + SafeLogExp + ScalarOperand + Neg + Sub<Array<F, D>, Output = Array<F, D>>,
{
    fn log_cumulative_density(&self, input: &ArrayBase<S, D>) -> Array<F, D> {
        let array: Array<F, D> = self.survival(input);
        (F::one() - array).mapv_into(F::ln)
    }
}

impl<S, D, F> LogCumulativeDensity<ArrayBase<S, D>, F> for WeibullDistribution<F>
where
    S: Data<Elem = F>,
    D: Dimension,
    F: Float + SafeLogExp + ScalarOperand + Neg + Sub<Array<F, D>, Output = Array<F, D>>,
{
    fn log_cumulative_density(&self, input: &ArrayBase<S, D>) -> F {
        let array: Array<F, D> = self.log_cumulative_density(input);
        array.sum()
    }
}

impl<S, F> InitialSolvePoint<WeibullDistribution<F>> for RightCensored<S, Ix1>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive,
{
    fn initial_solve_point(&self) -> WeibullDistribution<F> {
        let lambda = self.0.mean().unwrap_or_else(F::zero);
        WeibullDistribution {
            shape: F::one(),
            scale: lambda,
        }
    }
}

impl<S, F> InitialSolvePoint<WeibullDistribution<F>> for LeftCensored<S, Ix1>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive,
{
    fn initial_solve_point(&self) -> WeibullDistribution<F> {
        let lambda = self.0.mean().unwrap_or_else(F::zero);
        WeibullDistribution {
            shape: F::one(),
            scale: lambda,
        }
    }
}

impl<S, F> InitialSolvePoint<WeibullDistribution<F>> for IntervalCensored<S, Ix1>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive,
{
    fn initial_solve_point(&self) -> WeibullDistribution<F> {
        let lambda = self.start.mean().unwrap_or_else(F::zero);
        WeibullDistribution {
            shape: F::one(),
            scale: lambda,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::univariate::*;
    use crate::sample::*;
    use ndarray::prelude::*;

    const TOLERANCE_F32: f32 = 1e-5;
    const TOLERANCE_F64: f64 = 1e-5;

    #[test]
    fn log_hazard_f32() {
        let distribution = WeibullDistribution {
            shape: 0.5f32,
            scale: 1.,
        };

        let actual = distribution.log_hazard(&array![1., 2., 3., 4.]);
        let expected = array![-0.69314718, -1.03972077, -1.24245332, -1.38629436];
        assert_diff_within_tolerance!(&actual, &expected, TOLERANCE_F32);
    }

    #[test]
    fn log_hazard_f64() {
        let distribution = WeibullDistribution {
            shape: 1.5f64,
            scale: 2.,
        };

        let actual = distribution.log_hazard(&array![5., 6., 7., 8.]);
        let expected = array![0.17046329, 0.26162407, 0.33869941, 0.40546511];
        assert_diff_within_tolerance!(&actual, &expected, TOLERANCE_F64);
    }

    #[test]
    fn cumulative_hazard() {
        let distribution = WeibullDistribution {
            shape: 2.0f64,
            scale: 1.4,
        };

        let actual = distribution.cumulative_hazard(&array![[5., 6.], [7., 8.]]);
        let expected = array![[12.75510204, 18.36734694,], [25., 32.65306122,]];
        assert_diff_within_tolerance!(&actual, &expected, TOLERANCE_F64);
    }

    #[test]
    fn log_likelihood_right() {
        let distribution = WeibullDistribution {
            shape: 1.3,
            scale: 2.3,
        };

        let durations = array![1., 2., 3., 4.];

        let events: PartiallyObserved<_, _, RightCensored<_, _>> =
            PartiallyObserved::from_events(&durations.view(), &[true, false, true, false]);

        let actual: f64 = events.log_likelihood(&distribution);
        let expected = -5.949540344836688;

        assert!((actual - expected).abs() < TOLERANCE_F64);
    }

    #[test]
    fn log_likelihood_left() {
        let distribution = WeibullDistribution {
            shape: 0.7,
            scale: 0.5,
        };

        let events: PartiallyObserved<_, _, LeftCensored<_, _>> = PartiallyObserved::from_events(
            &array![1., 2., 3., 4.],
            array![true, false, true, false].view(),
        );

        let actual: f64 = events.log_likelihood(&distribution);
        let expected = -5.290127712264656;

        assert!((actual - expected).abs() < TOLERANCE_F64);
    }

    #[test]
    fn log_likelihood_interval() {
        let distribution = WeibullDistribution {
            shape: 1.7,
            scale: 0.5,
        };

        let events: PartiallyObserved<_, _, IntervalCensored<_, _>> =
            PartiallyObserved::from_events(
                &[(1., 5.), (2., 6.), (3., 7.), (4., 8.), (5., 9.)],
                &array![true, false, true, false, true],
            );

        let actual: f64 = events.log_likelihood(&distribution);
        let expected = -310.7517121011837;

        assert!((actual - expected).abs() < TOLERANCE_F64);
    }

    #[test]
    fn log_likelihood_interval_weights() {
        let distribution = WeibullDistribution {
            shape: 1.0170410407859767f64,
            scale: 2.0410538960706726,
        };

        let events = Weighted {
            time: IntervalCensored {
                start: array![0., 2., 5., 10.],
                stop: array![2., 5., 10., 1e10],
            },
            weight: array![1000. - 376., 376. - 82., 82. - 7., 7.],
        };

        let actual: f64 = events.log_likelihood(&distribution);
        let expected = -0.8832316840607934;

        assert!((actual - expected).abs() < TOLERANCE_F64);
    }
}
