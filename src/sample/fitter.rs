use super::{LogLikelihood, Weighted};
use crate::error::Error::*;
use argmin::prelude::*;
use argmin::solver::neldermead::NelderMead;
use ndarray::RawData;
use num_traits::{Float, FloatConst, FromPrimitive};
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

const NON_ZERO_DELTA: f64 = 0.05;
const ZERO_DELTA: f64 = 0.00025;

pub trait InitialSolvePoint<T> {
    fn initial_solve_point(&self) -> T;
}

impl<T, Distribution, W, D> InitialSolvePoint<Distribution> for Weighted<T, W, D>
where
    W: RawData,
    T: InitialSolvePoint<Distribution>,
{
    fn initial_solve_point(&self) -> Distribution {
        self.time.initial_solve_point()
    }
}

pub trait InitialNelderMeanSimplex<T> {
    fn initial_simplex(&self) -> Result<Vec<T>, crate::error::Error>;
}

impl<F> InitialNelderMeanSimplex<Vec<F>> for [F]
where
    F: Float + FromPrimitive,
{
    fn initial_simplex(&self) -> Result<Vec<Vec<F>>, crate::error::Error> {
        let initial_point: Vec<F> = self.into();

        let d = initial_point.len();
        let mut simplex = vec![initial_point; d + 1];
        for (index_within_point, point) in simplex.iter_mut().skip(1).enumerate() {
            if point[index_within_point] != F::zero() {
                let delta_multiple = 1.0 + NON_ZERO_DELTA;

                point[index_within_point] = F::from(delta_multiple)
                    .ok_or(NumericalConversion(delta_multiple))?
                    * point[index_within_point]
            } else {
                point[index_within_point] =
                    F::from(ZERO_DELTA).ok_or(NumericalConversion(ZERO_DELTA))?
            }
        }

        Ok(simplex)
    }
}

pub struct BaseFitter<S, D, F> {
    input_state: S,
    distribution: PhantomData<D>,
    float: PhantomData<F>,
    pub max_iterations: u64,
}

impl<S, D, F> BaseFitter<S, D, F> {
    pub fn new(data: S) -> Self {
        BaseFitter {
            input_state: data,
            distribution: PhantomData,
            float: PhantomData,
            max_iterations: 100,
        }
    }
}

impl<'f, S, D, F> ArgminOp for &'f BaseFitter<S, D, F>
where
    S: LogLikelihood<D, F>,
    D: for<'a> TryFrom<&'a [F], Error = crate::error::Error>,
    F: Float + FloatConst + FromPrimitive + Debug + Display + Serialize + for<'de> Deserialize<'de>,
{
    type Param = Vec<F>;
    type Output = F;
    type Hessian = ();
    type Jacobian = ();
    type Float = F;

    fn apply(&self, params: &Self::Param) -> Result<Self::Output, anyhow::Error> {
        let distribution = D::try_from(params)?;
        Ok(-self.input_state.log_likelihood(&distribution))
    }
}

pub trait Fitter<S, P> {
    fn fit(&self) -> Result<P, crate::error::Error>;
}

impl<S, D> Fitter<S, D> for BaseFitter<S, D, f64>
where
    S: LogLikelihood<D, f64> + InitialSolvePoint<D>,
    D: for<'a> TryFrom<&'a [f64], Error = crate::error::Error> + Into<Vec<f64>> + Debug,
{
    fn fit(&self) -> Result<D, crate::error::Error> {
        let initial_point: Vec<f64> = self.input_state.initial_solve_point().into();
        let initial_simplex = initial_point.initial_simplex()?;

        let solver = NelderMead::new().with_initial_params(initial_simplex);

        let res = Executor::new(self, solver, initial_point.clone())
            .max_iters(self.max_iterations)
            .run()?;

        D::try_from(&res.state.best_param)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::weibull::WeibullDistribution;
    use crate::sample::{IntervalCensored, Weighted};
    use ndarray::prelude::*;

    const TOLERANCE: f64 = 1e-5;

    #[test]
    fn test_fit() {
        let data = Weighted {
            time: IntervalCensored {
                start: array![0., 2., 5., 10.],
                stop: array![2., 5., 10., 1e10f64],
            },
            weight: array![1000. - 376., 376. - 82., 82. - 7., 7.],
        };

        let fitter: BaseFitter<_, WeibullDistribution<f64>, f64> = BaseFitter::new(data);

        let actual = fitter.fit().unwrap();
        let expected = WeibullDistribution {
            scale: 2.0410538960706726,
            shape: 1.0170410407859767,
        };

        assert!((actual.scale - expected.scale).abs() < TOLERANCE);
        assert!((actual.shape - expected.shape).abs() < TOLERANCE);
    }
}
