use argmin::prelude::*;
use argmin::solver::neldermead::NelderMead;
use ndarray::{ArrayBase, Data};
use num_traits::{Float, FloatConst, FromPrimitive};
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

pub mod univariate;

pub struct Uncensored<T, F, D>(ArrayBase<T, D>)
where
    T: Data<Elem = F>;

pub struct LeftCensored<T, F, D>(ArrayBase<T, D>)
where
    T: Data<Elem = F>;

pub struct IntervalCensored<T, F, D>
where
    T: Data<Elem = F>,
{
    pub start: ArrayBase<T, D>,
    pub stop: ArrayBase<T, D>,
}

pub struct Weighted<T, W, F, D>
where
    W: Data<Elem = F>,
{
    time: T,
    weight: ArrayBase<W, D>,
}

pub trait InitialSolvePoint<T> {
    fn initial_solve_point(&self) -> T;
}

pub trait InitialNelderMeanSimplex<T> {
    fn initial_simplex(&self) -> Vec<T>;
}

const NON_ZERO_DELTA: f64 = 0.05;
const ZERO_DELTA: f64 = 0.00025;

impl<F> InitialNelderMeanSimplex<Vec<F>> for [F]
where
    F: Float + FromPrimitive,
{
    fn initial_simplex(&self) -> Vec<Vec<F>> {
        let initial_point: Vec<F> = self.into();

        let d = initial_point.len();
        let mut simplex = vec![initial_point; d + 1];
        for (index_within_point, point) in simplex.iter_mut().skip(1).enumerate() {
            if point[index_within_point] != F::zero() {
                point[index_within_point] =
                    F::from(1.0 + NON_ZERO_DELTA).unwrap() * point[index_within_point]
            } else {
                point[index_within_point] = F::from(ZERO_DELTA).unwrap()
            }
        }

        simplex
    }
}

pub trait LogLikelihood<Distribution, F> {
    fn log_likelihood(&self, distribution: &Distribution) -> F;
}

pub struct BaseFitter<S, D, F> {
    input_state: S,
    _distribution: PhantomData<D>,
    _float: PhantomData<F>,
}

impl<'f, S, D, F> ArgminOp for &'f BaseFitter<S, D, F>
where
    S: LogLikelihood<D, F>,
    D: for<'a> From<&'a [F]>,
    F: Float + FloatConst + FromPrimitive + Debug + Display + Serialize + for<'de> Deserialize<'de>,
{
    type Param = Vec<F>;
    type Output = F;
    type Hessian = ();
    type Jacobian = ();
    type Float = F;

    fn apply(&self, params: &Self::Param) -> Result<Self::Output, Error> {
        let distribution = D::from(params);
        Ok(-self.input_state.log_likelihood(&distribution))
    }
}

pub trait Fitter<S, P> {
    fn fit(&self) -> Result<P, String>;
}

impl<S, D> Fitter<S, D> for BaseFitter<S, D, f64>
where
    S: LogLikelihood<D, f64> + InitialSolvePoint<D>,
    D: for<'a> From<&'a [f64]> + Into<Vec<f64>> + Debug,
{
    fn fit(&self) -> Result<D, String> {
        let initial_point: Vec<f64> = self.input_state.initial_solve_point().into();

        let solver = NelderMead::new().with_initial_params(initial_point.initial_simplex());

        let res = Executor::new(self, solver, initial_point.clone())
            .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
            .max_iters(100)
            .run()
            .unwrap();

        Ok(D::from(&res.state.best_param))
    }
}

#[cfg(test)]
mod tests {
    use super::univariate::{Events, IntervalCensoredDuration};
    use super::*;
    use crate::distribution::weibull::WeibullDistribution;
    use ndarray::prelude::*;

    const TOLERANCE: f64 = 1e-5;

    #[test]
    fn test_fit() {
        let input_state = Events {
            time: IntervalCensoredDuration {
                start_time: array![0., 2., 5., 10.],
                stop_time: array![2., 5., 10., 1e10f64],
            },
            observed: Array::from_elem((4,), false),
            weight: array![1000. - 376., 376. - 82., 82. - 7., 7.],
            truncation: (),
        };

        let fitter: BaseFitter<_, WeibullDistribution<f64>, f64> = BaseFitter {
            input_state,
            _distribution: PhantomData,
            _float: PhantomData,
        };

        let actual = fitter.fit().unwrap();
        let expected = WeibullDistribution {
            lambda: 2.0410538960706726,
            rho: 1.0170410407859767,
        };

        assert!((actual.lambda - expected.lambda).abs() < TOLERANCE);
        assert!((actual.rho - expected.rho).abs() < TOLERANCE);
    }
}
