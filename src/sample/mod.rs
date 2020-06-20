use argmin::prelude::*;
use argmin::solver::neldermead::NelderMead;
use num_traits::{Float, FloatConst, FromPrimitive};
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

pub mod univariate;

pub trait InitialSolvePoint<T> {
    fn initial_solve_point(&self) -> T;
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
    D: for<'a> From<&'a Vec<F>>,
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
    fn state(&self) -> &S;

    fn fit(&self) -> Result<P, String>;
}

const NON_ZERO_DELTA: f64 = 0.05;
const ZERO_DELTA: f64 = 0.00025;

impl<S, D> Fitter<S, D> for BaseFitter<S, D, f64>
where
    S: LogLikelihood<D, f64> + InitialSolvePoint<Option<D>>,
    D: for<'a> From<&'a Vec<f64>> + Into<Vec<f64>> + Debug,
{
    fn state(&self) -> &S {
        &self.input_state
    }

    fn fit(&self) -> Result<D, String> {
        let initial_point: Vec<f64> = if let Some(x) = self.input_state.initial_solve_point() {
            x.into()
        } else {
            vec![]
        };

        let d = initial_point.len();
        let mut simplex = vec![initial_point; d + 1];
        for (index_within_point, point) in simplex.iter_mut().skip(1).enumerate() {
            if point[index_within_point] != 0.0 {
                point[index_within_point] = (1.0 + NON_ZERO_DELTA) * point[index_within_point]
            } else {
                point[index_within_point] = ZERO_DELTA
            }
        }

        let solver = NelderMead::new().with_initial_params(simplex);

        let res = Executor::new(self, solver, vec![])
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
