use super::{InitialSolvePoint, LeftCensored, LogLikelihood, Uncensored};
use crate::distribution::{CumulativeHazard, LogCumulativeDensity, LogHazard, Survival};
use crate::utils::{filter, partition};
use ndarray::prelude::*;
use ndarray::Data;
use num_traits::{clamp, Float, FromPrimitive};
use std::iter::FromIterator;

pub struct Events<Censoring, Observation, Weight, Truncation> {
    pub time: Censoring,
    pub observed: Observation,
    pub weight: Weight,
    pub truncation: Truncation,
}

pub struct RightCensoredDuration<T, F>
where
    T: Data<Elem = F>,
{
    pub duration: ArrayBase<T, Ix1>,
}

pub struct LeftCensoredDuration<T, F>
where
    T: Data<Elem = F>,
{
    pub duration: ArrayBase<T, Ix1>,
}

pub struct IntervalCensoredDuration<T, F>
where
    T: Data<Elem = F>,
{
    pub start_time: ArrayBase<T, Ix1>,
    pub stop_time: ArrayBase<T, Ix1>,
}

impl<T, D, A, B, C> InitialSolvePoint<D> for Events<T, A, B, C>
where
    T: InitialSolvePoint<D>,
{
    fn initial_solve_point(&self) -> D {
        self.time.initial_solve_point()
    }
}

impl<D, F, T> LogLikelihood<D, F> for Uncensored<T, F, Ix1>
where
    D: LogHazard<ArrayBase<T, Ix1>, Array1<F>> + CumulativeHazard<ArrayBase<T, Ix1>, Array1<F>>,
    F: Float,
    T: Data<Elem = F>,
{
    fn log_likelihood(&self, distribution: &D) -> F {
        let Uncensored(time) = self;

        let log_hazard = distribution.log_hazard(time);
        let cumulative_hazard = distribution.cumulative_hazard(time);

        log_hazard.sum() - cumulative_hazard.sum()
    }
}

impl<D, F, T> LogLikelihood<D, Array1<F>> for Uncensored<T, F, Ix1>
where
    D: LogHazard<ArrayBase<T, Ix1>, Array1<F>> + CumulativeHazard<ArrayBase<T, Ix1>, Array1<F>>,
    F: Float,
    T: Data<Elem = F>,
{
    fn log_likelihood(&self, distribution: &D) -> Array1<F> {
        let Uncensored(time) = self;

        let log_hazard = distribution.log_hazard(time);
        let cumulative_hazard = distribution.cumulative_hazard(time);

        log_hazard - cumulative_hazard
    }
}

impl<D, F, T, B> LogLikelihood<D, F>
    for Events<RightCensoredDuration<T, F>, ArrayBase<B, Ix1>, (), ()>
where
    D: LogHazard<Array1<F>, Array1<F>> + CumulativeHazard<ArrayBase<T, Ix1>, Array1<F>>,
    F: Float,
    T: Data<Elem = F>,
    B: Data<Elem = bool>,
{
    fn log_likelihood(&self, distribution: &D) -> F {
        let Events {
            time: RightCensoredDuration { duration },
            observed,
            ..
        } = self;

        let observed_durations = filter(&duration, &observed);
        let log_hazard = distribution.log_hazard(&observed_durations);
        let cumulative_hazard = distribution.cumulative_hazard(&duration);

        log_hazard.sum() - cumulative_hazard.sum()
    }
}

impl<D, F, T, B, W> LogLikelihood<D, F>
    for Events<RightCensoredDuration<T, F>, ArrayBase<B, Ix1>, ArrayBase<W, Ix1>, ()>
where
    D: LogHazard<Array1<F>, Array1<F>> + CumulativeHazard<ArrayBase<T, Ix1>, Array1<F>>,
    F: Float,
    T: Data<Elem = F>,
    W: Data<Elem = F>,
    B: Data<Elem = bool>,
{
    fn log_likelihood(&self, distribution: &D) -> F {
        let Events {
            time: RightCensoredDuration { duration },
            observed,
            weight,
            ..
        } = self;

        let observed_durations = filter(&duration, &observed);
        let log_hazard = distribution.log_hazard(&observed_durations);
        let cumulative_hazard = distribution.cumulative_hazard(&duration);

        let observed_weight = filter(&weight, &observed);

        ((observed_weight * log_hazard).sum() - (weight * &cumulative_hazard).sum()) / weight.sum()
    }
}

impl<D, F, T> LogLikelihood<D, F> for LeftCensored<T, F, Ix1>
where
    D: LogCumulativeDensity<ArrayBase<T, Ix1>, Array1<F>>,
    F: Float,
    T: Data<Elem = F>,
{
    fn log_likelihood(&self, distribution: &D) -> F {
        let LeftCensored(time) = self;
        distribution.log_cumulative_density(&time).sum()
    }
}

impl<D, F, T, B> LogLikelihood<D, F>
    for Events<LeftCensoredDuration<T, F>, ArrayBase<B, Ix1>, (), ()>
where
    D: LogHazard<Array1<F>, Array1<F>>
        + CumulativeHazard<Array1<F>, Array1<F>>
        + LogCumulativeDensity<Array1<F>, Array1<F>>,
    F: Float,
    T: Data<Elem = F>,
    B: Data<Elem = bool>,
{
    fn log_likelihood(&self, distribution: &D) -> F {
        let Events {
            time: LeftCensoredDuration { duration },
            observed,
            ..
        } = self;

        let (observed_duration, censored_duration) = partition(duration, observed);
        let uncensored: F = Uncensored(observed_duration).log_likelihood(distribution);
        uncensored + LeftCensored(censored_duration).log_likelihood(distribution)
    }
}

impl<D, F, T, B, W> LogLikelihood<D, F>
    for Events<LeftCensoredDuration<T, F>, ArrayBase<B, Ix1>, ArrayBase<W, Ix1>, ()>
where
    D: LogHazard<Array1<F>, Array1<F>>
        + CumulativeHazard<Array1<F>, Array1<F>>
        + LogCumulativeDensity<Array1<F>, Array1<F>>,
    F: Float,
    T: Data<Elem = F>,
    W: Data<Elem = F>,
    B: Data<Elem = bool>,
{
    fn log_likelihood(&self, distribution: &D) -> F {
        let Events {
            time: LeftCensoredDuration { duration },
            observed,
            weight,
            ..
        } = self;

        let (observed_duration, censored_duration) = partition(duration, observed);
        let observed_log_hazard = distribution.log_hazard(&observed_duration);
        let observed_cumulative_hazard = distribution.cumulative_hazard(&observed_duration);

        let censored_log_cumulative_density =
            distribution.log_cumulative_density(&censored_duration);

        let (observed_weight, censored_weight) = partition(weight, observed);

        (observed_weight * (&observed_log_hazard - &observed_cumulative_hazard)).sum()
            + (censored_weight * &censored_log_cumulative_density).sum()
    }
}

impl<D, F, T, B> LogLikelihood<D, F>
    for Events<IntervalCensoredDuration<T, F>, ArrayBase<B, Ix1>, (), ()>
where
    D: LogHazard<Array1<F>, Array1<F>>
        + CumulativeHazard<Array1<F>, Array1<F>>
        + Survival<Array1<F>, Array1<F>>,
    F: Float + FromPrimitive,
    T: Data<Elem = F>,
    B: Data<Elem = bool>,
{
    fn log_likelihood(&self, distribution: &D) -> F {
        let Events {
            time:
                IntervalCensoredDuration {
                    start_time,
                    stop_time,
                },
            observed,
            ..
        } = self;

        let min = F::from_f64(-1e50).unwrap();
        let max = F::from_f64(1e50).unwrap();

        let censored_starts = filter(start_time, &!observed);
        let (observed_stops, censored_stops) = partition(stop_time, observed);

        let log_hazard = distribution.log_hazard(&observed_stops);
        let cumulative_hazard = distribution.cumulative_hazard(&observed_stops);
        let survival = (distribution.survival(&censored_starts)
            - distribution.survival(&censored_stops))
        .mapv_into(F::ln)
        .mapv_into(|x| clamp(x, min, max));

        log_hazard.sum() - cumulative_hazard.sum() + survival.sum()
    }
}

impl<D, F, T, B, W> LogLikelihood<D, F>
    for Events<IntervalCensoredDuration<T, F>, ArrayBase<B, Ix1>, ArrayBase<W, Ix1>, ()>
where
    D: LogHazard<Array1<F>, Array1<F>>
        + CumulativeHazard<Array1<F>, Array1<F>>
        + Survival<Array1<F>, Array1<F>>,
    F: Float + FromPrimitive,
    T: Data<Elem = F>,
    W: Data<Elem = F>,
    B: Data<Elem = bool>,
{
    fn log_likelihood(&self, distribution: &D) -> F {
        let Events {
            time:
                IntervalCensoredDuration {
                    start_time,
                    stop_time,
                },
            observed,
            weight,
            ..
        } = self;

        let min = F::from_f64(-1e50).unwrap();
        let max = F::from_f64(1e50).unwrap();

        let (observed_weights, censored_weights) = partition(&weight, &observed);
        let censored_starts = filter(start_time, &!observed);
        let (observed_stops, censored_stops) = partition(stop_time, observed);

        let log_hazard = &observed_weights * &distribution.log_hazard(&observed_stops);
        let cumulative_hazard =
            &observed_weights * &distribution.cumulative_hazard(&observed_stops);
        let survival = (censored_weights
            * (distribution.survival(&censored_starts) - distribution.survival(&censored_stops))
                .mapv_into(F::ln))
        .mapv_into(|x| clamp(x, min, max));

        (log_hazard.sum() - cumulative_hazard.sum() + survival.sum()) / weight.sum()
    }
}

impl<D, F, W, E> LogLikelihood<D, F> for (ArrayBase<W, Ix1>, ArrayBase<E, Ix1>)
where
    D: CumulativeHazard<Array1<F>, Array1<F>>,
    F: Float + FromPrimitive,
    W: Data<Elem = F>,
    E: Data<Elem = F>,
{
    fn log_likelihood(&self, distribution: &D) -> F {
        let (weight, entry) = self;

        let zero = F::zero();
        let entry_is_non_zero = entry.mapv(|x| x > zero);

        let non_zero_weights = filter(&weight, &entry_is_non_zero);
        let non_zero_entries = Array::from_iter(entry.iter().filter(|x| **x > zero).map(|x| *x));

        (non_zero_weights * distribution.cumulative_hazard(&non_zero_entries)).sum() / weight.sum()
    }
}
