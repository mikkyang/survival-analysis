use super::{
    InitialSolvePoint, IntervalCensored, LeftCensored, LogLikelihood, Uncensored, Weighted,
};
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

impl<D, F, T, W> LogLikelihood<D, Array1<F>> for Weighted<T, W, F, Ix1>
where
    T: LogLikelihood<D, Array1<F>>,
    F: Float,
    W: Data<Elem = F>,
{
    fn log_likelihood(&self, distribution: &D) -> Array1<F> {
        let Weighted { time, weight } = self;

        let log_likelihood = time.log_likelihood(distribution);
        weight * &log_likelihood
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

impl<D, F, T> LogLikelihood<D, Array1<F>> for LeftCensored<T, F, Ix1>
where
    D: LogCumulativeDensity<ArrayBase<T, Ix1>, Array1<F>>,
    F: Float,
    T: Data<Elem = F>,
{
    fn log_likelihood(&self, distribution: &D) -> Array1<F> {
        let LeftCensored(time) = self;
        distribution.log_cumulative_density(&time)
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
        let (observed_weight, censored_weight) = partition(weight, observed);

        let observed = Weighted {
            time: Uncensored(observed_duration),
            weight: observed_weight,
        }
        .log_likelihood(distribution);
        let censored = Weighted {
            time: LeftCensored(censored_duration),
            weight: censored_weight,
        }
        .log_likelihood(distribution);

        (observed.sum() + censored.sum()) / weight.sum()
    }
}

impl<D, F, T> LogLikelihood<D, F> for IntervalCensored<T, F, Ix1>
where
    D: Survival<ArrayBase<T, Ix1>, Array1<F>>,
    F: Float + FromPrimitive,
    T: Data<Elem = F>,
{
    fn log_likelihood(&self, distribution: &D) -> F {
        let IntervalCensored { start, stop } = self;

        let min = F::from_f64(-1e50).unwrap();
        let max = F::from_f64(1e50).unwrap();

        let survival = (distribution.survival(&start) - distribution.survival(&stop))
            .mapv_into(F::ln)
            .mapv_into(|x| clamp(x, min, max));

        survival.sum()
    }
}

impl<D, F, T> LogLikelihood<D, Array1<F>> for IntervalCensored<T, F, Ix1>
where
    D: Survival<ArrayBase<T, Ix1>, Array1<F>>,
    F: Float + FromPrimitive,
    T: Data<Elem = F>,
{
    fn log_likelihood(&self, distribution: &D) -> Array1<F> {
        let IntervalCensored { start, stop } = self;

        let min = F::from_f64(-1e50).unwrap();
        let max = F::from_f64(1e50).unwrap();

        let survival = (distribution.survival(&start) - distribution.survival(&stop))
            .mapv_into(F::ln)
            .mapv_into(|x| clamp(x, min, max));

        survival
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
        let censored_starts = filter(start_time, &!observed);
        let (observed_stops, censored_stops) = partition(stop_time, observed);

        let f: F = Uncensored(observed_stops).log_likelihood(distribution);
        f + IntervalCensored {
            start: censored_starts,
            stop: censored_stops,
        }
        .log_likelihood(distribution)
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
        let (observed_weight, censored_weight) = partition(&weight, &observed);
        let censored_start = filter(start_time, &!observed);
        let (observed_duration, censored_stop) = partition(stop_time, observed);

        let observed = Weighted {
            time: Uncensored(observed_duration),
            weight: observed_weight,
        }
        .log_likelihood(distribution);
        let censored = Weighted {
            time: IntervalCensored {
                start: censored_start,
                stop: censored_stop,
            },
            weight: censored_weight,
        }
        .log_likelihood(distribution);

        (observed.sum() + censored.sum()) / weight.sum()
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
