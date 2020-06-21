use super::{
    InitialSolvePoint, IntervalCensored, LeftCensored, LogLikelihood, PartiallyObserved,
    RightCensored, Uncensored, Weighted,
};
use crate::distribution::{CumulativeHazard, LogCumulativeDensity, LogHazard, Survival};
use crate::utils::{filter, partition};
use ndarray::prelude::*;
use ndarray::{Data, OwnedRepr, ScalarOperand};
use num_traits::{clamp, Float, FromPrimitive};
use std::iter::FromIterator;

pub struct Events<Censoring, Observation, Weight, Truncation> {
    pub time: Censoring,
    pub observed: Observation,
    pub weight: Weight,
    pub truncation: Truncation,
}

pub trait FromEvents<F> {
    fn from_events<S: Data<Elem = F>, B: Data<Elem = bool>>(
        events: &ArrayBase<S, Ix1>,
        event_observed: &ArrayBase<B, Ix1>,
    ) -> Self;
}

impl<F, C> FromEvents<F> for PartiallyObserved<OwnedRepr<F>, F, Ix1, C>
where
    F: Copy,
    C: From<Vec<F>>,
{
    fn from_events<S: Data<Elem = F>, B: Data<Elem = bool>>(
        events: &ArrayBase<S, Ix1>,
        event_observed: &ArrayBase<B, Ix1>,
    ) -> Self {
        let half_capacity = events.len() / 2;
        let mut observed_events = Vec::with_capacity(half_capacity);
        let mut censored_events = Vec::with_capacity(half_capacity);

        for (event, o) in events.iter().zip(event_observed.iter()) {
            if *o {
                observed_events.push(*event)
            } else {
                censored_events.push(*event)
            }
        }

        PartiallyObserved {
            observed: Uncensored(Array::from(observed_events)),
            censored: C::from(censored_events),
        }
    }
}

impl<F, C> FromEvents<(F, F)> for PartiallyObserved<OwnedRepr<F>, F, Ix1, C>
where
    F: Copy,
    C: From<Vec<(F, F)>>,
{
    fn from_events<S: Data<Elem = (F, F)>, B: Data<Elem = bool>>(
        events: &ArrayBase<S, Ix1>,
        event_observed: &ArrayBase<B, Ix1>,
    ) -> Self {
        let half_capacity = events.len() / 2;
        let mut observed_events = Vec::with_capacity(half_capacity);
        let mut censored_events = Vec::with_capacity(half_capacity);

        for (event, o) in events.iter().zip(event_observed.iter()) {
            if *o {
                let (_, time) = event;
                observed_events.push(*time)
            } else {
                censored_events.push(*event)
            }
        }

        PartiallyObserved {
            observed: Uncensored(Array::from(observed_events)),
            censored: C::from(censored_events),
        }
    }
}

pub struct RightCensoredDuration<T, F>
where
    T: Data<Elem = F>,
{
    pub duration: ArrayBase<T, Ix1>,
}

impl<F> From<Vec<F>> for RightCensored<OwnedRepr<F>, F, Ix1> {
    fn from(vec: Vec<F>) -> Self {
        RightCensored(Array::from(vec))
    }
}

pub struct LeftCensoredDuration<T, F>
where
    T: Data<Elem = F>,
{
    pub duration: ArrayBase<T, Ix1>,
}

impl<F> From<Vec<F>> for LeftCensored<OwnedRepr<F>, F, Ix1> {
    fn from(vec: Vec<F>) -> Self {
        LeftCensored(Array::from(vec))
    }
}

pub struct IntervalCensoredDuration<T, F>
where
    T: Data<Elem = F>,
{
    pub start_time: ArrayBase<T, Ix1>,
    pub stop_time: ArrayBase<T, Ix1>,
}

impl<F> From<Vec<(F, F)>> for IntervalCensored<OwnedRepr<F>, F, Ix1> {
    fn from(vec: Vec<(F, F)>) -> Self {
        let mut starts = Vec::with_capacity(vec.len());
        let mut stops = Vec::with_capacity(vec.len());
        for (start, stop) in vec.into_iter() {
            starts.push(start);
            stops.push(stop);
        }

        IntervalCensored {
            start: Array::from(starts),
            stop: Array::from(stops),
        }
    }
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
    F: Float + ScalarOperand,
    W: Data<Elem = F>,
{
    fn log_likelihood(&self, distribution: &D) -> Array1<F> {
        let Weighted { time, weight } = self;

        let log_likelihood = time.log_likelihood(distribution);
        (weight * &log_likelihood) / weight.sum()
    }
}

impl<D, F, T, W> LogLikelihood<D, F> for Weighted<T, W, F, Ix1>
where
    T: LogLikelihood<D, Array1<F>>,
    F: Float + ScalarOperand,
    W: Data<Elem = F>,
{
    fn log_likelihood(&self, distribution: &D) -> F {
        let array: Array1<F> = self.log_likelihood(distribution);
        array.sum()
    }
}

impl<D, F, T, C> LogLikelihood<D, F> for PartiallyObserved<T, F, Ix1, C>
where
    D: LogHazard<ArrayBase<T, Ix1>, Array1<F>> + CumulativeHazard<ArrayBase<T, Ix1>, Array1<F>>,
    F: Float,
    T: Data<Elem = F>,
    C: LogLikelihood<D, F>,
{
    fn log_likelihood(&self, distribution: &D) -> F {
        let PartiallyObserved { observed, censored } = self;

        let observed_log_likelihood: F = observed.log_likelihood(distribution);
        observed_log_likelihood + censored.log_likelihood(distribution)
    }
}

impl<D, F, T> LogLikelihood<D, F> for RightCensored<T, F, Ix1>
where
    D: LogHazard<ArrayBase<T, Ix1>, Array1<F>> + CumulativeHazard<ArrayBase<T, Ix1>, Array1<F>>,
    F: Float,
    T: Data<Elem = F>,
{
    fn log_likelihood(&self, distribution: &D) -> F {
        let RightCensored(time) = self;

        -distribution.cumulative_hazard(&time).sum()
    }
}

impl<D, F, T> LogLikelihood<D, Array1<F>> for RightCensored<T, F, Ix1>
where
    D: LogHazard<ArrayBase<T, Ix1>, Array1<F>> + CumulativeHazard<ArrayBase<T, Ix1>, Array1<F>>,
    F: Float,
    T: Data<Elem = F>,
{
    fn log_likelihood(&self, distribution: &D) -> Array1<F> {
        let RightCensored(time) = self;

        -distribution.cumulative_hazard(&time)
    }
}

impl<D, F, T, B> LogLikelihood<D, F>
    for Events<RightCensoredDuration<T, F>, ArrayBase<B, Ix1>, (), ()>
where
    D: LogHazard<Array1<F>, Array1<F>> + CumulativeHazard<Array1<F>, Array1<F>>,
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

        let (observed, censored) = partition(&duration, &observed);

        let f: F = Uncensored(observed).log_likelihood(distribution);
        f + RightCensored(censored).log_likelihood(distribution)
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
    F: Float + ScalarOperand,
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

        let observed: Array1<F> = Weighted {
            time: Uncensored(observed_duration),
            weight: observed_weight,
        }
        .log_likelihood(distribution);
        let censored: Array1<F> = Weighted {
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
    F: Float + FromPrimitive + ScalarOperand,
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

        let observed: Array1<F> = Weighted {
            time: Uncensored(observed_duration),
            weight: observed_weight,
        }
        .log_likelihood(distribution);
        let censored: Array1<F> = Weighted {
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
