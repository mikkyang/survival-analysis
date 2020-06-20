pub mod weibull;

pub trait Survival<Input, Output> {
    fn survival(&self, input: &Input) -> Output;
}
pub trait LogHazard<Input, Output> {
    fn log_hazard(&self, input: &Input) -> Output;
}

pub trait CumulativeHazard<Input, Output> {
    fn cumulative_hazard(&self, input: &Input) -> Output;
}

pub trait LogCumulativeDensity<Input, Output> {
    fn log_cumulative_density(&self, input: &Input) -> Output;
}
