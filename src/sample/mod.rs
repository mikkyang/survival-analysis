pub mod univariate;

pub trait LogLikelihood<Distribution, F> {
    fn log_likelihood(&self, distribution: &Distribution) -> F;
}
