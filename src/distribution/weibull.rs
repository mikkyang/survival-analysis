#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct WeibullDistribution<F> {
    pub rho: F,
    pub lambda: F,
}
