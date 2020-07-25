# Survival Analysis

A *experimental* survival analysis library, initially inspired by Python's [lifelines](https://github.com/CamDavidsonPilon/lifelines).

Current Limitations:

* Only supports fitting to a Weibull distribution
* Only supports distributions that can be fitted with Nelder-Mean
* No autodifferentiation support

## Fitting Data

```rust
use ndarray::prelude::*;
use survival_analysis::{IntervalCensored, Fitter, BaseFitter};
use survival_analysis::distribution::WeibullDistribution;

let data = IntervalCensored {
    start: array![0., 2., 5., 10.],
    stop: array![2., 5., 10., 1e10f64],
};

let fitter = BaseFitter::new(data);

let params: WeibullDistribution<f64> = fitter.fit().unwrap();

assert!((params.shape - 0.980).abs() < 1e-2);
assert!((params.scale - 7.187).abs() < 1e-2);
```
