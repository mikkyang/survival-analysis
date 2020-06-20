#[cfg(test)]
#[macro_use]
mod tests {
    #[macro_export]
    macro_rules! assert_diff_within_tolerance {
        ($actual: expr, $expected: expr, $tolerance: expr) => {
            for diff in ($actual - $expected).iter() {
                assert!(diff.abs() < $tolerance);
            }
        };
    }
}

pub mod distribution;
