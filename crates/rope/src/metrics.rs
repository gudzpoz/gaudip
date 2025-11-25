use crate::piece::{Sum, Summable};
use std::marker::PhantomData;
use std::ops::{Add, Sub};

/// A metric system for rope pieces
///
/// For the conceptual background, see the [rope science series].
///
/// [rope science series]: https://xi-editor.io/docs/rope_science_02.html
///
/// ## Base Metric
///
/// We provide a [BaseMetric] as the metric in base units of the rope.
pub trait Metric<T: Summable> {
    /// Returns the measurement of a [Sum]
    fn measure(sum: &T::S) -> usize;
}

/// The base metric measured by [Sum::len]
pub struct BaseMetric();
impl<T: Summable> Metric<T> for BaseMetric {
    fn measure(sum: &T::S) -> usize {
        sum.len()
    }
}

/// A measurement measured by `M` for `T`
pub struct Measured<T: Summable, M: Metric<T>> {
    /// The measured value
    pub value: usize,
    _metric: PhantomData<M>,
    _src_type: PhantomData<T>,
}
impl<T: Summable, M: Metric<T>> Measured<T, M> {
    /// Creates a measurement in the given metric
    pub fn new(value: usize) -> Self {
        Self {
            value,
            _metric: PhantomData,
            _src_type: PhantomData,
        }
    }
}
impl<T: Summable, M: Metric<T>> Copy for Measured<T, M> {}
impl<T: Summable, M: Metric<T>> Clone for Measured<T, M> {
    fn clone(&self) -> Self { *self }
}
impl<T: Summable, M: Metric<T>> Add<usize> for Measured<T, M> {
    type Output = Self;

    fn add(self, rhs: usize) -> Self::Output {
        Self::new(self.value + rhs)
    }
}
impl<T: Summable, M: Metric<T>> Sub<usize> for Measured<T, M> {
    type Output = Self;

    fn sub(self, rhs: usize) -> Self::Output {
        Self::new(self.value - rhs)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::roperig::Rope;
    use crate::roperig_test::Alphabet;

    #[test]
    fn test_cursor_creation() {
        let mut rb = Rope::<Alphabet>::default();
        rb.insert(0, "111".into());
        rb.insert(3, "222".into());

        let start = rb.cursor_at(0);
        assert!(start.is_some());
        let start = start.unwrap();
        assert_eq!(start.get(&rb.tree), &"111".into());
        assert_eq!(start.offset().value, 0);

        let mid = rb.cursor_at(3);
        assert!(mid.is_some());
        let mid = mid.unwrap();
        assert_eq!(mid.get(&rb.tree), &"111".into());
        assert_eq!(mid.offset().value, 3);

        let end = rb.cursor_at(6);
        assert!(end.is_some());
        let end = end.unwrap();
        assert_eq!(end.get(&rb.tree), &"222".into());
        assert_eq!(end.offset().value, 3);

        assert!(rb.cursor_at(7).is_none());
    }

    #[test]
    fn test_rel_node_far() {
        let mut rb = Rope::<Alphabet>::default();
        for c in ('a'..='z').rev() {
            rb.insert(0, c.into());
        }
        assert_eq!(26, rb.base_len());
        let mut start = rb.cursor_at(0).unwrap();
        assert_eq!(start.get(&rb.tree), &"a".into());
        assert_eq!(start.offset().value, 0);
        assert_eq!(0, start.position(&rb.tree));

        #[derive(Debug)]
        struct Step {
            step: isize,
            expected_offset: usize,
            expected_str: &'static str,
        }
        let steps = [
            Step { step: 0, expected_offset: 0, expected_str: "a" },
            Step { step: 26, expected_offset: 1, expected_str: "z" },
            Step { step: -26, expected_offset: 0, expected_str: "a" },
            Step { step: 13, expected_offset: 1, expected_str: "m" },
            Step { step: 13, expected_offset: 1, expected_str: "z" },
            Step { step: -13, expected_offset: 1, expected_str: "m" },
            Step { step: -13, expected_offset: 0, expected_str: "a" },
            Step { step: 1, expected_offset: 1, expected_str: "a" },
            Step { step: -1, expected_offset: 0, expected_str: "a" },
            Step { step: 13, expected_offset: 1, expected_str: "m" },
            Step { step: 1, expected_offset: 1, expected_str: "n" },
            Step { step: -1, expected_offset: 1, expected_str: "m" },
            Step { step: -13, expected_offset: 0, expected_str: "a" },
        ];

        let mut abs = 0usize;
        for step in steps {
            abs = abs.checked_add_signed(step.step).unwrap();
            let next = start.navigate(&rb.tree, step.step);
            assert!(next.is_some(), "step: {:?}", step);
            start = next.unwrap();
            assert_eq!(abs, start.position(&rb.tree));
            assert_eq!(start.offset().value, step.expected_offset);
            assert_eq!(start.get(&rb.tree), &step.expected_str.into());
        }
    }
}
