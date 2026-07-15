use std::iter::Sum;

use crate::{Array, ArrayView, OpsAvg, OpsSum};

impl<F> OpsSum<F> for ArrayView<'_, F> where for<'a> F: Copy {}

impl<F> OpsAvg<F> for ArrayView<'_, F> where for<'a> F: Copy {}
