use std::iter::Sum;

use crate::{Array, ArrayView, OpsSum};

impl<F> OpsSum<F> for ArrayView<'_, F> where for<'a> F: Copy {}
