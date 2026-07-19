use num_traits::Float;

use crate::prelude::*;

impl<F> OpsUnary<F> for ArrayView<'_, F> where F: Float {}
