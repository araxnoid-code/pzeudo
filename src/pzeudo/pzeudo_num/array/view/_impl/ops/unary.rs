use num_traits::Float;

use crate::{ArrayView, OpsUnary};

impl<F> OpsUnary<F> for ArrayView<'_, F> where F: Float {}
