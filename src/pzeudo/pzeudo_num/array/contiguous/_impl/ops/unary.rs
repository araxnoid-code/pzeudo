use num_traits::Float;

use crate::{Array, OpsUnary};

impl<F> OpsUnary<F> for Array<F> where F: Float {}
