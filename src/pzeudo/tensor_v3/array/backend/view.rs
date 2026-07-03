use ndarray::{ArrayD, ArrayViewD};
use num_traits::Float;

use crate::ArrayTrait;

impl<F> ArrayTrait for ArrayViewD<'_, F> where F: Float {}
