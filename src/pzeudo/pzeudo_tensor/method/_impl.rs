use num_traits::Zero;
use rand::distr::{Distribution, StandardUniform};

use crate::prelude::*;

impl<F> LinearTrait<F> for Linear<F>
where
    F: Clone + Zero,
    StandardUniform: Distribution<F>,
{
    fn new(in_features: usize, out_features: usize, module: &Module<F>) -> Result<Self, PzeudoErr>
    where
        Self: Sized,
    {
        Linear::new(in_features, out_features, module)
    }
}
