use crate::prelude::*;

pub trait LinearTrait<F> {
    fn new(in_features: usize, out_features: usize, module: &Module<F>) -> Result<Self, PzeudoErr>
    where
        Self: Sized;
}

impl<F> Module<F> {
    pub fn new_linear<Linear>(
        &self,
        in_features: usize,
        out_features: usize,
    ) -> Result<Linear, PzeudoErr>
    where
        Linear: LinearTrait<F>,
    {
        Linear::new(in_features, out_features, self)
    }
}
