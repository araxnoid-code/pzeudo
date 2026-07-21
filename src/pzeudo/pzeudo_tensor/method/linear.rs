use std::vec;

use num_traits::{Zero, zero};
use rand::{
    distr::{Distribution, StandardUniform},
    random,
};

use crate::prelude::*;

struct Linear<F> {
    in_features: usize,
    out_features: usize,
    weight: Tensor<F, Contiguous>,
    bias: Tensor<F, Contiguous>,
}

impl<F> Linear<F> {
    fn new(
        in_features: usize,
        out_features: usize,
        module: &Module<F>,
    ) -> Result<Linear<F>, PzeudoErr>
    where
        F: Clone + Zero,
        StandardUniform: Distribution<F>,
    {
        let len = in_features * out_features;
        let weight_vector = (0..len).map(|_| random::<F>()).collect::<Vec<F>>();
        let weight = Tensor::from_vector_with_shape(
            &weight_vector,
            &[in_features, out_features],
            module.storage.clone(),
            module.record.clone(),
        )?;

        let bias = Tensor::from_vector_with_shape(
            &vec![F::zero(); out_features],
            &[out_features],
            module.storage.clone(),
            module.record.clone(),
        )?;

        Ok(Self {
            in_features,
            out_features,
            bias,
            weight,
        })
    }

    fn forward<T, Tensor>(&self, input: Tensor)
    where
        Tensor: TensorTrait<F, T>,
    {
    }
}
