use crate::prelude::*;
use num_traits::{Zero, zero};
use rand::{
    distr::{Distribution, StandardUniform},
    random,
};
use std::vec;

pub struct Linear<F> {
    pub(crate) in_features: usize,
    pub(crate) out_features: usize,
    pub(crate) weight: Tensor<F, Contiguous>,
    pub(crate) bias: Tensor<F, Contiguous>,
}

impl<F> Linear<F> {
    pub fn new(
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
        let weight = Tensor::update_able_from_vector_with_shape(
            &weight_vector,
            &[in_features, out_features],
            module.storage.clone(),
            module.record.clone(),
        )?;

        let bias: Tensor<F, Contiguous> = Tensor::update_able_from_vector_with_shape(
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
}

impl Linear<f32> {
    pub fn forward<J>(&self, input: &Tensor<f32, J>) -> Result<Tensor<f32, Contiguous>, PzeudoErr>
    where
        for<'a> ArrayRef<'a, f32, J>: ArrayTrait<f32> + OpsAdd<f32> + OpsBroadcast<f32>,
        for<'a> ArrayRef<'a, f32, Contiguous>: ArrayTrait<f32> + OpsAdd<f32> + OpsBroadcast<f32>,
    {
        Ok(input.matmul_2d(&self.weight)?.add(&self.bias)?)
    }

    pub fn get_in_features(&self) -> usize {
        self.in_features
    }

    pub fn get_out_features(&self) -> usize {
        self.out_features
    }

    pub fn get_weight(&self) -> &Tensor<f32, Contiguous> {
        &self.weight
    }

    pub fn get_bias(&self) -> &Tensor<f32, Contiguous> {
        &self.bias
    }
}
