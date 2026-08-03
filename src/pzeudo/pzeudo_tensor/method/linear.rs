use crate::prelude::*;
use num_traits::Zero;
use rand::{
    distr::{Distribution, StandardUniform},
    random,
};
use std::vec;

pub struct Linear<F> {
    pub(crate) in_features: usize,
    pub(crate) out_features: usize,
    pub(crate) weight: Tensor<F, Contiguous, Grad>,
    pub(crate) bias: Tensor<F, Contiguous, Grad>,
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
        let weight = Tensor::permanent_from_vector_with_shape(
            &weight_vector,
            &[in_features, out_features],
            module.storage.clone(),
            module.record.clone(),
        )?;

        let bias: Tensor<F, Contiguous, Grad> = Tensor::permanent_from_vector_with_shape(
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
    pub fn forward<J, G>(
        &self,
        input: &Tensor<f32, J, G>,
    ) -> Result<Tensor<f32, Contiguous, Grad>, PzeudoErr>
    where
        for<'a> ArrayRef<'a, f32, J>: ArrayTrait<f32> + OpsAdd<f32> + OpsBroadcast<f32>,
        for<'a> ArrayRef<'a, f32, Contiguous>: ArrayTrait<f32> + OpsAdd<f32> + OpsBroadcast<f32>,
    {
        Ok(input.matmul_2d(&self.weight, Grad)?.add(&self.bias, Grad)?)
    }

    pub fn get_in_features(&self) -> usize {
        self.in_features
    }

    pub fn get_out_features(&self) -> usize {
        self.out_features
    }

    pub fn get_weight(&self) -> &Tensor<f32, Contiguous, Grad> {
        &self.weight
    }

    pub fn get_bias(&self) -> &Tensor<f32, Contiguous, Grad> {
        &self.bias
    }
}

impl Linear<f64> {
    pub fn forward<J, G>(
        &self,
        input: &Tensor<f64, J, G>,
    ) -> Result<Tensor<f64, Contiguous, Grad>, PzeudoErr>
    where
        for<'a> ArrayRef<'a, f64, J>: ArrayTrait<f64> + OpsAdd<f64> + OpsBroadcast<f64>,
        for<'a> ArrayRef<'a, f64, Contiguous>: ArrayTrait<f64> + OpsAdd<f64> + OpsBroadcast<f64>,
    {
        Ok(input.matmul_2d(&self.weight, Grad)?.add(&self.bias, Grad)?)
    }

    pub fn get_in_features(&self) -> usize {
        self.in_features
    }

    pub fn get_out_features(&self) -> usize {
        self.out_features
    }

    pub fn get_weight(&self) -> &Tensor<f64, Contiguous, Grad> {
        &self.weight
    }

    pub fn get_bias(&self) -> &Tensor<f64, Contiguous, Grad> {
        &self.bias
    }
}
