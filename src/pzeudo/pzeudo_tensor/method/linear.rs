use crate::prelude::*;
use num_traits::Zero;
use rand::{
    distr::{Distribution, StandardUniform},
    random,
};
use std::{ops::Add, vec};

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
        let weight = Tensor::param_from_vector_with_shape(
            &weight_vector,
            &[in_features, out_features],
            module,
            Grad,
        )?;

        let bias: Tensor<F, Contiguous, Grad> = Tensor::param_from_vector_with_shape(
            &vec![F::zero(); out_features],
            &[out_features],
            module,
            Grad,
        )?;

        Ok(Self {
            in_features,
            out_features,
            bias,
            weight,
        })
    }

    pub fn get_in_features(&self) -> usize {
        self.in_features
    }

    pub fn get_out_features(&self) -> usize {
        self.out_features
    }

    pub fn get_weight(&self) -> &Tensor<F, Contiguous, Grad> {
        &self.weight
    }

    pub fn get_bias(&self) -> &Tensor<F, Contiguous, Grad> {
        &self.bias
    }

    pub fn forward<J, G, ReqGrad>(
        &self,
        input: &Tensor<F, J, G>,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
    where
        F: F32F64MatmulTensor<F> + Copy + Add + Zero,
        ReqGrad: ReqGradTrait<F> + Copy,
        for<'a> ArrayRef<'a, F, Contiguous>: ArrayTrait<F>,
        for<'a> ArrayRef<'a, F, J>: ArrayTrait<F>,
    {
        Ok(F::matmul_2d(&input, &self.weight, requires_grad)?.add(&self.bias, requires_grad)?)
    }
}
