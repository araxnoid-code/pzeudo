use crate::prelude::*;
use num_traits::{Float, NumCast, One, Zero};
use rand::distr::{Distribution, StandardUniform};
use rand_distr::{Normal, StandardNormal};
use std::{
    ops::{Add, Div},
    vec,
};

/// Linear Layer
/// Accepts 2D input in the form [Batch, Features]. Must be exactly 2D.
///
/// formula:
/// linear = input * weight + bias
///
/// Shape specifications:
/// input shape: batch×in_features
/// weight shape: in_features×out_features
/// bias shape: out_features
///
/// Weight Initialization
/// Xavier:
///     mean    : 0
///     std_dev : 2/(in_features+out_features)
/// He:
///     mean: 0
///     std_dev : 2/in_features
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
        weight_init: WeightInit,
        module: &mut Module<F>,
    ) -> Result<Linear<F>, PzeudoErr>
    where
        F: Clone + Zero + One + NumCast + Div<Output = F> + Float,
        StandardUniform: Distribution<F>,
        StandardNormal: Distribution<F>,
    {
        let std = match weight_init {
            WeightInit::He => {
                (F::one() + F::one())
                    / F::from(in_features).ok_or(PzeudoErr::LayerErr(format!(
                        "Linear::new. Cannot perform data type casting on in_feature."
                    )))?
            }
            WeightInit::Xavier => {
                (F::one() + F::one())
                    / F::from(in_features + out_features).ok_or(PzeudoErr::LayerErr(format!(
                        "Linear::new. Cannot perform data type casting on in_feature."
                    )))?
            }
        };
        let normal =
            Normal::new(F::zero(), std).map_err(|err| PzeudoErr::RandDistrNormalErr(err))?;

        let len = in_features * out_features;
        let weight_vector = (0..len)
            .map(|_| normal.sample(module.get_rng_mut()))
            .collect::<Vec<F>>();
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
