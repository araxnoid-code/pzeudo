use num_traits::{One, Zero};

pub use crate::prelude::*;

pub struct BatchNorm<F> {
    gamma: Tensor<F, Contiguous, ReqGrad>,
    beta: Tensor<F, Contiguous, ReqGrad>,
}

impl<F> BatchNorm<F>
where
    F: Clone + Zero + One,
{
    pub fn new(
        hidden: usize,
        model_builder: &mut ModelBuilder<F>,
    ) -> Result<BatchNorm<F>, PzeudoErr> {
        let (gamma, beta) = if model_builder.is_params_load() {
            let gamma = model_builder
                .get_load_params()?
                .ok_or(PzeudoErr::LayerErr(String::from("LayerNorm::new. Unable to retrieve gamma data via load params because load params is undefined.")))?;

            let beta = model_builder
                .get_load_params()?
                .ok_or(PzeudoErr::LayerErr(String::from("LayerNorm::new. Unable to retrieve beta data via load params because load params is undefined.")))?;

            let module = model_builder.get_module();
            let gamma = Tensor::param_from_vector_with_shape(gamma, &[hidden], module, ReqGrad)?;
            let beta = Tensor::param_from_vector_with_shape(beta, &[hidden], module, ReqGrad)?;
            (gamma, beta)
        } else {
            let module = model_builder.get_module();
            let gamma = Tensor::param_ones(&[hidden], module, ReqGrad)?;
            let beta = Tensor::param_zeros(&[hidden], module, ReqGrad)?;
            (gamma, beta)
        };

        Ok(BatchNorm { beta, gamma })
    }
}
