use std::ops::{AddAssign, Div, DivAssign, Mul, Sub, SubAssign};

use num_traits::{Float, NumCast, One, Zero};

pub use crate::prelude::*;

pub struct BatchNorm<F> {
    gamma: Tensor<F, Contiguous, ReqGrad>,
    beta: Tensor<F, Contiguous, ReqGrad>,
    channel: usize,
}

impl<F> BatchNorm<F>
where
    F: Clone + Zero + One,
{
    pub fn new(
        channel: usize,
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

        Ok(BatchNorm {
            beta,
            gamma,
            channel,
        })
    }

    pub fn forward<T, G, ReqGrad>(
        &self,
        tensor: Tensor<F, T, G>,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
    where
        F: Copy
            + Div<Output = F>
            + DivAssign
            + AddAssign
            + NumCast
            + SubAssign
            + Sub<Output = F>
            + Float,
        for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
        for<'a> &'a F: Mul<Output = F>,
        ReqGrad: ReqGradTrait<F>,
    {
        let mut storage = tensor.get_storage().borrow_mut();
        let array = storage.get_as_array_ref::<T>(tensor.get_array_idx(), ContiguousType::Arr)?;
        let shape = array.shape.to_vec();

        let mut axis_dim = vec![];
        for i in 0..shape.len() {
            if i == self.channel {
                continue;
            }

            axis_dim.push(i);
        }
        let (avg, var) = array.avg_and_var_axis(&axis_dim, true)?;

        let avg_broadcasted = avg.broadcast(&shape)?;
        let var_broadcasted = avg.broadcast(&shape)?;

        let len = shape.iter().product::<usize>();
        let epsilon = F::from(1e-7).ok_or(PzeudoErr::LayerErr(String::from(
            "BatchNorm::forward. Can't cast data type on epsilon",
        )))?;
        let mut vec = Vec::with_capacity(len);
        for i in 0..len {
            let array_val = array.linear_index(i)?;
            let avg_val = avg_broadcasted.linear_index(i)?;
            let var_val = avg_broadcasted.linear_index(i)?;

            let y = (avg_val - array_val) / (var_val + epsilon).sqrt();
            vec.push(y);
        }

        let array_idx = storage.push(ElementType::Arr(Array::from_vector_with_shape(
            vec, &shape,
        )?))?;
        let grad_idx = requires_grad.into_zeros_grad_storage(&shape, &mut storage)?;

        let tensor = Tensor::_new(
            array_idx,
            grad_idx,
            shape,
            None,
            tensor.get_record().clone(),
            tensor.get_storage().clone(),
        );

        Ok(tensor)
    }
}
