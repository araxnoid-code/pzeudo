use crate::prelude::*;
use num_traits::Float;
use std::ops::{AddAssign, DivAssign, Mul, SubAssign};

pub struct LayerNorm {}

impl LayerNorm {
    pub fn forward<F, T, G, ReqGrad>(
        &self,
        tensor: &Tensor<F, T, G>,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<F, T, ReqGrad>, PzeudoErr>
    where
        F: AddAssign + DivAssign + Float + SubAssign,
        for<'a> &'a F: Mul<&'a F, Output = F>,
        ReqGrad: ReqGradTrait<F>,
        for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
    {
        let epsilon = F::from(1e-7).ok_or(PzeudoErr::LayerErr(format!(
            "LayerNorm::forward. Cannot cast to epsilon."
        )))?;
        let axis = tensor.shape.len() - 1;

        let mut storage = tensor.get_storage().borrow_mut();
        let array_tensor =
            storage.get_as_array_ref::<T>(tensor.get_array_idx(), ContiguousType::Arr)?;

        let (avg, var) = array_tensor.avg_and_var_axis(&[axis], true)?;

        let avg_broadcast = avg.broadcast(&tensor.shape)?;
        let var_broadcast = var.broadcast(&tensor.shape)?;

        let len = array_tensor.shape.iter().product::<usize>();
        let mut vec = Vec::with_capacity(len);
        for i in 0..len {
            let y = (array_tensor.linear_index(i)? - avg_broadcast.linear_index(i)?)
                / (var_broadcast.linear_index(i)? + epsilon).sqrt();
            vec.push(y);
        }

        let norm = storage.push(ElementType::Arr(Array::from_vector_with_shape(
            &vec,
            &tensor.shape,
        )?))?;
        let grad = requires_grad.into_zeros_grad_storage(&tensor.shape, &mut storage)?;

        let tensor = Tensor::_new(
            norm,
            grad,
            tensor.shape.to_vec(),
            None,
            tensor.get_record().clone(),
            tensor.get_storage().clone(),
        );

        Ok(tensor)
    }
}
