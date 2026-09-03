use std::ops::AddAssign;

use num_traits::Float;

use crate::prelude::*;

impl<F, T, G> Tensor<F, T, G> {
    pub fn softmax<J, TensorGrad, ReqGrad>(
        tensor: &Tensor<F, J, TensorGrad>,
        axis: usize,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
    where
        ReqGrad: ReqGradTrait<F>,
        for<'a> ArrayRef<'a, F, J>: ArrayTrait<F>,
        F: Float + AddAssign,
    {
        let mut storage = tensor.get_storage().borrow_mut();
        let tensor_array =
            storage.get_as_array_ref::<J>(tensor.get_array_idx(), ContiguousType::Arr)?;
        let shape = tensor_array.shape.to_vec();

        let exp_axis = tensor_array.sum_axis_closure(&[axis], true, |_, x| Ok(x.exp()))?;
        let exp_axis_broadcasted = exp_axis.broadcast(tensor_array.shape)?;

        let len = tensor_array.shape.iter().product::<usize>();
        let mut vec = Vec::with_capacity(len);
        for i in 0..len {
            vec.push(tensor_array.linear_index(i)?.exp() / exp_axis_broadcasted.linear_index(i)?);
        }

        let array_idx = storage.push(ElementType::Arr(Array::from_vector_with_shape(
            &vec, &shape,
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
