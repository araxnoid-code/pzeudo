use std::{
    fmt::Debug,
    iter::Sum,
    ops::{AddAssign, Mul, Sub},
};

use num_traits::{Float, Zero};

use crate::prelude::*;

// # Softmax
// softmax(x) = e^x/∑e^x
pub fn softmax<F, J, TensorGrad, ReqGrad>(
    tensor: &Tensor<F, J, TensorGrad>,
    axis: usize,
    requires_grad: ReqGrad,
) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
where
    ReqGrad: ReqGradTrait<F>,
    for<'a> ArrayRef<'a, F, J>: ArrayTrait<F>,
    F: Float + AddAssign + Debug,
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

    let mut storage = tensor.get_record().borrow_mut();
    let record_idx = Some(RecordStatus::Record(storage.len()));
    let record_label = RecordLabel::Softmax(array_idx, tensor.get_grad_idx(), axis, grad_idx);
    storage.push(record_label);

    let tensor = Tensor::_new(
        array_idx,
        grad_idx,
        shape,
        record_idx,
        tensor.get_record().clone(),
        tensor.get_storage().clone(),
    );

    Ok(tensor)
}

// dsoftmax(x)/dx = y(g  - ∑gy)
pub fn softmax_backward<F>(
    output: StorageType,
    array_grad_idx: Option<StorageType>,
    axis: usize,
    grad_idx: Option<StorageType>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    for<'a> F: Copy + Mul<Output = F> + Sum<&'a F> + AddAssign + Zero + Sub<Output = F> + Debug,
{
    if let Some(grad_idx) = grad_idx {
        if is_no_grad_or_time_not_match_or_no_update(grad_idx, storage)? {
            return Ok(());
        };

        if let Some(array_grad_idx) = array_grad_idx {
            storage.set_grad_update(array_grad_idx, true)?;
            if is_no_grad_or_time_not_match_or_no_update(array_grad_idx, storage)? {
                return Ok(());
            }

            let grad_take = storage.take_grad(grad_idx)?;
            let grad = grad_take.to_array_ref::<Contiguous>();

            let mut arr_grad_take = storage.take_grad(array_grad_idx)?;
            let mut arr_grad = arr_grad_take.to_array_ref_mut::<View>();

            let out_arr = storage.get_as_array_ref::<Contiguous>(output, ContiguousType::Arr)?;
            let out_mul_grad_axis = out_arr
                .sum_axis_closure(&[axis], true, |idx, out| Ok(out * grad.linear_index(idx)?))?;

            let out_mul_grad_axis_broadcasted = out_mul_grad_axis.broadcast(out_arr.shape)?;

            let len = arr_grad.shape.iter().product::<usize>();
            for i in 0..len {
                *arr_grad.linear_index_mut(i)? += out_arr.linear_index(i)?
                    * (grad.linear_index(i)? - out_mul_grad_axis_broadcasted.linear_index(i)?);
            }

            storage.replace_grad(grad_idx, grad_take)?;
            storage.replace_grad(array_grad_idx, arr_grad_take)?;
        }
    }

    Ok(())
}
