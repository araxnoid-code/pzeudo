use std::ops::AddAssign;

use num_traits::Float;

pub use crate::prelude::*;

/// ## Sigmoid
/// sigmoid(x) = 1/(1+e^{-x})
pub fn sigmoid<F, T, G, ReqGrad>(
    tensor: Tensor<F, T, G>,
    requires_grad: ReqGrad,
) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
where
    ReqGrad: ReqGradTrait<F>,
    F: Float + Copy,
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
{
    let mut storage = tensor.get_storage().borrow_mut();

    let array = storage.get_as_array_ref::<T>(tensor.get_array_idx(), ContiguousType::Arr)?;
    let shape = array.shape.to_vec();
    let len = shape.iter().product::<usize>();
    let mut vec = Vec::with_capacity(len);
    let one = F::one();

    for i in 0..len {
        let x = array.linear_index(i)?;
        let y = one / (one + (-x).exp());
        vec.push(y);
    }
    let result = Array::from_vector_with_shape(&vec, &shape)?;

    let array_idx = storage.push(ElementType::Arr(result))?;
    let grad_idx = requires_grad.into_zeros_grad_storage(&shape, &mut storage)?;
    let record_label = RecordLabel::Sigmoid(array_idx, tensor.get_grad_idx(), grad_idx);

    let mut record = tensor.get_record().borrow_mut();
    record.push(Some(record_label));
    let record_status = Some(RecordStatus::Record(record.len()));

    Ok(Tensor::_new(
        array_idx,
        grad_idx,
        shape,
        record_status,
        tensor.get_record().clone(),
        tensor.get_storage().clone(),
    ))
}

/// - sigmoid(x) = 1/(1+e^{-x})
/// - dsigmoid(x)/dx = sigmoid(x) * (1 - sigmoid(x)) * grad
pub fn sigmoid_backward<F>(
    output_idx: StorageType,
    array_grad_idx: Option<StorageType>,
    gradient_idx: Option<StorageType>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    F: Float + AddAssign,
{
    if let Some(gradient_idx) = gradient_idx {
        if is_no_grad_or_time_not_match_or_no_update(gradient_idx, storage)? {
            return Ok(());
        }

        if let Some(lhs_grad_idx) = array_grad_idx {
            if !is_no_grad_or_time_not_match_or_no_update(lhs_grad_idx, storage)? {
                let mut lhs_gradient = storage.take_grad(lhs_grad_idx)?;
                let mut lhs_gradient_ref = lhs_gradient.to_array_ref_mut::<View>();

                let gradient =
                    storage.get_as_array_ref::<Contiguous>(gradient_idx, ContiguousType::Grad)?;

                let output = storage.get_as_array_ref::<View>(output_idx, ContiguousType::Arr)?;
                let len = output.shape.iter().product::<usize>();

                let one = F::one();
                for i in 0..len {
                    let x = output.linear_index(i)?;
                    let y = x * (one - x) * gradient.linear_index(i)?;
                    *lhs_gradient_ref.linear_index_mut(i)? += y;
                }
                storage.replace_grad(lhs_grad_idx, lhs_gradient)?;
            }
        }
    }

    Ok(())
}
