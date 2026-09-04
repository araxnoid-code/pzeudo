use crate::prelude::*;
use num_traits::Float;
use std::ops::AddAssign;

/// ## Softplus
/// softplus(x) = ln(1+e^x)
pub fn softplus<F, T, G, ReqGrad>(
    tensor: &Tensor<F, T, G>,
    requires_grad: ReqGrad,
) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
where
    ReqGrad: ReqGradTrait<F>,
    F: Float + Copy,
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
{
    let mut storage = tensor.get_storage().borrow_mut();

    let array = storage.get_as_array_ref::<T>(tensor.get_array_idx(), ContiguousType::Arr)?;
    let len = array.shape.iter().product::<usize>();
    let mut vec = Vec::with_capacity(len);

    for i in 0..len {
        let y = (F::one() + array.linear_index(i)?.exp()).ln();
        vec.push(y);
    }

    let result = Array::from_vector_with_shape(&vec, array.shape)?;
    let shape = result.shape.to_vec();

    let array_idx = storage.push(ElementType::Arr(result))?;
    let grad_idx = requires_grad.into_zeros_grad_storage(&shape, &mut storage)?;
    let record_label =
        RecordLabel::Softplus(tensor.get_array_idx(), tensor.get_grad_idx(), grad_idx);

    let mut record = tensor.get_record().borrow_mut();
    let record_status = Some(RecordStatus::Record(record.len()));
    record.push(record_label);

    Ok(Tensor::_new(
        array_idx,
        grad_idx,
        shape,
        record_status,
        tensor.get_record().clone(),
        tensor.get_storage().clone(),
    ))
}

/// - softplus(x) = ln(1+e^x)
/// - softplus(x)/dx = e^x/(1+e^x) * grad
pub fn softplus_backward<F>(
    array_idx: StorageType,
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
            storage.set_grad_update(lhs_grad_idx, true)?;
            if !is_no_grad_or_time_not_match_or_no_update(lhs_grad_idx, storage)? {
                let mut lhs_gradient = storage.take_grad(lhs_grad_idx)?;
                let mut lhs_gradient_ref = lhs_gradient.to_array_ref_mut::<View>();

                let gradient =
                    storage.get_as_array_ref::<Contiguous>(gradient_idx, ContiguousType::Grad)?;

                let array = storage.get_as_array_ref::<View>(array_idx, ContiguousType::Arr)?;
                let len = array.shape.iter().product::<usize>();
                let one = F::one();
                for i in 0..len {
                    let exp = array.linear_index(i)?.exp();
                    let y = exp / (one + exp) * gradient.linear_index(i)?;
                    *lhs_gradient_ref.linear_index_mut(i)? += y;
                }
                storage.replace_grad(lhs_grad_idx, lhs_gradient)?;
            }
        }
    }

    Ok(())
}
