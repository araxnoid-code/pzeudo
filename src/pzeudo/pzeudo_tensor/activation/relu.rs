use std::ops::AddAssign;

use crate::prelude::*;
use num_traits::Float;

/// ## Relu
/// relu(x) = max(0, x)
pub fn relu<F, T, G, ReqGrad>(
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
    let mut output = Vec::with_capacity(len);
    for i in 0..len {
        let x = array.linear_index(i)?;
        output.push(x.max(F::zero()));
    }
    let result = Array::from_vector_with_shape(&output, &shape)?;

    let array_idx = storage.push(ElementType::Arr(result))?;
    let grad_idx = requires_grad.into_zeros_grad_storage(&shape, &mut storage)?;
    let record_label = RecordLabel::Relu(tensor.get_array_idx(), tensor.get_grad_idx(), grad_idx);
    tensor.get_record().borrow_mut().push(Some(record_label));

    Ok(Tensor::_new(
        array_idx,
        grad_idx,
        shape,
        tensor.get_record().clone(),
        tensor.get_storage().clone(),
    ))
}

/// - relu(x) = max(0, x)
/// - drelu(x)/dx = (if x > 0 {1} else {0}) * grad
pub fn relu_backward<F>(
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
            if !is_no_grad_or_time_not_match_or_no_update(lhs_grad_idx, storage)? {
                // e^x/(e^x+1) * gradient
                let mut lhs_gradient = storage.take_grad(lhs_grad_idx)?;
                let mut lhs_gradient_ref = lhs_gradient.to_array_ref_mut::<View>();

                let gradient =
                    storage.get_as_array_ref::<Contiguous>(gradient_idx, ContiguousType::Grad)?;

                let array = storage.get_as_array_ref::<View>(array_idx, ContiguousType::Arr)?;
                let len = array.shape.iter().product::<usize>();
                let zero = F::zero();

                for i in 0..len {
                    let x = array.linear_index(i)?;
                    let v = if x > zero {
                        gradient.linear_index(i)?
                    } else {
                        continue;
                    };
                    *lhs_gradient_ref.linear_index_mut(i)? += v;
                }

                storage.replace_grad(lhs_grad_idx, lhs_gradient)?;
            }
        }
    }

    Ok(())
}
