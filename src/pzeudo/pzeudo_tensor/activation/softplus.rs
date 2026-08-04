use crate::prelude::*;
use num_traits::Float;
use std::ops::AddAssign;

pub fn softplus<F, T, G, ReqGrad>(
    tensor: Tensor<F, T, G>,
    requires_grad: ReqGrad,
) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
where
    ReqGrad: RequiresGradTrait<F>,
    F: Float + Copy,
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
{
    // ln(1+e^x)
    let mut storage = tensor.get_storage().borrow_mut();

    let array = storage.get_as_array_ref::<T>(tensor.get_array_idx(), ContiguousType::Arr)?;
    let result = array.exp()?.add_scalar(F::one())?.ln()?;
    let shape = result.shape.to_vec();

    let array_idx = storage.push(ElementType::Arr(result))?;
    let grad_idx = requires_grad.into_zeros_grad(&shape, &mut storage)?;
    let record_label =
        RecordLabel::Softplus(tensor.get_array_idx(), tensor.get_grad_idx(), grad_idx);
    tensor.get_record().borrow_mut().push(record_label);

    Ok(Tensor::new(
        array_idx,
        grad_idx,
        shape,
        tensor.get_record().clone(),
        tensor.get_storage().clone(),
    ))
}

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
        if check_no_grad_or_time_not_match(gradient_idx, storage)? {
            return Ok(());
        }
        let gradient =
            storage.get_as_array_ref::<Contiguous>(gradient_idx, ContiguousType::Grad)?;

        if let Some(lhs_grad_idx) = array_grad_idx {
            if !check_no_grad_or_time_not_match(lhs_grad_idx, storage)? {
                // e^x/(e^x+1) * gradient
                let array = storage.get_as_array_ref::<View>(array_idx, ContiguousType::Arr)?;
                let exp = array.exp()?;
                let grad = exp.div(&exp.add_scalar(F::one())?)?.mul(&gradient)?;

                let mut array_grad =
                    storage.get_as_array_ref_mut::<View>(lhs_grad_idx, ContiguousType::Grad)?;
                array_grad.add_assign(&grad)?;
            }
        }
    }

    Ok(())
}
