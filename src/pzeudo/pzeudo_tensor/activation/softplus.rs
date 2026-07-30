use crate::prelude::*;
use num_traits::Float;
use std::ops::AddAssign;

pub fn softplus<F, T>(tensor: Tensor<F, T>) -> Result<Tensor<F, Contiguous>, PzeudoErr>
where
    F: Float + Copy,
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
{
    // ln(1+e^x)
    let mut storage = tensor.get_storage().borrow_mut();

    let array = storage.get_as_array_ref::<T>(tensor.get_array_idx(), ContiguousType::Arr)?;
    let result = array.exp()?.add_scalar(F::one())?.ln()?;
    let shape = result.shape.to_vec();

    let array_idx = storage.push(ElementType::Arr(result))?;
    let grad_idx = Some(storage.push(ElementType::Grad(Array::zeros(&shape)))?);
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
        let gradient =
            storage.get_as_array_ref::<Contiguous>(gradient_idx, ContiguousType::Grad)?;

        if let Some(array_grad_idx) = array_grad_idx {
            // e^x/(e^x+1) * gradient
            let array = storage.get_as_array_ref::<View>(array_idx, ContiguousType::Arr)?;
            let exp = array.exp()?;
            let grad = exp.div(&exp.add_scalar(F::one())?)?.mul(&gradient)?;

            let mut array_grad =
                storage.get_as_array_ref_mut::<View>(array_grad_idx, ContiguousType::Grad)?;
            array_grad.add_assign(&grad)?;
        }
    }

    Ok(())
}
