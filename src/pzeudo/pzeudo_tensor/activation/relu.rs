use std::ops::AddAssign;

use crate::prelude::*;
use num_traits::Float;

pub fn relu<F, T, G>(tensor: Tensor<F, T, G>) -> Result<Tensor<F, Contiguous, Grad>, PzeudoErr>
where
    F: Float + Copy,
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
{
    // max(0, x)
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
    let grad_idx = Some(storage.push(ElementType::Grad(Array::zeros(&shape)))?);
    let record_label = RecordLabel::Relu(tensor.get_array_idx(), tensor.get_grad_idx(), grad_idx);
    tensor.get_record().borrow_mut().push(record_label);

    Ok(Tensor::new(
        array_idx,
        grad_idx,
        shape,
        tensor.get_record().clone(),
        tensor.get_storage().clone(),
    ))
}

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
        let gradient =
            storage.get_as_array_ref::<Contiguous>(gradient_idx, ContiguousType::Grad)?;

        if let Some(array_grad_idx) = array_grad_idx {
            // e^x/(e^x+1) * gradient
            let array = storage.get_as_array_ref::<View>(array_idx, ContiguousType::Arr)?;
            let len = array.shape.iter().product::<usize>();
            let zero = F::zero();

            let mut vec = Vec::with_capacity(len);
            for i in 0..len {
                let x = array.linear_index(i)?;
                let v = if x > zero {
                    gradient.linear_index(i)?
                } else {
                    zero
                };
                vec.push(v);
            }
            let grad = Array::from_vector_with_shape(&vec, array.shape)?;

            let mut array_grad =
                storage.get_as_array_ref_mut::<View>(array_grad_idx, ContiguousType::Grad)?;
            array_grad.add_assign(&grad)?;
        }
    }

    Ok(())
}
