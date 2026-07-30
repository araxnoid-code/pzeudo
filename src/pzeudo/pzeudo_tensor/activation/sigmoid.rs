use std::ops::AddAssign;

use num_traits::Float;

pub use crate::prelude::*;

pub fn sigmoid<F, T>(tensor: Tensor<F, T>) -> Result<Tensor<F, Contiguous>, PzeudoErr>
where
    F: Float + Copy,
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
{
    // 1/(1+e^-x)
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
    let grad_idx = Some(storage.push(ElementType::Grad(Array::zeros(&shape)))?);
    let record_label = RecordLabel::Sigmoid(array_idx, tensor.get_grad_idx(), grad_idx);
    tensor.get_record().borrow_mut().push(record_label);

    Ok(Tensor::new(
        array_idx,
        grad_idx,
        shape,
        tensor.get_record().clone(),
        tensor.get_storage().clone(),
    ))
}

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
        let gradient =
            storage.get_as_array_ref::<Contiguous>(gradient_idx, ContiguousType::Grad)?;

        if let Some(array_grad_idx) = array_grad_idx {
            // sigmoid * (1 - sigmoid) * gradient
            let output = storage.get_as_array_ref::<View>(output_idx, ContiguousType::Arr)?;
            let len = output.shape.iter().product::<usize>();
            let mut vec = Vec::with_capacity(len);

            let one = F::one();
            for i in 0..len {
                let x = output.linear_index(i)?;
                let y = x * (one - x) * gradient.linear_index(i)?;
                vec.push(y);
            }

            let grad = Array::from_vector_with_shape(&vec, output.shape)?;

            let mut array_grad =
                storage.get_as_array_ref_mut::<View>(array_grad_idx, ContiguousType::Grad)?;
            array_grad.add_assign(&grad)?;
        }
    }

    Ok(())
}
