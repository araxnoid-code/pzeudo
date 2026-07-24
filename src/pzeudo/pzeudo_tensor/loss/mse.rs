use crate::prelude::*;
use num_traits::{Float, NumCast, One, Zero};
use std::{
    format,
    iter::Sum,
    ops::{Add, AddAssign, Div, Neg, Sub},
};

pub fn mse<F, T, J>(
    actual: &Tensor<F, J>,
    pred: &Tensor<F, T>,
) -> Result<Tensor<F, Contiguous>, PzeudoErr>
where
    for<'a> F: Sub<Output = F> + Copy + Float + Zero + Clone + Sum<&'a F> + AddAssign + NumCast,
    for<'a> ArrayRef<'a, F, T>: OpsSub<F> + OpsBroadcast<F>,
    for<'a> ArrayRef<'a, F, J>: OpsSub<F> + OpsBroadcast<F>,
{
    let mut storage = pred.storage.borrow_mut();

    let pred_array = storage.get_as_array_ref::<T>(pred.get_array_idx(), ContiguousType::Arr)?;
    let actual_array =
        storage.get_as_array_ref::<J>(actual.get_array_idx(), ContiguousType::Arr)?;

    if pred_array.shape != actual_array.shape {
        return Err(PzeudoErr::LossMseErr(format!(
            "mse. actual shape: {:?}, predicted shape: {:?}. The shape of both tensors must be the same",
            actual_array.shape, pred_array.shape
        )));
    }

    let len = F::from(actual_array.shape.iter().product::<usize>()).ok_or(
        PzeudoErr::LossMseErr(format!("mse. Unable to cast on length type")),
    )?;

    let array = actual_array
        .sub(&pred_array)?
        .powi(2)?
        .sum()?
        .div_scalar(len)?;

    let gradient = Array::<F>::zeros(&array.shape);

    let array_idx = storage.push(ElementType::Contiguous(array, ContiguousType::Arr))?;
    let grad_idx = Some(storage.push(ElementType::Contiguous(gradient, ContiguousType::Grad))?);

    let record_label = RecordLabel::LossMse(array_idx, pred.get_grad_idx(), grad_idx);
    pred.record.borrow_mut().push(record_label);

    Ok(Tensor {
        array_idx,
        grad_idx,
        record: pred.record.clone(),
        storage: pred.storage.clone(),
        _array_type: Default::default(),
    })
}

pub fn mse_backward<F>(
    grad_idx: Option<StorageType>,
    output_idx: StorageType,
    prediction_grad_idx: Option<StorageType>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    F: NumCast + One + Add<Output = F> + Div<Output = F> + Neg<Output = F> + Copy + AddAssign,
{
    if let Some(grad_idx) = grad_idx {
        let gradient = storage.get_as_array_ref::<Contiguous>(grad_idx, ContiguousType::Grad)?;

        if let Some(prediction_grad_idx) = prediction_grad_idx {
            let output_value =
                storage.get_as_array_ref::<Contiguous>(output_idx, ContiguousType::Arr)?;

            let scalar = -(F::one() + F::one())
                / F::from(output_value.shape.iter().product::<usize>()).ok_or(
                    PzeudoErr::LossMseBackwardErr(format!(
                        "mse_backward. Cannot cast on scalar length"
                    )),
                )?;

            let grad = output_value.mul_scalar(scalar)?.mul(&gradient)?;

            let mut prediction_grad =
                storage.get_as_array_ref_mut(prediction_grad_idx, ContiguousType::Grad)?;

            let grad_broadcast = grad.broadcast(prediction_grad.shape)?;
            prediction_grad.add_assign(&grad_broadcast)?;
        }
    }
    Ok(())
}
