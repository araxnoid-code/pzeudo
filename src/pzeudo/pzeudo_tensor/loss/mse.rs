use crate::prelude::*;
use num_traits::{Float, NumCast, One, Zero};
use std::{
    format,
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Neg, Sub},
    vec,
};

/// ## Mean Squared Error
/// - mse = 1/n * ∑(target - prediction)^2
/// - n: The total number of elements in the tensor.
/// - The result of mse will be summed into a scalar (since pzeudo does not yet support 0D tensors/scalars, it returns a 1D tensor containing a single value).
/// - The backward pass using mse_backward computes gradients only for the prediction.
pub fn mse<F, T, J, LhsGrad, RhsGrad, ReqGrad>(
    actual: &Tensor<F, J, LhsGrad>,
    prediction: &Tensor<F, T, RhsGrad>,
    requires_grad: ReqGrad,
) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
where
    ReqGrad: ReqGradTrait<F>,
    for<'a> F: Sub<Output = F>
        + Copy
        + Float
        + Zero
        + Clone
        + Sum<&'a F>
        + AddAssign
        + NumCast
        + DivAssign,
    for<'a> ArrayRef<'a, F, T>: OpsSub<F> + OpsBroadcast<F>,
    for<'a> ArrayRef<'a, F, J>: OpsSub<F> + OpsBroadcast<F>,
{
    let mut storage = prediction.storage.borrow_mut();

    let pred_array =
        storage.get_as_array_ref::<T>(prediction.get_array_idx(), ContiguousType::Arr)?;
    let actual_array =
        storage.get_as_array_ref::<J>(actual.get_array_idx(), ContiguousType::Arr)?;

    if pred_array.shape != actual_array.shape {
        return Err(PzeudoErr::LossErr(format!(
            "mse. actual shape: {:?}, predicted shape: {:?}. The shape of both tensors must be the same",
            actual_array.shape, pred_array.shape
        )));
    }

    let len = actual_array.shape.iter().product::<usize>();
    let mut sum = F::zero();
    for i in 0..len {
        let act = actual_array.linear_index(i)?;
        let pred = pred_array.linear_index(i)?;
        let y = act - pred;
        let y = y * y;
        sum += y;
    }
    sum /= F::from(len).ok_or(PzeudoErr::LossErr(format!(
        "mse. Unable to cast on length type"
    )))?;

    let loss_array = Array::from_vector(&[sum]);
    let array_idx = storage.push(ElementType::Arr(loss_array))?;
    let grad_idx = requires_grad.into_zeros_grad_storage(&[1], &mut storage)?;

    let record_label = RecordLabel::LossMse(
        actual.get_array_idx(),
        prediction.get_array_idx(),
        prediction.get_grad_idx(),
        grad_idx,
    );
    let mut record = prediction.get_record().borrow_mut();
    record.push(Some(record_label));
    let record_status = Some(RecordStatus::Record(record.len()));

    Ok(Tensor::_new(
        array_idx,
        grad_idx,
        vec![1],
        record_status,
        prediction.record.clone(),
        prediction.storage.clone(),
    ))
}

/// - mse = 1/n * ∑(target - prediction)^2
/// - dmse/dprediction = -gradient/n * 2 * (target - prediction)
pub fn mse_backward<F>(
    grad_idx: Option<StorageType>,
    actual_idx: StorageType,
    prediction_idx: StorageType,
    prediction_grad_idx: Option<StorageType>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    F: NumCast
        + One
        + Add<Output = F>
        + Div<Output = F>
        + Neg<Output = F>
        + Copy
        + AddAssign
        + Sub<Output = F>,
{
    if let Some(grad_idx) = grad_idx {
        if is_no_grad_or_time_not_match_or_no_update(grad_idx, storage)? {
            return Ok(());
        }

        let gradient = storage.take_grad(grad_idx)?;
        let gradient_ref = gradient.to_array_ref::<Contiguous>();
        let grad_val = gradient_ref.linear_index(0)?;

        if let Some(prediction_grad_idx) = prediction_grad_idx {
            if is_no_grad_or_time_not_match_or_no_update(prediction_grad_idx, storage)? {
                return Ok(());
            }

            let mut prediction_grad = storage.take_grad(prediction_grad_idx)?;
            let mut prediction_grad_ref = prediction_grad.to_array_ref_mut::<View>();

            let actual_value = storage.get_as_array_ref::<View>(actual_idx, ContiguousType::Arr)?;
            let prediction_value =
                storage.get_as_array_ref::<View>(prediction_idx, ContiguousType::Arr)?;

            let scalar = -(F::one() + F::one())
                / F::from(actual_value.shape.iter().product::<usize>()).ok_or(
                    PzeudoErr::BackwardErr(format!("mse_backward. Cannot cast on scalar length")),
                )?;

            let grad = grad_val * scalar;
            let len = actual_value.shape.iter().product::<usize>();
            for i in 0..len {
                let act = actual_value.linear_index(i)?;
                let pred = prediction_value.linear_index(i)?;
                let y = grad * (act - pred);
                *prediction_grad_ref.linear_index_mut(i)? += y;
            }

            storage.replace_grad(prediction_grad_idx, prediction_grad)?;
        }
        storage.replace_grad(grad_idx, gradient)?;
    }
    Ok(())
}
