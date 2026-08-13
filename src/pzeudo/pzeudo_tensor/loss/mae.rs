use crate::prelude::*;
use num_traits::{Float, NumCast};
use std::{
    iter::Sum,
    ops::{AddAssign, DivAssign, Sub},
};

/// ## Mean Absolute Error
/// - mae = 1/n * ∑|target - prediction|
/// - n: The total number of elements in the tensor.
/// - The result of mae will be summed into a scalar (since pzeudo does not yet support 0D tensors/scalars, it returns a 1D tensor containing a single value).
/// - The backward pass using mae_backward computes gradients only for the prediction.
pub fn mae<F, T, J, LhsGrad, RhsGrad, ReqGrad>(
    actual: &Tensor<F, T, LhsGrad>,
    prediction: &Tensor<F, J, RhsGrad>,
    require_grad: ReqGrad,
) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
where
    ReqGrad: ReqGradTrait<F>,
    for<'a> ArrayRef<'a, F, J>: ArrayTrait<F>,
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
    for<'a> F: Sub<Output = F> + Copy + NumCast + Float + Sum<&'a F> + AddAssign + DivAssign,
{
    let mut storage = prediction.storage.borrow_mut();
    let actual_array =
        storage.get_as_array_ref::<T>(actual.get_array_idx(), ContiguousType::Arr)?;
    let pred_array =
        storage.get_as_array_ref::<J>(prediction.get_array_idx(), ContiguousType::Arr)?;

    if actual_array.shape != pred_array.shape {
        return Err(PzeudoErr::LossErr(format!(
            "mae. actual shape: {:?}, predicted shape: {:?}. The shape of both tensors must be the same",
            actual_array.shape, pred_array.shape
        )));
    }

    let len = pred_array.shape.iter().product::<usize>();

    let mut sum = F::zero();
    for i in 0..len {
        let act = actual_array.linear_index(i)?;
        let pred = pred_array.linear_index(i)?;
        sum += (act - pred).abs();
    }
    sum /= F::from(len).ok_or(PzeudoErr::LossErr(format!("mae. cannot cast to data type")))?;

    let loss_array = Array::from_vector(&[sum]);
    let array_idx = storage.push(ElementType::Arr(loss_array))?;
    let grad_idx = require_grad.into_zeros_grad_storage(&[1], &mut storage)?;

    let record_label = RecordLabel::LossMae(
        actual.get_array_idx(),
        prediction.get_array_idx(),
        prediction.get_grad_idx(),
        grad_idx,
    );

    let mut record = prediction.get_record().borrow_mut();
    let record_status = Some(RecordStatus::Record(record.len()));
    record.push(record_label);

    Ok(Tensor::_new(
        array_idx,
        grad_idx,
        vec![1],
        record_status,
        prediction.get_record().clone(),
        prediction.get_storage().clone(),
    ))
}

/// - mae = 1/n * ∑|target - prediction|
/// - dmae/dprediction = -gradient/n * signum(target - prediction)
pub fn mae_backward<F>(
    grad_idx: Option<StorageType>,
    actual_idx: StorageType,
    prediction_idx: StorageType,
    prediction_grad_idx: Option<StorageType>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    F: NumCast + Float + Sub<Output = F> + AddAssign,
{
    if let Some(grad_idx) = grad_idx {
        if is_no_grad_or_time_not_match_or_no_update(grad_idx, storage)? {
            return Ok(());
        }

        let gradient = storage.take_grad(grad_idx)?;
        let gradient_ref = gradient.to_array_ref::<Contiguous>();
        let grad_val = gradient_ref.linear_index(0)?;

        if let Some(prediction_grad_idx) = prediction_grad_idx {
            storage.set_grad_update(prediction_grad_idx, true)?;
            if is_no_grad_or_time_not_match_or_no_update(prediction_grad_idx, storage)? {
                return Ok(());
            }

            let mut prediction_grad = storage.take_grad(prediction_grad_idx)?;
            let mut prediction_grad_ref = prediction_grad.to_array_ref_mut::<View>();

            let actual_value = storage.get_as_array_ref::<View>(actual_idx, ContiguousType::Arr)?;
            let prediction_value =
                storage.get_as_array_ref::<View>(prediction_idx, ContiguousType::Arr)?;

            let n = -F::from(actual_value.shape.iter().product::<usize>()).ok_or(
                PzeudoErr::BackwardErr(format!("mae_backward. Cannot cast on scalar length")),
            )?;

            let len = actual_value.shape.iter().product::<usize>();
            let grad = grad_val / n;
            for i in 0..len {
                let act = actual_value.linear_index(i)?;
                let pred = prediction_value.linear_index(i)?;
                let y = grad * (act - pred).signum();

                *prediction_grad_ref.linear_index_mut(i)? += y;
            }

            storage.replace_grad(prediction_grad_idx, prediction_grad)?;
        }

        storage.replace_grad(grad_idx, gradient)?;
    }

    Ok(())
}
