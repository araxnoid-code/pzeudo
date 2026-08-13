use crate::prelude::*;
use num_traits::{Float, NumCast};
use std::{
    iter::Sum,
    ops::{Add, AddAssign, Div, MulAssign, Neg},
};

/// ## Cross Entropy Loss
/// - H(p,q) = -∑p * ln(q + epsilon)
/// - epsilon = 1e-7
/// - p: target probability
/// - q: prediction probability
/// - The result of H(p,q) will be summed into a scalar (since pzeudo does not yet support 0D tensors/scalars, it returns a 1D tensor containing a single value).
/// - The backward pass using cross_entropy_loss_backward computes gradients only for the prediction.
pub fn cross_entropy_loss<F, T, J, LhsGrad, RhsGrad, ReqGrad>(
    actual: Tensor<F, T, LhsGrad>,
    prediction: Tensor<F, J, RhsGrad>,
    requieres_grad: ReqGrad,
) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
where
    ReqGrad: ReqGradTrait<F>,
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
    for<'a> ArrayRef<'a, F, J>: ArrayTrait<F>,
    for<'a> F: NumCast + Copy + Add<Output = F> + Float + Sum<&'a F> + AddAssign + MulAssign,
{
    let mut storage = prediction.get_storage().borrow_mut();

    let actual_array =
        storage.get_as_array_ref::<T>(actual.get_array_idx(), ContiguousType::Arr)?;
    let pred_array =
        storage.get_as_array_ref::<J>(prediction.get_array_idx(), ContiguousType::Arr)?;

    if actual_array.shape != pred_array.shape {
        return Err(PzeudoErr::LossErr(format!(
            "cross_entropy_loss. actual shape: {:?}, predicted shape: {:?}. The shape of both tensors must be the same",
            actual_array.shape, pred_array.shape
        )));
    }

    let epsilon = F::from(1e-7).ok_or(PzeudoErr::LossErr(format!(
        "cross_entropy_loss. cannot cast data type epsilon"
    )))?;

    let len = actual_array.shape.iter().product::<usize>();

    let mut sum = F::zero();
    for i in 0..len {
        let act = actual_array.linear_index(i)?;
        let pred = pred_array.linear_index(i)?;
        sum += act * (pred + epsilon).ln();
    }
    sum *= -F::one();

    let loss_array = Array::from_vector(&[sum]);
    let array_idx = storage.push(ElementType::Arr(loss_array))?;
    let grad_idx = requieres_grad.into_zeros_grad_storage(&[1], &mut storage)?;

    let record_label = RecordLabel::CrossEntropyLoss(
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
        prediction.get_record().clone(),
        prediction.get_storage().clone(),
    ))
}

/// - H(p,q) = -∑p * ln(q + epsilon)
/// - dH(p,q)/dq = -p/(q + epsilon) * gradient
/// - epsilon = 1e-7
/// - p: target probability
/// - q: prediction probability
pub fn cross_entropy_loss_backward<F>(
    actual_idx: StorageType,
    prediction_idx: StorageType,
    prediction_grad_idx: Option<StorageType>,
    gradient_idx: Option<StorageType>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    F: Copy + Div<Output = F> + Neg<Output = F> + AddAssign + Float + NumCast,
{
    if let Some(gradient_idx) = gradient_idx {
        if is_no_grad_or_time_not_match_or_no_update(gradient_idx, storage)? {
            return Ok(());
        }

        let gradient = storage.take_grad(gradient_idx)?;
        let gradient_ref = gradient.to_array_ref::<Contiguous>();
        let grad_val = gradient_ref.linear_index(0)?;

        if let Some(prediction_grad_idx) = prediction_grad_idx {
            storage.set_grad_update(prediction_grad_idx, true)?;
            if !is_no_grad_or_time_not_match_or_no_update(prediction_grad_idx, storage)? {
                let mut prediction_grad = storage.take_grad(prediction_grad_idx)?;
                let mut prediction_grad_ref = prediction_grad.to_array_ref_mut::<View>();

                let actual_value =
                    storage.get_as_array_ref::<View>(actual_idx, ContiguousType::Arr)?;
                let prediction_value =
                    storage.get_as_array_ref::<View>(prediction_idx, ContiguousType::Arr)?;

                let epsilon = F::from(1e-7).ok_or(PzeudoErr::BackwardErr(format!(
                    "cross_entropy_loss_backward. cannot cast data type epsilon"
                )))?;

                let len = actual_value.shape.iter().product::<usize>();
                for i in 0..len {
                    let p = actual_value.linear_index(i)?;
                    let q = prediction_value.linear_index(i)?;
                    let y = -p / (q + epsilon) * grad_val;
                    *prediction_grad_ref.linear_index_mut(i)? += y;
                }

                storage.replace_grad(prediction_grad_idx, prediction_grad)?;
            }
        }
        storage.replace_grad(gradient_idx, gradient)?;
    }
    Ok(())
}
