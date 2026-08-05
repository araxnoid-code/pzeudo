use crate::prelude::*;
use num_traits::{Float, NumCast};
use std::{
    iter::Sum,
    ops::{AddAssign, Sub},
};

pub fn mae<F, T, J, LhsGrad, RhsGrad, ReqGrad>(
    actual: &Tensor<F, T, LhsGrad>,
    prediction: &Tensor<F, J, RhsGrad>,
    require_grad: ReqGrad,
) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
where
    ReqGrad: ReqGradTrait<F>,
    for<'a> ArrayRef<'a, F, J>: ArrayTrait<F>,
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
    for<'a> F: Sub<Output = F> + Copy + NumCast + Float + Sum<&'a F> + AddAssign,
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

    let len = F::from(pred_array.shape.iter().product::<usize>())
        .ok_or(PzeudoErr::LossErr(format!("mae. cannot cast to data type")))?;

    let loss_array = actual_array
        .sub(&pred_array)?
        .abs()?
        .sum()?
        .div_scalar(len)?;

    let array_idx = storage.push(ElementType::Arr(loss_array))?;
    let grad_idx = require_grad.into_zeros_grad_storage(&[1], &mut storage)?;

    let record_label = RecordLabel::LossMae(
        actual.get_array_idx(),
        prediction.get_array_idx(),
        prediction.get_grad_idx(),
        grad_idx,
    );
    prediction.record.borrow_mut().push(record_label);

    Ok(Tensor::_new(
        array_idx,
        grad_idx,
        vec![1],
        prediction.get_record().clone(),
        prediction.get_storage().clone(),
    ))
}

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
        if check_no_grad_or_time_not_match(grad_idx, storage)? {
            return Ok(());
        }
        let gradient = storage.get_as_array_ref::<Contiguous>(grad_idx, ContiguousType::Grad)?;

        if let Some(prediction_grad_idx) = prediction_grad_idx {
            if check_no_grad_or_time_not_match(prediction_grad_idx, storage)? {
                return Ok(());
            }

            let actual_value = storage.get_as_array_ref::<View>(actual_idx, ContiguousType::Arr)?;
            let prediction_value =
                storage.get_as_array_ref::<View>(prediction_idx, ContiguousType::Arr)?;

            let n = -F::from(actual_value.shape.iter().product::<usize>()).ok_or(
                PzeudoErr::LossErr(format!("mae_backward. Cannot cast on scalar length")),
            )?;

            let grad = actual_value
                .sub(&prediction_value)?
                .signum()?
                .mul(&gradient.div_scalar(n)?)?;

            let mut prediction_grad =
                storage.get_as_array_ref_mut::<View>(prediction_grad_idx, ContiguousType::Grad)?;

            prediction_grad.add_assign(&grad)?;
        }
    }

    Ok(())
}
