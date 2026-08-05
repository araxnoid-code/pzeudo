use std::{
    iter::Sum,
    ops::{Add, AddAssign, Div, Neg},
};

use num_traits::{Float, NumCast};

use crate::prelude::*;

pub fn cross_entropy_loss<F, T, J, LhsGrad, RhsGrad, ReqGrad>(
    actual: Tensor<F, T, LhsGrad>,
    prediction: Tensor<F, J, RhsGrad>,
    requieres_grad: ReqGrad,
) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
where
    ReqGrad: ReqGradTrait<F>,
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
    for<'a> ArrayRef<'a, F, J>: ArrayTrait<F>,
    for<'a> F: NumCast + Copy + Add<Output = F> + Float + Sum<&'a F> + AddAssign,
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

    // -p_actual * ln(p_pred)
    let loss = pred_array
        .add_scalar(epsilon)?
        .ln()?
        .mul(&actual_array)?
        .sum()?
        .neg()?;

    let array_idx = storage.push(ElementType::Arr(loss))?;
    let grad_idx = requieres_grad.into_zeros_grad_storage(&[1], &mut storage)?;

    let record_label = RecordLabel::CrossEntropyLoss(
        actual.get_array_idx(),
        prediction.get_array_idx(),
        prediction.get_grad_idx(),
        grad_idx,
    );
    prediction.get_record().borrow_mut().push(record_label);

    Ok(Tensor::_new(
        array_idx,
        grad_idx,
        vec![1],
        prediction.get_record().clone(),
        prediction.get_storage().clone(),
    ))
}

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
        if check_no_grad_or_time_not_match(gradient_idx, storage)? {
            return Ok(());
        }
        let gradient =
            storage.get_as_array_ref::<Contiguous>(gradient_idx, ContiguousType::Grad)?;

        if let Some(prediction_grad_idx) = prediction_grad_idx {
            if !check_no_grad_or_time_not_match(prediction_grad_idx, storage)? {
                let actual_value =
                    storage.get_as_array_ref::<View>(actual_idx, ContiguousType::Arr)?;
                let prediction_value =
                    storage.get_as_array_ref::<View>(prediction_idx, ContiguousType::Arr)?;

                // -actual/prediction
                let epsilon = F::from(1e-7).ok_or(PzeudoErr::LossErr(format!(
                    "cross_entropy_loss_backward. cannot cast data type epsilon"
                )))?;

                let grad = actual_value
                    .div(&prediction_value.add_scalar(epsilon)?)?
                    .mul(&gradient.neg()?)?;

                let mut prediction_grad = storage
                    .get_as_array_ref_mut::<View>(prediction_grad_idx, ContiguousType::Grad)?;

                prediction_grad.add_assign(&grad)?;
            }
        }
    }
    Ok(())
}
