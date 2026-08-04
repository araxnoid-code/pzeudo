use crate::prelude::*;
use num_traits::{Float, NumCast, One, Zero};
use std::ops::AddAssign;

impl<F, T, G> Tensor<F, T, G>
where
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
    F: Clone + Zero + Float,
{
    pub fn powi<OutGrad>(
        &self,
        i: i32,
        requires_grad: OutGrad,
    ) -> Result<Tensor<F, Contiguous, OutGrad>, PzeudoErr>
    where
        OutGrad: ReqGradTrait<F>,
    {
        let mut storage = self.storage.borrow_mut();

        let array = storage.get_as_array_ref::<T>(self.get_array_idx(), ContiguousType::Arr)?;
        let arr_log = array.powi(i)?;
        let shape = arr_log.shape.to_vec();

        let array_idx = storage.push(ElementType::Arr(arr_log))?;
        let grad_idx = requires_grad.into_zeros_grad(&shape, &mut storage)?;

        let record_label =
            RecordLabel::Powi((self.get_array_idx(), self.get_grad_idx()), i, grad_idx);
        self.get_record().borrow_mut().push(record_label);

        Ok(Tensor::new(
            array_idx,
            grad_idx,
            shape,
            self.get_record().clone(),
            self.get_storage().clone(),
        ))
    }

    pub fn powf<OutGrad>(
        &self,
        f: F,
        requires_grad: OutGrad,
    ) -> Result<Tensor<F, Contiguous, OutGrad>, PzeudoErr>
    where
        OutGrad: ReqGradTrait<F>,
    {
        let mut storage = self.storage.borrow_mut();

        let array = storage.get_as_array_ref::<T>(self.get_array_idx(), ContiguousType::Arr)?;
        let arr_log = array.powf(f)?;
        let shape = arr_log.shape.to_vec();

        let array_idx = storage.push(ElementType::Arr(arr_log))?;
        let grad_idx = requires_grad.into_zeros_grad(&shape, &mut storage)?;

        let record_label =
            RecordLabel::Powf((self.get_array_idx(), self.get_grad_idx()), f, grad_idx);
        self.get_record().borrow_mut().push(record_label);

        Ok(Tensor::new(
            array_idx,
            grad_idx,
            shape,
            self.get_record().clone(),
            self.get_storage().clone(),
        ))
    }
}

pub fn powi_backward<F>(
    lhs_idx: StorageType,
    lhs_grad_idx: Option<StorageType>,
    i: i32,
    gradient_idx: Option<StorageType>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    F: Float + NumCast + AddAssign,
{
    if let Some(gradient_idx) = gradient_idx {
        if check_no_grad_or_time_not_match(gradient_idx, storage)? {
            return Ok(());
        }

        let gradient =
            storage.get_as_array_ref::<Contiguous>(gradient_idx, ContiguousType::Grad)?;

        if let Some(lhs_grad_idx) = lhs_grad_idx {
            if !check_no_grad_or_time_not_match(lhs_grad_idx, storage)? {
                let grad = storage
                    .get_as_array_ref::<View>(lhs_idx, ContiguousType::Arr)?
                    .powi(i - 1)?
                    .mul_scalar(F::from(i).ok_or(PzeudoErr::PowiBackwardErr(format!(
                        "powi_backward. cannot cast on i32 which has value {i}"
                    )))?)?
                    .mul(&gradient)?;

                let mut lhs_gradient =
                    storage.get_as_array_ref_mut::<View>(lhs_grad_idx, ContiguousType::Grad)?;
                lhs_gradient.add_assign(&grad)?;
            }
        }
    }
    Ok(())
}

pub fn powf_backward<F>(
    lhs_idx: StorageType,
    lhs_grad_idx: Option<StorageType>,
    f: F,
    gradient_idx: Option<StorageType>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    F: Float + NumCast + AddAssign + One,
{
    if let Some(gradient_idx) = gradient_idx {
        if check_no_grad_or_time_not_match(gradient_idx, storage)? {
            return Ok(());
        }
        let gradient =
            storage.get_as_array_ref::<Contiguous>(gradient_idx, ContiguousType::Grad)?;

        if let Some(lhs_grad_idx) = lhs_grad_idx {
            if !check_no_grad_or_time_not_match(lhs_grad_idx, storage)? {
                let grad = storage
                    .get_as_array_ref::<View>(lhs_idx, ContiguousType::Arr)?
                    .powf(f - F::one())?
                    .mul_scalar(f)?
                    .mul(&gradient)?;

                let mut lhs_gradient =
                    storage.get_as_array_ref_mut::<View>(lhs_grad_idx, ContiguousType::Grad)?;
                lhs_gradient.add_assign(&grad)?;
            }
        }
    }
    Ok(())
}
