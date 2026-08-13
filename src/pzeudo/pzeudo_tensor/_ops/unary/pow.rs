use crate::prelude::*;
use num_traits::{Float, NumCast, One, Zero};
use std::ops::AddAssign;

impl<F, T, G> Tensor<F, T, G>
where
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
    F: Clone + Zero + Float,
{
    /// - f(i, x) = x^i
    /// - i: intiger
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
        let grad_idx = requires_grad.into_zeros_grad_storage(&shape, &mut storage)?;

        let record_label =
            RecordLabel::Powi((self.get_array_idx(), self.get_grad_idx()), i, grad_idx);

        let mut record = self.get_record().borrow_mut();
        let record_status = Some(RecordStatus::Record(record.len()));
        record.push(record_label);

        Ok(Tensor::_new(
            array_idx,
            grad_idx,
            shape,
            record_status,
            self.get_record().clone(),
            self.get_storage().clone(),
        ))
    }

    /// - f(f, x) = x^f
    /// - f: float
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
        let grad_idx = requires_grad.into_zeros_grad_storage(&shape, &mut storage)?;

        let record_label =
            RecordLabel::Powf((self.get_array_idx(), self.get_grad_idx()), f, grad_idx);

        let mut record = self.get_record().borrow_mut();
        let record_status = Some(RecordStatus::Record(record.len()));
        record.push(record_label);

        Ok(Tensor::_new(
            array_idx,
            grad_idx,
            shape,
            record_status,
            self.get_record().clone(),
            self.get_storage().clone(),
        ))
    }
}

/// - f(i, x) = x^i
/// - df(i, x)/dx = ix^{i-1} * gradient
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
        if is_no_grad_or_time_not_match_or_no_update(gradient_idx, storage)? {
            return Ok(());
        }

        if let Some(lhs_grad_idx) = lhs_grad_idx {
            storage.set_grad_update(lhs_grad_idx, true)?;
            if !is_no_grad_or_time_not_match_or_no_update(lhs_grad_idx, storage)? {
                // - f(i, x) = x^i
                // - df(i, x)/dx = ix^{i-1} * gradient
                let gradient = storage.take_grad(gradient_idx)?;
                let gradient_ref = gradient.to_array_ref::<Contiguous>();

                let mut lhs_gradient = storage.take_grad(lhs_grad_idx)?;
                let mut lhs_gradient_ref = lhs_gradient.to_array_ref_mut::<View>();
                let lhs_value = storage.get_as_array_ref::<View>(lhs_idx, ContiguousType::Arr)?;

                let len = lhs_value.shape.iter().product::<usize>();
                let scalar = F::from(i).ok_or(PzeudoErr::BackwardErr(format!(
                    "powi_backward. cannot cast on i32 which has value {i}"
                )))?;
                for idx in 0..len {
                    *lhs_gradient_ref.linear_index_mut(idx)? +=
                        lhs_value.linear_index(idx)?.powi(i - 1)
                            * scalar
                            * gradient_ref.linear_index(idx)?;
                }

                storage.replace_grad(lhs_grad_idx, lhs_gradient)?;
                storage.replace_grad(gradient_idx, gradient)?;
            }
        }
    }
    Ok(())
}

/// - f(f, x) = x^f
/// - df(f, x)/dx = fx^{f-1.} * gradient
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
        if is_no_grad_or_time_not_match_or_no_update(gradient_idx, storage)? {
            return Ok(());
        }

        if let Some(lhs_grad_idx) = lhs_grad_idx {
            storage.set_grad_update(lhs_grad_idx, true)?;
            if !is_no_grad_or_time_not_match_or_no_update(lhs_grad_idx, storage)? {
                // - f(f, x) = x^f
                // - df(f, x)/dx = fx^{f-1.} * gradient
                let gradient = storage.take_grad(gradient_idx)?;
                let gradient_ref = gradient.to_array_ref::<Contiguous>();
                let mut lhs_gradient = storage.take_grad(lhs_grad_idx)?;
                let mut lhs_gradient_ref = lhs_gradient.to_array_ref_mut::<View>();
                let lhs_value = storage.get_as_array_ref::<View>(lhs_idx, ContiguousType::Arr)?;

                let len = lhs_value.shape.iter().product::<usize>();
                for idx in 0..len {
                    *lhs_gradient_ref.linear_index_mut(idx)? +=
                        lhs_value.linear_index(idx)?.powf(f - F::one())
                            * f
                            * gradient_ref.linear_index(idx)?;
                }
                storage.replace_grad(lhs_grad_idx, lhs_gradient)?;
                storage.replace_grad(gradient_idx, gradient)?;
            }
        }
    }
    Ok(())
}
