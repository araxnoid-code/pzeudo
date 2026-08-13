use crate::prelude::*;
use num_traits::{Float, NumCast, One, Zero};
use std::ops::AddAssign;

impl<F, T, G> Tensor<F, T, G>
where
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
    F: Clone + Zero + Float + One,
{
    /// f(x) = log(2, x)
    pub fn log2<OutGrad>(
        &self,
        requires_grad: OutGrad,
    ) -> Result<Tensor<F, Contiguous, OutGrad>, PzeudoErr>
    where
        OutGrad: ReqGradTrait<F>,
    {
        let mut storage = self.storage.borrow_mut();

        let array = storage.get_as_array_ref::<T>(self.get_array_idx(), ContiguousType::Arr)?;
        let arr_log = array.log2()?;
        let shape = arr_log.shape.to_vec();

        let array_idx = storage.push(ElementType::Arr(arr_log))?;
        let grad_idx = requires_grad.into_zeros_grad_storage(&shape, &mut storage)?;

        let record_label = RecordLabel::Log(
            (self.get_array_idx(), self.get_grad_idx()),
            F::one() + F::one(),
            grad_idx,
        );

        let mut record = self.get_record().borrow_mut();
        record.push(Some(record_label));
        let record_status = Some(RecordStatus::Record(record.len()));

        Ok(Tensor::_new(
            array_idx,
            grad_idx,
            shape,
            record_status,
            self.get_record().clone(),
            self.get_storage().clone(),
        ))
    }

    /// f(x) = log(10, x)
    pub fn log10<OutGrad>(
        &self,
        requires_grad: OutGrad,
    ) -> Result<Tensor<F, Contiguous, OutGrad>, PzeudoErr>
    where
        OutGrad: ReqGradTrait<F>,
        F: NumCast,
    {
        let mut storage = self.storage.borrow_mut();

        let array = storage.get_as_array_ref::<T>(self.get_array_idx(), ContiguousType::Arr)?;
        let arr_log = array.log10()?;
        let shape = arr_log.shape.to_vec();

        let array_idx = storage.push(ElementType::Arr(arr_log))?;
        let grad_idx = requires_grad.into_zeros_grad_storage(&shape, &mut storage)?;

        let record_label = RecordLabel::Log(
            (self.get_array_idx(), self.get_grad_idx()),
            F::from(10).ok_or(PzeudoErr::OpsErr(format!(
                "Tensor::log10. cannot cast 10 to data type"
            )))?,
            grad_idx,
        );
        let mut record = self.get_record().borrow_mut();
        record.push(Some(record_label));
        let record_status = Some(RecordStatus::Record(record.len()));

        Ok(Tensor::_new(
            array_idx,
            grad_idx,
            shape,
            record_status,
            self.get_record().clone(),
            self.get_storage().clone(),
        ))
    }

    /// f(x) = ln(x)
    pub fn ln<OutGrad>(
        &self,
        requires_grad: OutGrad,
    ) -> Result<Tensor<F, Contiguous, OutGrad>, PzeudoErr>
    where
        OutGrad: ReqGradTrait<F>,
    {
        let mut storage = self.storage.borrow_mut();

        let array = storage.get_as_array_ref::<T>(self.get_array_idx(), ContiguousType::Arr)?;
        let arr_log = array.ln()?;
        let shape = arr_log.shape.to_vec();

        let array_idx = storage.push(ElementType::Arr(arr_log))?;
        let grad_idx = requires_grad.into_zeros_grad_storage(&shape, &mut storage)?;

        let record_label = RecordLabel::Ln((self.get_array_idx(), self.get_grad_idx()), grad_idx);
        let mut record = self.get_record().borrow_mut();
        record.push(Some(record_label));
        let record_status = Some(RecordStatus::Record(record.len()));

        Ok(Tensor::_new(
            array_idx,
            grad_idx,
            shape,
            record_status,
            self.get_record().clone(),
            self.get_storage().clone(),
        ))
    }

    /// f(b, x) = log(b, x)
    pub fn log<OutGrad>(
        &self,
        base: F,
        requires_grad: OutGrad,
    ) -> Result<Tensor<F, Contiguous, OutGrad>, PzeudoErr>
    where
        OutGrad: ReqGradTrait<F>,
    {
        let mut storage = self.storage.borrow_mut();

        let array = storage.get_as_array_ref::<T>(self.get_array_idx(), ContiguousType::Arr)?;
        let arr_log = array.log(base)?;
        let shape = arr_log.shape.to_vec();

        let array_idx = storage.push(ElementType::Arr(arr_log))?;
        let grad_idx = requires_grad.into_zeros_grad_storage(&shape, &mut storage)?;

        let record_label =
            RecordLabel::Log((self.get_array_idx(), self.get_grad_idx()), base, grad_idx);
        let mut record = self.get_record().borrow_mut();
        record.push(Some(record_label));
        let record_status = Some(RecordStatus::Record(record.len()));

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

/// - f(x) = log(n, x)
/// - df(x)/x = 1/(x * ln(n)) * gradient = gradient/(x * ln(n))
pub fn log_backward<F>(
    lhs_idx: StorageType,
    lhs_grad_idx: Option<StorageType>,
    gradient_idx: Option<StorageType>,
    base: F,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    F: Float + One + AddAssign,
{
    if let Some(gradient_idx) = gradient_idx {
        if is_no_grad_or_time_not_match_or_no_update(gradient_idx, storage)? {
            return Ok(());
        }

        if let Some(lhs_grad_idx) = lhs_grad_idx {
            storage.set_grad_update(lhs_grad_idx, true)?;
            if !is_no_grad_or_time_not_match_or_no_update(lhs_grad_idx, storage)? {
                let gradient = storage.take_grad(gradient_idx)?;
                let gradient_ref = gradient.to_array_ref::<Contiguous>();
                // f(x) = log(n, x)
                // df(x)/x = 1/(x * ln(n)) * gradient = gradient/(x * ln(n))

                let mut lhs_gradient = storage.take_grad(lhs_grad_idx)?;
                let mut lhs_gradient_ref = lhs_gradient.to_array_ref_mut::<View>();

                let lhs_value = storage.get_as_array_ref::<View>(lhs_idx, ContiguousType::Arr)?;
                let shape = lhs_value.shape.iter().product::<usize>();
                let ln_base = base.ln();

                for i in 0..shape {
                    let x = gradient_ref.linear_index(i)? / (lhs_value.linear_index(i)? * ln_base);
                    *lhs_gradient_ref.linear_index_mut(i)? += x;
                }

                storage.replace_grad(lhs_grad_idx, lhs_gradient)?;
                storage.replace_grad(gradient_idx, gradient)?;
            }
        }
    }
    Ok(())
}

/// - f(x) = ln(x)
/// - df(x)/x = 1 / x * gradient = gradient / x
pub fn ln_backward<F>(
    lhs_idx: StorageType,
    lhs_grad_idx: Option<StorageType>,
    gradient_idx: Option<StorageType>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    F: Float + One + AddAssign,
{
    if let Some(gradient_idx) = gradient_idx {
        if is_no_grad_or_time_not_match_or_no_update(gradient_idx, storage)? {
            return Ok(());
        }

        if let Some(lhs_grad_idx) = lhs_grad_idx {
            storage.set_grad_update(lhs_grad_idx, true)?;
            if !is_no_grad_or_time_not_match_or_no_update(lhs_grad_idx, storage)? {
                // - f(x) = ln(x)
                // - df(x)/x = 1 / x * gradient = gradient / x
                let gradient = storage.take_grad(gradient_idx)?;
                let gradient_ref = gradient.to_array_ref::<Contiguous>();

                let mut lhs_gradient = storage.take_grad(lhs_grad_idx)?;
                let mut lhs_gradient_ref = lhs_gradient.to_array_ref_mut::<View>();

                let lhs_value = storage.get_as_array_ref::<View>(lhs_idx, ContiguousType::Arr)?;
                let len = lhs_value.shape.iter().product::<usize>();
                for i in 0..len {
                    let x = gradient_ref.linear_index(i)? / lhs_value.linear_index(i)?;
                    *lhs_gradient_ref.linear_index_mut(i)? += x;
                }

                storage.replace_grad(lhs_grad_idx, lhs_gradient)?;
                storage.replace_grad(gradient_idx, gradient)?;
            }
        }
    }
    Ok(())
}
