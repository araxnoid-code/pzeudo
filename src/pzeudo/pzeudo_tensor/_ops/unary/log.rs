use crate::prelude::*;
use num_traits::{Float, NumCast, One, Zero};
use std::ops::AddAssign;

impl<F, T> Tensor<F, T>
where
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
    F: Clone + Zero + Float + One,
{
    pub fn log2(&self) -> Result<Tensor<F, Contiguous>, PzeudoErr> {
        let mut storage = self.storage.borrow_mut();

        let array = storage.get_as_array_ref::<T>(self.get_array_idx(), ContiguousType::Arr)?;
        let arr_log = array.log2()?;
        let shape = arr_log.shape.to_vec();
        let gradient = Array::<F>::zeros(&shape);

        let array_idx = storage.push(ElementType::Arr(arr_log))?;
        let grad_idx = Some(storage.push(ElementType::Grad(gradient))?);

        let record_label = RecordLabel::Log(
            (self.get_array_idx(), self.get_grad_idx()),
            F::one() + F::one(),
            grad_idx,
        );
        self.get_record().borrow_mut().push(record_label);

        Ok(Tensor::new(
            array_idx,
            grad_idx,
            shape,
            self.get_record().clone(),
            self.get_storage().clone(),
        ))
    }

    pub fn log10(&self) -> Result<Tensor<F, Contiguous>, PzeudoErr>
    where
        F: NumCast,
    {
        let mut storage = self.storage.borrow_mut();

        let array = storage.get_as_array_ref::<T>(self.get_array_idx(), ContiguousType::Arr)?;
        let arr_log = array.log10()?;
        let shape = arr_log.shape.to_vec();
        let gradient = Array::<F>::zeros(&shape);

        let array_idx = storage.push(ElementType::Arr(arr_log))?;
        let grad_idx = Some(storage.push(ElementType::Grad(gradient))?);

        let record_label = RecordLabel::Log(
            (self.get_array_idx(), self.get_grad_idx()),
            F::from(10).ok_or(PzeudoErr::Log10Err(format!(
                "Tensor::log10. cannot cast 10 to data type"
            )))?,
            grad_idx,
        );
        self.get_record().borrow_mut().push(record_label);

        Ok(Tensor::new(
            array_idx,
            grad_idx,
            shape,
            self.get_record().clone(),
            self.get_storage().clone(),
        ))
    }

    pub fn ln(&self) -> Result<Tensor<F, Contiguous>, PzeudoErr> {
        let mut storage = self.storage.borrow_mut();

        let array = storage.get_as_array_ref::<T>(self.get_array_idx(), ContiguousType::Arr)?;
        let arr_log = array.ln()?;
        let shape = arr_log.shape.to_vec();
        let gradient = Array::<F>::zeros(&shape);

        let array_idx = storage.push(ElementType::Arr(arr_log))?;
        let grad_idx = Some(storage.push(ElementType::Grad(gradient))?);

        let record_label = RecordLabel::Ln((self.get_array_idx(), self.get_grad_idx()), grad_idx);
        self.get_record().borrow_mut().push(record_label);

        Ok(Tensor::new(
            array_idx,
            grad_idx,
            shape,
            self.get_record().clone(),
            self.get_storage().clone(),
        ))
    }

    pub fn log(&self, base: F) -> Result<Tensor<F, Contiguous>, PzeudoErr> {
        let mut storage = self.storage.borrow_mut();

        let array = storage.get_as_array_ref::<T>(self.get_array_idx(), ContiguousType::Arr)?;
        let arr_log = array.log(base)?;
        let shape = arr_log.shape.to_vec();
        let gradient = Array::<F>::zeros(&shape);

        let array_idx = storage.push(ElementType::Arr(arr_log))?;
        let grad_idx = Some(storage.push(ElementType::Grad(gradient))?);

        let record_label =
            RecordLabel::Log((self.get_array_idx(), self.get_grad_idx()), base, grad_idx);
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
        let gradient =
            storage.get_as_array_ref::<Contiguous>(gradient_idx, ContiguousType::Grad)?;

        if let Some(lhs_grad_idx) = lhs_grad_idx {
            let lhs_value = storage.get_as_array_ref::<View>(lhs_idx, ContiguousType::Arr)?;
            let grad = gradient.div(&lhs_value.mul_scalar(base.ln())?)?;

            let mut lhs_gradient =
                storage.get_as_array_ref_mut::<View>(lhs_grad_idx, ContiguousType::Grad)?;
            lhs_gradient.add_assign(&grad)?;
        }
    }
    Ok(())
}

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
        let gradient =
            storage.get_as_array_ref::<Contiguous>(gradient_idx, ContiguousType::Grad)?;

        if let Some(lhs_grad_idx) = lhs_grad_idx {
            let lhs_value = storage.get_as_array_ref::<View>(lhs_idx, ContiguousType::Arr)?;
            let grad = gradient.div(&lhs_value)?;

            let mut lhs_gradient =
                storage.get_as_array_ref_mut::<View>(lhs_grad_idx, ContiguousType::Grad)?;
            lhs_gradient.add_assign(&grad)?;
        }
    }
    Ok(())
}
