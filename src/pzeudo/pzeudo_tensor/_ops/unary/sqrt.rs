pub use crate::prelude::*;
use num_traits::{Float, One, Zero};
use std::ops::AddAssign;

impl<F, T> Tensor<F, T>
where
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
    F: Clone + Zero + Float,
{
    pub fn sqrt(&self) -> Result<Tensor<F, Contiguous>, PzeudoErr> {
        let mut storage = self.storage.borrow_mut();

        let array = storage.get_as_array_ref::<T>(self.get_array_idx(), ContiguousType::Arr)?;
        let arr_log = array.sqrt()?;
        let shape = arr_log.shape.to_vec();
        let gradient = Array::<F>::zeros(&shape);

        let array_idx = storage.push(ElementType::Arr(arr_log))?;
        let grad_idx = Some(storage.push(ElementType::Grad(gradient))?);

        let record_label = RecordLabel::Sqrt((array_idx, self.get_grad_idx()), grad_idx);
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

pub fn sqrt_backward<F>(
    out_idx: StorageType,
    lhs_grad_idx: Option<StorageType>,
    gradient_idx: Option<StorageType>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    F: One + Float + AddAssign,
{
    if let Some(gradient_idx) = gradient_idx {
        let gradient =
            storage.get_as_array_ref::<Contiguous>(gradient_idx, ContiguousType::Grad)?;

        if let Some(lhs_grad_idx) = lhs_grad_idx {
            let out_value = storage.get_as_array_ref::<Contiguous>(out_idx, ContiguousType::Arr)?;
            // 1/(2*sqrt(x)) * gradient = gradient/(2*sqrt(x))
            let grad = gradient.div(&out_value.mul_scalar(F::one() + F::one())?)?;

            let mut lhs_gradient =
                storage.get_as_array_ref_mut::<View>(lhs_grad_idx, ContiguousType::Grad)?;
            lhs_gradient.add_assign(&grad)?;
        }
    }
    Ok(())
}
