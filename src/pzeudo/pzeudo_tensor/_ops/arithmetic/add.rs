use crate::prelude::*;
use num_traits::Zero;
use std::ops::Add;

pub trait TensorAddOps<F, T>: TensorTrait<F, T> {
    fn add<Rhs, J>(&self, rhs: &Rhs) -> Result<Tensor<F>, PzeudoErr>
    where
        F: Copy + Add<Output = F> + Zero + Clone,
        Rhs: TensorTrait<F, J>,
        for<'a> ArrayRef<'a, F, T>: OpsAdd<F>,
        for<'a> ArrayRef<'a, F, J>: OpsAdd<F>,
    {
        let mut storage = self.get_storage().borrow_mut();

        let lhs_array_data = self.get_array_data();
        let lhs_array = metadata_to_array_ref(lhs_array_data, &storage)?;

        let rhs_array_data = rhs.get_array_data();
        let rhs_array = metadata_to_array_ref(rhs_array_data, &storage)?;

        let array = OpsAdd::add(&lhs_array, &rhs_array)?;
        let grad = Array::<F>::zeros(&array.shape);

        let array_idx = storage.push(array)?;
        let grad_idx = Some(storage.push(grad)?);

        let record_label = RecordLabel::Add(
            (self.get_array_idx(), self.get_grad_idx()),
            (rhs.get_array_idx(), rhs.get_grad_idx()),
            grad_idx,
        );
        self.get_record().borrow_mut().push(record_label);

        let tensor = Tensor {
            array_idx,
            grad_idx,
            record: self.get_record().clone(),
            storage: self.get_storage().clone(),
        };

        Ok(tensor)
    }
}

impl<F> TensorAddOps<F, Contiguous> for Tensor<F> {}
impl<F> TensorAddOps<F, View> for TensorView<F> {}
