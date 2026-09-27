use crate::prelude::*;

impl<F, T, G> Tensor<F, T, G> {
    pub fn dot<RhsT, RhsG>(&self, rhs: &Tensor<F, RhsT, RhsG>) -> Result<(), PzeudoErr>
    where
        for<'a> ArrayRef<'a, F, T>: ArrayTrait<F> + DotProductF32F64<F, T>,
        for<'a> ArrayRef<'a, F, RhsT>: ArrayTrait<F>,
    {
        let storage = self.get_storage().borrow_mut();
        let lhs_array = storage.get_as_array_ref::<T>(self.get_array_idx(), ContiguousType::Arr)?;
        let rhs_array =
            storage.get_as_array_ref::<RhsT>(rhs.get_array_idx(), ContiguousType::Arr)?;

        let dot = lhs_array.dot_f32f64(&rhs_array)?;

        Ok(())
    }
}
