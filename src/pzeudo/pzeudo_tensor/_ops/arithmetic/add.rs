use std::ops::Add;

use crate::prelude::*;

impl<F, T> Tensor<F, T> {
    pub fn add<Rhs>(&self, rhs: &Tensor<F, Rhs>)
    where
        F: Copy + Add<Output = F>,
    {
        let storage = self.get_storage().borrow_mut();

        let lhs_metadata = self.get_array_metadata();
        let lhs_array_index = self.get_array_idx();
        let lhs_array: ArrayRef<'_, F, T> =
            metadata_to_array_ref(lhs_metadata, lhs_array_index, &storage);

        let rhs_metadata = self.get_array_metadata();
        let rhs_array_index = self.get_array_idx();
        let rhs_array: ArrayRef<'_, F, Rhs> =
            metadata_to_array_ref(rhs_metadata, rhs_array_index, &storage);

        let result = OpsAdd::add(&lhs_array, &rhs_array);
    }
}
