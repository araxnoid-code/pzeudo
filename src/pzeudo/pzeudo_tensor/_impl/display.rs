use crate::prelude::*;
use std::fmt::{Debug, Display};

impl<F, T> Display for Tensor<F, T>
where
    F: Debug + Copy,
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let storage = self.storage.borrow();
        let array: ArrayRef<'_, F, T> = storage.get_as_array_ref(self.array_idx).unwrap();

        f.write_str(&format!("{}", array))
    }
}
