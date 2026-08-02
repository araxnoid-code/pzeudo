use crate::prelude::*;
use num_traits::Zero;

impl<F, T, G> Tensor<F, T, G> {
    pub fn view(&self) -> Result<Tensor<F, View, Grad>, PzeudoErr>
    where
        F: Clone + Zero,
    {
        let mut storage = self.storage.borrow_mut();
        let array = storage.get_as_array_ref::<T>(self.array_idx, ContiguousType::Arr)?;
        let shape = array.shape.to_vec();
        let arr_metadata = TensorMetadata::new(
            array.offset,
            array.shape.to_vec(),
            array.stride.to_vec(),
            self.array_idx.to_view_element_type()?,
        );
        let array_idx = storage.push(ElementType::View(arr_metadata))?;

        let grad_idx = self.grad_idx.map_or(Ok(None), |grad_idx| {
            let grad = storage.get_as_array_ref::<T>(grad_idx, ContiguousType::Grad)?;

            let grad_metadata = TensorMetadata::new(
                grad.offset,
                grad.shape.to_vec(),
                grad.stride.to_vec(),
                grad_idx.to_view_element_type()?,
            );
            Ok::<_, PzeudoErr>(Some(storage.push(ElementType::View(grad_metadata))?))
        })?;

        drop(storage);
        let view = Tensor::new(
            array_idx,
            grad_idx,
            shape,
            self.record.clone(),
            self.storage.clone(),
        );

        Ok(view)
    }
}
