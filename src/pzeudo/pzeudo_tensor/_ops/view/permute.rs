use crate::prelude::*;

impl<F, T, G> Tensor<F, T, G> {
    pub fn permute(&self, permute: &[usize]) -> Result<Tensor<F, View, G>, PzeudoErr>
    where
        for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
        F: Copy,
    {
        let mut storage = self.storage.borrow_mut();

        let array = storage.get_as_array_ref::<T>(self.array_idx, ContiguousType::Arr)?;
        let arr_permute = OpsPermute::permute(&array, permute)?;
        let offset = arr_permute.offset;
        let shape = arr_permute.shape.to_vec();
        let stride = arr_permute.stride.to_vec();

        let arr_metadata = TensorMetadata::new(
            offset,
            shape.to_vec(),
            stride.to_vec(),
            self.get_array_idx().to_view_element_type()?,
        );
        let array_idx = storage.push(ElementType::View(arr_metadata))?;

        let grad_idx = self.get_grad_idx().map_or(Ok(None), |grad_idx| {
            let grad_metadata = TensorMetadata::new(
                offset,
                shape.to_vec(),
                stride,
                grad_idx.to_view_element_type()?,
            );
            let grad_idx = storage.push(ElementType::View(grad_metadata))?;

            Ok(Some(grad_idx))
        })?;

        Ok(Tensor::new(
            array_idx,
            grad_idx,
            shape,
            self.get_record().clone(),
            self.get_storage().clone(),
        ))
    }

    pub fn t(&self) -> Result<Tensor<F, View, Grad>, PzeudoErr>
    where
        for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
        F: Copy,
    {
        let mut storage = self.storage.borrow_mut();

        let array = storage.get_as_array_ref::<T>(self.array_idx, ContiguousType::Arr)?;
        let arr_permute = array.t();
        let offset = arr_permute.offset;
        let shape = arr_permute.shape.to_vec();
        let stride = arr_permute.stride.to_vec();

        let arr_metadata = TensorMetadata::new(
            offset,
            shape.to_vec(),
            stride.to_vec(),
            self.get_array_idx().to_view_element_type()?,
        );
        let array_idx = storage.push(ElementType::View(arr_metadata))?;

        let grad_idx = self.get_grad_idx().map_or(Ok(None), |grad_idx| {
            let grad_metadata = TensorMetadata::new(
                offset,
                shape.to_vec(),
                stride,
                grad_idx.to_view_element_type()?,
            );
            let grad_idx = storage.push(ElementType::View(grad_metadata))?;

            Ok(Some(grad_idx))
        })?;

        Ok(Tensor::new(
            array_idx,
            grad_idx,
            shape,
            self.get_record().clone(),
            self.get_storage().clone(),
        ))
    }
}
