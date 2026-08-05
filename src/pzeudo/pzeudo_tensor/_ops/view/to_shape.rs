use crate::prelude::*;

impl<F, G> Tensor<F, Contiguous, G> {
    pub fn to_shape(&self, shape: &[usize]) -> Result<Tensor<F, View, G>, PzeudoErr>
    where
        for<'a> ArrayRef<'a, F, Contiguous>: ArrayTrait<F>,
        F: Copy,
    {
        let mut storage = self.storage.borrow_mut();

        let array = storage.get_as_array_ref::<Contiguous>(self.array_idx, ContiguousType::Arr)?;

        let result = array.to_shape(shape)?;
        let metadata = result.get_metadata();

        let offset = metadata.offset;
        let shape = metadata.shape.to_vec();
        let stride = metadata.stride.to_vec();

        let array_metadata = TensorMetadata::new(
            offset,
            shape.to_vec(),
            stride.to_vec(),
            self.get_array_idx().to_view_element_type()?,
        );
        let array_idx = storage.push(ElementType::View(array_metadata))?;

        let grad_idx = self.get_grad_idx().map_or(Ok(None), |grad_idx| {
            let grad_metadata = TensorMetadata::new(
                offset,
                shape.to_vec(),
                stride,
                grad_idx.to_view_element_type()?,
            );

            Ok(Some(storage.push(ElementType::View(grad_metadata))?))
        })?;

        Ok(Tensor::_new(
            array_idx,
            grad_idx,
            shape,
            self.record.clone(),
            self.storage.clone(),
        ))
    }

    pub fn reshape(&self, shape: &[i32]) -> Result<Tensor<F, View, Grad>, PzeudoErr>
    where
        for<'a> ArrayRef<'a, F, Contiguous>: ArrayTrait<F>,
        F: Copy,
    {
        let mut storage = self.storage.borrow_mut();

        let array = storage.get_as_array_ref::<Contiguous>(self.array_idx, ContiguousType::Arr)?;

        let result = array.reshape(shape)?;
        let metadata = result.get_metadata();

        let offset = metadata.offset;
        let shape = metadata.shape.to_vec();
        let stride = metadata.stride.to_vec();

        let array_metadata = TensorMetadata::new(
            offset,
            shape.to_vec(),
            stride.to_vec(),
            self.get_array_idx().to_view_element_type()?,
        );
        let array_idx = storage.push(ElementType::View(array_metadata))?;

        let grad_idx = self.get_grad_idx().map_or(Ok(None), |grad_idx| {
            let grad_metadata = TensorMetadata::new(
                offset,
                shape.to_vec(),
                stride,
                grad_idx.to_view_element_type()?,
            );

            Ok(Some(storage.push(ElementType::View(grad_metadata))?))
        })?;

        Ok(Tensor::_new(
            array_idx,
            grad_idx,
            shape,
            self.record.clone(),
            self.storage.clone(),
        ))
    }
}
