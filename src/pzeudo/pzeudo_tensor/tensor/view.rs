use num_traits::Zero;

use crate::prelude::*;

impl<F, T> Tensor<F, T> {
    pub fn view(&self) -> Result<Tensor<F, View>, PzeudoErr>
    where
        F: Clone + Zero,
    {
        let storage = self.get_storage().clone();
        let mut borrow_storage = storage.borrow_mut();

        let arr_metadata = self.get_array_metadata().clone();
        let gradient: Array<F> = Array::<F>::zeros(&arr_metadata.shape);

        let grad_metadata = TensorMetadata {
            offset: gradient.offset,
            shape: gradient.shape,
            stride: gradient.stride,
        };
        let grad_idx = Some(borrow_storage.push(gradient.data)?);

        drop(borrow_storage);
        let tensor = Tensor {
            array_idx: self.get_array_idx(),
            array_metadata: arr_metadata,
            grad_idx,
            grad_metadata: Some(grad_metadata),
            record: self.get_record().clone(),
            storage,
            _tensor_type: Default::default(),
        };

        Ok(tensor)
    }
}
