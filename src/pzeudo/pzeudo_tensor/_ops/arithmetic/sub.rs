use crate::prelude::*;
use num_traits::Zero;
use std::{marker::PhantomData, ops::Sub};

impl<F, T> Tensor<F, T>
where
    for<'a> ArrayRef<'a, F, T>: OpsAdd<F>,
{
    pub fn sub<Rhs>(&self, rhs: &Tensor<F, Rhs>) -> Result<Tensor<F, Contiguous>, PzeudoErr>
    where
        F: Copy + Sub<Output = F> + Zero + Clone,
        for<'a> ArrayRef<'a, F, Rhs>: ArrayTrait<F>,
    {
        let mut storage = self.get_storage().borrow_mut();

        let lhs_metadata = self.get_array_metadata();
        let lhs_array_index = self.get_array_idx();
        let lhs_array: ArrayRef<'_, F, T> =
            metadata_to_array_ref(lhs_metadata, lhs_array_index, &storage);

        let rhs_metadata = self.get_array_metadata();
        let rhs_array_index = self.get_array_idx();
        let rhs_array: ArrayRef<'_, F, Rhs> =
            metadata_to_array_ref(rhs_metadata, rhs_array_index, &storage);

        let result = OpsSub::sub(&lhs_array, &rhs_array)?;
        let gradient = Array::<F>::zeros(&result.shape);

        let arr_metadata = TensorMetadata {
            offset: result.offset,
            shape: result.shape,
            stride: result.stride,
        };
        let array_idx = storage.push(result.data)?;

        let grad_metadata = Some(TensorMetadata {
            offset: gradient.offset,
            shape: gradient.shape,
            stride: gradient.stride,
        });
        let grad_idx = Some(storage.push(gradient.data)?);

        let record_label = RecordLabel::Sub(
            (self.array_idx, self.grad_idx),
            (rhs.array_idx, rhs.grad_idx),
            grad_idx,
        );
        self.record.borrow_mut().push(record_label);

        drop(storage);
        let tensor: Tensor<F, Contiguous> = Tensor {
            array_idx,
            array_metadata: arr_metadata,
            grad_idx,
            grad_metadata,
            storage: self.get_storage().clone(),
            record: self.record.clone(),
            _tensor_type: PhantomData::default(),
        };

        Ok(tensor)
    }
}
