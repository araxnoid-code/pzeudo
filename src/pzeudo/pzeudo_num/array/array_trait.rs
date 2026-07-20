use std::vec;

use crate::prelude::*;

pub trait ArrayTrait<F> {
    // metadata
    fn get_metadata(&self) -> Metadata<'_, F>;

    // Method
    fn linear_index(&self, index: usize) -> Result<F, PzeudoErr>
    where
        F: Copy,
    {
        let Metadata {
            data,
            offset,
            stride,
            shape,
        } = self.get_metadata();

        let output_stride = shape_to_stride(shape);

        let mut idx = offset;
        for (shape, (stride, out_stride)) in
            shape.iter().zip(stride.iter().zip(output_stride.iter()))
        {
            let permute = (index / out_stride) % shape;
            idx += permute * stride;
        }

        if idx >= data.len() {
            return Err(PzeudoErr::LinearIndexErr(format!(
                "LinearIndexErr. linear_index. index points to {idx} but array only has length {:?}",
                data.len()
            )));
        }

        Ok(data[idx])
    }

    fn index(&self, index: &[usize]) -> Result<ArrayView<'_, F>, PzeudoErr> {
        let metadata = self.get_metadata();
        if index.len() > metadata.shape.len() {
            return Err(PzeudoErr::Index(format!("")));
        }

        let new_shape = if index.len() != metadata.shape.len() {
            metadata.shape[index.len()..].to_vec()
        } else {
            vec![1]
        };

        let new_stride = shape_to_stride(&new_shape);
        let mut new_offset = metadata.offset;
        for (i, dim) in index.iter().enumerate() {
            new_offset += dim * metadata.stride[i];
        }

        let array = ArrayView {
            data: metadata.data,
            offset: new_offset,
            shape: new_shape,
            stride: new_stride,
        };

        Ok(array)
    }
}
