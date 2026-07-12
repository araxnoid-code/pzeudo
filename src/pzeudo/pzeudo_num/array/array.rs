use std::ops::Add;

use crate::{Array, ArrayView, PzeudoNumErr, add, shape_to_stride};

pub struct Metadata<'a, F> {
    pub(crate) data: &'a [F],
    pub(crate) offset: usize,
    pub(crate) stride: &'a [usize],
    pub(crate) shape: &'a [usize],
}

pub trait ArrayTrait<F> {
    fn get_metadata(&self) -> Metadata<'_, F>;

    // Method
    fn linear_index(&self, index: usize) -> Result<F, PzeudoNumErr>
    where
        F: Copy,
    {
        let Metadata {
            data,
            offset,
            stride,
            shape,
        } = self.get_metadata();

        let mut idx = offset;
        for (shape, stride) in shape.iter().zip(stride.iter()) {
            let permute = (index / stride) % shape;
            idx += permute * stride;
        }

        if idx >= data.len() {
            return Err(PzeudoNumErr::LinearIndexErr(format!(
                "LinearIndexErr. linear_index. index points to {idx} but array only has length {:?}",
                data.len()
            )));
        }

        Ok(data[idx])
    }

    fn index(&self, index: &[usize]) -> Result<ArrayView<'_, F>, PzeudoNumErr> {
        let metadata = self.get_metadata();
        if index.len() > metadata.shape.len() {
            return Err(PzeudoNumErr::Index(format!("")));
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

    fn add<Rhs>(&self, rhs: &Rhs) -> Result<Array<F>, PzeudoNumErr>
    where
        Self: Sized,
        F: Copy,
        F: Add<Output = F>,
        Rhs: ArrayTrait<F>,
    {
        add(self, rhs)
    }
}
