use crate::prelude::*;

pub trait ArrayAssignTrait<F>: ArrayTrait<F> {
    fn get_mut_metadata(&mut self) -> MutMetadata<'_, F>;

    fn mut_linear_index(&mut self, index: usize) -> Result<&mut F, PzeudoErr>
    where
        F: Copy,
    {
        let MutMetadata {
            data,
            offset,
            stride,
            shape,
        } = self.get_mut_metadata();

        let output_stride = shape_to_stride(shape);

        let mut idx = offset;
        for (shape, (stride, out_stride)) in
            shape.iter().zip(stride.iter().zip(output_stride.iter()))
        {
            let permute = (index / out_stride) % shape;
            idx += permute * stride;
        }

        if idx >= data.len() {
            return Err(PzeudoErr::MutLinearIndexErr(format!(
                "ArrayAssignTrait::mut_linear_index. index points to {idx} but array only has length {:?}",
                data.len()
            )));
        }

        Ok(&mut data[idx])
    }
}

impl<F, T> ArrayAssignTrait<F> for ArrayRefMut<'_, F, T>
where
    F: Copy,
    for<'a> ArrayRefMut<'a, F, T>: ArrayTrait<F>,
{
    fn get_mut_metadata(&mut self) -> MutMetadata<'_, F> {
        MutMetadata {
            data: self.data,
            offset: self.offset,
            stride: self.stride,
            shape: self.shape,
        }
    }
}

impl<F> ArrayAssignTrait<F> for Array<F>
where
    F: Copy,
{
    fn get_mut_metadata(&mut self) -> MutMetadata<'_, F> {
        MutMetadata {
            data: &mut self.data,
            offset: self.offset,
            stride: &self.stride,
            shape: &self.shape,
        }
    }

    fn mut_linear_index(&mut self, index: usize) -> Result<&mut F, PzeudoErr>
    where
        F: Copy,
    {
        Ok(&mut self.data[index])
    }
}
