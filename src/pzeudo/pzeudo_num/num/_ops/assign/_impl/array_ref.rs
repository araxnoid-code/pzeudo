use crate::prelude::*;

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

    fn linear_index_mut(&mut self, index: usize) -> Result<&mut F, PzeudoErr> {
        if self.contiguous {
            Ok(&mut self.data[index])
        } else {
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
                return Err(PzeudoErr::OpsErr(format!(
                    "ArrayAssignTrait::mut_linear_index. index points to {idx} but array only has length {:?}",
                    data.len()
                )));
            }

            Ok(&mut data[idx])
        }
    }
}

impl<F, T> OpsAssign<F> for ArrayRefMut<'_, F, T>
where
    F: Copy,
    for<'a> ArrayRefMut<'a, F, T>: ArrayAssignTrait<F>,
{
}

impl<F, T> OpsAddAssign<F> for ArrayRefMut<'_, F, T>
where
    F: Copy,
    for<'a> ArrayRefMut<'a, F, T>: ArrayAssignTrait<F>,
{
}

impl<F, T> OpsDivAssign<F> for ArrayRefMut<'_, F, T>
where
    F: Copy,
    for<'a> ArrayRefMut<'a, F, T>: ArrayAssignTrait<F>,
{
}

impl<F, T> OpsMulAssign<F> for ArrayRefMut<'_, F, T>
where
    F: Copy,
    for<'a> ArrayRefMut<'a, F, T>: ArrayAssignTrait<F>,
{
}

impl<F, T> OpsSubAssign<F> for ArrayRefMut<'_, F, T>
where
    F: Copy,
    for<'a> ArrayRefMut<'a, F, T>: ArrayAssignTrait<F>,
{
}
