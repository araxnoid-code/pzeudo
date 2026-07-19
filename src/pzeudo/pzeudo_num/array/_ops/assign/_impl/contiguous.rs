use crate::prelude::*;

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

impl<F> OpsAssign<F> for Array<F> where F: Copy {}
impl<F> OpsAddAssign<F> for Array<F> where F: Copy {}
impl<F> OpsDivAssign<F> for Array<F> where F: Copy {}
impl<F> OpsMulAssign<F> for Array<F> where F: Copy {}
impl<F> OpsSubAssign<F> for Array<F> where F: Copy {}
