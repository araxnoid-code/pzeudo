use crate::{Array, ArrayTrait, Metadata, PzeudoErr};

impl<F> ArrayTrait<F> for Array<F>
where
    F: Copy,
{
    fn get_metadata(&self) -> crate::Metadata<'_, F> {
        Metadata {
            data: &self.data,
            offset: self.offset,
            shape: &self.shape,
            stride: &self.stride,
        }
    }

    fn linear_index(&self, index: usize) -> Result<F, crate::PzeudoErr> {
        if index >= self.data.len() {
            return Err(PzeudoErr::LinearIndexErr(
                "LinearIndex. linear_index\nindex points outside the size of the data".to_string(),
            ));
        }

        Ok(self.data[index])
    }
}

impl<F> ArrayTrait<F> for &Array<F>
where
    F: Copy,
{
    fn get_metadata(&self) -> crate::Metadata<'_, F> {
        Metadata {
            data: &self.data,
            offset: self.offset,
            shape: &self.shape,
            stride: &self.stride,
        }
    }

    fn linear_index(&self, index: usize) -> Result<F, crate::PzeudoErr> {
        if index >= self.data.len() {
            return Err(PzeudoErr::LinearIndexErr(
                "LinearIndex. linear_index\nindex points outside the size of the data".to_string(),
            ));
        }

        Ok(self.data[index])
    }
}
