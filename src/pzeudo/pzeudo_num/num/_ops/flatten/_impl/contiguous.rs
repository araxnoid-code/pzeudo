use crate::prelude::*;

impl<F> OpsFlatten<F> for Array<F>
where
    F: Copy,
{
    fn flatten(&self) -> Result<Array<F>, PzeudoErr>
    where
        F: Copy,
    {
        let len = self.shape.iter().product::<usize>();
        let vec = self.data.clone();

        Ok(Array {
            data: vec,
            offset: 0,
            shape: vec![len],
            stride: vec![1],
        })
    }

    fn into_flatten(self) -> Result<Array<F>, PzeudoErr>
    where
        F: Copy,
        Self: Sized,
    {
        let len = self.shape.iter().product::<usize>();
        Ok(Array {
            data: self.data,
            offset: 0,
            stride: vec![1],
            shape: vec![len],
        })
    }
}
