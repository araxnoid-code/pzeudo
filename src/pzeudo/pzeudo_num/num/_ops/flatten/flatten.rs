use crate::prelude::*;

pub trait OpsFlatten<F>: ArrayTrait<F> {
    fn flatten(&self) -> Result<Array<F>, PzeudoErr>
    where
        F: Copy,
    {
        let data = self.get_metadata();

        let len = data.shape.iter().product::<usize>();
        let mut vec = Vec::with_capacity(len);
        for i in 0..len {
            let value = self.linear_index(i)?;
            vec.push(value);
        }

        let array = Array {
            data: vec,
            offset: 0,
            shape: vec![len],
            stride: vec![1],
        };

        Ok(array)
    }

    fn into_flatten(self) -> Result<Array<F>, PzeudoErr>
    where
        F: Copy,
        Self: Sized,
    {
        let data = self.get_metadata();

        let len = data.shape.iter().product::<usize>();
        let mut vec = Vec::with_capacity(len);
        for i in 0..len {
            let value = self.linear_index(i)?;
            vec.push(value);
        }

        let array = Array {
            data: vec,
            offset: 0,
            shape: vec![len],
            stride: vec![1],
        };

        Ok(array)
    }
}
