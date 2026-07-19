use crate::{Array, ArrayTrait, ArrayView, PzeudoErr, shape_to_stride};

impl<'a, F> ArrayView<'a, F> {
    pub fn new(
        data: &'a [F],
        offset: usize,
        shape: Vec<usize>,
        stride: Vec<usize>,
    ) -> ArrayView<'a, F> {
        Self {
            data,
            offset,
            shape,
            stride,
        }
    }

    pub fn into_array(self) -> Result<Array<F>, PzeudoErr>
    where
        F: Clone + Copy,
    {
        let metadata = self.get_metadata();
        let len = metadata.shape.iter().product::<usize>();
        let mut vec = Vec::with_capacity(len);
        for i in 0..len {
            vec.push(self.linear_index(i)?);
        }

        Ok(Array {
            data: vec,
            offset: 0,
            stride: shape_to_stride(&self.shape),
            shape: self.shape,
        })
    }
}
