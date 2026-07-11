use crate::{Array, PzeudoNumErr, shape_to_stride};

impl<F> Array<F> {
    fn new(data: Vec<F>, offset: usize, stride: Vec<usize>, shape: Vec<usize>) -> Array<F> {
        Self {
            data,
            offset,
            stride,
            shape,
        }
    }

    pub fn from_vector(vector: &[F]) -> Array<F>
    where
        F: Clone,
    {
        let shape = vector.len();
        Self {
            data: vector.to_vec(),
            offset: 0,
            shape: vec![shape],
            stride: vec![1],
        }
    }

    pub fn from_vector_with_shape(vector: &[F], shape: &[usize]) -> Result<Array<F>, PzeudoNumErr>
    where
        F: Clone,
    {
        if vector.len() != shape.iter().product::<usize>() {
            return Err(PzeudoNumErr::ArrayNewErr(
                "ArrayNewErr. from_vector_with_shape\ncannot create array because shape size and vector length are different",
            ));
        }

        let stride = shape_to_stride(shape);
        Ok(Self {
            data: vector.to_vec(),
            offset: 0,
            shape: shape.to_vec(),
            stride,
        })
    }
}
