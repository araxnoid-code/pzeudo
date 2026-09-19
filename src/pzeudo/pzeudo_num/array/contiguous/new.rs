use num_traits::{NumCast, One, Zero, one, zero};

use crate::{Array, PzeudoErr, SlicingRangeTrait, shape_to_stride};

impl<F> Array<F> {
    pub fn new(data: Vec<F>, offset: usize, stride: Vec<usize>, shape: Vec<usize>) -> Array<F> {
        Self {
            data,
            offset,
            stride,
            shape,
        }
    }

    pub fn from_vector(vector: Vec<F>) -> Array<F>
    where
        F: Clone,
    {
        let shape = vector.len();
        Self {
            data: vector,
            offset: 0,
            shape: vec![shape],
            stride: vec![1],
        }
    }

    pub fn from_slice(vector: &[F]) -> Array<F>
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

    pub fn from_slice_with_shape(vector: &[F], shape: &[usize]) -> Result<Array<F>, PzeudoErr>
    where
        F: Clone,
    {
        if vector.len() != shape.iter().product::<usize>() {
            return Err(PzeudoErr::ArrayErr(format!(
                "Array::from_vector_with_shape. Cannot create array because a vector of size {} cannot be stored in shape {:?}",
                vector.len(),
                shape
            )));
        }

        let stride = shape_to_stride(shape);
        Ok(Self {
            data: vector.to_vec(),
            offset: 0,
            shape: shape.to_vec(),
            stride,
        })
    }

    pub fn from_vector_with_shape(vector: Vec<F>, shape: &[usize]) -> Result<Array<F>, PzeudoErr> {
        if vector.len() != shape.iter().product::<usize>() {
            return Err(PzeudoErr::ArrayErr(format!(
                "Array::from_vector_with_shape. Cannot create array because a vector of size {} cannot be stored in shape {:?}",
                vector.len(),
                shape
            )));
        }

        let stride = shape_to_stride(shape);
        Ok(Self {
            data: vector,
            offset: 0,
            shape: shape.to_vec(),
            stride,
        })
    }

    pub fn zeros(shape: &[usize]) -> Array<F>
    where
        F: Zero + Clone,
    {
        let vector: Vec<F> = vec![zero(); shape.iter().product::<usize>()];
        let stride = shape_to_stride(shape);

        Array {
            data: vector,
            offset: 0,
            shape: shape.to_vec(),
            stride,
        }
    }

    pub fn ones(shape: &[usize]) -> Array<F>
    where
        F: One + Clone,
    {
        let vector: Vec<F> = vec![one(); shape.iter().product::<usize>()];
        let stride = shape_to_stride(shape);

        Array {
            data: vector,
            offset: 0,
            shape: shape.to_vec(),
            stride,
        }
    }

    pub fn from_range<R>(range: R, shape: &[usize]) -> Result<Array<F>, PzeudoErr>
    where
        F: NumCast,
        R: SlicingRangeTrait,
    {
        let len = shape.iter().product::<usize>();
        if let Some(end) = range.end() {
            if len != (end - range.start().unwrap_or(0)) {
                return Err(PzeudoErr::ArrayErr(format!(
                    "Array::from_range_with_shape. The len of the range is {} and is not the same as the len of the shape is {}.",
                    (end - range.start().unwrap_or(0)),
                    len
                )));
            }
        }

        let mut vec = Vec::with_capacity(len);
        let start = range.start().unwrap_or(0);
        for x in start..range.end().unwrap_or(len + start) {
            vec.push(F::from(x).ok_or(PzeudoErr::ArrayErr(String::from(
                "Array::from_range_with_shape. Cannot cast value to data type F",
            )))?);
        }
        let stride = shape_to_stride(shape);

        Ok(Array {
            data: vec,
            offset: 0,
            shape: shape.to_vec(),
            stride,
        })
    }

    pub fn from_range_fn<R>(
        range: R,
        shape: &[usize],
        f: impl Fn(F) -> F,
    ) -> Result<Array<F>, PzeudoErr>
    where
        F: NumCast,
        R: SlicingRangeTrait,
    {
        let len = shape.iter().product::<usize>();
        if let Some(end) = range.end() {
            if len != (end - range.start().unwrap_or(0)) {
                return Err(PzeudoErr::ArrayErr(format!(
                    "Array::from_range_with_shape. The len of the range is {} and is not the same as the len of the shape is {}.",
                    (end - range.start().unwrap_or(0)),
                    len
                )));
            }
        }

        let mut vec = Vec::with_capacity(len);
        for x in range.start().unwrap_or(0)..range.end().unwrap_or(len) {
            vec.push(f(F::from(x).ok_or(PzeudoErr::ArrayErr(String::from(
                "Array::from_range_with_shape. Cannot cast value to data type F",
            )))?));
        }
        let stride = shape_to_stride(shape);

        Ok(Array {
            data: vec,
            offset: 0,
            shape: shape.to_vec(),
            stride,
        })
    }
}
