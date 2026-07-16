use std::ops::{AddAssign, Mul};

use num_traits::{Zero, zero};

use crate::{Array, ArrayTrait, PzeudoErr};

pub trait OpsDotProduct<F>: ArrayTrait<F> {
    fn dot<Rhs>(&self, rhs: &Rhs) -> Result<Array<F>, PzeudoErr>
    where
        F: Copy + Zero + Mul<Output = F> + AddAssign,
        Rhs: ArrayTrait<F>,
    {
        let lhs_metadata = self.get_metadata();
        let rhs_metadata = rhs.get_metadata();

        if lhs_metadata.shape.len() != 1 || rhs_metadata.shape.len() != 1 {
            return Err(PzeudoErr::DotProductErr(format!(
                "DotProductErr. OpsDotProduct::dot. cannot do dot product on arrays with shape {:?} and arrays with shape {:?} because they are not 1-dimensional.",
                lhs_metadata.shape, rhs_metadata.shape
            )));
        }

        let len = lhs_metadata.shape.iter().product::<usize>();
        if len != rhs_metadata.shape.iter().product::<usize>() {
            return Err(PzeudoErr::DotProductErr(format!(
                "DotProductErr. OpsDotProduct::dot. cannot do dot product on an array with shape {:?} and an array with shape {:?} because they do not have the same length.",
                lhs_metadata.shape, rhs_metadata.shape
            )));
        }

        let mut sum = zero::<F>();
        for idx in 0..len {
            let lhs_value = self.linear_index(idx)?;
            let rhs_value = rhs.linear_index(idx)?;
            sum += lhs_value * rhs_value;
        }

        Ok(Array {
            data: vec![sum],
            offset: 0,
            shape: vec![1],
            stride: vec![1],
        })
    }
}

pub trait OpsDotProductF32: ArrayTrait<f32> {
    fn dot_f32<Rhs>(&self, rhs: &Rhs) -> Result<Array<f32>, PzeudoErr>
    where
        Rhs: ArrayTrait<f32>,
    {
        let lhs_metadata = self.get_metadata();
        let rhs_metadata = rhs.get_metadata();

        if lhs_metadata.shape.len() != 1 || rhs_metadata.shape.len() != 1 {
            return Err(PzeudoErr::DotProductErr(format!(
                "DotProductErr. OpsDotProduct::dot. cannot do dot product on arrays with shape {:?} and arrays with shape {:?} because they are not 1-dimensional.",
                lhs_metadata.shape, rhs_metadata.shape
            )));
        }

        let len = lhs_metadata.shape.iter().product::<usize>();
        if len != rhs_metadata.shape.iter().product::<usize>() {
            return Err(PzeudoErr::DotProductErr(format!(
                "DotProductErr. OpsDotProduct::dot. cannot do dot product on an array with shape {:?} and an array with shape {:?} because they do not have the same length.",
                lhs_metadata.shape, rhs_metadata.shape
            )));
        }

        let mut sum = vec![0.];
        unsafe {
            matrixmultiply::sgemm(
                1,
                len,
                1,
                1.,
                lhs_metadata.data.as_ptr().add(lhs_metadata.offset),
                lhs_metadata.stride[0] as isize,
                1,
                rhs_metadata.data.as_ptr().add(rhs_metadata.offset),
                1,
                1,
                0.,
                sum.as_mut_ptr(),
                1,
                1,
            );
        }

        Ok(Array {
            data: sum,
            offset: 0,
            shape: vec![1],
            stride: vec![1],
        })
    }
}

pub trait OpsDotProductF64: ArrayTrait<f64> {
    fn dot_f64<Rhs>(&self, rhs: &Rhs) -> Result<Array<f64>, PzeudoErr>
    where
        Rhs: ArrayTrait<f64>,
    {
        let lhs_metadata = self.get_metadata();
        let rhs_metadata = rhs.get_metadata();

        if lhs_metadata.shape.len() != 1 || rhs_metadata.shape.len() != 1 {
            return Err(PzeudoErr::DotProductErr(format!(
                "DotProductErr. OpsDotProduct::dot. cannot do dot product on arrays with shape {:?} and arrays with shape {:?} because they are not 1-dimensional.",
                lhs_metadata.shape, rhs_metadata.shape
            )));
        }

        let len = lhs_metadata.shape.iter().product::<usize>();
        if len != rhs_metadata.shape.iter().product::<usize>() {
            return Err(PzeudoErr::DotProductErr(format!(
                "DotProductErr. OpsDotProduct::dot. cannot do dot product on an array with shape {:?} and an array with shape {:?} because they do not have the same length.",
                lhs_metadata.shape, rhs_metadata.shape
            )));
        }

        let mut sum = vec![0.];
        unsafe {
            matrixmultiply::dgemm(
                1,
                len,
                1,
                1.,
                lhs_metadata.data.as_ptr().add(lhs_metadata.offset),
                lhs_metadata.stride[0] as isize,
                1,
                rhs_metadata.data.as_ptr().add(rhs_metadata.offset),
                1,
                1,
                0.,
                sum.as_mut_ptr(),
                1,
                1,
            );
        }

        Ok(Array {
            data: sum,
            offset: 0,
            shape: vec![1],
            stride: vec![1],
        })
    }
}
