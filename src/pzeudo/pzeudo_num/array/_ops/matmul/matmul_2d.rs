use std::vec;

use crate::prelude::*;

pub trait OpsMatmul2DF32: ArrayTrait<f32> {
    fn matmul_2d<Rhs>(&self, rhs: &Rhs) -> Result<Array<f32>, PzeudoErr>
    where
        Rhs: ArrayTrait<f32>,
    {
        let lhs_meta = self.get_metadata();
        let rhs_meta = rhs.get_metadata();

        if lhs_meta.shape.len() != 2 || rhs_meta.shape.len() != 2 {
            return Err(PzeudoErr::Matmul2DErr(format!(
                "Matmul2DErr. OpsMatmul2Df32::matmul_2d. cannot perform matmul_2d on lhs which has shape {:?} and rhs has shape {:?}. dimension must be 2-dimensional.",
                lhs_meta.shape, rhs_meta.shape
            )));
        }

        if lhs_meta.shape[1] != rhs_meta.shape[0] {
            return Err(PzeudoErr::Matmul2DErr(format!(
                "Matmul2DErr. OpsMatmul2Df32. cannot perform matmul_2d on lhs which has shape {:?} and rhs has shape {:?}. does not satisfy the form 'mxk * kxn'.",
                lhs_meta.shape, rhs_meta.shape
            )));
        }

        let m = lhs_meta.shape[0];
        let k = lhs_meta.shape[1];
        let n = rhs_meta.shape[1];

        let output_shape = vec![m, n];
        let output_stride = shape_to_stride(&output_shape);
        let mut result = vec![0.; output_shape.iter().product::<usize>()];

        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.,
                lhs_meta.data.as_ptr().add(lhs_meta.offset),
                lhs_meta.stride[0] as isize,
                lhs_meta.stride[1] as isize,
                rhs_meta.data.as_ptr().add(rhs_meta.offset),
                rhs_meta.stride[0] as isize,
                rhs_meta.stride[1] as isize,
                0.,
                result.as_mut_ptr(),
                output_stride[0] as isize,
                output_stride[1] as isize,
            );
        }

        let array = Array::new(result, 0, output_stride, output_shape);

        Ok(array)
    }
}

pub trait OpsMatmul2DF64: ArrayTrait<f64> {
    fn matmul_2d<Rhs>(&self, rhs: &Rhs) -> Result<Array<f64>, PzeudoErr>
    where
        Rhs: ArrayTrait<f64>,
    {
        let lhs_meta = self.get_metadata();
        let rhs_meta = rhs.get_metadata();

        if lhs_meta.shape.len() != 2 || rhs_meta.shape.len() != 2 {
            return Err(PzeudoErr::Matmul2DErr(format!(
                "Matmul2DErr. OpsMatmul2Df32::matmul_2d. cannot perform matmul_2d on lhs which has shape {:?} and rhs has shape {:?}. dimension must be 2-dimensional.",
                lhs_meta.shape, rhs_meta.shape
            )));
        }

        if lhs_meta.shape[1] != rhs_meta.shape[0] {
            return Err(PzeudoErr::Matmul2DErr(format!(
                "Matmul2DErr. OpsMatmul2Df32. cannot perform matmul_2d on lhs which has shape {:?} and rhs has shape {:?}. does not satisfy the form 'm×k * k×n'.",
                lhs_meta.shape, rhs_meta.shape
            )));
        }

        let m = lhs_meta.shape[0];
        let k = lhs_meta.shape[1];
        let n = rhs_meta.shape[1];

        let output_shape = vec![m, n];
        let output_stride = shape_to_stride(&output_shape);
        let mut result = vec![0.; output_shape.iter().product::<usize>()];

        unsafe {
            matrixmultiply::dgemm(
                m,
                k,
                n,
                1.,
                lhs_meta.data.as_ptr().add(lhs_meta.offset),
                lhs_meta.stride[0] as isize,
                lhs_meta.stride[1] as isize,
                rhs_meta.data.as_ptr().add(rhs_meta.offset),
                rhs_meta.stride[0] as isize,
                rhs_meta.stride[1] as isize,
                0.,
                result.as_mut_ptr(),
                output_stride[0] as isize,
                output_stride[1] as isize,
            );
        }

        let array = Array::new(result, 0, output_stride, output_shape);

        Ok(array)
    }
}
