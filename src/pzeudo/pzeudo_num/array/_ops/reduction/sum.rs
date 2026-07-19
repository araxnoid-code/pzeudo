use crate::prelude::*;
use num_traits::{Zero, zero};
use std::{ops::AddAssign, vec};

pub trait OpsSum<F>: ArrayTrait<F> {
    fn sum(&self) -> Result<Array<F>, PzeudoErr>
    where
        F: Copy + AddAssign + Zero,
    {
        let shape = self.get_metadata().shape.iter().product::<usize>();
        let mut sum = zero::<F>();
        for i in 0..shape {
            sum += self.linear_index(i)?;
        }

        let array = Array {
            data: vec![sum],
            offset: 0,
            shape: vec![1],
            stride: vec![1],
        };

        Ok(array)
    }

    fn sum_axis(&self, axis: &[usize], keep_dim: bool) -> Result<Array<F>, PzeudoErr>
    where
        F: AddAssign + Zero + Copy,
    {
        let metadata = self.get_metadata();

        if axis.len() <= 0 {
            return Err(PzeudoErr::SumAxisErr(format!(
                "SumAxisErr. OpsSum::sum_axis. Cannot perform sum_axis because axis is empty.",
            )));
        } else if axis.len() > metadata.shape.len() {
            return Err(PzeudoErr::SumAxisErr(format!(
                "SumAxisErr. OpsSum::sum_axis. Unable to perform sum_axis because axis {:?} is out of bounds.",
                axis
            )));
        }

        let mut axis_shape = Vec::new();
        let mut out_shape = metadata.shape.to_vec();
        let mut check = vec![false; metadata.shape.len()];
        let mut prev = None;
        for dim in axis {
            if *dim >= metadata.shape.len() {
                return Err(PzeudoErr::SumAxisErr(format!(
                    "SumAxisErr. OpsSum::sum_axis. Cannot perform sum_axis because axis {:?}, number {} is out of array dimension bounds.",
                    axis, dim,
                )));
            } else if check[*dim] {
                return Err(PzeudoErr::SumAxisErr(format!(
                    "SumAxisErr. OpsSum::sum_axis. Cannot perform sum_axis because there is a repeating number on axis {:?}.",
                    axis,
                )));
            }

            if let Some(prev) = &mut prev {
                if *prev > *dim {
                    return Err(PzeudoErr::SumAxisErr(format!(
                        "SumAxisErr. OpsSum::sum_axis. Cannot do sum_axis because axis {:?} must be arranged in order from smallest to largest.",
                        axis,
                    )));
                }

                *prev = *dim;
            } else {
                prev = Some(*dim);
            }
            check[*dim] = true;

            axis_shape.push(out_shape[*dim]);
            out_shape[*dim] = 1;
        }

        let axis_stride = shape_to_stride(&axis_shape);
        let mut out_stride = shape_to_stride(&out_shape);

        let len = out_shape.iter().product::<usize>();
        let axis_len = axis_shape.iter().product::<usize>();
        let mut vec = Vec::with_capacity(len);
        for i in 0..len {
            let mut offset = metadata.offset;
            for ((out_shape, out_stride), meta_stride) in out_shape
                .iter()
                .zip(out_stride.iter())
                .zip(metadata.stride.iter())
            {
                let permute = (i / out_stride) % out_shape;
                offset += permute * meta_stride;
            }

            let mut sum = zero::<F>();
            for ii in 0..axis_len {
                let mut index = offset;

                for ((axis_shape, axis_stride), axis) in
                    axis_shape.iter().zip(axis_stride.iter()).zip(axis.iter())
                {
                    let permute = (ii / axis_stride) % axis_shape;

                    index += permute * metadata.stride[*axis];
                }
                sum += metadata.data[index];
            }
            vec.push(sum);
        }

        if !keep_dim {
            if axis.len() == out_shape.len() {
                out_shape = vec![1];
                out_stride = vec![1];
            } else {
                for dim in axis.iter().rev() {
                    out_shape.remove(*dim);
                }
                out_stride = shape_to_stride(&out_shape);
            }
        }

        let array = Array {
            data: vec,
            offset: 0,
            shape: out_shape,
            stride: out_stride,
        };

        Ok(array)
    }
}
