use num_traits::Float;

use crate::prelude::*;

pub trait OpsMin<F>: ArrayTrait<F> {
    fn min(&self) -> Result<Array<F>, PzeudoErr>
    where
        F: Float,
    {
        let metadata = self.get_metadata();
        let len = metadata.shape.iter().product::<usize>();

        let mut min = None;
        for i in 0..len {
            let j = self.linear_index(i)?;
            match &mut min {
                None => min = Some(j),
                Some(min) => {
                    *min = min.min(self.linear_index(i)?);
                }
            }
        }

        Ok(Array::from_vector(vec![min.unwrap()]))
    }

    fn min_axis(&self, axis: &[usize], keep_dim: bool) -> Result<Array<F>, PzeudoErr>
    where
        F: Float,
    {
        let metadata = self.get_metadata();

        if axis.len() <= 0 {
            return Err(PzeudoErr::OpsErr(format!(
                "OpsMax::max_axis. Cannot perform max_axis because axis is empty.",
            )));
        } else if axis.len() > metadata.shape.len() {
            return Err(PzeudoErr::OpsErr(format!(
                "OpsMax::max_axis. Unable to perform max_axis because axis {:?} is out of bounds.",
                axis
            )));
        }

        let mut axis_shape = Vec::new();
        let mut out_shape = metadata.shape.to_vec();
        let mut check = vec![false; metadata.shape.len()];
        let mut prev = None;
        for dim in axis {
            if *dim >= metadata.shape.len() {
                return Err(PzeudoErr::OpsErr(format!(
                    "OpsMax::max_axis. Cannot perform max_axis because axis {:?}, number {} is out of array dimension bounds.",
                    axis, dim,
                )));
            } else if check[*dim] {
                return Err(PzeudoErr::OpsErr(format!(
                    "OpsMax::max_axis. Cannot perform max_axis because there is a repeating number on axis {:?}.",
                    axis,
                )));
            }

            if let Some(prev) = &mut prev {
                if *prev > *dim {
                    return Err(PzeudoErr::OpsErr(format!(
                        "OpsMax::max_axis. Cannot do max_axis because axis {:?} must be arranged in order from smallest to largest.",
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

            let mut min = None;
            for ii in 0..axis_len {
                let mut index = offset;

                for ((axis_shape, axis_stride), axis) in
                    axis_shape.iter().zip(axis_stride.iter()).zip(axis.iter())
                {
                    let permute = (ii / axis_stride) % axis_shape;

                    index += permute * metadata.stride[*axis];
                }
                let value = metadata.data[index];
                match &mut min {
                    None => min = Some(value),
                    Some(min) => *min = min.min(value),
                }
            }
            vec.push(min.unwrap());
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

    fn argmin(&self) -> Result<Array<F>, PzeudoErr>
    where
        F: Float,
    {
        let metadata = self.get_metadata();
        let len = metadata.shape.iter().product::<usize>();

        let mut min = None;
        let mut idx = None;
        for i in 0..len {
            let j = self.linear_index(i)?;
            match &mut idx {
                None => {
                    idx = Some(i);
                    min = Some(j);
                }
                Some(idx) => {
                    if min.unwrap() > j {
                        min = Some(j);
                        *idx = i;
                    }
                }
            }
        }

        Ok(Array::from_vector(vec![F::from(idx.unwrap()).ok_or(
            PzeudoErr::OpsErr(String::from("OpsArgMax::argmax_idx. Cannot cast on index")),
        )?]))
    }

    fn argmin_axis(&self, axis: &[usize], keep_dim: bool) -> Result<Array<F>, PzeudoErr>
    where
        F: Float,
    {
        let metadata = self.get_metadata();

        if axis.len() <= 0 {
            return Err(PzeudoErr::OpsErr(format!(
                "OpsMax::argmax_axis. Cannot perform argmax_axis because axis is empty.",
            )));
        } else if axis.len() > metadata.shape.len() {
            return Err(PzeudoErr::OpsErr(format!(
                "OpsMax::argmax_axis. Unable to perform argmax_axis because axis {:?} is out of bounds.",
                axis
            )));
        }

        let mut axis_shape = Vec::new();
        let mut out_shape = metadata.shape.to_vec();
        let mut check = vec![false; metadata.shape.len()];
        let mut prev = None;
        for dim in axis {
            if *dim >= metadata.shape.len() {
                return Err(PzeudoErr::OpsErr(format!(
                    "OpsMax::argmax_axis. Cannot perform argmax_axis because axis {:?}, number {} is out of array dimension bounds.",
                    axis, dim,
                )));
            } else if check[*dim] {
                return Err(PzeudoErr::OpsErr(format!(
                    "OpsMax::argmax_axis. Cannot perform argmax_axis because there is a repeating number on axis {:?}.",
                    axis,
                )));
            }

            if let Some(prev) = &mut prev {
                if *prev > *dim {
                    return Err(PzeudoErr::OpsErr(format!(
                        "OpsMax::argmax_axis. Cannot do argmax_axis because axis {:?} must be arranged in order from smallest to largest.",
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

            let mut min = None;
            let mut min_idx = None;
            for ii in 0..axis_len {
                let mut index = offset;

                for ((axis_shape, axis_stride), axis) in
                    axis_shape.iter().zip(axis_stride.iter()).zip(axis.iter())
                {
                    let permute = (ii / axis_stride) % axis_shape;

                    index += permute * metadata.stride[*axis];
                }
                let value = metadata.data[index];
                match &mut min {
                    None => {
                        min_idx = Some(F::from(ii).ok_or(PzeudoErr::OpsErr(String::from(
                            "OpsMax::argmax_axis. Cannot cast value to type F",
                        )))?);
                        min = Some(value);
                    }
                    Some(min) => {
                        if *min > value {
                            min_idx = Some(F::from(ii).ok_or(PzeudoErr::OpsErr(String::from(
                                "OpsMax::argmax_axis. Cannot cast value to type F",
                            )))?);
                            *min = value;
                        }
                    }
                }
            }
            vec.push(min_idx.unwrap());
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
