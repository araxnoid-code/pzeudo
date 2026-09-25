use num_traits::Float;

use crate::prelude::*;

pub trait OpsMax<F>: ArrayTrait<F> {
    fn max(&self) -> Result<Array<F>, PzeudoErr>
    where
        F: Float,
    {
        let metadata = self.get_metadata();
        let len = metadata.shape.iter().product::<usize>();

        let mut max = None;
        for i in 0..len {
            let j = self.linear_index(i)?;
            match &mut max {
                None => max = Some(j),
                Some(max) => {
                    *max = max.max(self.linear_index(i)?);
                }
            }
        }

        Ok(Array::from_vector(vec![max.unwrap()]))
    }

    fn max_axis(&self, axis: &[usize], keep_dim: bool) -> Result<Array<F>, PzeudoErr>
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

            let mut max = None;
            for ii in 0..axis_len {
                let mut index = offset;

                for ((axis_shape, axis_stride), axis) in
                    axis_shape.iter().zip(axis_stride.iter()).zip(axis.iter())
                {
                    let permute = (ii / axis_stride) % axis_shape;

                    index += permute * metadata.stride[*axis];
                }
                let value = metadata.data[index];
                match &mut max {
                    None => max = Some(value),
                    Some(max) => *max = max.max(value),
                }
            }
            vec.push(max.unwrap());
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

    fn argmax(&self) -> Result<Array<F>, PzeudoErr>
    where
        F: Float,
    {
        let metadata = self.get_metadata();
        let len = metadata.shape.iter().product::<usize>();

        let mut max = None;
        let mut idx = None;
        for i in 0..len {
            let j = self.linear_index(i)?;
            match &mut idx {
                None => {
                    idx = Some(i);
                    max = Some(j);
                }
                Some(idx) => {
                    if max.unwrap() < j {
                        max = Some(j);
                        *idx = i;
                    }
                }
            }
        }

        Ok(Array::from_vector(vec![F::from(idx.unwrap()).ok_or(
            PzeudoErr::OpsErr(String::from("OpsArgMax::argmax_idx. Cannot cast on index")),
        )?]))
    }

    fn argmax_axis(&self, axis: &[usize], keep_dim: bool) -> Result<Array<F>, PzeudoErr>
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

            let mut max = None;
            let mut max_idx = None;
            for ii in 0..axis_len {
                let mut index = offset;

                for ((axis_shape, axis_stride), axis) in
                    axis_shape.iter().zip(axis_stride.iter()).zip(axis.iter())
                {
                    let permute = (ii / axis_stride) % axis_shape;

                    index += permute * metadata.stride[*axis];
                }
                let value = metadata.data[index];
                match &mut max {
                    None => {
                        max_idx = Some(F::from(ii).ok_or(PzeudoErr::OpsErr(String::from(
                            "OpsMax::argmax_axis. Cannot cast value to type F",
                        )))?);
                        max = Some(value);
                    }
                    Some(max) => {
                        if *max < value {
                            max_idx = Some(F::from(ii).ok_or(PzeudoErr::OpsErr(String::from(
                                "OpsMax::argmax_axis. Cannot cast value to type F",
                            )))?);
                            *max = value;
                        }
                    }
                }
            }
            vec.push(max_idx.unwrap());
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

    fn max_and_argmax(&self) -> Result<(Array<F>, Array<F>), PzeudoErr>
    where
        F: Float,
    {
        let metadata = self.get_metadata();
        let len = metadata.shape.iter().product::<usize>();

        let mut max = None;
        let mut idx = None;
        for i in 0..len {
            let j = self.linear_index(i)?;
            match &mut idx {
                None => {
                    idx = Some(i);
                    max = Some(j);
                }
                Some(idx) => {
                    if max.unwrap() < j {
                        max = Some(j);
                        *idx = i;
                    }
                }
            }
        }

        let argmax = Array::from_vector(vec![F::from(idx.unwrap()).ok_or(PzeudoErr::OpsErr(
            String::from("OpsArgMax::argmax_idx. Cannot cast on index"),
        ))?]);
        let max = Array::from_vector(vec![F::from(max.unwrap()).ok_or(PzeudoErr::OpsErr(
            String::from("OpsArgMax::argmax_idx. Cannot cast on index"),
        ))?]);

        Ok((max, argmax))
    }

    fn max_and_argmax_axis(
        &self,
        axis: &[usize],
        keep_dim: bool,
    ) -> Result<(Array<F>, Array<F>), PzeudoErr>
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
        let mut max_vec = Vec::with_capacity(len);
        let mut max_idx_vec = Vec::with_capacity(len);
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

            let mut max = None;
            let mut max_idx = None;
            for ii in 0..axis_len {
                let mut index = offset;

                for ((axis_shape, axis_stride), axis) in
                    axis_shape.iter().zip(axis_stride.iter()).zip(axis.iter())
                {
                    let permute = (ii / axis_stride) % axis_shape;

                    index += permute * metadata.stride[*axis];
                }
                let value = metadata.data[index];
                match &mut max {
                    None => {
                        max_idx = Some(F::from(ii).ok_or(PzeudoErr::OpsErr(String::from(
                            "OpsMax::argmax_axis. Cannot cast value to type F",
                        )))?);
                        max = Some(value);
                    }
                    Some(max) => {
                        if *max < value {
                            max_idx = Some(F::from(ii).ok_or(PzeudoErr::OpsErr(String::from(
                                "OpsMax::argmax_axis. Cannot cast value to type F",
                            )))?);
                            *max = value;
                        }
                    }
                }
            }
            max_vec.push(max.unwrap());
            max_idx_vec.push(max_idx.unwrap());
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

        let max_idx_array = Array {
            data: max_idx_vec.to_vec(),
            offset: 0,
            shape: out_shape.to_vec(),
            stride: out_stride.to_vec(),
        };

        let max_array = Array {
            data: max_vec,
            offset: 0,
            shape: out_shape,
            stride: out_stride,
        };

        Ok((max_array, max_idx_array))
    }

    fn max_axis_with_flatten_index(
        &self,
        axis: &[usize],
        keep_dim: bool,
    ) -> Result<(Array<F>, Vec<usize>), PzeudoErr>
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
        let mut max_vec = Vec::with_capacity(len);
        let mut max_idx_vec = Vec::with_capacity(len);
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

            let mut max = None;
            let mut max_idx = None;
            for ii in 0..axis_len {
                let mut index = offset;

                for ((axis_shape, axis_stride), axis) in
                    axis_shape.iter().zip(axis_stride.iter()).zip(axis.iter())
                {
                    let permute = (ii / axis_stride) % axis_shape;

                    index += permute * metadata.stride[*axis];
                }
                let value = metadata.data[index];
                match &mut max {
                    None => {
                        max_idx = Some(index);
                        max = Some(value);
                    }
                    Some(max) => {
                        if *max < value {
                            max_idx = Some(index);
                            *max = value;
                        }
                    }
                }
            }
            max_vec.push(max.unwrap());
            max_idx_vec.push(max_idx.unwrap());
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

        let max_array = Array {
            data: max_vec,
            offset: 0,
            shape: out_shape,
            stride: out_stride,
        };

        Ok((max_array, max_idx_vec))
    }
}
