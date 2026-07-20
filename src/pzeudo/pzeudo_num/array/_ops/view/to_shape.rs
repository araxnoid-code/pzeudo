use crate::prelude::*;

pub trait OpsToShape<F>: ArrayTrait<F> {
    fn to_shape(&self, shape: &[usize]) -> Result<ArrayView<'_, F>, PzeudoErr> {
        let metadata = self.get_metadata();
        if metadata.shape.iter().product::<usize>() != shape.iter().product::<usize>() {
            return Err(PzeudoErr::ToShapeErr(format!(
                "OpsToShape::to_shape. Cannot change array with shape {:?} to {shape:?} due to unequal sizes",
                metadata.shape
            )));
        }

        Ok(ArrayView {
            data: metadata.data,
            offset: metadata.offset,
            stride: shape_to_stride(shape),
            shape: shape.to_vec(),
        })
    }

    fn reshape(&self, reshape: &[i32]) -> Result<ArrayView<'_, F>, PzeudoErr> {
        let metadata = self.get_metadata();

        let mut label_idx = None::<usize>;
        let mut product = 1;
        let mut to_shape = Vec::new();
        for (idx, dim) in reshape.iter().enumerate() {
            if *dim == -1 {
                if let Some(_) = label_idx {
                    return Err(PzeudoErr::ToShapeErr(format!(
                        "OpsToShape::reshape. in shape {reshape:?}, cannot perform size inference on more than 1 dimension at once.",
                    )));
                } else {
                    label_idx = Some(idx);
                    to_shape.push(0);
                }
            } else if *dim == 0 || *dim < -1 {
                return Err(PzeudoErr::ToShapeErr(format!(
                    "OpsToShape::reshape. cannot reshape on {reshape:?}, only allowed sizes are x > 0 or x == -1",
                )));
            } else {
                product *= *dim;
                to_shape.push(*dim as usize);
            }
        }

        if let Some(label_idx) = label_idx {
            let rest = metadata.shape.iter().product::<usize>() / product as usize;
            to_shape[label_idx] = rest;
        }

        Ok(self.to_shape(&to_shape)?)
    }
}
