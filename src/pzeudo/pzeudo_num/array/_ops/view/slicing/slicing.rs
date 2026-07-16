use crate::{ArrayTrait, ArrayView, PzeudoErr, SliceRange};

pub trait OpsSlicing<F>: ArrayTrait<F> {
    fn slicing(&self, range: &[SliceRange]) -> Result<ArrayView<'_, F>, PzeudoErr> {
        let arr_metadata = self.get_metadata();

        if arr_metadata.shape.len() < range.len() {
            return Err(PzeudoErr::SlicingErr(format!(
                "Slicing. OpsSlicing::slicing, An array with shape {:?} cannot perform slicing {:?} because the slicing is out of bounds",
                arr_metadata.shape, range
            )));
        }

        let mut new_shape = vec![];
        let mut new_offset = arr_metadata.offset;
        for (idx, range) in range.iter().enumerate() {
            let start = range.start.unwrap_or(0);
            let end = range.end.unwrap_or(arr_metadata.shape[idx]);

            if start >= end {
                return Err(PzeudoErr::SlicingErr(format!(
                    "Slicing. OpsSlicing::slicing, An array with shape {:?} cannot perform slicing {:?} because it has start >= end",
                    arr_metadata.shape, range
                )));
            }

            new_offset += start * arr_metadata.stride[idx];
            new_shape.push(end - start);
        }

        new_shape.extend_from_slice(&arr_metadata.shape[range.len()..]);

        let array = ArrayView {
            data: arr_metadata.data,
            offset: new_offset,
            stride: arr_metadata.stride.to_vec(),
            shape: new_shape,
        };

        Ok(array)
    }
}
