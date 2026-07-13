use crate::{
    ArrayTrait, ArrayView,
    PzeudoNumErr::{self, PermuteErr},
};

pub trait OpsPermute<F>: ArrayTrait<F> {
    fn permute(&self, permute: &[usize]) -> Result<ArrayView<'_, F>, PzeudoNumErr> {
        let metadata = self.get_metadata();

        if permute.len() == 0 {
            return Err(PermuteErr(format!(
                "PermuteErr. OpsPermute::permute. permute cannot be empty",
            )));
        }

        if metadata.shape.len() != permute.len() {
            return Err(PermuteErr(format!(
                "PermuteErr. OpsPermute::permute. array has shape {:?} cannot do permutation with {:?} because the dimensions are not the same",
                metadata.shape, permute
            )));
        }

        let len = metadata.shape.len();
        let mut new_stride = Vec::new();
        let mut new_shape = Vec::new();
        let mut check = vec![false; len];

        for dim in permute {
            if *dim >= len {
                return Err(PermuteErr(format!(
                    "PermuteErr. OpsPermute::permute. in permute {:?}, there is an index that is out of bounds",
                    permute
                )));
            } else if check[*dim] {
                return Err(PermuteErr(format!(
                    "PermuteErr. OpsPermute::permute. on permute {:?}, a repeated number is detected",
                    permute
                )));
            }

            new_stride.push(metadata.stride[*dim]);
            new_shape.push(metadata.shape[*dim]);
            check[*dim] = true;
        }

        let array = ArrayView {
            data: metadata.data,
            offset: metadata.offset,
            shape: new_shape,
            stride: new_stride,
        };

        Ok(array)
    }
}
