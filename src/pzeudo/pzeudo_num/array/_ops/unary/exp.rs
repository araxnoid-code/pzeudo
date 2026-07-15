use num_traits::Float;

use crate::{Array, ArrayTrait, PzeudoNumErr};

trait OpsExp<F>: ArrayTrait<F> {
    fn exp(&self) -> Result<Array<F>, PzeudoNumErr>
    where
        F: Float + Copy,
    {
        let metadata = self.get_metadata();

        let len = metadata.shape.iter().product::<usize>();
        let mut vec = Vec::with_capacity(len);
        for idx in 0..len {
            let value = self.linear_index(idx)?.exp();
            vec.push(value);
        }

        let array = Array {
            data: vec,
            offset: 0,
            shape: metadata.shape.to_vec(),
            stride: metadata.stride.to_vec(),
        };

        Ok(array)
    }
}
