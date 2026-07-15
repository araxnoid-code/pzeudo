use std::ops::{Div, Mul, Sub};

use crate::{
    Array, ArrayTrait,
    PzeudoNumErr::{self, DivErr, MulErr, SubErr},
    shape_to_stride,
};

pub trait OpsSub<F>: ArrayTrait<F> {
    fn sub<Rhs>(&self, rhs: &Rhs) -> Result<Array<F>, PzeudoNumErr>
    where
        F: Copy + Sub<Output = F>,
        Rhs: ArrayTrait<F>,
    {
        let lhs_metadata = self.get_metadata();
        let rhs_metadata = rhs.get_metadata();

        if lhs_metadata.shape != rhs_metadata.shape {
            return Err(SubErr(format!(
                "SubErr. sub. cannot reduce arrays of the form {:?} and {:?} because they are of different shapes",
                lhs_metadata.shape, rhs_metadata.shape
            )));
        }

        let len = lhs_metadata.shape.iter().product::<usize>();
        let mut output = Vec::with_capacity(len);
        for i in 0..len {
            let lhs_value = self.linear_index(i)?;
            let rhs_value = rhs.linear_index(i)?;
            output.push(lhs_value - rhs_value);
        }

        let shape = lhs_metadata.shape.to_vec();
        let array = Array::new(output, 0, shape_to_stride(&shape), shape);

        Ok(array)
    }
}
