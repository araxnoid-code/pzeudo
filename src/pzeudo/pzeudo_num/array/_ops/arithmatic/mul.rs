use std::ops::{Div, Mul};

use crate::{
    Array, ArrayTrait,
    PzeudoNumErr::{self, DivErr, MulErr},
    shape_to_stride,
};

pub trait OpsMul<F>: ArrayTrait<F> {
    fn mul<Rhs>(&self, rhs: &Rhs) -> Result<Array<F>, PzeudoNumErr>
    where
        F: Copy + Mul<Output = F>,
        Rhs: ArrayTrait<F>,
    {
        let lhs_metadata = self.get_metadata();
        let rhs_metadata = rhs.get_metadata();

        if lhs_metadata.shape != rhs_metadata.shape {
            return Err(MulErr(format!(
                "MulErr. mul. cannot multiply arrays of the form {:?} and {:?} because they are different in shape",
                lhs_metadata.shape, rhs_metadata.shape
            )));
        }

        let len = lhs_metadata.shape.iter().product::<usize>();
        let mut output = Vec::with_capacity(len);
        for i in 0..len {
            let lhs_value = self.linear_index(i)?;
            let rhs_value = rhs.linear_index(i)?;
            output.push(lhs_value * rhs_value);
        }

        let shape = lhs_metadata.shape.to_vec();
        let array = Array::new(output, 0, shape_to_stride(&shape), shape);

        Ok(array)
    }
}
