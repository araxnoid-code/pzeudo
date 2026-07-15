use std::ops::Div;

use crate::{
    Array, ArrayTrait,
    PzeudoNumErr::{self, DivErr},
    shape_to_stride,
};

pub trait OpsDiv<F>: ArrayTrait<F> {
    fn div<Rhs>(&self, rhs: &Rhs) -> Result<Array<F>, PzeudoNumErr>
    where
        F: Copy + Div<Output = F>,
        Rhs: ArrayTrait<F>,
    {
        let lhs_metadata = self.get_metadata();
        let rhs_metadata = rhs.get_metadata();

        if lhs_metadata.shape != rhs_metadata.shape {
            return Err(DivErr(format!(
                "DivErr. div. cannot divide arrays of the form {:?} and {:?} because the forms are different",
                lhs_metadata.shape, rhs_metadata.shape
            )));
        }

        let len = lhs_metadata.shape.iter().product::<usize>();
        let mut output = Vec::with_capacity(len);
        for i in 0..len {
            let lhs_value = self.linear_index(i)?;
            let rhs_value = rhs.linear_index(i)?;
            output.push(lhs_value / rhs_value);
        }

        let shape = lhs_metadata.shape.to_vec();
        let array = Array::new(output, 0, shape_to_stride(&shape), shape);

        Ok(array)
    }

    fn div_scalar(&self, scalar: F) -> Result<Array<F>, PzeudoNumErr>
    where
        F: Copy + Div<Output = F>,
    {
        let metadata = self.get_metadata();
        let len = metadata.shape.iter().product::<usize>();
        let mut output = Vec::with_capacity(len);
        for idx in 0..len {
            let value = self.linear_index(idx)?;
            output.push(value / scalar);
        }

        let shape = metadata.shape.to_vec();
        let array = Array::new(output, 0, shape_to_stride(&shape), shape);

        Ok(array)
    }

    fn scalar_div(&self, scalar: F) -> Result<Array<F>, PzeudoNumErr>
    where
        F: Copy + Div<Output = F>,
    {
        let metadata = self.get_metadata();
        let len = metadata.shape.iter().product::<usize>();
        let mut output = Vec::with_capacity(len);
        for idx in 0..len {
            let value = self.linear_index(idx)?;
            output.push(scalar / value);
        }

        let shape = metadata.shape.to_vec();
        let array = Array::new(output, 0, shape_to_stride(&shape), shape);

        Ok(array)
    }
}
