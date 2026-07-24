use std::ops::Mul;

use crate::{
    PzeudoErr::MulErr,
    prelude::{PzeudoErr, *},
};

pub trait OpsMul<F>: ArrayTrait<F> {
    fn mul<Rhs>(&self, rhs: &Rhs) -> Result<Array<F>, PzeudoErr>
    where
        Self: OpsBroadcast<F>,
        F: Copy + Mul<Output = F>,
        Rhs: ArrayTrait<F> + OpsBroadcast<F>,
    {
        let lhs_metadata = self.get_metadata();
        let rhs_metadata = rhs.get_metadata();

        let (lhs_broadcasted, rhs_broadcasted, len, shape) = if lhs_metadata.shape
            == rhs_metadata.shape
        {
            (
                None,
                None,
                lhs_metadata.shape.iter().product::<usize>(),
                lhs_metadata.shape.to_vec(),
            )
        } else if able_broadcast(lhs_metadata.shape, rhs_metadata.shape).is_ok() {
            (
                Some(self.broadcast(rhs_metadata.shape)?),
                None,
                rhs_metadata.shape.iter().product::<usize>(),
                rhs_metadata.shape.to_vec(),
            )
        } else if able_broadcast(rhs_metadata.shape, lhs_metadata.shape).is_ok() {
            (
                None,
                Some(rhs.broadcast(lhs_metadata.shape)?),
                lhs_metadata.shape.iter().product::<usize>(),
                lhs_metadata.shape.to_vec(),
            )
        } else {
            return Err(MulErr(format!(
                "OpsAdd::add. cannot add arrays of shape {:?} and {:?} because they have different shapes",
                lhs_metadata.shape, rhs_metadata.shape
            )));
        };

        let mut output = Vec::with_capacity(len);
        for i in 0..len {
            let lhs_value = if let Some(lhs) = &lhs_broadcasted {
                lhs.linear_index(i)?
            } else {
                self.linear_index(i)?
            };

            let rhs_value = if let Some(rhs) = &rhs_broadcasted {
                rhs.linear_index(i)?
            } else {
                rhs.linear_index(i)?
            };
            output.push(lhs_value * rhs_value);
        }

        let array = Array::new(output, 0, shape_to_stride(&shape), shape);
        Ok(array)
    }

    fn mul_scalar(&self, scalar: F) -> Result<Array<F>, PzeudoErr>
    where
        F: Copy + Mul<Output = F>,
    {
        let metadata = self.get_metadata();
        let len = metadata.shape.iter().product::<usize>();
        let mut output = Vec::with_capacity(len);
        for idx in 0..len {
            let value = self.linear_index(idx)?;
            output.push(value * scalar);
        }

        let shape = metadata.shape.to_vec();
        let array = Array::new(output, 0, shape_to_stride(&shape), shape);

        Ok(array)
    }

    fn scalar_mul(&self, scalar: F) -> Result<Array<F>, PzeudoErr>
    where
        F: Copy + Mul<Output = F>,
    {
        let metadata = self.get_metadata();
        let len = metadata.shape.iter().product::<usize>();
        let mut output = Vec::with_capacity(len);
        for idx in 0..len {
            let value = self.linear_index(idx)?;
            output.push(scalar * value);
        }

        let shape = metadata.shape.to_vec();
        let array = Array::new(output, 0, shape_to_stride(&shape), shape);

        Ok(array)
    }
}
