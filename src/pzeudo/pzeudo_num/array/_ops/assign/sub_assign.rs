use std::ops::SubAssign;

use crate::prelude::*;

pub trait OpsSubAssign<F>: ArrayAssignTrait<F> {
    fn sub_assign<Assign>(&mut self, assign: &Assign) -> Result<(), PzeudoErr>
    where
        Assign: ArrayTrait<F>,
        F: Copy + SubAssign,
    {
        let lhs_metadata = self.get_mut_metadata();
        let rhs_metadata = assign.get_metadata();

        if lhs_metadata.shape != rhs_metadata.shape {
            return Err(PzeudoErr::SubAssignErr(format!(
                "OpsSubAssign::sub_assign. cannot sub_assign arrays of shape {:?} and {:?} because they have different shapes",
                lhs_metadata.shape, rhs_metadata.shape
            )));
        }

        let len = lhs_metadata.shape.iter().product::<usize>();
        for i in 0..len {
            let lhs_value = self.mut_linear_index(i)?;
            let rhs_value = assign.linear_index(i)?;
            *lhs_value -= rhs_value;
        }

        Ok(())
    }
}
