use std::ops::{AddAssign, Div};

use num_traits::{NumCast, Zero};

use crate::{Array, ArrayTrait, OpsDiv, OpsSum, PzeudoNumErr};

pub trait OpsAvg<F>: OpsSum<F> {
    fn avg(&self) -> Result<Array<F>, PzeudoNumErr>
    where
        F: AddAssign + Copy + Zero + NumCast + Div<Output = F>,
    {
        let metadata = self.get_metadata();
        let len = F::from(metadata.shape.iter().product::<usize>()).ok_or(PzeudoNumErr::AvgErr(
            format!("AvgErr. OpsAvg::avg. error while casting data type"),
        ))?;

        let sum = self.sum()?;
        let avg = sum.data[0] / len;

        let array = Array {
            data: vec![avg],
            offset: 0,
            shape: vec![1],
            stride: vec![1],
        };

        Ok(array)
    }
}
