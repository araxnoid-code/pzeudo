use std::ops::{AddAssign, Div, MulAssign};

use num_traits::{NumCast, One, Zero, one};

use crate::{Array, OpsSum, PzeudoErr};

pub trait OpsAvg<F>: OpsSum<F> {
    fn avg(&self) -> Result<Array<F>, PzeudoErr>
    where
        F: AddAssign + Copy + Zero + NumCast + Div<Output = F>,
    {
        let metadata = self.get_metadata();
        let len = F::from(metadata.shape.iter().product::<usize>()).ok_or(PzeudoErr::AvgErr(
            String::from("AvgErr. OpsAvg::avg. error while casting data type"),
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

    fn avg_axis(&self, axis: &[usize], keep_dim: bool) -> Result<Array<F>, PzeudoErr>
    where
        F: One + MulAssign + NumCast + AddAssign + Copy + Zero + Div<Output = F>,
    {
        let metadata = self.get_metadata();
        let len = NumCast::from(axis.iter().fold(1, |acc, dim| acc * metadata.shape[*dim])).ok_or(
            PzeudoErr::SumAxisErr(String::from(
                "AvgErr. OpsAvg::avg. error while casting data type",
            )),
        )?;

        let sum_axis = self.sum_axis(axis, keep_dim)?;
        let avg = sum_axis.data.iter().map(|v| *v / len).collect::<Vec<F>>();

        let array = Array {
            data: avg,
            offset: 0,
            shape: sum_axis.shape,
            stride: sum_axis.stride,
        };

        Ok(array)
    }
}
