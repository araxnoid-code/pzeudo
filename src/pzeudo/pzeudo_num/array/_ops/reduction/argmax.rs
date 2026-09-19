use num_traits::Float;

use crate::prelude::*;

pub trait OpsMax<F>: ArrayTrait<F> {
    fn max(&self) -> Result<Array<F>, PzeudoErr>
    where
        F: Float,
    {
        let metadata = self.get_metadata();
        let len = metadata.shape.iter().product::<usize>();

        let mut max = None;
        for i in 0..len {
            let j = self.linear_index(i)?;
            match &mut max {
                None => max = Some(j),
                Some(max) => {
                    *max = max.max(self.linear_index(i)?);
                }
            }
        }

        Ok(Array::from_vector(vec![max.unwrap()]))
    }

    fn max_axis(&self, axis: &[usize], keep_dim: bool) -> Result<(), PzeudoErr> {
        let metadata = self.get_metadata();

        if axis.len() <= 0 {
            return Err(PzeudoErr::OpsErr(format!(
                "SumAxisErr. OpsSum::sum_axis. Cannot perform sum_axis because axis is empty.",
            )));
        } else if axis.len() > metadata.shape.len() {
            return Err(PzeudoErr::OpsErr(format!(
                "SumAxisErr. OpsSum::sum_axis. Unable to perform sum_axis because axis {:?} is out of bounds.",
                axis
            )));
        }

        Ok(())
    }

    fn argmax(&self) -> Result<Array<F>, PzeudoErr>
    where
        F: Float,
    {
        let metadata = self.get_metadata();
        let len = metadata.shape.iter().product::<usize>();

        let mut max = None;
        let mut idx = None;
        for i in 0..len {
            let j = self.linear_index(i)?;
            match &mut idx {
                None => {
                    idx = Some(i);
                    max = Some(j);
                }
                Some(idx) => {
                    if max.unwrap() < j {
                        max = Some(j);
                        *idx = i;
                    }
                }
            }
        }

        Ok(Array::from_vector(vec![F::from(idx.unwrap()).ok_or(
            PzeudoErr::OpsErr(String::from("OpsArgMax::argmax_idx. Cannot cast on index")),
        )?]))
    }
}
