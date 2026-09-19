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
