use num_traits::Float;

use crate::prelude::*;

pub trait OpsMax<F>: ArrayTrait<F> {
    fn max(&self) -> Result<Array<F>, PzeudoErr>
    where
        F: Float,
    {
        let metadata = self.get_metadata();
        let len = metadata.shape.iter().product::<usize>();

        let mut max = F::zero();
        for i in 0..len {
            max = max.max(self.linear_index(i)?);
        }

        Ok(Array::from_vector(vec![max]))
    }

    fn argmax(&self) -> Result<Array<F>, PzeudoErr>
    where
        F: Float,
    {
        let metadata = self.get_metadata();
        let len = metadata.shape.iter().product::<usize>();

        let mut max = F::zero();
        let mut idx = None;
        for i in 0..len {
            match &mut idx {
                None => {
                    idx = Some(i);
                }
                Some(idx) => {
                    let j = self.linear_index(i)?;
                    if max < j {
                        max = j;
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
