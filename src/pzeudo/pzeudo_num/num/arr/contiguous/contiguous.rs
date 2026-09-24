use num_traits::{One, Zero};

use crate::PzeudoErr;

pub struct Array<F> {
    pub(crate) data: Vec<F>,       // 24
    pub(crate) stride: Vec<usize>, // 24
    pub(crate) shape: Vec<usize>,  // 24
    pub(crate) offset: usize,      // 8
}

impl<F> Array<F> {
    pub fn to_zeros(&mut self)
    where
        F: Zero,
    {
        for x in &mut self.data {
            *x = F::zero();
        }
    }

    pub fn get_data(&self) -> &Vec<F> {
        &self.data
    }

    pub fn to_ones(&mut self)
    where
        F: One,
    {
        for x in &mut self.data {
            *x = F::one();
        }
    }

    pub fn vec_equal(&self, vec: &Vec<F>) -> Result<(), PzeudoErr>
    where
        Vec<F>: PartialEq<Vec<F>>,
    {
        if &self.data != vec {
            return Err(PzeudoErr::ArrayErr(String::from(
                "Array::vec_equal. data and vector not equal",
            )));
        }

        Ok(())
    }
}
