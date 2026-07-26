use num_traits::{One, Zero};

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

    pub fn to_ones(&mut self)
    where
        F: One,
    {
        for x in &mut self.data {
            *x = F::one();
        }
    }
}
