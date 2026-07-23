use num_traits::{One, Zero};

pub struct Array<F> {
    pub(crate) data: Vec<F>,
    pub(crate) offset: usize,
    pub(crate) stride: Vec<usize>,
    pub(crate) shape: Vec<usize>,
}

impl<F> Array<F> {
    pub fn to_zero(&mut self)
    where
        F: Zero,
    {
        for x in &mut self.data {
            *x = F::zero();
        }
    }

    pub fn to_one(&mut self)
    where
        F: One,
    {
        for x in &mut self.data {
            *x = F::one();
        }
    }
}
