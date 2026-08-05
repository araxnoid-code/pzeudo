use num_traits::Zero;

use crate::prelude::*;

impl<F> Module<F> {
    pub fn tensor_from_array(
        &self,
        array: Array<F>,
    ) -> Result<Tensor<F, Contiguous, Grad>, PzeudoErr>
    where
        F: Clone + Zero,
    {
        Tensor::from_array(array, self.storage.clone(), self.record.clone())
    }

    pub fn tensor_from_vector_with_shape<G>(
        &self,
        vec: &[F],
        shape: &[usize],
    ) -> Result<Tensor<F, Contiguous, G>, PzeudoErr>
    where
        G: ReqGradTrait<F>,
        F: Clone + Zero,
    {
        Tensor::from_vector_with_shape(vec, shape, self.storage.clone(), self.record.clone())
    }

    pub fn param_from_vector_with_shape<G>(
        &self,
        vec: &[F],
        shape: &[usize],
    ) -> Result<Tensor<F, Contiguous, G>, PzeudoErr>
    where
        G: ReqGradTrait<F>,
        F: Clone + Zero,
    {
        Tensor::param_from_vector_with_shape(vec, shape, self.storage.clone(), self.record.clone())
    }
}
