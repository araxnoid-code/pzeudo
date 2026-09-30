use num_traits::Float;

use crate::prelude::*;

impl<F, T, G> Tensor<F, T, G>
where
    for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
    F: Copy + Float + FToUsize,
{
    pub fn argmax(&self) -> Result<Tensor<F, Contiguous, ReqNoGrad>, PzeudoErr> {
        let mut storage = self.get_storage().borrow_mut();
        let array = storage.get_as_array_ref::<T>(self.get_array_idx(), ContiguousType::Arr)?;
        let argmax = array.argmax()?;

        let array_idx = storage.push(ElementType::Arr(argmax))?;

        let tensor = Tensor::_new(
            array_idx,
            None,
            vec![1],
            None,
            self.get_record().clone(),
            self.get_storage().clone(),
        );
        Ok(tensor)
    }

    pub fn argmax_axis(
        &self,
        axis: &[usize],
        keep_dim: bool,
    ) -> Result<Tensor<F, Contiguous, ReqNoGrad>, PzeudoErr> {
        let mut storage = self.get_storage().borrow_mut();
        let array = storage.get_as_array_ref::<T>(self.get_array_idx(), ContiguousType::Arr)?;
        let argmax = array.argmax_axis(axis, keep_dim)?;
        let shape = argmax.shape.to_vec();

        let array_idx = storage.push(ElementType::Arr(argmax))?;

        let tensor = Tensor::_new(
            array_idx,
            None,
            shape,
            None,
            self.get_record().clone(),
            self.get_storage().clone(),
        );
        Ok(tensor)
    }
}
