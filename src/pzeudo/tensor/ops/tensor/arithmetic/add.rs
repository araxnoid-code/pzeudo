use std::{cell::RefCell, rc::Rc};

use ndarray::ArrayD;
use num_traits::Float;

use crate::{PzeudoOpsErr, StorageTrait, Tensor, TensorNDArray, add};

impl<F, A, GradStorage> Tensor<F, A, GradStorage>
where
    GradStorage: StorageTrait<ArrayD<F>>,
    A: TensorNDArray<F>,
    F: Float,
{
    pub fn add<Rhs>(
        &self,
        rhs: &Tensor<F, Rhs, GradStorage>,
    ) -> Result<Tensor<F, ArrayD<F>, GradStorage>, PzeudoOpsErr>
    where
        Rhs: TensorNDArray<F>,
    {
        let result = add(self.array._view(), rhs.array._view())?;
        let grad = ArrayD::<F>::zeros(result.shape());

        let new_tensor = Tensor::new(result, Some(grad), self.grad_storage.clone())
            .map_err(|err| PzeudoOpsErr::AddErr(err.to_string()))?;

        Ok(new_tensor)
    }
}
