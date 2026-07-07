use std::{cell::RefCell, fmt::Display, rc::Rc, sync::atomic::Ordering};

use ndarray::ArrayD;
use num_traits::Float;

use crate::{OpsLabel, PzeudoOpsErr, StorageTrait, Tensor, TensorNDArray, add, div};

impl<'ops_label, F, A, GradStorage> Tensor<'ops_label, F, A, GradStorage>
where
    GradStorage: StorageTrait<ArrayD<F>>,
    A: TensorNDArray<F> + Display,
    F: Float + Display,
{
    pub fn div<Rhs>(
        &'ops_label self,
        rhs: &'ops_label Tensor<F, Rhs, GradStorage>,
    ) -> Result<Tensor<'ops_label, F, ArrayD<F>, GradStorage>, PzeudoOpsErr>
    where
        Rhs: TensorNDArray<F> + Display,
    {
        let result = div(self.array._view(), rhs.array._view())?;
        let grad = ArrayD::<F>::zeros(result.shape());
        let label = OpsLabel::Div(
            (self.array._view(), self.grad),
            (rhs.array._view(), rhs.grad),
        );

        let new_tensor = Tensor::new(
            result,
            Some(grad),
            self.grad_storage.clone(),
            Some(label),
            self.record_storage.clone(),
        )
        .map_err(|err| PzeudoOpsErr::AddErr(err.to_string()))?;

        Ok(new_tensor)
    }
}
