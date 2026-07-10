use ndarray::ArrayD;
use num_traits::Float;

use crate::{NDArray, OpsLabel, PzeudoOpsErr, StorageTrait, Tensor, matmul_2d};

impl<'ops_label, F, A, GradStorage> Tensor<'ops_label, F, A, GradStorage>
where
    GradStorage: StorageTrait<ArrayD<F>>,
    A: NDArray<F>,
    F: Float + 'static,
{
    pub fn matmul_2d<Rhs>(
        &'ops_label self,
        rhs: &'ops_label Tensor<F, Rhs, GradStorage>,
    ) -> Result<Tensor<'ops_label, F, ArrayD<F>, GradStorage>, PzeudoOpsErr>
    where
        Rhs: NDArray<F>,
    {
        let result = matmul_2d(self.array._view(), rhs.array._view())?;
        let grad = ArrayD::<F>::zeros(result.shape());
        let label = OpsLabel::Matmul2d(
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
        .map_err(|err| PzeudoOpsErr::Matmul2dErr(err.to_string()))?;

        Ok(new_tensor)
    }
}
