use ndarray::{ArrayD, ArrayViewD};
use num_traits::Float;

use crate::{NDArray, OpsLabel, PzeudoOpsErr, StorageTrait, Tensor};

impl<'ops_label, F, A, GradStorage> Tensor<'ops_label, F, A, GradStorage>
where
    GradStorage: StorageTrait<ArrayD<F>>,
    A: NDArray<F>,
    F: Float,
{
    pub fn view(
        &self,
    ) -> Result<Tensor<'ops_label, F, ArrayViewD<'_, F>, GradStorage>, PzeudoOpsErr> {
        let view = self.array._view();
        let grad = ArrayD::<F>::zeros(view.shape());
        let label = OpsLabel::View(self.grad);

        let new_tensor = Tensor::new(
            view,
            Some(grad),
            self.grad_storage.clone(),
            Some(label),
            self.record_storage.clone(),
        )
        .map_err(|err| PzeudoOpsErr::ViewErr(err.to_string()))?;

        Ok(new_tensor)
    }
}
