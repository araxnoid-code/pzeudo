use std::ops::AddAssign;

use ndarray::ArrayD;

use crate::{GradientStorage, PzeudoOpsErr, StorageTrait};

pub fn view_backward<F, GradStorage: StorageTrait<ArrayD<F>>>(
    own_gradient_idx: Option<usize>,
    gradient_idx: Option<usize>,
    grad_storage: &mut GradStorage,
) -> Result<(), PzeudoOpsErr>
where
    F: Clone + AddAssign + Copy,
{
    let grad = if let Some(gradient_idx) = gradient_idx {
        Some(
            grad_storage
                .get_element(gradient_idx)
                .map_err(|err| PzeudoOpsErr::ViewBackwardErr(err.to_string()))?
                .to_owned(),
        )
    } else {
        None
    };

    if let (Some(assign), Some(grad_idx)) = (grad, own_gradient_idx) {
        let data = grad_storage
            .get_mut_element(grad_idx)
            .map_err(|err| PzeudoOpsErr::ViewBackwardErr(err.to_string()))?;

        data.add_assign(&assign);
    }
    Ok(())
}
