use crate::prelude::*;

pub fn shape_to_stride(shape: &[usize]) -> Vec<usize> {
    (0..shape.len())
        .map(|idx| shape[idx + 1..].iter().product::<usize>())
        .collect::<Vec<usize>>()
}

pub fn check_no_grad_or_time_not_match<F>(
    storage_type: StorageType,
    storage: &ArrayStorage<F>,
) -> Result<bool, PzeudoErr> {
    if let StorageType::Arr(idx, grad_time) = storage_type {
        return Ok(storage.grad_storage.check_no_grad(idx)?
            || storage.grad_storage.check_time_not_match(
                idx,
                grad_time.ok_or(PzeudoErr::CheckNoGradOrTimeNotMatchErr(format!(
                    "check_no_grad_or_time_not_match. gradient does not have time_grad"
                )))?,
            )?);
    } else {
        Ok(false)
        // return Err(PzeudoErr::CheckNoGradOrTimeNotMatchErr(format!(
        //     "check_no_grad_or_time_not_match. no_grad cannot be done on permanent or view"
        // )));
    }
}
