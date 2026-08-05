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
    match storage_type {
        StorageType::Arr(idx, grad_time) => {
            return Ok(storage.grad_storage.check_no_grad(idx)?
                || storage.grad_storage.check_time_not_match(
                    idx,
                    grad_time.ok_or(PzeudoErr::TensorToolsErr(format!(
                        "check_no_grad_or_time_not_match. gradient does not have time_grad"
                    )))?,
                )?);
        }
        StorageType::View(view_idx) => {
            let metadata = storage.view_storage.get_metadata(view_idx)?;
            match metadata.arr_index{
                ViewStorageType::Storage(idx, grad_time) => {
                    return Ok(storage.grad_storage.check_no_grad(idx)?
                        || storage.grad_storage.check_time_not_match(
                            idx,
                            grad_time.ok_or(PzeudoErr::TensorToolsErr(format!(
                                "check_no_grad_or_time_not_match. gradient does not have time_grad"
                            )))?,
                        )?);
                },
                ViewStorageType::Param(idx) => {
                    return Ok(storage
                        .params_storage.storage
                        .get(idx)
                        .ok_or(PzeudoErr::TensorToolsErr(format!("check_no_grad_or_time_not_match. metadata index {idx} points to an invalid location on params storage")))?.grad.is_none())

                }
            }
        }
        StorageType::Param(idx) => {
            return Ok(storage
                .params_storage.storage
                .get(idx)
                .ok_or(PzeudoErr::TensorToolsErr(format!("check_no_grad_or_time_not_match. index {idx} points to an invalid location on params storage")))?.grad.is_none())
        }
    }
}
