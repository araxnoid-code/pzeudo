use crate::prelude::*;

pub fn is_no_grad_or_time_not_match_or_no_update<F>(
    storage_type: StorageType,
    storage: &ArrayStorage<F>,
) -> Result<bool, PzeudoErr> {
    match storage_type {
        StorageType::Arr(idx, grad_time) => {
            return Ok(storage.grad_storage.is_no_grad(idx)?
                || storage.grad_storage.is_time_not_match(
                    idx,
                    grad_time.ok_or(PzeudoErr::TensorToolsErr(format!(
                        "check_no_grad_or_time_not_match. gradient does not have time_grad"
                    )))?,
                )? || !storage.grad_storage.is_update(idx)?);
        }
        StorageType::View(view_idx) => {
            let metadata = storage.view_storage.get_metadata(view_idx)?;
            match metadata.arr_index{
                ViewStorageType::Storage(idx, grad_time) => {
                    return Ok(storage.grad_storage.is_no_grad(idx)?
                        || storage.grad_storage.is_time_not_match(
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
                .ok_or(PzeudoErr::TensorToolsErr(format!("check_no_grad_or_time_not_match. index {idx} points to an invalid location on params storage")))?.grad.is_none() || storage.params_storage.is_update(idx)?)
        }
    }
}
