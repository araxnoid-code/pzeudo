use crate::PzeudoOpsErr::{self, AbleBroadcastErr};

pub fn able_broadcast(origin_shape: &[usize], to_shape: &[usize]) -> Result<(), PzeudoOpsErr> {
    if origin_shape.len() > to_shape.len() {
        return Err(AbleBroadcastErr(format!(
            "AbleBroadcastErr, shape {:?} cannot be broadcast to {:?}",
            origin_shape, to_shape
        )));
    }

    if origin_shape.len() == 0 {
        return Ok(());
    }

    let mut origin_shape_idx = origin_shape.len() - 1;
    for to_shape_dim in to_shape.iter().rev() {
        let origin_dim = origin_shape[origin_shape_idx];

        if origin_dim != 1 && origin_dim != *to_shape_dim {
            return Err(AbleBroadcastErr(format!(
                "AbleBroadcastErr, shape {:?} cannot be broadcast to {:?}",
                origin_shape, to_shape
            )));
        }

        if origin_shape_idx == 0 {
            break;
        }
        origin_shape_idx -= 1;
    }

    Ok(())
}
