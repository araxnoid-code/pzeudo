use crate::PzeudoOpsErr::{self, AbleBroadcastErr};

pub fn able_broadcast(origin_shape: &[usize], to_shape: &[usize]) -> Result<(), PzeudoOpsErr> {
    if origin_shape.len() > to_shape.len() {
        return Err(AbleBroadcastErr(format!(
            "AbleBroadcastErr, shape {:?} cannot be broadcast to {:?}",
            origin_shape, to_shape
        )));
    }

    let d = to_shape.len() - origin_shape.len();
    let mut padding_shape = vec![1; d];
    padding_shape.extend_from_slice(origin_shape);

    padding_shape
        .iter()
        .zip(to_shape.iter())
        .try_for_each(|(origin_dim, to_dim)| {
            if *origin_dim != 1 && origin_dim != to_dim {
                return Err(AbleBroadcastErr(format!(
                    "AbleBroadcastErr, shape {:?} cannot be broadcast to {:?}",
                    origin_shape, to_shape
                )));
            }
            Ok(())
        })
}
