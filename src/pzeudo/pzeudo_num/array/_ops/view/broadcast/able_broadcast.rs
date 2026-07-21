use crate::PzeudoErr::{self, BroadcastErr};

pub fn able_broadcast(shape: &[usize], to: &[usize]) -> Result<(), PzeudoErr> {
    if shape == to {
        return Ok(());
    }

    if shape.len() > to.len() {
        return Err(BroadcastErr(format!(
            "BroadcastErr. able_broadcast. The shape {:?} cannot be broadcast to shape {:?} because the target shape is smaller.",
            shape, to
        )));
    }

    if shape.len() == 0 || to.len() == 0 {
        return Err(BroadcastErr(format!(
            "BroadcastErr. able_broadcast. It is currently not possible to broadcast with 0D"
        )));
    }

    let d = to.len() - shape.len();
    for (idx, to_dim) in to.iter().enumerate().rev() {
        let index = idx - d;
        let shape_dim = shape[index];
        if shape_dim != 1 && shape_dim != *to_dim {
            return Err(PzeudoErr::BroadcastErr(format!(
                "BroadcastErr. able_broadcast. cannot broadcast the shape {:?} to {:?} because {:?} cannot be broadcast to {:?}",
                shape, to, shape_dim, to_dim
            )));
        }

        if index == 0 {
            break;
        }
    }

    Ok(())
}

pub fn get_broadcast_dim(shape: &[usize], to: &[usize]) -> Result<Vec<usize>, PzeudoErr> {
    if shape.len() > to.len() {
        return Err(BroadcastErr(format!(
            "BroadcastErr. get_broadcast_dim. The shape {:?} cannot be broadcast to shape {:?} because the target shape is smaller.",
            shape, to
        )));
    }

    if shape.len() == 0 || to.len() == 0 {
        return Err(BroadcastErr(format!(
            "BroadcastErr. get_broadcast_dim. It is currently not possible to broadcast with 0D"
        )));
    }

    let d = to.len() - shape.len();
    let mut dim = (0..d).map(|idx| idx).collect::<Vec<usize>>();
    for (idx, to_dim) in to.iter().enumerate().rev() {
        let index = idx - d;
        let shape_dim = shape[index];
        if shape_dim != 1 && shape_dim != *to_dim {
            return Err(PzeudoErr::BroadcastErr(format!(
                "BroadcastErr. get_broadcast_dim. cannot broadcast the shape {:?} to {:?} because {:?} cannot be broadcast to {:?}",
                shape, to, shape_dim, to_dim
            )));
        } else if shape_dim == 1 {
            dim.push(idx);
        }

        if index == 0 {
            break;
        }
    }

    Ok(dim)
}
