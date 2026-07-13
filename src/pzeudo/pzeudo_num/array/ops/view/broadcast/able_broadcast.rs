use crate::PzeudoNumErr::{self, BroadcastErr};

pub fn able_broadcast(shape: &[usize], to: &[usize]) -> Result<(), PzeudoNumErr> {
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
    for (idx, to) in to.iter().enumerate().rev() {
        let index = idx - d;
        let shape = shape[index];
        if shape != 1 && shape != *to {
            return Err(PzeudoNumErr::BroadcastErr(format!("")));
        }

        if index == 0 {
            break;
        }
    }

    Ok(())
}

pub fn get_broadcast_dim(shape: &[usize], to: &[usize]) -> Result<(), PzeudoNumErr> {
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
    for (idx, to) in to.iter().enumerate().rev() {
        let index = idx - d;
        let shape = shape[index];
        if shape != 1 && shape != *to {
            return Err(PzeudoNumErr::BroadcastErr(format!("")));
        } else if shape == 1 {
            dim.push(idx);
        }

        if index == 0 {
            break;
        }
    }

    println!("{:?}", dim);

    Ok(())
}
