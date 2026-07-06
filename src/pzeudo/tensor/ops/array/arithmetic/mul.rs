use std::{
    cell::RefCell,
    ops::{AddAssign, Mul},
    rc::Rc,
};

use ndarray::{ArrayBase, ArrayD, ArrayViewD, Axis, Dim, IxDynImpl, OwnedRepr};
use num_traits::Zero;

use crate::{PzeudoOpsErr, StorageTrait, able_broadcast, broadcast_handling};

pub fn mul<F>(
    lhs: ArrayViewD<F>,
    rhs: ArrayViewD<F>,
) -> Result<ArrayBase<OwnedRepr<F>, Dim<IxDynImpl>, F>, PzeudoOpsErr>
where
    F: Mul<Output = F> + Copy,
{
    if lhs.shape().len() < rhs.shape().len() {
        able_broadcast(lhs.shape(), rhs.shape())?;
    } else {
        able_broadcast(rhs.shape(), lhs.shape())?;
    }

    Ok(&lhs * &rhs)
}

pub fn mul_backward<F, GradStorage>(
    lhs: ArrayViewD<F>,
    lhs_grad_idx: Option<usize>,
    rhs: ArrayViewD<F>,
    rhs_grad_idx: Option<usize>,
    gradient_idx: Option<usize>,
    grad_storage: &mut GradStorage,
) -> Result<(), PzeudoOpsErr>
where
    F: Mul<Output = F> + AddAssign + Copy + Zero + Clone,
    GradStorage: StorageTrait<ArrayD<F>>,
{
    let storage = grad_storage.get_mut_storage();
    let mut lhs_assign = None;
    let mut rhs_assign = None;

    // f(x, y) = x * y
    if let Some(grad_idx) = gradient_idx {
        let gradient = storage
            .get(grad_idx)
            .ok_or(PzeudoOpsErr::MulBackwardErr(format!(
                "MulBackwardErr. index {} is not available on gradient storage",
                grad_idx
            )))?
            .as_ref()
            .ok_or(PzeudoOpsErr::MulBackwardErr(format!(
                "MulBackwardErr. index {} points to a value that is None in Storage",
                grad_idx
            )))?;

        // df(lhs, rhs)/dlhs = rhs
        if let Some(lhs) = lhs_grad_idx {
            let lhs = storage.get(lhs).unwrap().as_ref().unwrap();
            if gradient.shape() == lhs.shape() {
                let gradient = &rhs * gradient;
                lhs_assign = Some(gradient);
            } else {
                able_broadcast(lhs.shape(), gradient.shape())?;
                let init = &rhs * gradient;
                let gradient = broadcast_handling(lhs.view(), gradient.view(), init.view());
                lhs_assign = Some(gradient.to_owned());
            }
        }

        // df(lhs, rhs)/drhs =  lhs
        if let Some(rhs) = rhs_grad_idx {
            let rhs = storage.get(rhs).unwrap().as_ref().unwrap();
            if gradient.shape() == rhs.shape() {
                let gradient = &lhs * gradient;
                rhs_assign = Some(gradient);
            } else {
                able_broadcast(rhs.shape(), gradient.shape())?;
                let init = &lhs * gradient;
                let gradient = broadcast_handling(rhs.view(), gradient.view(), init.view());
                rhs_assign = Some(gradient.to_owned());
            }
        }
    }

    if let (Some(assign), Some(idx)) = (lhs_assign, lhs_grad_idx) {
        let grad = storage.get_mut(idx).unwrap().as_mut().unwrap();
        grad.add_assign(&assign);
    }

    if let (Some(assign), Some(idx)) = (rhs_assign, rhs_grad_idx) {
        let grad = storage.get_mut(idx).unwrap().as_mut().unwrap();
        grad.add_assign(&assign);
    }

    Ok(())
}
