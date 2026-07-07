use std::{
    cell::RefCell,
    fmt::Display,
    ops::{Add, AddAssign, Div, Mul, Neg},
    rc::Rc,
};

use ndarray::{ArrayBase, ArrayD, ArrayViewD, Dim, IxDynImpl, OwnedRepr};
use num_traits::{Float, One, Zero, one};

use crate::{PzeudoOpsErr, StorageTrait, able_broadcast, broadcast_handling, neg, pow2};

pub fn div<F>(
    lhs: ArrayViewD<F>,
    rhs: ArrayViewD<F>,
) -> Result<ArrayBase<OwnedRepr<F>, Dim<IxDynImpl>, F>, PzeudoOpsErr>
where
    F: Div<Output = F> + Copy + Float,
{
    if lhs.shape().len() < rhs.shape().len() {
        able_broadcast(lhs.shape(), rhs.shape())?;
    } else {
        able_broadcast(rhs.shape(), lhs.shape())?
    }

    Ok(&lhs / &rhs)
}

pub fn div_backward<F, GradStorage>(
    lhs: ArrayViewD<F>,
    lhs_grad_idx: Option<usize>,
    rhs: ArrayViewD<F>,
    rhs_grad_idx: Option<usize>,
    gradient_idx: Option<usize>,
    grad_storage: &mut GradStorage,
) -> Result<(), PzeudoOpsErr>
where
    F: Div<Output = F>
        + Copy
        + Mul<Output = F>
        + Zero
        + AddAssign<F>
        + Clone
        + Neg<Output = F>
        + One
        + Display,
    GradStorage: StorageTrait<ArrayD<F>>,
{
    let storage = grad_storage.get_mut_storage();
    let mut lhs_assign = None;
    let mut rhs_assign = None;

    // f(x, y) = x / y
    if let Some(grad_idx) = gradient_idx {
        let gradient = storage
            .get(grad_idx)
            .ok_or(PzeudoOpsErr::DivBackwardErr(format!(
                "DivBackwardErr. index {} is not available on gradient storage",
                grad_idx
            )))?
            .as_ref()
            .ok_or(PzeudoOpsErr::DivBackwardErr(format!(
                "DivBackwardErr. index {} points to a value that is None in Storage",
                grad_idx
            )))?;

        // df(lhs, rhs)/dlhs = 1/rhs
        if let Some(lhs_grad_idx) = lhs_grad_idx {
            let lhs_grad = storage.get(lhs_grad_idx).unwrap().as_ref().unwrap();
            if gradient.shape() == lhs_grad.shape() {
                let gradient = scalar_div(one(), rhs.view()) * gradient;
                lhs_assign = Some(gradient);
            } else {
                able_broadcast(lhs_grad.shape(), gradient.shape())?;
                let init = scalar_div(one(), rhs.view()) * gradient;
                let gradient = broadcast_handling(lhs_grad.view(), gradient.view(), init.view());
                lhs_assign = Some(gradient.to_owned());
            }
        }

        // df(lhs, rhs)/drhs =  -lhs/rhs^2
        if let Some(rhs_grad_idx) = rhs_grad_idx {
            let rhs_grad = storage.get(rhs_grad_idx).unwrap().as_ref().unwrap();
            if gradient.shape() == rhs_grad.shape() {
                // println!("{}", rhs);
                let gradient = neg(lhs.view()) / pow2(rhs.view()) * gradient;
                rhs_assign = Some(gradient);
            } else {
                able_broadcast(rhs_grad.shape(), gradient.shape())?;
                let init = neg(lhs.view()) / pow2(rhs_grad.view()) * gradient;
                let gradient = broadcast_handling(rhs_grad.view(), gradient.view(), init.view());
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

fn scalar_div<F>(scalar: F, arr: ArrayViewD<F>) -> ArrayD<F>
where
    F: Div<Output = F> + Clone + Copy,
{
    arr.mapv(|x| scalar / x)
}
