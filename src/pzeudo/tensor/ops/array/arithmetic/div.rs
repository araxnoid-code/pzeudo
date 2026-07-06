use std::{
    cell::RefCell,
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
        + One,
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
        if let Some(lhs) = lhs_grad_idx {
            let lhs = storage.get(lhs).unwrap().as_ref().unwrap();
            if gradient.shape() == lhs.shape() {
                let gradient = scalar_div(one(), rhs.view()) * gradient;
                lhs_assign = Some(gradient);
            } else {
                able_broadcast(lhs.shape(), gradient.shape())?;
                let init = scalar_div(one(), rhs.view()) * gradient;
                let gradient = broadcast_handling(lhs.view(), gradient.view(), init.view());
                lhs_assign = Some(gradient.to_owned());
            }
        }

        // df(lhs, rhs)/drhs =  -lhs/rhs^2
        if let Some(rhs) = rhs_grad_idx {
            let rhs = storage.get(rhs).unwrap().as_ref().unwrap();
            if gradient.shape() == rhs.shape() {
                let gradient = neg(lhs.view()) / pow2(rhs.view()) * gradient;
                rhs_assign = Some(gradient);
            } else {
                able_broadcast(rhs.shape(), gradient.shape())?;
                let init = neg(lhs.view()) / pow2(rhs.view()) * gradient;
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

// pub fn div_backward<F>(
//     lhs: ArrayViewD<F>,
//     lhs_grad: &Option<Rc<RefCell<ArrayD<F>>>>,
//     rhs: ArrayViewD<F>,
//     rhs_grad: &Option<Rc<RefCell<ArrayD<F>>>>,
//     gradient: &Option<Rc<RefCell<ArrayD<F>>>>,
// ) where
//     F: Div<Output = F>
//         + Copy
//         + Mul<Output = F>
//         + Zero
//         + AddAssign<F>
//         + Clone
//         + Neg<Output = F>
//         + One,
// {
//     // f(lhs, rhs) = lhs / rhs
//     if let Some(gradient) = gradient {
//         let gradient = gradient.borrow();

//         // df(lhs, rhs)/dlhs = 1/rhs
//         if let Some(lhs_grad) = lhs_grad {
//             let mut lhs_grad = lhs_grad.borrow_mut();

//             if gradient.shape() == lhs_grad.shape() {
//                 let assign = &scalar_div(one(), rhs.view()).view() * &gradient.view();
//                 lhs_grad.add_assign(&assign);
//             } else {
//                 let gradient_shape = gradient.shape();
//                 let mut lhs_grad_shape = lhs_grad.shape().to_vec();
//                 let distance = gradient_shape.len() - lhs_grad_shape.len();
//                 if distance != 0 {
//                     let ones = vec![1; distance];
//                     lhs_grad_shape = [ones, lhs_grad_shape].concat();
//                 }

//                 let mut gradient_axis: Option<ArrayBase<OwnedRepr<F>, Dim<IxDynImpl>, F>> =
//                     Some(scalar_div(one(), rhs.view()) * &gradient.view());

//                 for (idx, (gradient_shape, lhs_grad_shape)) in
//                     gradient_shape.iter().zip(lhs_grad_shape.iter()).enumerate()
//                 {
//                     if *lhs_grad_shape == 1 && *gradient_shape != 1 {
//                         if let Some(grad) = gradient_axis {
//                             gradient_axis = Some(grad.sum_axis(Axis(idx)).insert_axis(Axis(idx)));
//                         }
//                     }
//                 }

//                 let gradient = gradient_axis.unwrap();
//                 let gradient = gradient.to_shape(lhs_grad.shape()).unwrap();
//                 lhs_grad.add_assign(&gradient);
//             }
//         }

// df(lhs, rhs)/drhs =  -lhs/rhs^2
//         if let Some(rhs_grad) = rhs_grad {
//             let mut rhs_grad = rhs_grad.borrow_mut();

//             if gradient.shape() == rhs_grad.shape() {
//                 rhs_grad.add_assign(&(neg(lhs) / pow2(rhs.view()) * &gradient.view()));
//             } else {
//                 let gradient_shape = gradient.shape();
//                 let mut rhs_grad_shape = rhs_grad.shape().to_vec();
//                 let distance = gradient_shape.len() - rhs_grad_shape.len();
//                 if distance != 0 {
//                     let ones = vec![1; distance];
//                     rhs_grad_shape = [ones, rhs_grad_shape].concat();
//                 }

//                 // pow2(rhs.view()) * &gradient.view();
//                 let mut gradient_axis: Option<ArrayBase<OwnedRepr<F>, Dim<IxDynImpl>, F>> =
//                     Some(neg(lhs) / pow2(rhs.view()) * &gradient.view());

//                 for (idx, (gradient_shape, rhs_shape)) in
//                     gradient_shape.iter().zip(rhs_grad_shape.iter()).enumerate()
//                 {
//                     if *rhs_shape == 1 && *gradient_shape != 1 {
//                         if let Some(grad) = gradient_axis {
//                             gradient_axis = Some(grad.sum_axis(Axis(idx)).insert_axis(Axis(idx)));
//                         }
//                     }
//                 }

//                 let gradient = gradient_axis.unwrap();
//                 let gradient = gradient.to_shape(rhs_grad.shape()).unwrap();
//                 rhs_grad.add_assign(&gradient);
//             }
//         }
//     }
// }

fn scalar_div<F>(scalar: F, arr: ArrayViewD<F>) -> ArrayD<F>
where
    F: Div<Output = F> + Clone + Copy,
{
    arr.mapv(|x| scalar / x)
}
