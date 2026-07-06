use std::ops::{Add, AddAssign};

use ndarray::{ArrayBase, ArrayD, ArrayViewD, Axis, Dim, IxDynImpl, OwnedRepr};
use num_traits::Zero;

use crate::{PzeudoOpsErr, StorageTrait, able_broadcast};

pub fn add<F>(
    lhs: ArrayViewD<F>,
    rhs: ArrayViewD<F>,
) -> Result<ArrayBase<OwnedRepr<F>, Dim<IxDynImpl>, F>, PzeudoOpsErr>
where
    F: Add<Output = F> + Copy,
{
    if lhs.shape().len() < rhs.shape().len() {
        able_broadcast(lhs.shape(), rhs.shape())?;
    } else {
        able_broadcast(rhs.shape(), lhs.shape())?;
    }

    Ok(&lhs + &rhs)
}

fn broadcast_handling<F: Clone + Zero>(input: ArrayViewD<F>, gradient: ArrayViewD<F>) -> ArrayD<F> {
    let gradient_shape = gradient.shape();
    let mut input_shape = input.shape().to_vec();
    let distance = gradient_shape.len() - input_shape.len();
    if distance != 0 {
        let ones = vec![1; distance];
        input_shape = [ones, input_shape].concat();
    }

    let mut gradient_axis: Option<ArrayBase<OwnedRepr<F>, Dim<IxDynImpl>, F>> = None;
    for (idx, (gradient_shape, lhs_shape)) in
        gradient_shape.iter().zip(input_shape.iter()).enumerate()
    {
        if *lhs_shape == 1 && *gradient_shape != 1 {
            if let Some(grad) = gradient_axis {
                gradient_axis = Some(grad.sum_axis(Axis(idx)).insert_axis(Axis(idx)));
            } else {
                gradient_axis = Some(gradient.sum_axis(Axis(idx)).insert_axis(Axis(idx)));
            }
        }
    }

    let gradient = gradient_axis.unwrap();
    let gradient = gradient.to_shape(input.shape()).unwrap();
    gradient.to_owned()
}

pub fn add_backward<F, GradStorage>(
    lhs_grad_idx: Option<usize>,
    rhs_grad_idx: Option<usize>,
    gradient_idx: Option<usize>,
    grad_storage: &mut GradStorage,
) -> Result<(), PzeudoOpsErr>
where
    F: Clone + Zero + AddAssign<F>,
    GradStorage: StorageTrait<ArrayD<F>>,
{
    let storage = grad_storage.get_mut_storage();
    let mut lhs_assign = None;
    let mut rhs_assign = None;

    // f(x, y) = x + y
    if let Some(grad_idx) = gradient_idx {
        let gradient = storage
            .get(grad_idx)
            .ok_or(PzeudoOpsErr::AddBackwardErr(format!(
                "AddBackwardErr. index {} is not available on gradient storage",
                grad_idx
            )))?
            .as_ref()
            .ok_or(PzeudoOpsErr::AddBackwardErr(format!(
                "AddBackwardErr. index {} points to a value that is None in Storage",
                grad_idx
            )))?;

        // df(x, y)/dx = 1
        if let Some(lhs) = lhs_grad_idx {
            let lhs = storage.get(lhs).unwrap().as_ref().unwrap();
            if gradient.shape() == lhs.shape() {
                lhs_assign = Some(gradient.to_owned());
            } else {
                able_broadcast(lhs.shape(), gradient.shape())?;
                let gradient = broadcast_handling(lhs.view(), gradient.view());
                lhs_assign = Some(gradient.to_owned());
            }
        }

        // df(x, y)/dy = 1
        if let Some(rhs) = rhs_grad_idx {
            let rhs = storage.get(rhs).unwrap().as_ref().unwrap();
            if gradient.shape() == rhs.shape() {
                rhs_assign = Some(gradient.to_owned());
            } else {
                able_broadcast(rhs.shape(), gradient.shape())?;
                let gradient = broadcast_handling(rhs.view(), gradient.view());
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
