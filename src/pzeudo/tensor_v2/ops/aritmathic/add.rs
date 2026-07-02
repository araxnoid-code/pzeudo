use std::{cell::RefCell, rc::Rc};

use crate::{AddOps, ArrayAble, ArrayMutAble, OpsAble, SumOps, ToShape};

fn add<Ops>(lhs: Ops, rhs: Ops) -> <Ops as AddOps>::Out
where
    Ops: OpsAble + AddOps,
{
    lhs._add(&rhs)
}

fn add_bacward<Array, Ops>(
    lhs_grad: &Option<Rc<RefCell<Array>>>,
    rhs_grad: &Option<Rc<RefCell<Array>>>,
    gradient: &Option<Rc<RefCell<Array>>>,
) where
    for<'a> Array: ArrayMutAble<Ops<'a> = Ops> + 'a,
    for<'a> Ops: OpsAble + SumOps<Out = Array> + ToShape,
{
    if let Some(gradient) = gradient {
        let gradient = gradient.borrow();

        if let Some(lhs_grad) = lhs_grad {
            let mut lhs_grad = lhs_grad.borrow_mut();

            let gradient_shape = gradient.shape();
            let lhs_grad_shape = lhs_grad.shape();

            if gradient_shape == lhs_grad_shape {
                lhs_grad._add_assign(gradient.into_ops());
            } else {
                // broadcast handler
                let mut acc: Option<Array> = None;
                let mut lhs_grad_dim_idx = lhs_grad_shape.len() - 1;
                let mut flow_handler = false;
                for dim in (0..gradient_shape.len()).rev() {
                    let gradient_dim = gradient_shape[dim];
                    let lhs_grad_dim = lhs_grad_shape[lhs_grad_dim_idx];

                    if (lhs_grad_dim == 1 || flow_handler) && gradient_dim != 1 {
                        if let Some(_acc) = &acc {
                            acc = Some(_acc.into_ops()._sum_axis(dim))
                        } else {
                            acc = Some(gradient.into_ops()._sum_axis(dim));
                        }
                    }

                    if lhs_grad_dim_idx == 0 && gradient_dim != 1 {
                        flow_handler = true;
                        continue;
                    }

                    lhs_grad_dim_idx -= 1
                }

                let acc = acc.unwrap();
                let ops = acc.into_ops();
                let reshape = ops._to_shape(lhs_grad.shape());
                let ops = reshape.into_ops();
                // lhs_grad._add_assign(reshape.into_ops());
            }
        }
    }
}

// pub fn add_backward(
// lhs_grad: &Option<Rc<RefCell<ArrayD<f32>>>>,
// rhs_grad: &Option<Rc<RefCell<ArrayD<f32>>>>,
// gradient: &Option<Rc<RefCell<ArrayD<f32>>>>,
// ) {
//     // f(x, y) = x + y
//     if let Some(gradient) = gradient {
//         let gradient = gradient.borrow();

//         // df(x, y)/dx = 1
//         if let Some(lhs_grad) = lhs_grad {
//             let mut lhs = lhs_grad.borrow_mut();
//             if gradient.shape() == lhs.shape() {
//                 lhs.add_assign(&gradient.view());
//             } else {
//                 let gradient_shape = gradient.shape();
//                 let mut lhs_shape = lhs.shape().to_vec();
//                 let distance = gradient_shape.len() - lhs_shape.len();
//                 if distance != 0 {
//                     let ones = vec![1; distance];
//                     lhs_shape = [ones, lhs_shape].concat();
//                 }

//                 let mut gradient_axis: Option<ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>, f32>> =
//                     None;
//                 for (idx, (gradient_shape, lhs_shape)) in
//                     gradient_shape.iter().zip(lhs_shape.iter()).enumerate()
//                 {
//                     if *lhs_shape == 1 && *gradient_shape != 1 {
//                         if let Some(grad) = gradient_axis {
//                             gradient_axis = Some(grad.sum_axis(Axis(idx)).insert_axis(Axis(idx)));
//                         } else {
//                             gradient_axis =
//                                 Some(gradient.sum_axis(Axis(idx)).insert_axis(Axis(idx)));
//                         }
//                     }
//                 }

//                 let gradient = gradient_axis.unwrap();
//                 let gradient = gradient.to_shape(lhs.shape()).unwrap();
//                 lhs.add_assign(&gradient);
//             }
//         }

//         // df(x, y)/dy = 1
//         if let Some(rhs_grad) = rhs_grad {
//             let mut rhs = rhs_grad.borrow_mut();
//             if gradient.shape() == rhs.shape() {
//                 rhs.add_assign(&gradient.view());
//             } else {
//                 let gradient_shape = gradient.shape();
//                 let mut rhs_shape = rhs.shape().to_vec();
//                 let distance = gradient_shape.len() - rhs_shape.len();
//                 if distance != 0 {
//                     let ones = vec![1; distance];
//                     rhs_shape = [ones, rhs_shape].concat();
//                 }

//                 let mut gradient_axis: Option<ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>, f32>> =
//                     None;
//                 for (idx, (gradient_shape, rhs_shape)) in
//                     gradient_shape.iter().zip(rhs_shape.iter()).enumerate()
//                 {
//                     if *rhs_shape == 1 && *gradient_shape != 1 {
//                         if let Some(grad) = gradient_axis {
//                             gradient_axis = Some(grad.sum_axis(Axis(idx)).insert_axis(Axis(idx)));
//                         } else {
//                             gradient_axis =
//                                 Some(gradient.sum_axis(Axis(idx)).insert_axis(Axis(idx)));
//                         }
//                     }
//                 }

//                 let gradient = gradient_axis.unwrap();
//                 let gradient = gradient.to_shape(rhs.shape()).unwrap();
//                 rhs.add_assign(&gradient);
//             }
//         }
//     }
// }
