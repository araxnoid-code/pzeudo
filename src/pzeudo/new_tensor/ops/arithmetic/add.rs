use std::{cell::RefCell, rc::Rc};

use crate::{OpsAble, OwnAble, RefAble, TensorAble, TensorMutAble};

fn add<Ops, Own>(lhs: &Ops, rhs: &Ops) -> Own
where
    Own: TensorAble,
    Ops: OpsAble<TensorType = Own>,
{
    // f(x, y) = x + y
    lhs.add(&rhs)
}

fn add_backward<T>(
    lhs_grad: &Option<Rc<RefCell<T>>>,
    rhs_grad: &Option<Rc<RefCell<T>>>,
    gradient: &Option<Rc<RefCell<T>>>,
) where
    T: TensorAble + TensorMutAble,
{
    // f(x, y) = x + y
    if let Some(gradient) = gradient {
        let gradient = gradient.borrow();

        // df(x, y)/dx = 1
        if let Some(lhs_grad) = lhs_grad {
            let mut lhs_grad = lhs_grad.borrow_mut();

            if lhs_grad.get_shape() == gradient.get_shape() {
                let ops = gradient.to_ops_able();
                lhs_grad.tensor_add_assign(ops);
            } else {
                // broadcasting handler
                let gradient_shape = gradient.get_shape();
                let lhs_shape = lhs_grad.get_shape();

                let mut lhs_dim_idx = lhs_shape.len() - 1;
                let mut sum_axis: Option<<<T as TensorAble>::OpsType<'_> as OpsAble>::OwnType> =
                    None;

                for (dim, gradient_dim_idx) in gradient_shape.iter().enumerate().rev() {
                    let lhs_dim = lhs_shape[lhs_dim_idx];
                    let gradint_dim = gradient_shape[*gradient_dim_idx];

                    if lhs_dim == 1 && gradint_dim != 1 {
                        if let Some(sum) = sum_axis {
                            sum.to_ops_able();
                            // sum.to_ops_able();
                            // sum_axis = Some(sum);
                        } else {
                            let sum = gradient.to_ops_able().ops_sum_axis(dim);
                            sum_axis = Some(sum);
                        }
                    }

                    if lhs_dim_idx == 0 {
                        continue;
                    }
                    lhs_dim_idx -= 1;
                }
            }
        }
    }
}

// pub fn add_backward(
//     lhs_grad: &Option<Rc<RefCell<ArrayD<f32>>>>,
//     rhs_grad: &Option<Rc<RefCell<ArrayD<f32>>>>,
//     gradient: &Option<Rc<RefCell<ArrayD<f32>>>>,
// ) {
// f(x, y) = x + y
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
