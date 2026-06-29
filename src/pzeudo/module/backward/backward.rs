use crate::{BackwardLabel, TensorTrait, add_backward, div_backward, mul_backward, sub_backward};

pub trait Backward<'a>: TensorTrait<'a> {
    fn backward(&'a self) {
        if let Err(_) = self.set_gradient_ones() {
            return;
        };

        let mut queue = vec![(self.get_share_backward_label(), self.get_share_gradient())];
        while let Some((backward_label, gradient)) = queue.pop() {
            match backward_label {
                Some(backward_label) => {
                    //
                    match &*backward_label {
                        BackwardLabel::Add(lhs, rhs) => {
                            add_backward(&lhs.1, &rhs.1, &gradient);
                            queue.push((lhs.2.clone(), lhs.1.clone()));
                            queue.push((rhs.2.clone(), rhs.1.clone()));
                        }
                        BackwardLabel::Sub(lhs, rhs) => {
                            sub_backward(&lhs.1, &rhs.1, &gradient);
                            queue.push((lhs.2.clone(), lhs.1.clone()));
                            queue.push((rhs.2.clone(), rhs.1.clone()));
                        }
                        BackwardLabel::Mul(lhs, rhs) => {
                            mul_backward(lhs.0.view(), &lhs.1, rhs.0.view(), &rhs.1, &gradient);
                            queue.push((lhs.2.clone(), lhs.1.clone()));
                            queue.push((rhs.2.clone(), rhs.1.clone()));
                        }
                        BackwardLabel::Div(lhs, rhs) => {
                            div_backward(lhs.0.view(), &lhs.1, rhs.0.view(), &rhs.1, &gradient);
                            queue.push((lhs.2.clone(), lhs.1.clone()));
                            queue.push((rhs.2.clone(), rhs.1.clone()));
                        }
                    }
                }
                None => continue,
            }
        }
    }
}
