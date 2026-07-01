use std::sync::Arc;

use crate::{BackwardLabel, TensorTrait, add_backward, div_backward, mul_backward, sub_backward};

pub trait Backward<'bacward_label, F>: TensorTrait<'bacward_label, F> {
    fn backward(&self, record: &Vec<Option<Arc<BackwardLabel<'bacward_label>>>>) {
        if let Err(_) = self.set_gradient_ones() {
            return;
        };

        for backward in record.iter().rev() {
            if let Some(backward_label) = backward {
                match &**backward_label {
                    BackwardLabel::Add(lhs, rhs, gradient) => {
                        add_backward(&lhs.1, &rhs.1, gradient);
                    }
                    BackwardLabel::Sub(lhs, rhs, gradient) => {
                        sub_backward(&lhs.1, &rhs.1, gradient);
                    }
                    BackwardLabel::Mul(lhs, rhs, gradient) => {
                        mul_backward(lhs.0.view(), &lhs.1, rhs.0.view(), &rhs.1, gradient);
                    }
                    BackwardLabel::Div(lhs, rhs, gradient) => {
                        div_backward(lhs.0.view(), &lhs.1, rhs.0.view(), &rhs.1, gradient);
                    }
                }
            }
        }
    }
}
