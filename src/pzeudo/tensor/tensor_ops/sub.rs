use std::{
    cell::RefCell,
    rc::Rc,
    sync::{Arc, atomic::AtomicBool},
};

use ndarray::{ArrayD, ArrayViewD};

use crate::{BackwardLabel, Tensor, TensorTrait, sub};

impl<'a> Tensor<'a> {
    pub fn sub<Rhs>(
        &'a self,
        rhs: &'a Rhs,
        record: &mut Vec<Option<Arc<BackwardLabel<'a>>>>,
    ) -> Tensor<'a>
    where
        Rhs: TensorTrait<'a>,
    {
        if !self.get_label_ops() {
            self.set_label_ops(true);
            record.push(self.get_share_backward_label());
        }

        if !rhs.get_label_ops() {
            rhs.set_label_ops(true);
            record.push(rhs.get_share_backward_label());
        }

        let result = sub(self.get_array_view(), rhs.get_array_view());
        let grad = Rc::new(RefCell::new(ArrayD::<f32>::zeros(result.shape())));
        let backward_label = Arc::new(BackwardLabel::Div(
            (self.get_array_view(), self.get_share_gradient()),
            (rhs.get_array_view(), rhs.get_share_gradient()),
            Some(grad.clone()),
        ));

        let tensor = Tensor {
            array: result,
            gradient: Some(grad),
            backward_label: Some(backward_label.clone()),
            label_ops: AtomicBool::new(true),
        };

        record.push(Some(backward_label));

        tensor
    }
}
