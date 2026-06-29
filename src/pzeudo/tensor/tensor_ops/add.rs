use std::{
    cell::{Ref, RefCell},
    rc::Rc,
    sync::Arc,
};

use ndarray::{ArrayD, ArrayViewD};

use crate::{BackwardLabel, Tensor, TensorTrait, add};

impl<'a> Tensor<'a> {
    pub fn add<Rhs>(
        &'a mut self,
        rhs: &'a mut Rhs,
        record: &mut Vec<Option<Arc<BackwardLabel<'a>>>>,
    ) -> Tensor<'a>
    where
        Rhs: TensorTrait<'a>,
    {
        if self.get_label_ops() {
            self.set_label_ops(true);
            record.push(self.get_share_backward_label());
        }

        if rhs.get_label_ops() {
            rhs.set_label_ops(true);
            record.push(rhs.get_share_backward_label());
        }

        let add = add(self.get_array_view(), rhs.get_array_view());
        let grad = ArrayD::<f32>::zeros(add.shape());
        let backward_label = Arc::new(BackwardLabel::Add(
            (
                self.get_array_view(),
                self.get_share_gradient(),
                self.get_share_backward_label(),
            ),
            (
                rhs.get_array_view(),
                rhs.get_share_gradient(),
                rhs.get_share_backward_label(),
            ),
        ));

        let tensor = Tensor {
            array: add,
            gradient: Some(Rc::new(RefCell::new(grad))),
            backward_label: Some(backward_label.clone()),
            label_ops: true,
        };

        record.push(Some(backward_label));

        tensor
    }
}
