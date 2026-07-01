use std::{
    cell::RefCell,
    rc::Rc,
    sync::{Arc, atomic::AtomicBool},
};

use ndarray::ArrayD;

use crate::{BackwardLabel, PzeudoErr, Tensor, TensorTrait, sub};

impl<'backward_label> Tensor<'backward_label> {
    pub fn sub<Rhs>(
        &'backward_label self,
        rhs: &'backward_label Rhs,
        record: &mut Vec<Option<Arc<BackwardLabel<'backward_label>>>>,
    ) -> Result<Tensor<'backward_label>, PzeudoErr>
    where
        Rhs: TensorTrait<'backward_label>,
    {
        let result = sub(self.get_array_view(), rhs.get_array_view())?;
        let grad = Rc::new(RefCell::new(ArrayD::<f32>::zeros(result.shape())));

        if !self.get_label_ops() {
            self.set_label_ops(true);
            record.push(self.get_share_backward_label());
        }

        if !rhs.get_label_ops() {
            rhs.set_label_ops(true);
            record.push(rhs.get_share_backward_label());
        }

        let backward_label = Arc::new(BackwardLabel::Sub(
            (self.get_array_view(), self.get_share_gradient()),
            (rhs.get_array_view(), rhs.get_share_gradient()),
            Some(grad.clone()),
        ));

        let tensor = Tensor {
            array: result,
            gradient: Some(grad),
            backward_label: Some(backward_label),
            label_ops: AtomicBool::new(true),
        };

        record.push(tensor.get_share_backward_label());

        Ok(tensor)
    }
}
