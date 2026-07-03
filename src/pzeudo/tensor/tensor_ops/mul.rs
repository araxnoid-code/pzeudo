use std::{
    cell::RefCell,
    rc::Rc,
    sync::{Arc, atomic::AtomicBool},
};

use ndarray::ArrayD;
use num_traits::Float;

use crate::{BackwardLabel, PzeudoErr, Tensor, TensorTrait, mul};

pub trait PzeudoOpsMul<'backward_label, F>: TensorTrait<'backward_label, F> {
    fn mul<Rhs>(
        &'backward_label self,
        rhs: &'backward_label Rhs,
        record: &mut Vec<Option<Arc<BackwardLabel<'backward_label, F>>>>,
    ) -> Result<Tensor<'backward_label, F>, PzeudoErr>
    where
        Rhs: TensorTrait<'backward_label, F>,
        F: Float,
    {
        let result = mul(self.get_array_view(), rhs.get_array_view())?;
        let grad = Rc::new(RefCell::new(ArrayD::<F>::zeros(result.shape())));

        if !self.get_label_ops() {
            self.set_label_ops(true);
            record.push(self.get_share_backward_label());
        }

        if !rhs.get_label_ops() {
            rhs.set_label_ops(true);
            record.push(rhs.get_share_backward_label());
        }

        let backward_label = Arc::new(BackwardLabel::Mul(
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
