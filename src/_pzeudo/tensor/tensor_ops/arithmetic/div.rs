use std::{
    cell::RefCell,
    ops::Div,
    rc::Rc,
    sync::{Arc, atomic::AtomicBool},
};

use ndarray::{ArrayD, ArrayViewD};
use num_traits::Float;

use crate::{BackwardLabel, GradientStorage, PzeudoErr, Tensor, TensorTrait, div, mul};

pub trait PzeudoOpsDiv<'backward_label, F, Grad, GradStorage>:
    TensorTrait<'backward_label, F, ArrayD<F>, GradStorage>
where
    GradStorage: GradientStorage<ArrayD<F>>,
{
    fn div<Rhs>(
        &'backward_label self,
        rhs: &'backward_label Rhs,
        record: &mut Vec<Option<&BackwardLabel<'backward_label, F>>>,
    ) -> Result<Tensor<'backward_label, F, GradStorage>, PzeudoErr>
    where
        F: Div<Output = F> + Copy + Float,
        Rhs: TensorTrait<'backward_label, F, ArrayD<F>, GradStorage>,
    {
        let result = div(self.get_array_view(), rhs.get_array_view())?;
        let grad = ArrayD::<F>::zeros(result.shape());
        let idx_grad = self.get_storage().borrow_mut().push_grad(grad);

        if !self.get_label_ops() {
            self.set_label_ops(true);
            record.push(self.get_share_backward_label());
        }

        if !rhs.get_label_ops() {
            rhs.set_label_ops(true);
            record.push(rhs.get_share_backward_label());
        }

        let backward_label = BackwardLabel::Div(
            (self.get_array_view(), self.get_share_gradient()),
            (rhs.get_array_view(), rhs.get_share_gradient()),
            Some(idx_grad),
        );

        let tensor = Tensor {
            array: result,
            gradient: Some(idx_grad),
            backward_label: Some(backward_label),
            label_ops: AtomicBool::new(true),
            grad_storage: self.get_storage().clone(),
        };

        record.push(tensor.get_share_backward_label());

        Ok(tensor)
    }
}
