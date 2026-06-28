use ndarray::{ArrayD, ArrayViewD};

use crate::{BackwardLabel, Tensor, TensorTrait, sub};

impl<'a> Tensor<'a> {
    pub fn sub<Rhs>(&'a self, rhs: &'a Rhs) -> Tensor<'a>
    where
        Rhs: TensorTrait<'a>,
    {
        let sub = sub(self.array_view(), rhs.array_view());
        let grad = ArrayD::<f32>::zeros(sub.shape());
        let backward_label = BackwardLabel::Sub(
            (self.array_view(), self.share_gradient()),
            (rhs.array_view(), rhs.share_gradient()),
        );

        let tensor = Tensor::new(sub, Some(grad), Some(backward_label));
        tensor
    }
}
