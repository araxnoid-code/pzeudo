use ndarray::{ArrayD, ArrayViewD};

use crate::{BackwardLabel, Tensor, TensorTrait, sub};

impl<'a> Tensor<'a> {
    pub fn sub<Rhs>(&'a self, rhs: &'a Rhs) -> Tensor<'a>
    where
        Rhs: TensorTrait<'a>,
    {
        let sub = sub(self.get_array_view(), rhs.get_array_view());
        let grad = ArrayD::<f32>::zeros(sub.shape());
        let backward_label = BackwardLabel::Sub(
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
        );

        let tensor = Tensor::new(sub, Some(grad), Some(backward_label));
        tensor
    }
}
