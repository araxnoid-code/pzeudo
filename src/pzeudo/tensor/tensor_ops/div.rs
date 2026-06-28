use ndarray::{ArrayD, ArrayViewD};

use crate::{BackwardLabel, Tensor, TensorTrait, div, mul};

impl<'a> Tensor<'a> {
    pub fn div<Rhs>(&'a self, rhs: &'a Rhs) -> Tensor<'a>
    where
        Rhs: TensorTrait<'a>,
    {
        let div = div(self.get_array_view(), rhs.get_array_view());
        let grad = ArrayD::<f32>::zeros(div.shape());
        let backward_label = BackwardLabel::Div(
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

        let tensor = Tensor::new(div, Some(grad), Some(backward_label));
        tensor
    }
}
