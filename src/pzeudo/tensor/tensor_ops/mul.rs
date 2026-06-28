use ndarray::{ArrayD, ArrayViewD};

use crate::{BackwardLabel, Tensor, TensorTrait, mul};

impl<'a> Tensor<'a> {
    pub fn mul<Rhs>(&'a self, rhs: &'a Rhs) -> Tensor<'a>
    where
        Rhs: TensorTrait<'a>,
    {
        let mul = mul(self.get_array_view(), rhs.get_array_view());
        let grad = ArrayD::<f32>::zeros(mul.shape());
        let backward_label = BackwardLabel::Mul(
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

        let tensor = Tensor::new(mul, Some(grad), Some(backward_label));
        tensor
    }
}
