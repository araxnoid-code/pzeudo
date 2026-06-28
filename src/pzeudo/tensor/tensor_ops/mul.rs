use ndarray::{ArrayD, ArrayViewD};

use crate::{BackwardLabel, Tensor, TensorTrait, mul};

impl<'a> Tensor<'a> {
    pub fn mul<Rhs>(&'a self, rhs: &'a Rhs) -> Tensor<'a>
    where
        Rhs: TensorTrait<'a>,
    {
        let mul = mul(self.array_view(), rhs.array_view());
        let grad = ArrayD::<f32>::zeros(mul.shape());
        let backward_label = BackwardLabel::Mul(
            self.array_view(),
            rhs.array_view(),
            self.share_gradient(),
            rhs.share_gradient(),
        );

        let tensor = Tensor::new(mul, Some(grad), Some(backward_label));
        tensor
    }
}
