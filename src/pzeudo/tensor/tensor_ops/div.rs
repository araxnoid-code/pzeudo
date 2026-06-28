use ndarray::{ArrayD, ArrayViewD};

use crate::{BackwardLabel, Tensor, TensorTrait, div, mul};

impl<'a> Tensor<'a> {
    pub fn div<Rhs>(&'a self, rhs: &'a Rhs) -> Tensor<'a>
    where
        Rhs: TensorTrait<'a>,
    {
        let div = div(self.array_view(), rhs.array_view());
        let grad = ArrayD::<f32>::zeros(div.shape());
        let backward_label = BackwardLabel::Div(
            (self.array_view(), self.share_gradient()),
            (rhs.array_view(), rhs.share_gradient()),
        );

        let tensor = Tensor::new(div, Some(grad), Some(backward_label));
        tensor
    }
}
