use ndarray::{ArrayD, ArrayViewD};

use crate::{BackwardLabel, Tensor, TensorTrait, add};

impl<'a> Tensor<'a> {
    pub fn add<Rhs>(&'a self, rhs: &'a Rhs) -> Tensor<'a>
    where
        Rhs: TensorTrait<'a>,
    {
        let add = add(self.get_array_view(), rhs.get_array_view());
        let grad = ArrayD::<f32>::zeros(add.shape());
        let backward_label = BackwardLabel::Add(
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

        let tensor = Tensor::new(add, Some(grad), Some(backward_label));
        tensor
    }
}
