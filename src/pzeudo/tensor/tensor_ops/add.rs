use ndarray::{ArrayD, ArrayViewD};

use crate::{BackwardLabel, Tensor, TensorTrait, add};

impl<'a> Tensor<'a> {
    pub fn add<Rhs>(&'a self, rhs: &'a Rhs) -> Tensor<'a>
    where
        Rhs: TensorTrait<'a>,
    {
        let add = add(self.array_view(), rhs.array_view());
        let grad = ArrayD::<f32>::zeros(add.shape());
        let backward_label = BackwardLabel::Add(
            (self.array_view(), self.share_gradient()),
            (rhs.array_view(), rhs.share_gradient()),
        );

        let tensor = Tensor::new(add, Some(grad), Some(backward_label));
        tensor
    }
}
