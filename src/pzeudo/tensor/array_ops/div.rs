use ndarray::{ArrayBase, ArrayView, ArrayViewD, Dim, IxDynImpl, OwnedRepr};

pub fn div(
    lhs: ArrayViewD<f32>,
    rhs: ArrayViewD<f32>,
) -> ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>, f32> {
    &lhs / &rhs
}
