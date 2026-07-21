use crate::Array;

pub struct UpdateAbleTensor<F> {
    pub(crate) array: Array<F>,
    pub(crate) grad: Array<F>,
}
