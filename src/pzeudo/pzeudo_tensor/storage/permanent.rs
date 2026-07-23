use crate::Array;

pub struct PermanentTensor<F> {
    pub(crate) array: Array<F>,
    pub(crate) grad: Array<F>,
}
