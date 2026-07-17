use crate::ArrayView;

impl<'a, F> ArrayView<'a, F> {
    pub fn new(
        data: &'a [F],
        offset: usize,
        shape: Vec<usize>,
        stride: Vec<usize>,
    ) -> ArrayView<'a, F> {
        Self {
            data,
            offset,
            shape,
            stride,
        }
    }
}
