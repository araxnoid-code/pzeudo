use ndarray::{ArrayD, ArrayViewD, CowArray, Dim, IxDynImpl};
use num_traits::Float;

/// NDarray Trait
/// This trait is specifically for ndarrays
pub trait NDArray<F> {
    fn _view(&self) -> ArrayViewD<'_, F>;
    fn shape(&self) -> &[usize];
}

// IMPL
impl<F> NDArray<F> for ArrayD<F>
where
    F: Float,
{
    fn _view(&self) -> ArrayViewD<'_, F> {
        self.view()
    }

    fn shape(&self) -> &[usize] {
        self.shape()
    }
}

impl<F> NDArray<F> for ArrayViewD<'_, F>
where
    F: Float,
{
    fn _view(&self) -> ArrayViewD<'_, F> {
        self.view()
    }

    fn shape(&self) -> &[usize] {
        self.shape()
    }
}

impl<F> NDArray<F> for CowArray<'_, F, Dim<IxDynImpl>>
where
    F: Float,
{
    fn _view(&self) -> ArrayViewD<'_, F> {
        self.view()
    }

    fn shape(&self) -> &[usize] {
        self.shape()
    }
}
