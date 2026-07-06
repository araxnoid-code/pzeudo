use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use ndarray::{ArrayD, ArrayViewD, CowArray, Dim, IxDynImpl};
use num_traits::Float;

use crate::{PzeudoStorageErr, StorageTrait};

pub struct Tensor<F, A, GradStore>
where
    A: NDArray<F>,
    F: Float,
    GradStore: StorageTrait<ArrayD<F>>,
{
    array: A,
    grad: Option<usize>,
    grad_storage: Rc<RefCell<GradStore>>,
    _float_type: PhantomData<F>,
}

impl<F, A, GradStorage> Tensor<F, A, GradStorage>
where
    GradStorage: StorageTrait<ArrayD<F>>,
    A: NDArray<F>,
    F: Float,
{
    pub fn new(
        array: A,
        gradient: Option<ArrayD<F>>,
        grad_storage: Rc<RefCell<GradStorage>>,
    ) -> Result<Tensor<F, A, GradStorage>, PzeudoStorageErr> {
        Ok(Self {
            array,
            grad: gradient.map_or(Ok(None::<usize>), |grad| {
                Ok(Some(grad_storage.borrow_mut().push_element(grad)?))
            })?,
            grad_storage,
            _float_type: PhantomData::default(),
        })
    }
}

// trait
pub trait NDArray<F> {
    fn _view(&self) -> ArrayViewD<'_, F>;
}

impl<F> NDArray<F> for ArrayD<F>
where
    F: Float,
{
    fn _view(&self) -> ArrayViewD<'_, F> {
        self.view()
    }
}

impl<F> NDArray<F> for ArrayViewD<'_, F>
where
    F: Float,
{
    fn _view(&self) -> ArrayViewD<'_, F> {
        self.view()
    }
}

impl<F> NDArray<F> for CowArray<'_, F, Dim<IxDynImpl>>
where
    F: Float,
{
    fn _view(&self) -> ArrayViewD<'_, F> {
        self.view()
    }
}
