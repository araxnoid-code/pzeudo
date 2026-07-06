use std::{
    cell::RefCell,
    fmt::{Debug, Display},
    marker::PhantomData,
    rc::Rc,
};

use ndarray::{ArrayD, ArrayViewD, CowArray, Dim, IxDynImpl};
use num_traits::Float;

use crate::{PzeudoStorageErr, StorageTrait};

/// Tensor Trait
/// This trait is specifically for ndarrays
pub trait TensorNDArray<F> {
    fn _view(&self) -> ArrayViewD<'_, F>;
}

/// tensor using ndarray
pub struct Tensor<F, A, GradStore>
where
    A: TensorNDArray<F>,
    F: Float,
    GradStore: StorageTrait<ArrayD<F>>,
{
    pub(crate) array: A,
    pub(crate) grad: Option<usize>,
    pub(crate) grad_storage: Rc<RefCell<GradStore>>,
    pub(crate) is_record: bool,
    _float_type: PhantomData<F>,
}

impl<F, A, GradStorage> Tensor<F, A, GradStorage>
where
    GradStorage: StorageTrait<ArrayD<F>>,
    A: TensorNDArray<F>,
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
            is_record: false,
            _float_type: PhantomData::default(),
        })
    }
}

impl<F, A, GradStorage> Display for Tensor<F, A, GradStorage>
where
    GradStorage: StorageTrait<ArrayD<F>>,
    A: TensorNDArray<F> + Display,
    F: Float,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("{}", self.array))
    }
}

// impl trait
impl<F> TensorNDArray<F> for ArrayD<F>
where
    F: Float,
{
    fn _view(&self) -> ArrayViewD<'_, F> {
        self.view()
    }
}

impl<F> TensorNDArray<F> for ArrayViewD<'_, F>
where
    F: Float,
{
    fn _view(&self) -> ArrayViewD<'_, F> {
        self.view()
    }
}

impl<F> TensorNDArray<F> for CowArray<'_, F, Dim<IxDynImpl>>
where
    F: Float,
{
    fn _view(&self) -> ArrayViewD<'_, F> {
        self.view()
    }
}
