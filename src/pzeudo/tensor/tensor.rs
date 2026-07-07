use std::{
    cell::RefCell,
    fmt::{Debug, Display},
    marker::PhantomData,
    rc::Rc,
    sync::atomic::AtomicBool,
};

use ndarray::{ArrayD, ArrayViewD, CowArray, Dim, IxDynImpl};
use num_traits::Float;

use crate::{OpsLabel, PzeudoStorageErr, StorageTrait};

/// Tensor Trait
/// This trait is specifically for ndarrays
pub trait TensorNDArray<F> {
    fn _view(&self) -> ArrayViewD<'_, F>;
    fn shape(&self) -> &[usize];
}

/// tensor using ndarray
pub struct Tensor<'ops_label, F, A, GradStore>
where
    A: TensorNDArray<F>,
    F: Float,
    GradStore: StorageTrait<ArrayD<F>>,
{
    pub(crate) array: A,

    pub(crate) grad: Option<usize>,
    pub(crate) grad_storage: Rc<RefCell<GradStore>>,

    pub(crate) record: Option<usize>,
    pub(crate) record_storage: Rc<RefCell<Vec<(OpsLabel<'ops_label, F>, Option<usize>)>>>,
    _float_type: PhantomData<F>,
}

impl<'ops_label, F, A, GradStorage> Tensor<'ops_label, F, A, GradStorage>
where
    A: TensorNDArray<F>,
    F: Float,
    GradStorage: StorageTrait<ArrayD<F>>,
{
    pub fn new(
        array: A,

        gradient: Option<ArrayD<F>>,
        grad_storage: Rc<RefCell<GradStorage>>,

        record: Option<OpsLabel<'ops_label, F>>,
        record_storage: Rc<RefCell<Vec<(OpsLabel<'ops_label, F>, Option<usize>)>>>,
    ) -> Result<Tensor<'ops_label, F, A, GradStorage>, PzeudoStorageErr> {
        let mut grad_idx = None;
        Ok(Self {
            array,
            grad: gradient.map_or(Ok(None::<usize>), |grad| {
                grad_idx = Some(grad_storage.borrow_mut().push_element(grad)?);
                Ok(grad_idx)
            })?,
            grad_storage,

            record: record.map(|record| {
                let mut borrow = record_storage.borrow_mut();
                borrow.push((record, grad_idx));
                borrow.len()
            }),
            record_storage,
            _float_type: PhantomData::default(),
        })
    }

    pub fn from_array(
        array: A,
        grad_storage: Rc<RefCell<GradStorage>>,
        record_storage: Rc<RefCell<Vec<(OpsLabel<'ops_label, F>, Option<usize>)>>>,
    ) -> Result<Tensor<'ops_label, F, A, GradStorage>, PzeudoStorageErr> {
        let gradient = ArrayD::<F>::zeros(array.shape());
        let grad_idx = grad_storage.borrow_mut().push_element(gradient)?;

        Ok(Self {
            array,
            grad: Some(grad_idx),
            grad_storage: grad_storage,
            record: None,
            record_storage,
            _float_type: Default::default(),
        })
    }

    pub fn get_grad_idx(&self) -> Option<usize> {
        self.grad
    }
}

impl<'ops_label, F, A, GradStorage> Display for Tensor<'ops_label, F, A, GradStorage>
where
    for<'a> GradStorage: StorageTrait<ArrayD<F>>,
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

    fn shape(&self) -> &[usize] {
        self.shape()
    }
}

impl<F> TensorNDArray<F> for ArrayViewD<'_, F>
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

impl<F> TensorNDArray<F> for CowArray<'_, F, Dim<IxDynImpl>>
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
