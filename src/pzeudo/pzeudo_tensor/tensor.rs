use crate::prelude::*;
use num_traits::{Float, One};
use std::{cell::RefCell, format, iter::Sum, marker::PhantomData, ops::AddAssign, rc::Rc};

pub struct Tensor<F, T> {
    pub(crate) array_idx: StorageType,
    pub(crate) grad_idx: Option<StorageType>,
    pub(crate) record: Rc<RefCell<Vec<RecordLabel>>>,
    pub(crate) storage: Rc<RefCell<ArrayStorage<F>>>,
    pub(crate) _array_type: PhantomData<T>,
}

impl<F, T> Tensor<F, T> {
    pub fn backward(&self) -> Result<(), PzeudoErr>
    where
        ArrayStorage<F>: StorageF32F64,
        for<'a> F: Clone + One + AddAssign + Float + Sum<&'a F>,
        for<'a> ArrayRef<'a, F, Contiguous>: OpsBroadcast<F>,
        for<'a> ArrayRef<'a, F, View>: OpsBroadcast<F>,
    {
        let mut storage = self.storage.borrow_mut();
        if let Some(grad_idx) = self.grad_idx {
            match storage.get_element_mut(grad_idx)? {
                GetElementMutOutput::Permanent(permanent) => {
                    permanent.grad.to_one();
                }

                GetElementMutOutput::Storage(element) => match element {
                    ElementType::View(_, _) => {
                        return Err(PzeudoErr::BackwardErr(format!(
                            "Tensor::backward. The gradient index on a tensor has a value of {grad_idx:?} which points to an element of the View data type. The gradient of a tensor must be contiguous."
                        )));
                    }
                    ElementType::Contiguous(array, contiguous_type) => {
                        if let ContiguousType::Arr = contiguous_type {
                            return Err(PzeudoErr::BackwardErr(format!(
                                "Tensor::backward. The gradient index on the tensor points to {grad_idx:?} which is a contiguous element of type Contiguous::Arr, the gradient index must point to Contiguous::Grad"
                            )));
                        }
                        let ones = Array::<F>::ones(&array.shape);
                        *array = ones;
                    }
                },
            }
        }

        let mut record = self.record.borrow_mut();

        for record in record.iter().rev() {
            record.backward(&mut storage)?;
        }

        record.clear();

        Ok(())
    }
}
