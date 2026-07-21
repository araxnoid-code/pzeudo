use crate::prelude::*;
use num_traits::{Float, One};
use std::{cell::RefCell, marker::PhantomData, ops::AddAssign, rc::Rc};

pub struct Tensor<F, T> {
    pub(crate) array_idx: usize,
    pub(crate) grad_idx: Option<usize>,
    pub(crate) record: Rc<RefCell<Vec<RecordLabel>>>,
    pub(crate) storage: Rc<RefCell<ArrayStorage<F>>>,
    pub(crate) _array_type: PhantomData<T>,
}

impl<F, T> Tensor<F, T> {
    pub fn backward(&self) -> Result<(), PzeudoErr>
    where
        F: Clone + One + AddAssign + Float,
    {
        let mut storage = self.storage.borrow_mut();
        if let Some(grad_idx) = self.grad_idx {
            match storage.get_element_mut(grad_idx)? {
                ElementType::Contiguous(array) => {
                    let ones = Array::<F>::ones(&array.shape);
                    *array = ones;
                }
                ElementType::UpdateableTensor(p_idx) => {
                    let p_idx = *p_idx;
                    let permanent = storage.get_mut_update_able_storage();
                    let array = permanent.get_mut(p_idx).unwrap();
                    let ones = Array::<F>::ones(&array.array.shape);
                    array.grad = ones;
                }
                ElementType::View(_, _) => {
                    return Err(PzeudoErr::BackwardErr(format!(
                        "Tensor::BackwardErr. The gradient index on a tensor has a value of {grad_idx} which points to an element of the View data type. The gradient of a tensor must be contiguous."
                    )));
                }
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
