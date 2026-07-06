use std::{
    cell::RefCell,
    ops::{AddAssign, Div, Neg},
    rc::Rc,
    slice::Iter,
};

use ndarray::ArrayD;
use num_traits::{Float, One, Zero};

use crate::{OpsLabel, PzeudoOpsErr, StorageTrait, Tensor, TensorNDArray};

pub trait BackwardTrait<'ops_label, F, GradStorage>
where
    F: AddAssign + Clone + Zero + Div<Output = F> + Copy + One + Neg<Output = F> + 'ops_label,
    GradStorage: StorageTrait<ArrayD<F>>,
{
    fn get_record(&'ops_label self) -> Rc<RefCell<Vec<(OpsLabel<'ops_label, F>, Option<usize>)>>>;

    fn get_storage(&self) -> Rc<RefCell<GradStorage>>;

    fn backward(&'ops_label self) -> Result<(), PzeudoOpsErr> {
        let record = self.get_record();
        let mut record_mut = record.borrow_mut();

        let storage = self.get_storage();
        let mut storage_mut = storage.borrow_mut();
        for (record, grad_idx) in record_mut.iter() {
            record.backward(*grad_idx, &mut *storage_mut)?;
        }

        record_mut.clear();
        Ok(())
    }
}

impl<'ops_label, F, A, GradStorage> BackwardTrait<'ops_label, F, GradStorage>
    for Tensor<'ops_label, F, A, GradStorage>
where
    A: TensorNDArray<F>,
    F: AddAssign + Clone + Zero + Div<Output = F> + Copy + One + Neg<Output = F> + Float,
    GradStorage: StorageTrait<ArrayD<F>>,
{
    fn get_record(&'ops_label self) -> Rc<RefCell<Vec<(OpsLabel<'ops_label, F>, Option<usize>)>>> {
        self.record_storage.clone()
    }

    fn get_storage(&self) -> Rc<RefCell<GradStorage>> {
        self.grad_storage.clone()
    }
}
