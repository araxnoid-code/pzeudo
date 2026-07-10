use std::{
    cell::RefCell,
    fmt::Display,
    ops::{AddAssign, Div, Neg},
    rc::Rc,
    slice::Iter,
};

use ndarray::{Array2, ArrayD, ArrayView2, linalg::Dot};
use num_traits::{Float, One, Zero};

use crate::{NDArray, OpsLabel, PzeudoOpsErr, StorageTrait, Tensor};

pub trait BackwardTrait<'ops_label, F, GradStorage>
where
    F: AddAssign
        + Clone
        + Zero
        + Div<Output = F>
        + Copy
        + One
        + Neg<Output = F>
        + 'ops_label
        + Display,
    for<'a> ArrayView2<'a, F>: Dot<ArrayView2<'a, F>, Output = Array2<F>>,
    GradStorage: StorageTrait<ArrayD<F>>,
{
    fn grad_to_ones(&self);

    fn get_record(&'ops_label self) -> Rc<RefCell<Vec<(OpsLabel<'ops_label, F>, Option<usize>)>>>;

    fn get_storage(&self) -> Rc<RefCell<GradStorage>>;

    fn backward(&'ops_label self) -> Result<(), PzeudoOpsErr> {
        self.grad_to_ones();

        let record = self.get_record();
        let mut record_mut = record.borrow_mut();

        let storage = self.get_storage();
        let mut storage_mut = storage.borrow_mut();
        for (record, grad_idx) in record_mut.iter().rev() {
            record.backward(*grad_idx, &mut *storage_mut)?;
        }

        record_mut.clear();
        Ok(())
    }
}

impl<'ops_label, F, A, GradStorage> BackwardTrait<'ops_label, F, GradStorage>
    for Tensor<'ops_label, F, A, GradStorage>
where
    A: NDArray<F>,
    F: AddAssign + Clone + Zero + Div<Output = F> + Copy + One + Neg<Output = F> + Float + Display,
    for<'a> ArrayView2<'a, F>: Dot<ArrayView2<'a, F>, Output = Array2<F>>,
    GradStorage: StorageTrait<ArrayD<F>>,
{
    fn grad_to_ones(&self) {
        if let Some(grad_idx) = self.grad {
            let ones = ArrayD::<F>::ones(self.array.shape());
            *self
                .grad_storage
                .borrow_mut()
                .get_mut_storage()
                .get_mut(grad_idx)
                .as_mut()
                .unwrap()
                .as_mut()
                .unwrap() = ones;
        }
    }

    fn get_record(&'ops_label self) -> Rc<RefCell<Vec<(OpsLabel<'ops_label, F>, Option<usize>)>>> {
        self.record_storage.clone()
    }

    fn get_storage(&self) -> Rc<RefCell<GradStorage>> {
        self.grad_storage.clone()
    }
}
