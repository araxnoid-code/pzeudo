use std::{cell::RefCell, fmt::Display, ops::AddAssign, rc::Rc};

use ndarray::{ArrayD, ArrayView, ArrayViewD, ArrayViewMutD};
use num_traits::{Float, Zero};

use crate::{PzeudoBackwardErr, StorageTrait, add_backward};

pub enum OpsLabel<'ops_label, F> {
    Init,
    // Arith
    Add(
        (ArrayViewD<'ops_label, F>, Option<usize>),
        (ArrayViewD<'ops_label, F>, Option<usize>),
    ),
    Sub(
        (ArrayViewD<'ops_label, F>, Option<usize>),
        (ArrayViewD<'ops_label, F>, Option<usize>),
    ),
    Mul(
        (ArrayViewD<'ops_label, F>, Option<usize>),
        (ArrayViewD<'ops_label, F>, Option<usize>),
    ),
    Div(
        (ArrayViewD<'ops_label, F>, Option<usize>),
        (ArrayViewD<'ops_label, F>, Option<usize>),
    ),
}

impl<'ops_label, F> OpsLabel<'ops_label, F> {
    fn backward<GradStorage>(
        self,
        gradient: ArrayViewD<F>,
        grad_storage: Rc<RefCell<GradStorage>>,
    ) -> Result<(), &'static str>
    where
        GradStorage: StorageTrait<ArrayD<F>>,
        F: AddAssign + Clone + Zero,
    {
        let mut borrow = grad_storage.borrow_mut();
        let storage = borrow.get_mut_storage();
        let storage_len = storage.len();
        match self {
            Self::Add(lhs, rhs) => {}
            _ => (),
        }

        Ok(())
    }
}

impl<'ops_label, F> Display for OpsLabel<'ops_label, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpsLabel::Init => f.write_str("OpsLabel: Init"),
            // Arith
            Self::Add(lhs, rhs) => f.write_str(&format!(
                "OpsLabel: Add | Lhs_grad_idx: {:?} | Rhs_grad_idx: {:?}",
                lhs.1, rhs.1
            )),
            Self::Sub(lhs, rhs) => f.write_str(&format!(
                "OpsLabel: Sub | Lhs_grad_idx: {:?} | Rhs_grad_idx: {:?}",
                lhs.1, rhs.1
            )),
            Self::Mul(lhs, rhs) => f.write_str(&format!(
                "OpsLabel: Mul | Lhs_grad_idx: {:?} | Rhs_grad_idx: {:?}",
                lhs.1, rhs.1
            )),
            Self::Div(lhs, rhs) => f.write_str(&format!(
                "OpsLabel: Div | Lhs_grad_idx: {:?} | Rhs_grad_idx: {:?}",
                lhs.1, rhs.1
            )),
        }
    }
}
