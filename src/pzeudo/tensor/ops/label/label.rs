use std::{
    cell::RefCell,
    fmt::Display,
    ops::{AddAssign, Div, Neg},
    rc::Rc,
};

use ndarray::{Array2, ArrayD, ArrayView, ArrayView2, ArrayViewD, ArrayViewMutD, linalg::Dot};
use num_traits::{Float, One, Zero};

use crate::{
    PzeudoOpsErr, StorageTrait, add_backward, div_backward, matmul_2d_backward, mul_backward,
    sub_backward,
};

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

    // Matmul
    Matmul2d(
        (ArrayViewD<'ops_label, F>, Option<usize>),
        (ArrayViewD<'ops_label, F>, Option<usize>),
    ),
}

impl<'ops_label, F> OpsLabel<'ops_label, F> {
    pub fn backward<GradStorage>(
        &self,
        gradient_idx: Option<usize>,
        storage: &mut GradStorage,
    ) -> Result<(), PzeudoOpsErr>
    where
        GradStorage: StorageTrait<ArrayD<F>>,
        for<'a> ArrayView2<'a, F>: Dot<ArrayView2<'a, F>, Output = Array2<F>>,
        F: AddAssign + Clone + Zero + Div<Output = F> + Copy + One + Neg<Output = F> + Display,
    {
        match self {
            OpsLabel::Init => (),
            Self::Add(lhs, rhs) => {
                add_backward(lhs.1, rhs.1, gradient_idx, &mut *storage)?;
            }
            OpsLabel::Sub(lhs, rhs) => {
                sub_backward(lhs.1, rhs.1, gradient_idx, &mut *storage)?;
            }
            OpsLabel::Div(lhs, rhs) => {
                // println!("{}", rhs.0);
                div_backward(
                    lhs.0.view(),
                    lhs.1,
                    rhs.0.view(),
                    rhs.1,
                    gradient_idx,
                    &mut *storage,
                )?
            }
            OpsLabel::Mul(lhs, rhs) => mul_backward(
                lhs.0.view(),
                lhs.1,
                rhs.0.view(),
                rhs.1,
                gradient_idx,
                &mut *storage,
            )?,

            OpsLabel::Matmul2d(lhs, rhs) => matmul_2d_backward(
                lhs.0.view(),
                lhs.1,
                rhs.0.view(),
                rhs.1,
                gradient_idx,
                &mut *storage,
            )?,
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

            // Matmul
            Self::Matmul2d(lhs, rhs) => f.write_str(&format!(
                "OpsLabel: Div | Lhs_grad_idx: {:?} | Rhs_grad_idx: {:?}",
                lhs.1, rhs.1
            )),
        }
    }
}
