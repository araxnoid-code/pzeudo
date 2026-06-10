use std::fmt::Debug;

use crate::{
    Arr, PzeudoDataType,
    tensor::backend_conf::{self, arr},
};

///
pub trait PzeudoBackend<'s, A>
where
    A: Arr<'s> + 's,
{
    //
    type BackendArrType: Arr<'s>;
    fn backend() -> impl Debug;

    // desc
    fn get_array(&'s self) -> &'s <<Self as backend_conf::backend::PzeudoBackend<'s, A>>::BackendArrType as backend_conf::arr::Arr<'s>>::ArrType{
        self.get_backend_arr().get_array()
    }

    //
    fn get_backend_arr(&self) -> &Self::BackendArrType;
    fn get_backend_grad_as_mut(&mut self) -> &mut Option<Self::BackendArrType>;
    fn arr_into(arr: Self::BackendArrType, grad: bool) -> Self;

    //
    fn add(&self, rhs: &Self) -> Self
    where
        Self: Sized,
    {
        let lhs = self.get_backend_arr();
        let rhs = rhs.get_backend_arr();
        let output = lhs.add(&rhs);

        Self::arr_into(output, true)
    }

    fn sub(&self, rhs: &Self) -> Self
    where
        Self: Sized,
    {
        let lhs = self.get_backend_arr();
        let rhs = rhs.get_backend_arr();
        let output = lhs.sub(&rhs);

        Self::arr_into(output, true)
    }

    fn mul(&self, rhs: &Self) -> Self
    where
        Self: Sized,
    {
        let lhs = self.get_backend_arr();
        let rhs = rhs.get_backend_arr();
        let output = lhs.mul(&rhs);

        Self::arr_into(output, true)
    }

    fn div(&self, rhs: &Self) -> Self
    where
        Self: Sized,
    {
        let lhs = self.get_backend_arr();
        let rhs = rhs.get_backend_arr();
        let output = lhs.div(&rhs);

        Self::arr_into(output, true)
    }

    // dx/dy
    fn add_d(grad_arr: &mut Self, rhs: &mut Self, lhs: &mut Self)
    where
        Self: Sized,
    {
        if let Some(cr_grad) = grad_arr.get_backend_grad_as_mut() {
            if let Some(rhs_grad) = rhs.get_backend_grad_as_mut() {
                rhs_grad.add_to(cr_grad);
            }

            if let Some(lhs_grad) = lhs.get_backend_grad_as_mut() {
                lhs_grad.add_to(cr_grad);
            }
        }
    }

    fn sub_d(grad_arr: &mut Self, rhs: &mut Self, lhs: &mut Self)
    where
        Self: Sized,
    {
        if let Some(cr_grad) = grad_arr.get_backend_grad_as_mut() {
            if let Some(rhs_grad) = rhs.get_backend_grad_as_mut() {
                rhs_grad.add_to(cr_grad);
            }

            if let Some(lhs_grad) = lhs.get_backend_grad_as_mut() {
                let scalar = Self::BackendArrType::from_scalar(10.);
                lhs_grad.add_to(&cr_grad.mul(&scalar));
            }
        }
    }
}
