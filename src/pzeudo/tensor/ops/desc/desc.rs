use crate::{Arr, PzeudoBackend, PzeudoDataType, Tensor};

impl<'s, A, B> Tensor<'s, A, B>
where
    A: Arr<'s, ScalarType = PzeudoDataType> + 's,
    B: PzeudoBackend<'s, A>,
{
    // pub fn get_arr(
    //     &'s self,
    // ) -> &'s <<B as PzeudoBackend<'s, A>>::BackendArrType as Arr<'s>>::ArrType {
    // }

    // pub fn get_grad(&self) {}
}
