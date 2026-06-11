use std::{
    marker::PhantomData,
    ops::Add,
    process::Child,
    sync::{Arc, Mutex, MutexGuard},
};

use ndarray::{
    Array2, ArrayBase, ArrayD, ArrayView, Dim, IxDynImpl, OwnedRepr, ViewRepr, linalg::Dot,
};
use pzeudo::{F64Base, NDArrayArr, NDArrayBackend, PzeudoBackend, PzeudoDataType, Tensor};

fn main() {
    let array_a: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>, f64> = ArrayD::<f64>::zeros(vec![2, 2]);

    let array_b: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>, f64> = ArrayD::<f64>::zeros(vec![2, 2]);

    let f64_base: F64Base = F64Base::new(array_a);

    let ndarray_arr: NDArrayArr<F64Base> = NDArrayArr::new(f64_base);

    let backend: NDArrayBackend<'_, NDArrayArr<F64Base>> = NDArrayBackend::new(ndarray_arr, None);

    let tensor: Tensor<'_, NDArrayArr<F64Base>, NDArrayBackend<'_, NDArrayArr<F64Base>>> =
        Tensor::new(backend, None);
}

struct Penduduk<'PendudukLT> {
    nama: String,
    identity_card: i32,
    _phantom: PhantomData<&'PendudukLT i32>,
}

//
struct Data<'PendudukLT, P: PendudukTrait<'PendudukLT>> {
    penduduk: Arc<Mutex<P>>,
    _phantom: PhantomData<&'PendudukLT i32>,
}

impl<'PendudukLT, P: PendudukTrait<'PendudukLT, IdentityCrad = &'PendudukLT i32>>
    Data<'PendudukLT, P>
{
    pub fn pinjam_identifier(&'PendudukLT self) -> &Arc<Mutex<P>> {
        &self.penduduk
    }
}

//

trait PendudukTrait<'PendudukLT> {
    type IdentityCrad;
    fn pinjamkan_identity_card(&'PendudukLT self) -> Self::IdentityCrad;
}

impl<'PendudukLT> PendudukTrait<'PendudukLT> for Penduduk<'PendudukLT> {
    type IdentityCrad = &'PendudukLT i32;
    fn pinjamkan_identity_card(&'PendudukLT self) -> Self::IdentityCrad {
        &self.identity_card
    }
}
