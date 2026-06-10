use std::{ops::Add, process::Child};

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

// trait Animal {}

// trait IndonesiaCountry {}

// trait GajahSumatera {}

// trait HarimauSumatera {}

// trait AfricanCountry {}

// trait Kangguru {}

// trait HarimauSumatera {}
