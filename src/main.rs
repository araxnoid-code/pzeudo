use std::{ops::Add, process::Child};

use ndarray::{
    Array2, ArrayBase, ArrayD, ArrayView, Dim, IxDynImpl, OwnedRepr, ViewRepr, linalg::Dot,
};
use pzeudo::{F64Base, NDArrayArr, NDArrayBackend, PzeudoBackend, PzeudoDataType, Tensor};

fn main() {
    let array_a: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>, f64> = ArrayD::<f64>::zeros(vec![2, 2]);

    let array_b: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>, f64> = ArrayD::<f64>::zeros(vec![2, 2]);

    let tensor_a: Tensor<NDArrayArr<F64Base>, NDArrayBackend<F64Base>> = Tensor::new(
        NDArrayBackend::new(NDArrayArr::new(F64Base::new(array_a)), None),
        None,
    );

    let tensor_b: Tensor<NDArrayArr<F64Base>, _> = Tensor::new(
        NDArrayBackend::new(NDArrayArr::new(F64Base::new(array_b)), None),
        None,
    );

    let tensor_c = tensor_a.add(&tensor_b);

    // tensor_c.add(rhs);

    // let tensor_a = Tensor::new(
    //     NDarrayBackend::new(F64Base::new(array_a, Some(ArrayD::zeros(vec![2, 2])))),
    //     None,
    // );
    // let tensor_b = Tensor::new(
    //     NDarrayBackend::new(F64Base::new(array_b, Some(ArrayD::zeros(vec![2, 2])))),
    //     None,
    // );
    // let tensor_c = tensor_a.add(&tensor_b);
}

// trait A {
//     type TypeData;
//     fn from_my_type(data: Self::TypeData);
// }

// trait B {
//     type Child: A;
//     fn execute(&self) {
//         let data = Self::Child::from_my_type(10.);
//     }
// }
