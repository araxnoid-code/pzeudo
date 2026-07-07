use std::{
    cell::RefCell,
    ops::{Add, AddAssign},
    rc::Rc,
    sync::Arc,
};

use ndarray::{ArrayD, CowArray, array};
use pzeudo::{BackwardTrait, GradStorage, StorageTrait, Tensor};

fn main() {
    let storage = Rc::new(RefCell::new(GradStorage::new(None)));
    let record = Rc::new(RefCell::new(vec![]));

    let array_a =
        Tensor::from_array(array![5.].into_dyn(), storage.clone(), record.clone()).unwrap();
    let array_b =
        Tensor::from_array(array![4.].into_dyn(), storage.clone(), record.clone()).unwrap();
    let array_c =
        Tensor::from_array(array![6.].into_dyn(), storage.clone(), record.clone()).unwrap();

    let array_d = Tensor::from_array(array![2.].into_dyn(), storage.clone(), record).unwrap();

    let ops_a = array_a.add(&array_c).unwrap();
    let ops_b = array_b.mul(&array_d).unwrap();

    let ops_c = ops_a.div(&ops_b).unwrap();

    let ops_d = ops_b.sub(&ops_c).unwrap();

    // println!("{}", ops_b);

    ops_d.backward().unwrap();

    let grad_idx = array_a.get_grad_idx().unwrap();
    let _storage = storage.borrow_mut();
    let storage = _storage
        .get_storage()
        .get(grad_idx)
        .unwrap()
        .as_ref()
        .unwrap();
    println!("{}", storage);
}
