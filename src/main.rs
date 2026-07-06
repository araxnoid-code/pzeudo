use std::{
    cell::RefCell,
    ops::{Add, AddAssign},
    rc::Rc,
    sync::Arc,
};

use ndarray::{ArrayD, CowArray};
use pzeudo::{BackwardTrait, GradStorage, Tensor};

fn main() {
    // let num = Arc::new(1);
    // let a = num.add_assign(10);
    // println!("{}", num);

    let array = ArrayD::<f32>::ones(vec![10, 10]);
    let array_view = array.view();
    let array_cow = CowArray::from(&array);

    let grad_storage = Rc::new(RefCell::new(GradStorage::new(None)));
    let record_storage = Rc::new(RefCell::new(Vec::new()));
    // let tensor = Tensor::new(array, None, grad_storage.clone()).unwrap();
    let mut tensor_view = Tensor::new(
        array_view,
        None,
        grad_storage.clone(),
        None,
        record_storage.clone(),
    )
    .unwrap();

    let tensor_cow = Tensor::new(
        array_cow,
        None,
        grad_storage.clone(),
        None,
        record_storage.clone(),
    )
    .unwrap();

    let mut tensor_c = tensor_view.add(&tensor_cow).unwrap();
    let tensor_d = tensor_c.add(&tensor_view).unwrap();
    let tensor_e = tensor_d.add(&tensor_d).unwrap();
    tensor_e.backward().unwrap();
    // println!("{}", tensor_e);

    // for i in record_storage.borrow().iter() {
    //     // i.0.backward(gradient_idx, grad_storage)
    //     // println!("grad_idx {:?}", i.1);
    //     // println!("{}", i.0);
    //     // println!("===========")
    // }
}
