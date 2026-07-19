use std::{cell::RefCell, error::Error, rc::Rc};

use pzeudo::{
    Array, ArrayRef, ArrayStorage, ArrayTrait, Contiguous, OpsAdd, OpsAddAssign, PzeudoErr, Tensor,
    TensorAddOps, TensorDivOps, TensorMulOps, TensorSubOps, TensorTrait,
};

fn main() -> Result<(), PzeudoErr> {
    let storage = Rc::new(RefCell::new(ArrayStorage::new(None)));
    let record = Rc::new(RefCell::new(Vec::new()));

    let tensor_a = Tensor::from_vector_with_shape(&[3.], &[1], storage.clone(), record.clone())?;
    let tensor_b = Tensor::from_vector_with_shape(&[2.], &[1], storage.clone(), record.clone())?;
    let tensor_c = Tensor::from_vector_with_shape(&[4.], &[1], storage.clone(), record.clone())?;

    let result_d = tensor_a.add(&tensor_b)?;

    let result_e = tensor_c.sub(&tensor_a)?;
    let result_f = result_e.mul(&result_d)?;
    let result_g = tensor_b.div(&result_e)?;
    let result_h = result_f.mul(&result_g)?;
    let result_i = tensor_a.sub(&result_h)?;
    let result_j = result_i.div(&tensor_c)?;
    // println!("{result_j}");
    result_j.backward()?;

    println!(
        "{}",
        storage
            .borrow()
            .get_as_array_ref::<Contiguous>(tensor_a.get_grad_idx().unwrap())
            .unwrap()
    );

    println!(
        "{}",
        storage
            .borrow()
            .get_as_array_ref::<Contiguous>(tensor_b.get_grad_idx().unwrap())
            .unwrap()
    );

    println!(
        "{}",
        storage
            .borrow()
            .get_as_array_ref::<Contiguous>(tensor_c.get_grad_idx().unwrap())
            .unwrap()
    );

    println!(
        "{}",
        storage
            .borrow()
            .get_as_array_ref::<Contiguous>(result_d.get_grad_idx().unwrap())
            .unwrap()
    );

    println!(
        "{}",
        storage
            .borrow()
            .get_as_array_ref::<Contiguous>(result_e.get_grad_idx().unwrap())
            .unwrap()
    );

    println!(
        "{}",
        storage
            .borrow()
            .get_as_array_ref::<Contiguous>(result_f.get_grad_idx().unwrap())
            .unwrap()
    );

    println!(
        "{}",
        storage
            .borrow()
            .get_as_array_ref::<Contiguous>(result_g.get_grad_idx().unwrap())
            .unwrap()
    );

    println!(
        "{}",
        storage
            .borrow()
            .get_as_array_ref::<Contiguous>(result_h.get_grad_idx().unwrap())
            .unwrap()
    );

    println!(
        "{}",
        storage
            .borrow()
            .get_as_array_ref::<Contiguous>(result_i.get_grad_idx().unwrap())
            .unwrap()
    );

    println!(
        "{}",
        storage
            .borrow()
            .get_as_array_ref::<Contiguous>(result_j.get_grad_idx().unwrap())
            .unwrap()
    );

    Ok(())
}

// fn main() {
// let storage = Rc::new(RefCell::new(ArrayStorage::new(None)));
// let record = Rc::new(RefCell::new(Vec::new()));

//     let shape = [3, 4];
//     let vec_a = (0..shape.iter().product::<usize>())
//         .map(|idx| idx as f32)
//         .collect::<Vec<f32>>();
//     let tensor_a =
//         Tensor::from_vector_with_shape(&vec_a, &shape, storage.clone(), record.clone()).unwrap();
//     println!("{}", tensor_a);

//     let shape = [3, 4];
//     let vec_b = (10..10 + shape.iter().product::<usize>())
//         .map(|idx| idx as f32)
//         .collect::<Vec<f32>>();
//     let tensor_b =
//         Tensor::from_vector_with_shape(&vec_b, &shape, storage.clone(), record.clone()).unwrap();
//     println!("{}", tensor_b);

//     let result_c = tensor_a.mul(&tensor_b).unwrap();
//     println!("{}", result_c);
//     result_c.backward().unwrap();
// }
