use std::println;

use pzeudo::{
    Array, ArrayTrait, Contiguous, Module, OpsMatmul2DF32, OpsMatmul2DF64, OpsMatmulNDF32,
    OpsMatmulNDF64, OpsPermute, OpsSlicing, StorageType, TensorTrait, r,
};

fn main() {

    // let module = Module::new();

    // let shape = [2, 3, 4, 5];
    // let vec_a = (0..shape.iter().product::<usize>())
    //     .map(|idx| idx as f32)
    //     .collect::<Vec<f32>>();
    // let array_a = module
    //     .tensor_from_vector_with_shape(&vec_a, &shape)
    //     .unwrap();

    // let shape = [2, 3, 5, 6];
    // let vec_b = (0..shape.iter().product::<usize>())
    //     .map(|idx| idx as f32 + 100.)
    //     .collect::<Vec<f32>>();
    // let array_b = module
    //     .tensor_from_vector_with_shape(&vec_b, &shape)
    //     .unwrap();

    // let tensor_c = array_a.matmul_nd(&array_b).unwrap();
    // tensor_c.backward().unwrap();

    // let storeage = module.get_storage().borrow();
    // let tensor_c_result = module
    //     .get_storage()
    //     .borrow()
    //     .get_as_array_ref::<Contiguous>(
    //         array_a.get_grad_idx().unwrap(),
    //         pzeudo::ContiguousType::Grad,
    //     )
    //     .unwrap();

    // let shape = [2, 3, 4, 6];
    // let ones = Array::<f32>::ones(&shape);

    // let array_b = storeage
    //     .get_as_array_ref::<Contiguous>(array_b.get_array_idx(), pzeudo::ContiguousType::Arr)
    //     .unwrap();

    // let permute = array_b.permute(&[0, 1, 3, 2]).unwrap();
    // let check = ones.matmul_nd(&permute).unwrap();

    // let array_a = storeage
    //     .get_as_array_ref::<Contiguous>(array_a.get_array_idx(), pzeudo::ContiguousType::Arr)
    //     .unwrap();

    // let permute = array_a.permute(&[0, 1, 3, 2]).unwrap();
    // let check = permute.matmul_nd(&ones).unwrap();
}
