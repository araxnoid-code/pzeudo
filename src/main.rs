use pzeudo::{Array, ArrayTrait, OpsAdd, OpsMatmul2DF64, shape_to_stride};

fn main() {
    let array_a = Array::from_vector_with_shape(
        &vec![0., 0., 0., 0., 0., 0., 1., 2., 3., 4., 5., 6.],
        &[2, 2, 3],
    )
    .unwrap();
    let array_b = Array::from_vector_with_shape(&vec![1., 2., 3., 4., 5., 6.], &[3, 2]).unwrap();

    let index_a = array_a.index(&[1]).unwrap();
    // let index_b = array_b.index(&[1]).unwrap();

    println!("{}", index_a.to_string().unwrap());
    // println!("{}", index_b.to_string().unwrap());

    let array_c = index_a.matmul_2d(&array_b).unwrap();
    println!("{}", array_c.to_string());

    // let array_c = array_a.matmul_2d(&array_b).unwrap();
    // println!("{}", array_c.to_string());

    // println!("{}", array_a.to_string());
    // println!("{}", array_b.to_string());

    // let mut output = vec![0.; 4];
    // unsafe {
    //     let meta_a = array_a.get_metadata();
    //     let meta_b = array_b.get_metadata();

    //     matrixmultiply::sgemm(
    //         2,
    //         3,
    //         2,
    //         1.,
    //         meta_a.data.as_ptr(),
    //         3,
    //         1,
    //         meta_b.data.as_ptr(),
    //         2,
    //         1,
    //         1.,
    //         output.as_mut_ptr(),
    //         2,
    //         1,
    //     );
    // }

    // let array_c = Array::from_vector_with_shape(&output, &[2, 2]).unwrap();

    // println!("{}", array_c.to_string());
}
