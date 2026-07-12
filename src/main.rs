use pzeudo::{Array, ArrayTrait, OpsAdd, shape_to_stride};

fn main() {
    let array_a = Array::from_vector_with_shape(&vec![1., 2., 3., 4., 5., 6.], &[2, 3]).unwrap();
    let array_b = Array::from_vector_with_shape(&vec![1., 2., 3., 4., 5., 6.], &[3, 2]).unwrap();

    println!("{}", array_a.to_string());
    println!("{}", array_b.to_string());

    let mut output = vec![0.; 4];
    unsafe {
        let meta_a = array_a.get_metadata();
        let meta_b = array_b.get_metadata();

        matrixmultiply::sgemm(
            2,
            3,
            2,
            1.,
            meta_a.data.as_ptr(),
            3,
            1,
            meta_b.data.as_ptr(),
            2,
            1,
            1.,
            output.as_mut_ptr(),
            2,
            1,
        );
    }

    let array_c = Array::from_vector_with_shape(&output, &[2, 2]).unwrap();

    println!("{}", array_c.to_string());
}
