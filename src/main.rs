use pzeudo::{Array, ArrayTrait, add, shape_to_stride};

fn main() {
    let shape = [2, 2, 3, 4];
    let array_a = Array::from_vector_with_shape(
        &(0..shape.iter().product::<usize>())
            .map(|idx| idx)
            .collect::<Vec<usize>>(),
        &shape,
    )
    .unwrap();

    println!("{}", array_a.to_string());

    let index = array_a.index(&[1, 1]).unwrap();

    println!("{}", index.to_string().unwrap());

    let index = index.index(&[2]).unwrap();

    println!("{}", index.to_string().unwrap());

    let index = index.index(&[2]).unwrap();

    println!("{}", index.to_string().unwrap());

    // let array_b =
    //     Array::from_vector_with_shape(&vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], &[3, 4])
    //         .unwrap();

    // let result = add(&array_a, &array_b).unwrap();
    // println!("{}", result.to_string());
}
