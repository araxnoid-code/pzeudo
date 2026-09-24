use pzeudo::{Array, ArrayTrait, OpsPermute};

fn main() {
    let array: Array<f32> = Array::from_shape(&[3, 2, 2]).unwrap();
    println!("{}", array);

    let permute = array.permute(&[0, 2, 1]).unwrap();
    println!("{}", permute);

    let indexing = permute.index(&[1, 3, 1]).unwrap();
    println!("{}", indexing);
}
