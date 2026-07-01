use ndarray::{ArrayD, array};
use pzeudo::{Backward, Tensor, TensorTrait};

fn main() {
    let mut record = vec![];
    let tensor_a = Tensor::from_array(ArrayD::<f32>::ones(vec![2, 1, 1]));
    let tensor_b = Tensor::from_array(ArrayD::<f32>::ones(vec![1, 2, 3, 5]));

    let tensor_c = tensor_b.mul(&tensor_a, &mut record);

    tensor_c.backward(&mut record);

    let grad = tensor_a.get_share_gradient().unwrap();
    println!("{}", grad.borrow());
}
