use ndarray::{ArrayD, array};
use pzeudo::{Backward, Tensor, TensorTrait};

fn main() {
    let mut record = vec![];
    let tensor_a = Tensor::from_array(ArrayD::<f32>::zeros(vec![2, 1, 5]));
    let tensor_b = Tensor::from_array(ArrayD::<f32>::zeros(vec![1, 2, 3, 5]));

    let tensor_c = tensor_a.add(&tensor_b, &mut record);

    tensor_c.backward(&mut record);

    let grad = tensor_a.get_share_gradient().unwrap();
    println!("{}", grad.borrow());
}
