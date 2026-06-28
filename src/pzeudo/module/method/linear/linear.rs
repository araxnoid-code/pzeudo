// use std::sync::{Arc, RwLock};

// use ndarray::ArrayD;

// use crate::{Module, Tensor};

// pub struct Linear {
//     weight: Arc<RwLock<Tensor>>,
//     bias: Arc<RwLock<Tensor>>,
// }

// impl Module {
//     pub fn new_linear(&self, in_feature: usize, out_feature: usize) {
//         let value = (0..in_feature * out_feature).map(|i| i as f32).collect();
//         let weight = ArrayD::<f32>::from_shape_vec(vec![in_feature, out_feature], value).unwrap();
//         let weight_grad = ArrayD::<f32>::zeros(weight.shape());

//         let weight_shared = Arc::new(RwLock::new(weight));
//         let weight_grad_shared = Arc::new(RwLock::new(weight_grad));

//         let value = (0..out_feature).map(|i| i as f32).collect();
//         let bias = ArrayD::<f32>::from_shape_vec(vec![out_feature], value).unwrap();
//         let bias_grad = ArrayD::<f32>::zeros(vec![out_feature]);

//         // Linear{

//         // }
//     }
// }
