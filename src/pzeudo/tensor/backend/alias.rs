use ndarray::ArrayD;
use num_traits::Float;

use crate::{GradientStorage, Module, OpsLabel, Tensor};

pub type ModuleNDArrayBackend<'ops_label, F> =
    Module<GradientStorage<F>, ArrayD<F>, OpsLabel<'ops_label, F>>;

pub type TensorNDArrayOwn<'ops_label, F> = Tensor<'ops_label, F, ArrayD<F>, GradientStorage<F>>;
