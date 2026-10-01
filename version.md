# Version 0.0.4
## Fixed a Bug in ArrayTrait::index
A bug was discovered in the ArrayTrait::index method. The bug occurred due to an error in the stride calculation, which caused errors when using ArrayTrait::index on an ArrayView or other View-like Array. This bug has been fixed.

ArrayTrait::index has added additional validation to ensure indexing remains within the tensor dimension range.

## Fixed a bug in the testing code pzeudo_num::_test::matmul::matmul_2d::matmul_2d_test_3
The bug occurs due to an indexing error because ArrayTrait::index does not validate indexing that crosses the array boundary.

## BatchNorm Addition
```rs
fn main() {
    let mut module_builder: ModuleBuilder<f32> = ModuleBuilder::new(42);
    let mut model_builder = module_builder.model_builder();
    let batch_norm = BatchNorm::new(2, 16, &mut model_builder).unwrap();
}
```
For each parameter in BatchNorm::new
```rs
pub fn new(
        channel: usize, // index dimension that becomes a channel on the input
        channel_size: usize, // size of channel to be input
        model_builder: &mut ModelBuilder<F>,
    ) -> Result<BatchNorm<F>, PzeudoErr>
    where
        F: NumCast,
    {
        // ...
    }
```
#### Default Value
```
gamma = ones tensor
beta = zeros tensor
momentum = 0.1
running_avg = 0
running_var = 1
```

#### Formula
```
avg = E[x]
variance = E[x^2] - E[x]^2
epsilon = 1e-7
norm = x - avg/sqrt(variance + epsilon)
y = norm * gamma + beta
```

#### Training Phase
During the training phase, BatchNorm will calculate the mean and variance of the input tensor, then use these to normalize the tensor and update running_avg and running_var.
```
momentum = 0.1 (default)
running_avg = (1 - momentum) * running_avg + momentum * avg
running_var = (1 - momentum) * running_var + momentum * var
```

#### Testing Phase
During the testing phase, BatchNorm will use running_avg and running_var to normalize the tensor.
```
norm = x - running_avg/sqrt(running_var + epsilon)
y = norm * gamma + beta
```

## Adding Tensor::dot
```rs
use pzeudo::{ModuleBuilder, NoneModel, ReqGrad, Tensor};

fn main() {
    let module: pzeudo::Module<f32, NoneModel> = ModuleBuilder::new(42).build(NoneModel);
    let tensor_a = Tensor::from_vector_with_shape(
        vec![1., 2., 3., 4., 5., 6., 7., 8.],
        &[8],
        &module,
        ReqGrad,
    )
    .unwrap();

    let tensor_b = Tensor::from_vector_with_shape(
        vec![1., 2., 3., 4., 5., 6., 7., 8.],
        &[8],
        &module,
        ReqGrad,
    )
    .unwrap();

    let dot = tensor_a.dot(&tensor_b, ReqGrad).unwrap();
}
```

## Added new residual OpsMax and OpsMin traits
- OpsMax::max
- OpsMax::max_axis
- OpsMax::argmax
- OpsMax::argmax_axis
- OpsMax::max_axis_with_flatten_index
- OpsMin::min
- OpsMin::min_axis
- OpsMin::argmin
- OpsMin::argmin_axis
- OpsMin::min_axis_with_flatten_index

## Added new residual methods to tensors
#### Tensor::max
Returns the largest value of a tensor.

#### Tensor::max_axis
Returns the largest value of a tensor based on its axis.

#### Tensor::argmax
Returns the index of the largest value in a tensor.
This method cannot perform backpropagation.

#### Tensor::argmax_axis
Returns the index of the largest value of a tensor based on its axes.
This method cannot perform backpropagation.

#### Tensor::Min
Returns the smallest value of a tensor.

#### Tensor::max_axis
Returns the smallest value of a tensor based on its axis.

#### Tensor::argmax
Returns the index of the smallest value in a tensor.
This method cannot perform backpropagation.

#### Tensor::argmax_axis
Returns the index of the smallest value of a tensor based on its axes.
This method cannot perform backpropagation.

## New Array Initialization Method
- Array::from_range
- Array::from_range_fn
- Array::from_shape
- Array::from_shape_fn

## Implementing SlicingRangeTrait for usize
When using usize in the r() function, it will automatically set start to the usize value and end to start + 1.

## Changes to LayerNorm
Now LayerNorm will directly have ReqGrad status without having to manually notate it during initialization.

## Added alg_optim method for each optimizer
alg_optim is a method that uses algebraic methods in its operations. Its development has been postponed for the future and is not a top priority.
