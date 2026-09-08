# Version 0.0.4-dev.1
## Adding scalar operations to tensors
allows operating tensors in cases that require scalars in their operations. The added operations include:
- Tensor::add_scalar, Tensor + scalar
- Tensor::scalar_add, scalar + Tensor
- Tensor::sub_scalar, Tensor - scalar
- Tensor::scalar_sub, scalar - Tensor
- Tensor::mul_scalar, Tensor * scalar
- Tensor::scalar_mul, scalar * Tensor
- Tensor::div_scalar, Tensor / scalar
- Tensor::scalar_div, scalar / Tensor
Note: the above operation will accept scalars directly, not tensors containing scalars.

## Changes and additions to Array and Tensor initialization
The methods below:
- Array::from_vector
- Array::from_vector_with_shape
- Tensor::from_vector_with_shape
- Tensor::param_from_vector_with_shape
The methods above have been modified. The method that originally accepted a slice has now been replaced with a vector. The method for accepting slices has now been redirected to the new methods below:
- Array::from_slice
- Array::from_slice_with_shape
- Tensor::from_slice_with_shape
- Tensor::param_from_slice_with_shape
The method that accepts slices has the disadvantage that each slice will be cloned or converted to a vector.


## Added loss function softmax_cross_entropy

## Indexing optimization on ArrayRef and ArrayRefMut

## WeightInit Changes
WeightInit::He has been split into WeightInit::HeIn and WeightInit::HeOut
  - WeightInit::HeIn uses in_features
  - WeightInit::HeOut uses out_features
  
Added WeightInit::Custom(mean, std_dev)
  - WeightInit::Custom allows for manual configuration of mean and std_dev
