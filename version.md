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

## Allows array initialization within epoch
Previously, tensor initialization required a ModuleBuilder, whether for regular tensors or params. Now, ModuleBuilder is only required for Params initialization. Tensor initialization now uses a Module, allowing for direct tensor initialization within an epoch.

It should be noted that the initialization inside the epoch is directly an array that is not permanent and will be deleted at each epoch, use it wisely and if the tensor is a constant value or trainable, then use Tensor Param outside the epoch.

## Added loss function softmax_cross_entropy
Added a combined loss function for softmax and cross entropy loss, named softmax_cross_entropy. This loss function was added because it is mathematically simpler in the backpropagation section than the chain rules for softmax and cross entropy loss, which are separated.

This loss function will accept a target (must be in the form of one hot tensor) and a logit (raw result from the model), the logit will automatically be converted into a probability by the function.

## Indexing optimization on ArrayRef and ArrayRefMut
Optimization on ArrayRef and ArrayRefMut, because in backpropogation often use ArrayRef and ArrayRefMut which are of type View as a safe choice for many situations, but will reduce performance if the Array is actually contiguous but is read in view (view has an additional stage in its indexing). therefore added one handler in the form of a label for ArrayRef and ArrayRefMut as an additional determinant whether the array is read in contiguous or view form besides only referring to Contiguous and View only.

## WeightInit Changes
WeightInit::He has been split into WeightInit::HeIn and WeightInit::HeOut
  - WeightInit::HeIn uses in_features
  - WeightInit::HeOut uses out_features
  
Added WeightInit::Custom(mean, std_dev)
  - WeightInit::Custom allows for manual configuration of mean and std_dev

to see the development progress: [0.0.4_plan.md](0.0.4_plan.md)
