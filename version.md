# Version 0.0.3-dev.2
## Bug Fixes
- Fixed a bug in the is_no_grad_or_time_not_match_or_no_update function where the update status of an ArrayView pointing to an Array was not being checked.

## New
- Added reduction methods, including:
  - Tensor::sum
  - Tensor::sum_axis
  - Tensor::avg
  - Tensor::avg_axis
  
- added new unary methods:
  - Tensor::sin
  - Tensor::cos
  - Tensor::tan

- Added concat methods for vectors or arrays
  - ConcatVector::tensor_concat
    - Used for vectors or arrays storing owned tensors. 
  - ConcatVectorRef::tensor_concat
    - Used for vectors or arrays storing tensor references.

to see the development progress: [0.0.3_plan.md](0.0.3_plan.md)
