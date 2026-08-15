# Version 0.0.3-dev.1
- Development of the 'unrecord' concept:
  - This feature allows a record label to be skipped during gradient calculation, this ensures that the entire backward chain associated with that label receives no updates (calculations are bypassed), even though the record label's backward function is executed (no_update).

- Enhancement of ParamsStorage and GradStorage to check whether a specific gradient has been updated.

- Enhancement of Record to determine whether a record label should be executed or skipped.

- Updating methods and backward functions to align with the 'no_update' and 'unrecord' features.

- Updating Tensor to include the following methods:
  - Tensor::unrecord
  - Tensor::record
  - Note: These can only be used on `Tensor<F, Contiguous, G>`, they are not applicable to Tensor Views, as Tensor Views do not possess gradients and do not generate record labels during operations.
