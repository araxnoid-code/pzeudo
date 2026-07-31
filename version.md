# Version 0.0.1
### BUG FIXES
- Fixed a bug in matmul_nd f32 and f64, namely an error in dimension slicing that caused incorrect index access.

- fix mean square error (mse) due to formula error in mse_backward.

### Update
- Changes to ArrayStorage
  - ArrayStorage now has three components
    - permanent_storage: used to store array pairs (array as value and array as grad) that are not intended for deletion.
    - arr_storage, of type ArrStorage, used to store Array.
    - view_storage, of type ViewStorage, used to store Metadata.
    - grad_storage, of type GradStorage, used to store grad values (array).
  - Internal changes to the ArrayStorage method, but almost all of them do not interfere with the method's use by other code.

- ArrayRefMut Update
  - Gradient can now be retrieved using ArrayRefMut<'_, View>
  - Added ArrayRefMut::to_ones method

- Array Update
  - OpsUnary.
    - Added OpsUnary::abs.
    - Added OpsUnary::signum.

- Tensor Update
  - Matmul
    - Added Tensor::matmul_nd(f32 and f64) methods to tensors.
  - View Operations
    - Added Tensor::slice
    - Added Tensor::broadcast
    - Added Tensor::permute
    - Added Tensor::t
    - Added Tensor::index
    - Added Tensor::to_shape
    - Added Tensor::reshape
    - Tensor::view method, which can now be used backward.
  - getter method
    - Added Tensor::get_shape
  - Unary
    - Added Tensor::log
    - Added Tensor::log2
    - Added Tensor::log10
    - Added Tensor::ln
    - Added Tensor::powi
    - Added Tensor::powf
    - Added Tensor::sqrt
    - Added Tensor::exp
  - Loss Function
    - Added mae (Mean Absolute Error)
    - Added cross_entropy_loss
  - Activation
    - softplus 
    - relu 
    - sigmoid 
    - tanh

- Module Update
  - update the function Module::epoch which will accept a function that will return Result<O, PzeudoErr>, O is a generic type.

more details on: [0.0.1-dev.7_plan.md](./0.0.1-dev.7_plan.md)
