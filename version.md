# Version 0.0.2-dev.2
### Fix Bug
- Fixed an bug where the ArrayStorage::get_grad_storage_mut method was incorrectly returning ArrayStorage.

### Update
#### Internal
- Added mechanisms to take and replace gradients for:
  - GradStorage
    - GradStorage::take_grad
    - GradStorage::replace_grad
  - ParamsStorage
    - ParamsStorage::take_grad
    - ParamsStorage::replace_grad
  - ViewStorage
    - ViewStorage::take_metadata
    - ViewStorage::replace_metadata
  - ArrayStorage (as an API combining GradStorage, ParamsStorage, and ViewStorage)
    - ArrayStorage::take_grad
    - ArrayStorage::replace_grad

#### Optimize
- take_replace: pzeudo utilizes ArrayStorage to store arrays and gradients (which are also represented as arrays). However, backpropagation requires the gradient array to be mutable, simultaneously borrowing other arrays from storage previously caused borrowing conflicts. Consequently, in version 0.0.1 and earlier, it was necessary to clone the data or perform operations that allocated new, owned temporary data before the gradient could be updated. In this version, the storage supports a take operation on the gradient, effectively transferring ownership of it. Since the gradient's mutability is no longer tied to the storage, it can be borrowed directly, allowing backward computation to proceed alongside other arrays borrowed from the storage simultaneously. Once the gradient has been "taken" it must be "replaced" back into its original index.

- Optimizing. Optimization achieved by combining several operations that would otherwise require multiple allocations into a single operation that performs a single allocation to store the output.

- implement take_replace
  - add_backward
  - sub_backward
  - mul_backward
  - div_backward
  - log_backward
  - ln_backward
  - powi_backward
  - powf_backward
  - sqrt_backward
  - mse_backward
  - mae_backward
  - cross_entropy_loss_backward
  - relu_backward
  - sigmoid_backward
  - softplus_backward
  - tanh_backward

- optimizing
  - mse(Mean Squared Error) function
  - mae(Mean Absolute Error) function
  - cross_entropy_loss function
  - optimizing softplus

to see the development progress: [0.0.2_plan.md](0.0.2_plan.md)
