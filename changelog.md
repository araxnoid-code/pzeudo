## ChangeLog
### Version 0.0.3-dev.2
##### Bug Fixes
- Fixed a bug in the is_no_grad_or_time_not_match_or_no_update function where the update status of an ArrayView pointing to an Array was not being checked.

##### New
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
    
### Version 0.0.3-dev.1
- Development of the 'unrecord' concept:
  - This feature allows a record label to be skipped during gradient calculation, this ensures that the entire backward chain associated with that label receives no updates (calculations are bypassed), even though the record label's backward function is executed (no_update).

- Enhancement of ParamsStorage and GradStorage to check whether a specific gradient has been updated.

- Enhancement of Record to determine whether a record label should be executed or skipped.

- Updating methods and backward functions to align with the 'no_update' and 'unrecord' features.

- Updating Tensor to include the following methods:
  - Tensor::unrecord
  - Tensor::record
  - Note: These can only be used on `Tensor<F, Contiguous, G>`, they are not applicable to Tensor Views, as Tensor Views do not possess gradients and do not generate record labels during operations.


### Version 0.0.2
##### BUG FIXES
- Fixed an bug where the ArrayStorage::get_grad_storage_mut method was incorrectly returning ArrayStorage.

##### Update
###### Internal
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

###### Optimize
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


### Version 0.0.2-dev.1
##### BUG FIXES
- Fixed a bug where the array was not being replaced at the empty index location in GradStorage::grad_push.
- Fixed a bug where the array was not being replaced at the empty index location in ArrStorage::push_arr.
- Fixed a bug where epoch records were not cleared upon epoch completion; the Module::epoch method now clears record data after each epoch finishes.

##### Main Update
- rename permanent to params
- permanent_storage to params_storage
- permanent_tensor to params_tensor

- Updating PzeudoErr to be more concise by using fields.
  - TensorToolsErr
  - ArrayErr
  - ReqGradErr
  - OpsErr
  - StorageErr
  - StorageNoGradErr
  - StorageTimeErr
  - BackwardErr
  - LossErr
  - LayerErr
  - OptimErr
  - ModuleErr
  - RandDistrNormalErr
  - SerdeJsonErr
  - IOErr

- adding Grad and NoGrad features
  - With this change, the tensor structure becomes `Tensor<F, T, G>`.
    - F: Type
    - T: Contiguous/view
    - G: Grad/NoGrad
  - A tensor can be converted to Grad or NoGrad via...
    - Tensor::no_grad, for `Tensor<F, Contiguous, Grad>`
    - Tensor::with_grad, for `Tensor<F, Contiguous, NoGrad>`
  - The Grad/NoGrad status of a Tensor View (or `Tensor<F, View, G>`) cannot be changed because a Tensor View does not possess a gradient of its own; its gradient status follows that of its parent contiguous tensor.
  - When a tensor with Grad status performs an operation and generates a stored record, switching to NoGrad status causes its gradient to be discarded; if it subsequently reverts to Grad status, a new gradient is allocated, but the previous record remains unchanged, resulting in the calculation being skipped because the gradient stored in that record is no longer valid.

- The addition of the Grad/NoGrad feature has resulted in the following changes:
  - Updates to GradStorage to handle tensors that transition to the NoGrad state. 
  - All methods are now affected by Grad/NoGrad.
  - All backward functions.
  - The removal mechanism in GradStorage has been made private (as there is currently no handling for manual removal cases).

- Modules can no longer initialize tensors and layers; instead, initialization now directly accesses the relevant data type and takes the Module as an argument.

- Added the Module::reset method, useful for clearing all data in ArrayStorage (except params_storage) and records.

- added a enum WeightInit for weight initialization.
  - field:
    - WeightInit::He
    - WeightInit::Xavier
  - rng uses SmallRng.
  - implementing WeightInit for Linear.

- Created ModelBuilder (pzeudo_module/model_builder.rs) to facilitate layer creation within the model. 
  - Workflow change: model creation now requires a ModelBuilder, which is obtained via Module::model_builder.
  - Changes affecting Linear and Optimizer:
    - Linear (layer) requires a &mut ModelBuilder argument to register its Parameters Tensor. 
    - The Optimizer consumes ownership of the ModelBuilder, preventing the creation of new layers afterward (unless a new ModelBuilder is created, which effectively amounts to creating a new model sharing the same storage as the previous one). 
  - This ensures the model is fully defined before being stored and processed by the optimizer. Additionally, since the ModelBuilder holds a &mut Module, it prevents the creation of tensors (specifically parameter tensors) during the building phase; this guarantees that model parameters are laid out contiguously within the parameter storage. Tensors can only be created after the Optimizer has consumed the ModelBuilder. 
  - Due to this change, the Optimizer stores the range spanning from the start point (when the ModelBuilder was initialized) to the end point (when the Optimizer consumed the ModelBuilder); only the parameter tensors within this range are processed by the optimizer. This allows for the distinction between trainable model parameters, non-trainable parameters (e.g., datasets), and trainable parameters belonging to other models.

- new optimizer
  - SgdMomentum
  - AdaGrad
  - RMSProp
  - Adam

- added the trait SaveParamsTrait(pzeudo_module/model/save_params.rs), which functions to provide a method for the Optimizer to be able to save parameters. 
  - The method in SaveParamsTrait that is related to save is SaveParamsTrait::save_parasm 
  - has been implemented on 
    - Sgd 
    - SgdMomentum 
    - AdaGrad 
    - RMSProp 
    - Adam 
  - The save params mechanism uses serde_json by specifying the path for saving. parameters are saved in the form of a flat array

- added ModelBuilder methods including: 
  - ModelBuilder::load_params, functions as the main method for commanding load parameters that have been saved. 
  - ModelBuilder::get_load_params, functions to access LoadParams and retrieve params data(not public). 
  - ModelBuilder::get_load_else_generate_vec, functions for a layer requesting parameters, if the ModelBuilder contains LoadParams, it will directly access the load parameters and if not it will immediately generate new parameters(not public).

- in saving, only save parameters, then the model must be saved separately. 
  - the model that uses load parameters must be the same as the model that uses save parameters, because the save and load methods used are really affected by the order in the storage array.


### Version 0.0.1
##### BUG FIXES
- Fixed a bug in matmul_nd f32 and f64, namely an error in dimension slicing that caused incorrect index access.

- fix mean square error (mse) due to formula error in mse_backward.

##### Update
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


### Version 0.0.1-dev.6
##### fix bugs
- Fixed a bug in Array::matmul_2d due to an offset error.

- Fixed an issue with the get_broadcast_dim function that wasn't returning broadcast dimensions sequentially.

##### New
- Added Auto-broadcast to Calculations
  - Add
  - Sub
  - Mul
  - Div

- Added matmul_2d method (for f32 and f64) for tensors

- Added Module struct, which will be the main environment providing the source for deep learning purposes.

- Added EpochBuilder, which will be the main structure in the training process configuration later.

- The training mechanism will be via Module::epoch.

- Added Array methods
  - Array::to_ones
  - Array::to_zeros

- Added Optimizer
  - Sgd

- Added Loss Function
  - Mean Square Error (MSE).

- Added linear method that allows for the creation of linear layers.

- Changed the way tensor arrays are stored; there are now two ways to store arrays in ArrayStorage: regular storage and permanent_storage.
  - Storage is required to allow temporary storage of arrays required for the backpropagation flow. To manage memory, storage is periodically deleted.
    - Automatic storage deletion per epoch via Module::epoch. After deletion, the storage is ready for use in the next epoch without retaining unnecessary data from the previous epoch.
  - permanent_storage: permanent_storage is not intended for deletion because it functions to store arrays permanently.
    - Permanent array storage is required to store arrays that will be updated by the optimizer.

- Changes to ArrayStorage now allow storing updateable arrays (for optimizer purposes), array views, and arrays, while eliminating unnecessary relationships between stored arrays.
  - Due to this change, the structure that stores array indexes has undergone changes in how it stores indexes and handles output from storage, but the flow of most of the structure remains unchanged.


### Version 0.0.1-dev.5
- Added new array types, namely ArrayRef and ArrayRefMut
  - ArrayRef is a form of array that is only used for wrapping data and metadata, in contrast to ArrayRef which has its own metadata, ArrayRef metadata is borrowed from other array metadata.
  - ArrayRefMut is a mutable form of Array, allowing mutability for using OpsAssign.
- Creating Tensor
  - ArrayStorage.
  - Contiguous and View.
  - The main tensor methods that will be created include:
    - arithmetic
      - sub
      - div
      - mul
      - add
- Create forward and backward flows.
  - forward
  - backward
    - RecordLabel, development of a recorder that functions to record or store every operation carried out.
- method addition to array.
  - The main methods that will be added include:
    - assign
    - add_assign
    - sub_assign
    - mul_assign
    - div_assign
    - sqrt
    - reshape
    - to_shape
    - flatten
    - zeros
    - ones

### Version 0.0.1-dev.4
- Focus on developing an array (named pzeudo_num) that will be used in deep learning. Deep learning itself hasn't been created yet.
- pzeudo_num development.
  - Array (Core)
    - `Metadata`
    - `ArrayTrait`
      - `ArrayTrait::index`
      - `ArrayTrait::linear_index`
  - operation(ops)
    - Arithmetic
      - `OpsAdd`
        - `OpsAdd::add`
        - `OpsAdd::add_scalar`
        - `OpsAdd::scalar_add`
      - `OpsSub`
        - `OpsSub::sub`
        - `OpsSub::sub_scalar`
        - `OpsSub::scalar_sub`
      - `OpsMul`
        - `OpsMul::mul`
        - `OpsMul::mul_scalar`
        - `OpsMul::scalar_mul`
      - `OpsDiv`
        - `OpsDiv::div`
        - `OpsDiv::div_scalar`
        - `OpsDiv::scalar_div`
    - Dot Product
      - `OpsDotProduct`
        - `OpsDotProduct::dot`
      - `OpsDotProductF32`
        - `OpsDotProductF32::f32`
      - `OpsDotProductF64`
        - `OpsDotProductF64::f64`
    - Matmul
      - Matmul 2D
        - `OpsMatmul2DF32`
          - `OpsMatmul2DF32::matmul_2d`
        - `OpsMatmul2DF64`
          - `OpsMatmul2DF64::matmul_2d`
      - Matmul ND
        - `OpsMatmulNDF32`
          - `OpsMatmulNDF32::matmul_nd`
        - `OpsMatmulNDF64`
          - `OpsMatmulNDF64::matmul_nd`
    - Reduction
      - `OpsAvg`
        - `OpsAvg::avg`
        - `OpsAvg::avg_axis`
      - `OpSum`
        - `OpSum::sum`
        - `OpSum::sum_axis`
    - Unary
      - `OpsUnary`
        - `OpsUnary::exp`
        - `OpsUnary::ln`
        - `OpsUnary::log2`
        - `OpsUnary::log10`
        - `OpsUnary::powi`
        - `OpsUnary::powf`
    - View Method
      - Broadcast
        - `able_broadcast`
        - `get_broadcast_dim`
        - `OpsBroadcast`
          - `OpsBroadcast::broadcast`
      - Slicing
        - `SliceRange`
        - `r`
        - `SlicingRangeTrait`
        - `OpsSlicing`
          - `OpsSlicing::slicing`
      - permute
        - `OpsPermute`
          - `OpsPermute::permute`
          - `OpsPermute::t`
  - Array (contiguous)
    - array will store data linearly using vector.
    - Initialization
      - `Array::new`
      - `Array::from_vector`
      - `Array::from_vector_with_shape`
  - ArrayView (view)
    - In reading data, array view uses shape, stride and offset.
  - _test
    - Arithmetic
      - add
      - sub
      - mul
      - div
