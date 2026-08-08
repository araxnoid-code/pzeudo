# Version 0.0.2
### BUG FIXES
- Fixed a bug where the array was not being replaced at the empty index location in GradStorage::grad_push.
- Fixed a bug where the array was not being replaced at the empty index location in ArrStorage::push_arr.
- Fixed a bug where epoch records were not cleared upon epoch completion; the Module::epoch method now clears record data after each epoch finishes.

### Main Update
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

See more details at: [0.0.2-dev.1.md](./0.0.2-dev.1.md)
