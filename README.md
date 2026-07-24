# Pzeudo
a deep learning project for fun.

## Goal
can be used to create AI models, that's all.

## Stable?
still far from stable, but will continue to be developed.
## what's new in 0.0.1-dev.6
Update 0.0.1-dev.6 berfokus pada pengembangan alur backpropogation dan pembuatan model deep learning. Perubahan yang terjadi antara lain:


#### fix bugs
- Fixed a bug in Array::matmul_2d due to an offset error.

- Fixed an issue with the get_broadcast_dim function that wasn't returning broadcast dimensions sequentially.

#### New
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

see the development process in more detail at: [0.0.1-dev.6_plan.md](0.0.1-dev.6_plan.md)
