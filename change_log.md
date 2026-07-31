## Change Log
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
