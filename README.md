# Pzeudo
a deep learning project for fun.

## Goal
can be used to create AI models, that's all.

## Stable?
still far from stable, but will continue to be developed.

## what's new in 0.0.1-dev.5
- Added new array types, namely ArrayRef and ArrayRefMut
  - ArrayRef is a form of array that is only used for wrapping data and metadata, in contrast to ArrayRef which has its own metadata, ArrayRef metadata is borrowed from other array metadata.
  - ArrayRefMut is a mutable form of Array, allowing mutability for using OpsAssign.
- Creating Tensor
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

see more details: [0.0.1-dev.5_plan.md](./0.0.1-dev.5_plan.md)


## Next Version Plan (0.0.1-dev.6)
*coming soon*
