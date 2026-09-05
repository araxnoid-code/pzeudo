# Version 0.0.3
## Adding TrainPhase and EvalPhase structs
- TrainPhase
  - also implements ReqGradTrait.
  - The function is proportional to Grad.
- EvalPhase
  - also implements ReqGradTrait.
  - The function is proportional to NoGrad.
  
This phase can function as either Grad or NoGrad, making it useful for certain methods. It also adds clarity to the model phase.

## Adding a Dropout Layer
using an inverted dropout
- p: probability of a zero value
    - example: p=0.7, Consequently, 70 percent of the values or elements will be dropped (at random locations, depending on the Bernoulli distribution and the specified seed).
### Formula
element * drop_value / (1 - p)
### Phase
- TrainPhase
Executes a formula based on the Bernoulli distribution and produces a new tensor.
- EvelPhase
It immediately returns a new tensor containing the same array data, state, and record as the previous tensor. The only difference is that the gradient is `None` (the previous tensor remains unaffected).

## Adding a LayerNorm Layer
Normalizing the last axis.
### formula:
```
avg = E[x]
variance = E[x^2] - E[x]^2
epsilon = 1e-7
norm = x - avg/sqrt(variance + epsilon)
```
### optional gamma and beta
```
y * gamma + beta
```
```rs
// ...
let mut module_builder: ModuleBuilder<f32> = ModuleBuilder::new(42);
let mut model_builder = module_builder.model_builder();
// without gamma and beta
let layer_norm = LayerNorm::new(None, &mut model_builder, ReqGrad).unwrap();
// With gamma dan beta
let layer_norm = LayerNorm::new(Some(16), &mut model_builder, ReqGrad).unwrap();
// gamma, 1-dimensional, shape [16], initialization as a tensor of ones
// beta, 1-dimensional, shape [16], initialization as a tensor of zeros
```

## Adding a Embedding Layer
```
embedding_num = The number of weights to be made
embedding_dim = The length of the weight parameters to be created
```
Initialization using a normal distribution with:
```
mean = 0
std_dev = 1
```

## Adding the Softmax activation function
Forward
```
softmax(x) = e^x/∑e^x
```

Backward
```
dsoftmax(x)/dx = y(g  - ∑gy)
```

## Changes to bias initialization in the Linear Layer
Bias initialization in the linear layer initially followed the initialization used for weights via the `WeightInit` enum. Now, biases will be initialized directly to 0.

## renamed Grad and NoGrad to ReqGrad and ReqNoGrad.
Since the names `Grad` and `NoGrad` are also used in several enums, the names of the `Grad` and `NoGrad` structs have been changed to avoid misuse.

## Adding supporting methods and traits
- OpsSum::sum_axis_closure
- OpsVar::avg_and_var_axis
- ModelBuilder::is_params_load
- ModelBuilder::get_load_else_generate_zeros

see development progress: [0.0.3_plan.md](./0.0.3_plan.md)
