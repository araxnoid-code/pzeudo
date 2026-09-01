use crate::prelude::*;
use num_traits::{Float, One, Zero};
use std::{
    fmt::{Debug, Display},
    iter::Sum,
    ops::{AddAssign, DivAssign, Mul, MulAssign, SubAssign},
};

/// # Layer Norm
/// Normalizing the last axis.
/// ## formula:
/// ```sh
/// avg = E[x]
/// variance = E[x^2] - E[x]^2
/// epsilon = 1e-7
/// norm = x - avg/sqrt(variance + epsilon)
/// ```
/// ## optional gamma and beta
/// ```sh
/// y * gamma + beta
/// ```
/// ```rs
/// let mut module_builder: ModuleBuilder<f32> = ModuleBuilder::new(42);
/// let mut model_builder = module_builder.model_builder();
///
/// // without gamma and beta
/// let layer_norm = LayerNorm::new(None, &mut model_builder, ReqGrad).unwrap();
/// // With gamma dan beta
/// let layer_norm = LayerNorm::new(Some(16), &mut model_builder, ReqGrad).unwrap();
/// // gamma, 1-dimensional, shape [16], initialization as a tensor of ones
/// // beta, 1-dimensional, shape [16], initialization as a tensor of zeros
/// ```
pub struct LayerNorm<F, ReqGrad> {
    gamma: Option<Tensor<F, Contiguous, ReqGrad>>,
    beta: Option<Tensor<F, Contiguous, ReqGrad>>,
}

impl<F, G> LayerNorm<F, G> {
    pub fn new(
        hidden: Option<usize>,
        model_builder: &mut ModelBuilder<F>,
        requires_grad: G,
    ) -> Result<LayerNorm<F, G>, PzeudoErr>
    where
        F: Clone + One + Zero,
        G: ReqGradTrait<F>,
    {
        let (gamma, beta) = if let Some(hidden) = hidden {
            if model_builder.is_params_load() {
                let gamma = model_builder
                    .get_load_params()?
                    .ok_or(PzeudoErr::LayerErr(String::from("LayerNorm::new. Unable to retrieve gamma data via load params because load params is undefined.")))?;

                let beta = model_builder
                    .get_load_params()?
                    .ok_or(PzeudoErr::LayerErr(String::from("LayerNorm::new. Unable to retrieve beta data via load params because load params is undefined.")))?;

                let module = model_builder.get_module();
                let gamma =
                    Tensor::param_from_vector_with_shape(&gamma, &[hidden], module, requires_grad)?;
                let beta =
                    Tensor::param_from_vector_with_shape(&beta, &[hidden], module, requires_grad)?;
                (Some(gamma), Some(beta))
            } else {
                let module = model_builder.get_module();
                let gamma = Tensor::param_ones(&[hidden], module, requires_grad)?;
                let beta = Tensor::param_zeros(&[hidden], module, requires_grad)?;
                (Some(gamma), Some(beta))
            }
        } else {
            (None, None)
        };

        Ok(LayerNorm { beta, gamma })
    }

    pub fn forward<T, TensorGrad, ReqGrad>(
        &self,
        tensor: &Tensor<F, T, TensorGrad>,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
    where
        F: AddAssign + DivAssign + Float + SubAssign + Debug + MulAssign,
        for<'a> &'a F: Mul<&'a F, Output = F>,
        ReqGrad: ReqGradTrait<F>,
        for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
    {
        let epsilon = F::from(1e-7).ok_or(PzeudoErr::LayerErr(format!(
            "LayerNorm::forward. Cannot cast to epsilon."
        )))?;
        let axis = tensor.shape.len() - 1;

        let mut storage = tensor.get_storage().borrow_mut();
        let array_tensor =
            storage.get_as_array_ref::<T>(tensor.get_array_idx(), ContiguousType::Arr)?;

        let (avg, var) = array_tensor.avg_and_var_axis(&[axis], true)?;

        let avg_broadcast = avg.broadcast(&tensor.shape)?;
        let var_broadcast = var.broadcast(&tensor.shape)?;

        let len = array_tensor.shape.iter().product::<usize>();
        let mut vec = Vec::with_capacity(len);

        for i in 0..len {
            let y = (array_tensor.linear_index(i)? - avg_broadcast.linear_index(i)?)
                / (var_broadcast.linear_index(i)? + epsilon).sqrt();
            vec.push(y);
        }

        let norm = storage.push(ElementType::Arr(Array::from_vector_with_shape(
            &vec,
            &tensor.shape,
        )?))?;
        let grad = requires_grad.into_zeros_grad_storage(&tensor.shape, &mut storage)?;

        let record_idx = if requires_grad.is_grad() {
            let mut record = tensor.get_record().borrow_mut();
            let record_idx = RecordStatus::Record(record.len());
            let record_label = RecordLabel::LayerNorm(norm, tensor.get_grad_idx(), var.data, grad);
            record.push(record_label);
            Some(record_idx)
        } else {
            None
        };

        let tensor: Tensor<F, Contiguous, ReqGrad> = Tensor::_new(
            norm,
            grad,
            tensor.shape.to_vec(),
            record_idx,
            tensor.get_record().clone(),
            tensor.get_storage().clone(),
        );

        if let (Some(gamma), Some(beta)) = (&self.gamma, &self.beta) {
            drop(storage);
            let mul = tensor.mul::<Contiguous, _, _>(gamma, requires_grad)?;
            let add = mul.add::<Contiguous, _, _>(beta, requires_grad)?;
            return Ok(add);
        }

        Ok(tensor)
    }
}

pub fn layer_norm_backward<F>(
    output_arr_idx: StorageType,
    array_grad_idx: Option<StorageType>,
    var: &[F],
    grad_idx: Option<StorageType>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    for<'a> F: Float + AddAssign + MulAssign + Sum<&'a F> + Display + Debug,
{
    if let Some(grad_idx) = grad_idx {
        if is_no_grad_or_time_not_match_or_no_update(grad_idx, storage)? {
            return Ok(());
        }

        if let Some(arr_grad_idx) = array_grad_idx {
            storage.set_grad_update(arr_grad_idx, true)?;
            if is_no_grad_or_time_not_match_or_no_update(arr_grad_idx, storage)? {
                return Ok(());
            }
            let epsilon = F::from(1e-7).ok_or(PzeudoErr::BackwardErr(format!(
                "layer_norm_backwar. tidak dapat melakukan casting pada epsilon."
            )))?;

            let mut arr_grad = storage.take_grad(arr_grad_idx)?;

            let mut arr_grad_ref = arr_grad.to_array_ref_mut::<View>();
            let axis = arr_grad_ref.shape.len() - 1;

            let grad = storage.take_grad(grad_idx)?;
            let grad_ref = grad.to_array_ref::<Contiguous>();

            let out_array =
                storage.get_as_array_ref::<Contiguous>(output_arr_idx, ContiguousType::Arr)?;

            let mut s = arr_grad_ref.shape.to_vec();
            s[axis] = 1;
            let mut stride = shape_to_stride(&s);
            stride[axis] = 0;

            let var_arr: ArrayRef<'_, F, View> = ArrayRef {
                data: var,
                offset: 0,
                shape: &arr_grad_ref.shape,
                stride: &stride,
                _array_type: Default::default(),
            };

            let avg_grad_mul_arr = grad_ref.mul(&out_array)?.avg_axis(&[axis], true)?;

            let avg_grad_ref = grad_ref.avg_axis(&[axis], true)?;

            let avg_grad_mul_arr_broadcasted = avg_grad_mul_arr.broadcast(arr_grad_ref.shape)?;
            let avg_grad_ref_broadcasted = avg_grad_ref.broadcast(arr_grad_ref.shape)?;

            let len = arr_grad_ref.shape.iter().product::<usize>();
            let one = F::one();

            for i in 0..len {
                let std = (var_arr.linear_index(i)? + epsilon).sqrt();

                // 1
                let mut y = one / std * grad_ref.linear_index(i)?;

                // 2
                y += -avg_grad_ref_broadcasted.linear_index(i)? / std;

                // 3
                y += -out_array.linear_index(i)? * avg_grad_mul_arr_broadcasted.linear_index(i)?
                    / std;

                *arr_grad_ref.linear_index_mut(i)? += y;
            }

            storage.replace_grad(grad_idx, grad)?;
            storage.replace_grad(arr_grad_idx, arr_grad)?;
        }
    }

    Ok(())
}
