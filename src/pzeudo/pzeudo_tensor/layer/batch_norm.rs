use std::ops::{AddAssign, Div, DivAssign, Mul, Sub, SubAssign};

use num_traits::{Float, NumCast, One, Zero};

pub use crate::prelude::*;

pub struct BatchNorm<F> {
    gamma: Tensor<F, Contiguous, ReqGrad>,
    beta: Tensor<F, Contiguous, ReqGrad>,
    channel: usize,
}

impl<F> BatchNorm<F>
where
    F: Clone + Zero + One,
{
    pub fn new(
        channel: usize,
        hidden: usize,
        model_builder: &mut ModelBuilder<F>,
    ) -> Result<BatchNorm<F>, PzeudoErr> {
        let (gamma, beta) = if model_builder.is_params_load() {
            let gamma = model_builder
                .get_load_params()?
                .ok_or(PzeudoErr::LayerErr(String::from("LayerNorm::new. Unable to retrieve gamma data via load params because load params is undefined.")))?;

            let beta = model_builder
                .get_load_params()?
                .ok_or(PzeudoErr::LayerErr(String::from("LayerNorm::new. Unable to retrieve beta data via load params because load params is undefined.")))?;

            let module = model_builder.get_module();
            let gamma = Tensor::param_from_vector_with_shape(gamma, &[hidden], module, ReqGrad)?;
            let beta = Tensor::param_from_vector_with_shape(beta, &[hidden], module, ReqGrad)?;
            (gamma, beta)
        } else {
            let module = model_builder.get_module();
            let gamma = Tensor::param_ones(&[hidden], module, ReqGrad)?;
            let beta = Tensor::param_zeros(&[hidden], module, ReqGrad)?;
            (gamma, beta)
        };

        Ok(BatchNorm {
            beta,
            gamma,
            channel,
        })
    }

    pub fn forward<T, G, ReqGrad>(
        &self,
        tensor: Tensor<F, T, G>,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
    where
        F: Copy
            + Div<Output = F>
            + DivAssign
            + AddAssign
            + NumCast
            + SubAssign
            + Sub<Output = F>
            + Float,
        for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
        for<'a> &'a F: Mul<Output = F>,
        ReqGrad: ReqGradTrait<F>,
    {
        let mut storage = tensor.get_storage().borrow_mut();
        let array = storage.get_as_array_ref::<T>(tensor.get_array_idx(), ContiguousType::Arr)?;
        let shape = array.shape.to_vec();

        let mut axis_dim = vec![];

        for i in 0..shape.len() {
            if i == self.channel {
                continue;
            }

            axis_dim.push(i);
        }
        let (avg, var) = array.avg_and_var_axis(&axis_dim, true)?;

        let avg_broadcasted = avg.broadcast(&shape)?;
        let var_broadcasted = var.broadcast(&shape)?;

        let mut scale_shape = vec![1; shape.len()];
        scale_shape[self.channel] = shape[self.channel];
        let gamma = storage
            .get_as_array_ref::<Contiguous>(self.gamma.get_array_idx(), ContiguousType::Arr)?;
        let beta = storage
            .get_as_array_ref::<Contiguous>(self.beta.get_array_idx(), ContiguousType::Arr)?;
        let gamma_to_shape = gamma.to_shape(&scale_shape)?;
        let gamma_broadcasted = gamma_to_shape.broadcast(&shape)?;
        let beta_to_shape = beta.to_shape(&scale_shape)?;
        let beta_broadcasted = beta_to_shape.broadcast(&shape)?;

        let len = shape.iter().product::<usize>();
        let epsilon = F::from(1e-7).ok_or(PzeudoErr::LayerErr(String::from(
            "BatchNorm::forward. Can't cast data type on epsilon",
        )))?;
        let mut vec = Vec::with_capacity(len);
        for i in 0..len {
            let array_val = array.linear_index(i)?;
            let avg_val = avg_broadcasted.linear_index(i)?;
            let var_val = var_broadcasted.linear_index(i)?;

            let y = (avg_val - array_val) / (var_val + epsilon).sqrt();
            vec.push(y * gamma_broadcasted.linear_index(i)? + beta_broadcasted.linear_index(i)?);
        }

        let array_idx = storage.push(ElementType::Arr(Array::from_vector_with_shape(
            vec, &shape,
        )?))?;
        let grad_idx = requires_grad.into_zeros_grad_storage(&shape, &mut storage)?;

        let tensor = Tensor::_new(
            array_idx,
            grad_idx,
            shape,
            None,
            tensor.get_record().clone(),
            tensor.get_storage().clone(),
        );

        Ok(tensor)
    }
}

pub fn batch_norm_backward<F>(
    arr_grdient_idx: Option<StorageType>,
    output_idx: StorageType,
    var_vec: &[F],
    gamma_idx: StorageType,
    gamma_grad_idx: StorageType,
    beta_idx: StorageType,
    beta_grad_idx: StorageType,
    channel: usize,
    grad_idx: Option<StorageType>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    F: Mul<Output = F> + Copy + AddAssign + Sub<Output = F> + Div<Output = F> + Float,
{
    if let Some(grad_idx) = grad_idx {
        if is_no_grad_or_time_not_match_or_no_update(grad_idx, storage)? {
            return Ok(());
        };

        if let Some(arr_gradient_idx) = arr_grdient_idx {
            storage.set_grad_update(arr_gradient_idx, true)?;
            if is_no_grad_or_time_not_match_or_no_update(arr_gradient_idx, storage)? {
                return Ok(());
            }

            // TAKE
            // Fetch gradient from storage, to overcome borrow rule problem
            let mut arr_gradient_take = storage.take_grad(arr_gradient_idx)?;
            let mut arr_gradient = arr_gradient_take.to_array_ref_mut::<View>();

            let mut gamma_grad_take = storage.take_grad(gamma_grad_idx)?;
            let gamma_grad_arr = gamma_grad_take.to_array_ref_mut::<Contiguous>();

            let mut beta_grad_take = storage.take_grad(beta_grad_idx)?;
            let beta_grad_arr = beta_grad_take.to_array_ref_mut::<Contiguous>();

            // AS
            // Retrieve the required data in the form of ArrayRef
            let output_arr =
                storage.get_as_array_ref::<Contiguous>(output_idx, ContiguousType::Arr)?;
            let shape = output_arr.shape;
            let len = shape.iter().product::<usize>();

            let gradient_arr =
                storage.get_as_array_ref::<Contiguous>(grad_idx, ContiguousType::Grad)?;

            let beta_arr = storage.get_as_array_ref::<Contiguous>(beta_idx, ContiguousType::Arr)?;

            let gamma_arr =
                storage.get_as_array_ref::<Contiguous>(gamma_idx, ContiguousType::Arr)?;

            // SCALE
            // Making adjustments to the tensor scale in batch norm
            let mut scale_broadcasted_stride = shape_to_stride(shape);
            for i in 0..shape.len() {
                if i == channel {
                    continue;
                }
                scale_broadcasted_stride[i] = 0;
            }
            let gamma_arr_broadcasted = ArrayRef::<F, View> {
                _array_type: Default::default(),
                contiguous: false,
                data: gamma_arr.data,
                offset: 0,
                shape,
                stride: &scale_broadcasted_stride,
            };
            let beta_arr_broadcasted = ArrayRef::<F, View> {
                _array_type: Default::default(),
                contiguous: false,
                data: beta_arr.data,
                offset: 0,
                shape,
                stride: &scale_broadcasted_stride,
            };
            let mut beta_grad_arr_broadcasted = ArrayRefMut::<F, View> {
                _array_type: Default::default(),
                contiguous: false,
                data: beta_grad_arr.data,
                offset: 0,
                shape,
                stride: &scale_broadcasted_stride,
            };
            let mut gamma_grad_arr_broadcasted = ArrayRefMut::<F, View> {
                _array_type: Default::default(),
                contiguous: false,
                data: gamma_grad_arr.data,
                offset: 0,
                shape,
                stride: &scale_broadcasted_stride,
            };

            // SCALE GRADIENT COMPUTATION
            let mut gradient_norm_vec_1 = Vec::with_capacity(len);
            let mut gradient_norm_vec_2 = Vec::with_capacity(len);
            let epsilon = F::from(1e-7).ok_or(PzeudoErr::BackwardErr(String::from(
                "batch_norm_backward. Can't cast data type on epsilon",
            )))?;
            for i in 0..len {
                let output = output_arr.linear_index(i)?;
                let gamma = gamma_arr_broadcasted.linear_index(i)?;
                let beta = beta_arr_broadcasted.linear_index(i)?;
                let scale_grad = gradient_arr.linear_index(i)?;

                // BETA GRADIENT
                *beta_grad_arr_broadcasted.linear_index_mut(i)? += scale_grad;

                // GAMMA GRADIENT
                // output = (norm * gamma) + bias
                // norm = (output - bias) / gamma
                let norm = (output - beta) / gamma;
                let gamma_g = norm * scale_grad;
                *gamma_grad_arr_broadcasted.linear_index_mut(i)? += gamma_g;

                // NORM GRADIENT
                let norm_grad = scale_grad * gamma;
                gradient_norm_vec_1.push(norm_grad);
                gradient_norm_vec_2.push(norm_grad * norm);
            }

            // BATCHNORM GRADIENT COMPUTATION
            let mut axis_dim = vec![];
            for i in 0..shape.len() {
                if i == channel {
                    continue;
                }

                axis_dim.push(i);
            }

            let gradient_norm_arr_1 = ArrayRef::<F, Contiguous> {
                data: &gradient_norm_vec_1,
                offset: 0,
                shape: output_arr.shape,
                stride: output_arr.stride,
                contiguous: true,
                _array_type: Default::default(),
            };
            let avg_gradient_norm_arr_1 = gradient_norm_arr_1.avg_axis(&axis_dim, true)?;
            let avg_gradient_norm_arr_1_broadcasted =
                avg_gradient_norm_arr_1.broadcast(output_arr.shape)?;

            let gradient_norm_arr_2 = ArrayRef::<F, Contiguous> {
                data: &gradient_norm_vec_2,
                offset: 0,
                shape: output_arr.shape,
                stride: output_arr.stride,
                contiguous: true,
                _array_type: Default::default(),
            };
            let avg_gradient_norm_arr_2 = gradient_norm_arr_2.avg_axis(&axis_dim, true)?;
            let avg_gradient_norm_arr_2_broadcasted =
                avg_gradient_norm_arr_2.broadcast(output_arr.shape)?;

            let norm_arr = ArrayRef::<F, Contiguous> {
                data: &var_vec,
                offset: 0,
                shape: &avg_gradient_norm_arr_2.shape,
                stride: &avg_gradient_norm_arr_2.stride,
                contiguous: true,
                _array_type: Default::default(),
            };
            let norm_arr_broadcasted = norm_arr.broadcast(output_arr.shape)?;

            for i in 0..len {
                let output = output_arr.linear_index(i)?;
                let gamma = gamma_arr_broadcasted.linear_index(i)?;
                let beta = beta_arr_broadcasted.linear_index(i)?;
                let norm = (output - beta) / gamma;

                let std = (norm_arr_broadcasted.linear_index(i)? + epsilon).sqrt();
                let y = (gradient_norm_vec_1[i]
                    - avg_gradient_norm_arr_1_broadcasted.linear_index(i)?
                    - (norm * avg_gradient_norm_arr_2_broadcasted.linear_index(i)?))
                    / std;

                *arr_gradient.linear_index_mut(i)? += y;
            }

            storage.replace_grad(arr_gradient_idx, arr_gradient_take)?;
            storage.replace_grad(gamma_grad_idx, gamma_grad_take)?;
            storage.replace_grad(beta_grad_idx, beta_grad_take)?;
        }
    }

    Ok(())
}
