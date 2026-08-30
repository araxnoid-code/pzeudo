use std::ops::{AddAssign, Div, Mul};

use crate::prelude::*;
use num_traits::{Float, NumCast, Zero};
use rand::{SeedableRng, rngs::SmallRng};
use rand_distr::{Bernoulli, Distribution};

/// # Dropout
/// using an inverted dropout
/// - p: probability of a zero value
///     - example: p=0.7, Consequently, 70 percent of the values or elements will be dropped (at random locations, depending on the Bernoulli distribution and the specified seed).
/// ## Formula
/// element * drop_value / (1 - p)
/// ## Phase
/// - TrainPhase
/// Executes a formula based on the Bernoulli distribution and produces a new tensor.
/// - EvelPhase
/// It immediately returns a new tensor containing the same array data, state, and record as the previous tensor. The only difference is that the gradient is `None` (the previous tensor remains unaffected).
pub struct Dropout {
    rng: rand::rngs::SmallRng,
    p: f64,
}

impl Dropout {
    /// # Dropout
    /// using an inverted dropout
    /// - p: probability of a zero value
    ///     - example: p=0.7, Consequently, 70 percent of the values or elements will be dropped (at random locations, depending on the Bernoulli distribution and the specified seed).
    pub fn new(p: f64, seed: u64) -> Dropout {
        Self {
            p,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    /// # Dropout
    /// using an inverted dropout
    /// - p: probability of a zero value
    ///     - example: p=0.7, Consequently, 70 percent of the values or elements will be dropped (at random locations, depending on the Bernoulli distribution and the specified seed).
    /// ## Formula
    /// element * drop_value / (1 - p)
    /// ## Phase
    /// - TrainPhase
    /// Executes a formula based on the Bernoulli distribution and produces a new tensor.
    /// - EvelPhase
    /// It immediately returns a new tensor containing the same array data, state, and record as the previous tensor. The only difference is that the gradient is `None` (the previous tensor remains unaffected).
    pub fn forward<F, T, G, Phase>(
        &mut self,
        tensor: &Tensor<F, T, G>,
        phase: Phase,
    ) -> Result<Tensor<F, T, Phase>, PzeudoErr>
    where
        F: Float,
        Phase: PhaseStatus<F>,
        for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
    {
        if phase.is_eval() {
            let tensor = Tensor::_new(
                tensor.get_array_idx(),
                None,
                tensor.shape.to_vec(),
                tensor.record_status,
                tensor.get_record().clone(),
                tensor.get_storage().clone(),
            );

            return Ok(tensor);
        }
        let len = tensor.shape.iter().product::<usize>();

        let bernoulli = Bernoulli::new(self.p).map_err(|err| PzeudoErr::BernoulliErr(err))?;
        let zero = F::zero();

        let mut storage = tensor.get_storage().borrow_mut();
        let tensor_array =
            storage.get_as_array_ref::<T>(tensor.get_array_idx(), ContiguousType::Arr)?;

        let q = F::one()
            - F::from(self.p).ok_or(PzeudoErr::LayerErr(format!(
                "Droput::forward. Unable to cast on p"
            )))?;

        let mut out_vec = Vec::with_capacity(len);
        let mut mask: Vec<u8> = Vec::new();
        for i in 0..len {
            if bernoulli.sample(&mut self.rng) {
                mask.push(0);
                out_vec.push(zero);
                continue;
            }

            let y = tensor_array.linear_index(i)? / q;
            mask.push(1);
            out_vec.push(y);
        }

        let result_arr = Array::from_vector_with_shape(&out_vec, &tensor.shape)?;
        let array_idx = storage.push(ElementType::Arr(result_arr))?;
        let grad_idx = phase.into_zeros_grad_storage(&tensor.shape, &mut storage)?;

        let mut record = tensor.record.borrow_mut();
        let record_idx = record.len();
        let record_label = RecordLabel::Dropout(mask, q, grad_idx, tensor.get_grad_idx());
        record.push(record_label);

        let tensor = Tensor::_new(
            array_idx,
            grad_idx,
            tensor.shape.to_vec(),
            Some(RecordStatus::Record(record_idx)),
            tensor.record.clone(),
            tensor.storage.clone(),
        );

        Ok(tensor)
    }
}

pub fn dropout_backward<F>(
    mask: &[u8],
    q: F,
    arr_grad_idx: Option<StorageType>,
    grad_idx: Option<StorageType>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    F: AddAssign + Copy + Mul<Output = F> + Div<Output = F> + NumCast + Zero + PartialEq,
{
    if let Some(grad_idx) = grad_idx {
        if is_no_grad_or_time_not_match_or_no_update(grad_idx, storage)? {
            return Ok(());
        };

        if let Some(arr_grad_idx) = arr_grad_idx {
            storage.set_grad_update(arr_grad_idx, true)?;
            if is_no_grad_or_time_not_match_or_no_update(arr_grad_idx, storage)? {
                return Ok(());
            };

            let grad = storage.take_grad(grad_idx)?;
            let grad_ref = grad.to_array_ref::<Contiguous>();

            let mut arr_grad = storage.take_grad(arr_grad_idx)?;
            let mut arr_grad_ref = arr_grad.to_array_ref_mut::<View>();

            let len = arr_grad_ref.shape.iter().product::<usize>();
            for i in 0..len {
                if mask[i] != 0 {
                    *arr_grad_ref.linear_index_mut(i)? += grad_ref.linear_index(i)? / q;
                }
            }

            storage.replace_grad(grad_idx, grad)?;
            storage.replace_grad(arr_grad_idx, arr_grad)?;
        }
    }

    Ok(())
}
