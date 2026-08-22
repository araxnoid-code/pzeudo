use crate::prelude::*;
use num_traits::Float;
use rand::{SeedableRng, rngs::SmallRng};
use rand_distr::{Bernoulli, Distribution};

/// # Dropout
/// using an inverted dropout
/// - p: probability of a zero value
///     - contoh: p=0.7, Consequently, 70 percent of the values or elements will be dropped (at random locations, depending on the Bernoulli distribution and the specified seed).
/// ## Formula
/// element * drop_value / (1 - p)
pub struct Dropout {
    rng: rand::rngs::SmallRng,
    p: f64,
}

impl Dropout {
    /// # Dropout
    /// using an inverted dropout
    /// - p: probability of a zero value
    ///     - contoh: p=0.7, Consequently, 70 percent of the values or elements will be dropped (at random locations, depending on the Bernoulli distribution and the specified seed).
    pub fn new(p: f64, seed: u64) -> Dropout {
        Self {
            p,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    /// # Dropout
    /// using an inverted dropout
    /// - p: probability of a zero value
    ///     - contoh: p=0.7, Consequently, 70 percent of the values or elements will be dropped (at random locations, depending on the Bernoulli distribution and the specified seed).
    /// ## Formula
    /// element * drop_value / (1 - p)
    pub fn forward<F, T, G, ReqGrad>(
        &mut self,
        tensor: Tensor<F, T, G>,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
    where
        F: Float,
        ReqGrad: ReqGradTrait<F>,
        for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
    {
        let len = tensor.shape.iter().product::<usize>();

        let bernoulli = Bernoulli::new(self.p).map_err(|err| PzeudoErr::BernoulliErr(err))?;
        let zero = F::zero();

        let mut storage = tensor.get_storage().borrow_mut();
        let tensor_array =
            storage.get_as_array_ref::<T>(tensor.get_array_idx(), ContiguousType::Arr)?;

        let prop = F::one()
            - F::from(self.p).ok_or(PzeudoErr::LayerErr(format!(
                "Droput::forward. Unable to cast on p"
            )))?;
        let mut out_vec = Vec::with_capacity(len);
        for i in 0..len {
            if bernoulli.sample(&mut self.rng) {
                out_vec.push(zero);
                continue;
            }

            let y = tensor_array.linear_index(i)? / prop;
            out_vec.push(y);
        }

        let result_arr = Array::from_vector_with_shape(&out_vec, &tensor.shape)?;
        let array_idx = storage.push(ElementType::Arr(result_arr))?;
        let grad_idx = requires_grad.into_zeros_grad_storage(&tensor.shape, &mut storage)?;

        let tensor = Tensor::_new(
            array_idx,
            grad_idx,
            tensor.shape.to_vec(),
            None,
            tensor.record.clone(),
            tensor.storage.clone(),
        );

        Ok(tensor)
    }
}
