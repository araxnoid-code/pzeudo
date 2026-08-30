use num_traits::Float;
use rand_distr::{Distribution, Normal, StandardNormal};
use std::array;

use crate::prelude::*;

pub struct Embedding<const NUM: usize, F, Grad> {
    embedding_dim: usize,
    weights: [Tensor<F, Contiguous, Grad>; NUM],
}

impl<const NUM: usize, F, Grad> Embedding<NUM, F, Grad> {
    pub fn new(
        embedding_dim: usize,
        model_builder: &mut ModelBuilder<F>,
        requires_grad: Grad,
    ) -> Result<Embedding<NUM, F, Grad>, PzeudoErr>
    where
        Grad: ReqGradTrait<F>,
        F: Float,
        StandardNormal: Distribution<F>,
    {
        let module = model_builder.get_module();

        let zero = F::zero();
        let one = F::one();
        let normal = Normal::new(zero, one).map_err(|err| PzeudoErr::RandDistrNormalErr(err))?;
        let weights: [Tensor<F, Contiguous, Grad>; NUM] = array::from_fn(|_| {
            let mut vec = Vec::with_capacity(embedding_dim);
            for _ in 0..embedding_dim {
                vec.push(normal.sample(&mut module.rng));
            }
            Tensor::<_, _, Grad>::from_vector_with_shape(
                &vec,
                &[embedding_dim],
                module,
                requires_grad,
            )
            .unwrap()
        });

        Ok(Embedding {
            embedding_dim,
            weights: weights,
        })
    }

    pub fn forward<T, ReqGrad>(
        &self,
        tensor: Tensor<F, T, ReqGrad>,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
    where
        for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
        F: Copy + EmbbedingIndex,
        ReqGrad: ReqGradTrait<F>,
    {
        let mut storage = tensor.get_storage().borrow_mut();
        let mut n_shape = tensor.get_shape().to_vec();
        let len = n_shape.iter().product::<usize>();
        n_shape.push(self.embedding_dim);

        let tensor_array =
            storage.get_as_array_ref::<T>(tensor.get_array_idx(), ContiguousType::Arr)?;

        let mut embedding_grads = Vec::with_capacity(len);
        let mut vec = Vec::with_capacity(len);
        for i in 0..len {
            let idx = tensor_array.linear_index(i)?.into_usize();
            let storage_idx = self.weights[idx].get_array_idx();
            let arr = storage.get_as_array_ref::<Contiguous>(storage_idx, ContiguousType::Arr)?;
            vec.extend_from_slice(arr.data);
            embedding_grads.push(self.weights[idx].get_grad_idx());
        }

        let array = Array::from_vector_with_shape(&vec, &n_shape)?;
        let array_idx = storage.push(ElementType::Arr(array))?;
        let grad_idx = requires_grad.into_zeros_grad_storage(&n_shape, &mut storage)?;

        let record_status = if requires_grad.is_grad() {
            let mut record = tensor.get_record().borrow_mut();
            let record_label =
                RecordLabel::Embedding(embedding_grads, tensor.get_array_idx(), grad_idx);
            let record_idx = RecordStatus::Record(record.len());
            record.push(record_label);
            Some(record_idx)
        } else {
            None
        };

        let tensor = Tensor::_new(
            array_idx,
            grad_idx,
            n_shape,
            record_status,
            tensor.get_record().clone(),
            tensor.get_storage().clone(),
        );

        Ok(tensor)
    }
}
