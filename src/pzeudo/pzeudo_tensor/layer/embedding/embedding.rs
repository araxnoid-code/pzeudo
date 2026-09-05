use num_traits::Float;
use rand_distr::{Distribution, Normal, StandardNormal};
use std::ops::AddAssign;

use crate::prelude::*;

/// # Embedding
/// ```md
/// embedding_num = The number of weights to be made
/// embedding_dim = The length of the weight parameters to be created
/// ```
/// Initialization using a normal distribution with:
/// ```md
/// mean = 0
/// std_dev = 1
/// ```
pub struct Embedding<F, Grad> {
    embedding_dim: usize,
    weights: Vec<Tensor<F, Contiguous, Grad>>,
}

impl<F, Grad> Embedding<F, Grad> {
    pub fn new(
        embedding_num: usize,
        embedding_dim: usize,
        model_builder: &mut ModelBuilder<F>,
        requires_grad: Grad,
    ) -> Result<Embedding<F, Grad>, PzeudoErr>
    where
        Grad: ReqGradTrait<F>,
        F: Float,
        StandardNormal: Distribution<F>,
    {
        let mut weights = Vec::with_capacity(embedding_num);
        let normal =
            Normal::new(F::zero(), F::one()).map_err(|err| PzeudoErr::RandDistrNormalErr(err))?;
        for _ in 0..embedding_num {
            let vec = model_builder.get_load_else_generate_vec(embedding_dim, &normal)?;
            let tensor = Tensor::param_from_vector_with_shape(
                &vec,
                &[embedding_dim],
                model_builder.get_module(),
                requires_grad,
            )?;
            weights.push(tensor);
        }

        Ok(Embedding {
            embedding_dim,
            weights: weights,
        })
    }

    pub fn forward<T, TensorGrad, ReqGrad>(
        &self,
        tensor: &Tensor<F, T, TensorGrad>,
        requires_grad: ReqGrad,
    ) -> Result<Tensor<F, Contiguous, ReqGrad>, PzeudoErr>
    where
        for<'a> ArrayRef<'a, F, T>: ArrayTrait<F>,
        F: Copy + EmbeddingIndex,
        ReqGrad: ReqGradTrait<F>,
    {
        let mut storage = tensor.get_storage().borrow_mut();
        let mut n_shape = tensor.get_shape().to_vec();
        let len = n_shape.iter().product::<usize>();
        n_shape.push(self.embedding_dim);

        let tensor_array =
            storage.get_as_array_ref::<T>(tensor.get_array_idx(), ContiguousType::Arr)?;
        // println!("test");

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
            let record_label = RecordLabel::Embedding(embedding_grads, grad_idx);
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

pub fn embedding_backward<F>(
    embedding_grads: &[Option<StorageType>],
    grad_idx: Option<StorageType>,
    storage: &mut ArrayStorage<F>,
) -> Result<(), PzeudoErr>
where
    F: AddAssign + Copy,
{
    if let Some(grad_idx) = grad_idx {
        if is_no_grad_or_time_not_match_or_no_update(grad_idx, storage)? {
            return Ok(());
        };

        let grad_take = storage.take_grad(grad_idx)?;
        let grad_ref = grad_take.to_array_ref::<Contiguous>();
        let embedding_dim = grad_ref.shape[grad_ref.shape.len() - 1];

        for (idx, embedding_grad_idx) in embedding_grads.iter().enumerate() {
            if let Some(embedding_grad_idx) = embedding_grad_idx {
                // println!("{:?}", embedding_grad_idx);
                storage.set_grad_update(*embedding_grad_idx, true)?;
                if is_no_grad_or_time_not_match_or_no_update(*embedding_grad_idx, storage)? {
                    continue;
                };

                let array = storage.get_as_array_ref_mut::<Contiguous>(
                    *embedding_grad_idx,
                    ContiguousType::Grad,
                )?;

                let start = idx * embedding_dim;
                let end = start + embedding_dim;
                for (array_val, grad_val) in
                    array.data.iter_mut().zip(grad_ref.data[start..end].iter())
                {
                    *array_val += *grad_val;
                }
            }
        }

        storage.replace_grad(grad_idx, grad_take)?;
    }

    Ok(())
}
