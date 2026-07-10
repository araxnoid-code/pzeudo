use std::{cell::RefCell, rc::Rc};

use ndarray::ArrayD;
use num_traits::Float;

use crate::{GradientStorage, Module, NDArray, OpsLabel, PzeudoModuleErr, StorageTrait, Tensor};

// New Tensor
impl<'ops_label, F> Module<GradientStorage<F>, ArrayD<F>, OpsLabel<'ops_label, F>>
where
    F: Float,
{
    pub fn new_tensor<A>(
        &'ops_label self,
        array: A,
        gradient: Option<ArrayD<F>>,
        record: Option<OpsLabel<'ops_label, F>>,
    ) -> Result<Tensor<'ops_label, F, A, GradientStorage<F>>, PzeudoModuleErr>
    where
        F: Float,
        A: NDArray<F>,
    {
        let tensor = Tensor::new(
            array,
            gradient,
            self.grad_storage.clone(),
            record,
            self.record_storage.clone(),
        )
        .map_err(|err| PzeudoModuleErr::NewTensorErr(err.to_string()))?;

        Ok(tensor)
    }

    pub fn tensor_from_vector(
        &self,
        vector: Vec<F>,
        shape: &[usize],
    ) -> Result<Tensor<'ops_label, F, ArrayD<F>, GradientStorage<F>>, PzeudoModuleErr> {
        if shape.iter().product::<usize>() != vector.len() {
            return Err(PzeudoModuleErr::NewTensorErr(format!("")));
        }

        let array = ArrayD::<F>::from_shape_vec(shape, vector).map_err(|shape_err| {
            PzeudoModuleErr::NewTensorErr(format!("NewTensorErr. ndarray error: {shape_err}"))
        })?;

        let tensor = Tensor::from_array(
            array,
            self.grad_storage.clone(),
            self.record_storage.clone(),
        )
        .map_err(|err| PzeudoModuleErr::NewTensorErr(err.to_string()))?;

        Ok(tensor)
    }
}
