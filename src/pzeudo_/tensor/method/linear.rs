use num_traits::Float;

use crate::{GradientStorage, NDArray, PzeodoMethodErr, Tensor, TensorNDArrayOwn};

pub struct Linear<'ops_label, F>
where
    F: Float,
{
    //
    pub in_features: usize,
    pub out_features: usize,

    //
    pub weight: TensorNDArrayOwn<'ops_label, F>,
    pub bias: TensorNDArrayOwn<'ops_label, F>,
}

impl<F> Linear<'_, F>
where
    F: Float + 'static,
{
    pub fn forward<'a, A>(
        &'a self,
        input: &'a Tensor<'a, F, A, GradientStorage<F>>,
    ) -> Result<(), PzeodoMethodErr>
    where
        A: NDArray<F>,
    {
        let matmul = input
            .matmul_2d(&self.weight)
            .map_err(|err| PzeodoMethodErr::LinearErr(format!("{err:?}")))?;

        println!("matmul added");
        // let result = matmul.add(&self.bias);

        Ok(())
    }
}
