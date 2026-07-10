use num_traits::Float;

use crate::{GradientStorage, NDArray, PzeodoMethodErr, Tensor, TensorNDArrayOwn};

struct Linear<'ops_label, F>
where
    F: Float,
{
    //
    in_features: usize,
    out_features: usize,

    //
    weight: TensorNDArrayOwn<'ops_label, F>,
    bias: TensorNDArrayOwn<'ops_label, F>,
}

impl<'ops_label, F> Linear<'ops_label, F>
where
    F: Float + 'static,
{
    fn forward<A>(
        &'ops_label self,
        input: &'ops_label Tensor<'ops_label, F, A, GradientStorage<F>>,
    ) -> Result<(), PzeodoMethodErr>
    where
        A: NDArray<F>,
    {
        let matmul = input
            .matmul_2d(&self.weight)
            .map_err(|err| PzeodoMethodErr::LinearErr(format!("{err:?}")))?;

        // let result = matmul.add(&self.bias);

        Ok(())
    }
}
