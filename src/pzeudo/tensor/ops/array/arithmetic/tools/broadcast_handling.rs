use ndarray::{ArrayBase, ArrayD, ArrayViewD, Axis, Dim, IxDynImpl, OwnedRepr};
use num_traits::Zero;

pub fn broadcast_handling<F: Clone + Zero>(
    input: ArrayViewD<F>,
    gradient: ArrayViewD<F>,
    init: ArrayViewD<'_, F>,
) -> ArrayD<F> {
    let gradient_shape = gradient.shape();
    let mut input_shape = input.shape().to_vec();
    let distance = gradient_shape.len() - input_shape.len();
    if distance != 0 {
        let ones = vec![1; distance];
        input_shape = [ones, input_shape].concat();
    }

    let mut gradient_axis: Option<ArrayBase<OwnedRepr<F>, Dim<IxDynImpl>, F>> = None;
    for (idx, (gradient_shape, lhs_shape)) in
        gradient_shape.iter().zip(input_shape.iter()).enumerate()
    {
        if *lhs_shape == 1 && *gradient_shape != 1 {
            if let Some(grad) = gradient_axis {
                gradient_axis = Some(grad.sum_axis(Axis(idx)).insert_axis(Axis(idx)));
            } else {
                gradient_axis = Some(init.sum_axis(Axis(idx)).insert_axis(Axis(idx)));
            }
        }
    }

    let gradient = gradient_axis.unwrap();
    let gradient = gradient.to_shape(input.shape()).unwrap();
    gradient.to_owned()
}
