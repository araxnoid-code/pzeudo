use std::ops::{Add, Sub};

use crate::{
    Array, ArrayTrait,
    PzeudoNumErr::{self, AddErr, SubErr},
    shape_to_stride,
};

pub fn sub<Lhs, Rhs, F>(lhs: &Lhs, rhs: &Rhs) -> Result<Array<F>, PzeudoNumErr>
where
    F: Copy + Sub<Output = F>,
    Lhs: ArrayTrait<F>,
    Rhs: ArrayTrait<F>,
{
    let lhs_metadata = lhs.get_metadata();
    let rhs_metadata = rhs.get_metadata();

    if lhs_metadata.shape != rhs_metadata.shape {
        return Err(SubErr(format!(
            "AddErr. add\ncannot reduce arrays of the form {:?} and {:?} because they are of different shapes",
            lhs_metadata.shape, rhs_metadata.shape
        )));
    }

    let len = lhs_metadata.shape.iter().product::<usize>();
    let mut output = Vec::with_capacity(len);
    for i in 0..len {
        let lhs_value = lhs.linear_index(i)?;
        let rhs_value = rhs.linear_index(i)?;
        output.push(lhs_value - rhs_value);
    }

    let shape = lhs_metadata.shape.to_vec();
    let array = Array::new(output, 0, shape_to_stride(&shape), shape);

    Ok(array)
}
