use crate::get_broadcast_dim;

pub fn broadcast_detech(
    shape_a: &[usize],
    shape_b: &[usize],
) -> (Option<Vec<usize>>, Option<Vec<usize>>) {
    if shape_a == shape_b {
        return (None, None);
    }

    if let Ok(mut dim) = get_broadcast_dim(shape_a, shape_b) {
        dim.reverse();
        return (Some(dim), None);
    } else if let Ok(mut dim) = get_broadcast_dim(shape_b, shape_a) {
        dim.reverse();
        return (None, Some(dim));
    }

    (None, None)
}
