pub fn shape_to_stride(shape: &[usize]) -> Vec<usize> {
    (0..shape.len())
        .map(|idx| shape[idx + 1..].iter().product::<usize>())
        .collect::<Vec<usize>>()
}
