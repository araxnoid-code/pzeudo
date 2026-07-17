pub struct Metadata<'a, F> {
    pub data: &'a [F],
    pub offset: usize,
    pub stride: &'a [usize],
    pub shape: &'a [usize],
}
