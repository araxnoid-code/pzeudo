pub trait EmbeddingIndex {
    fn into_usize(&self) -> usize;
}

impl EmbeddingIndex for f32 {
    fn into_usize(&self) -> usize {
        *self as usize
    }
}

impl EmbeddingIndex for f64 {
    fn into_usize(&self) -> usize {
        *self as usize
    }
}
