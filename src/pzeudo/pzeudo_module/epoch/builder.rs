pub struct EpochBuilder<M, T> {
    pub epoch: usize,
    pub model: M,
    pub arg: T,
}

impl<M, T> EpochBuilder<M, T> {
    pub fn new(epoch: usize, model: M, arg: T) -> EpochBuilder<M, T> {
        Self { epoch, model, arg }
    }
}
