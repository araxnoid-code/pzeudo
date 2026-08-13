pub struct EpochBuilder<T> {
    pub epoch: usize,
    pub arg: T,
}

impl<T> EpochBuilder<T> {
    pub fn new(epoch: usize, arg: T) -> EpochBuilder<T> {
        Self { epoch, arg }
    }
}
