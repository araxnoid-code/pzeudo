pub enum WeightInit<F> {
    Xavier,
    HeIn,
    HeOut,
    Costum(F, F),
}
