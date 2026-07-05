use ndarray::ArrayD;

pub(crate) enum Element<F> {
    Own(ArrayD<F>),
    View(ArrayD<F>),
    Cow(ArrayD<F>),
}
