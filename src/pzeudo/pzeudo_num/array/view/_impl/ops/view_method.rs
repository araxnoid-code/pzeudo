use crate::{ArrayView, OpsSlicing};

impl<F> OpsSlicing<F> for ArrayView<'_, F> where F: Copy {}
