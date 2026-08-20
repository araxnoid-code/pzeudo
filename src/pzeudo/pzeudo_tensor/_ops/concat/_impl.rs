use crate::prelude::*;

impl<F, T, G> ConcatVector<F, T, G> for Vec<Tensor<F, T, G>> {
    fn _iter(&self) -> std::slice::Iter<'_, Tensor<F, T, G>> {
        self.iter()
    }

    fn _len(&self) -> usize {
        self.len()
    }

    fn _get(&self, idx: usize) -> Option<&Tensor<F, T, G>> {
        self.get(idx)
    }
}

impl<F, T, G> ConcatVectorRef<F, T, G> for Vec<&Tensor<F, T, G>> {
    fn _get(&self, idx: usize) -> Option<&&Tensor<F, T, G>> {
        self.get(idx)
    }
    fn _iter(&self) -> std::slice::Iter<'_, &Tensor<F, T, G>> {
        self.iter()
    }
    fn _len(&self) -> usize {
        self.len()
    }
}

impl<F, T, G, const N: usize> ConcatVector<F, T, G> for [Tensor<F, T, G>; N] {
    fn _iter(&self) -> std::slice::Iter<'_, Tensor<F, T, G>> {
        self.iter()
    }

    fn _len(&self) -> usize {
        self.len()
    }

    fn _get(&self, idx: usize) -> Option<&Tensor<F, T, G>> {
        self.get(idx)
    }
}

impl<F, T, G, const N: usize> ConcatVectorRef<F, T, G> for [&Tensor<F, T, G>; N] {
    fn _get(&self, idx: usize) -> Option<&&Tensor<F, T, G>> {
        self.get(idx)
    }
    fn _iter(&self) -> std::slice::Iter<'_, &Tensor<F, T, G>> {
        self.iter()
    }
    fn _len(&self) -> usize {
        self.len()
    }
}
