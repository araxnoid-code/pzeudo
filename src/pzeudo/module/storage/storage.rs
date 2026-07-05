pub trait StorageDeref {
    type Storage;
    type Empty;

    fn get_mut_storage(&mut self) -> &mut Self::Storage;
    fn get_mut_empty(&mut self) -> &mut Self::Empty;
}

pub trait GradientStorage<Grad>:
    StorageDeref<Storage = Vec<Option<Grad>>, Empty = Vec<usize>>
{
    fn push_grad(&mut self, grad: Grad) -> usize {
        if let Some(idx) = self.get_mut_empty().pop() {
            self.get_mut_storage()[idx] = Some(grad);
            idx
        } else {
            self.get_mut_storage().push(Some(grad));
            self.get_mut_storage().len()
        }
    }

    fn remove_grad(&mut self, idx: usize) -> Result<(), &'static str> {
        self.get_mut_storage()
            .get(idx)
            .ok_or("error deleting gradient because index not found in storage")?
            .as_ref()
            .ok_or("error deleting gradient because element has been deleted")?;
        self.get_mut_storage()[idx] = None;
        self.get_mut_empty().push(idx);
        Ok(())
    }

    fn clear(&mut self) {
        self.get_mut_storage().clear();
        self.get_mut_empty().clear();
    }
}
