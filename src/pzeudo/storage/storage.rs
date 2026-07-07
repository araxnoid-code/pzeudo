use crate::PzeudoStorageErr;

pub trait StorageTrait<Element> {
    fn get_storage(&self) -> &Vec<Option<Element>>;
    fn get_empty_idx(&self) -> &Vec<usize>;

    fn get_mut_storage(&mut self) -> &mut Vec<Option<Element>>;
    fn get_mut_empty_idx(&mut self) -> &mut Vec<usize>;

    fn push_element(&mut self, elem: Element) -> Result<usize, PzeudoStorageErr> {
        let idx = if let Some(idx) = self.get_mut_empty_idx().pop() {
            self
                .get_mut_storage()
                .get_mut(idx)
                .ok_or(PzeudoStorageErr::IndexNotFoundErr(
                    String::from("PzeudoStorageErr, IndexNotFoundErr. the index of empty_idx is obtained. However, this index is not available in storage.")
                ))?
                .replace(elem)
                .map_or(Ok(()), |_| Err(PzeudoStorageErr::PushElementErr(String::from(""))
                ))?;
            idx
        } else {
            let storage = self.get_mut_storage();
            storage.push(Some(elem));
            storage.len() - 1
        };

        Ok(idx)
    }

    fn remove_element(&mut self, index: usize) -> Result<(), PzeudoStorageErr> {
        self.get_mut_storage()
            .get_mut(index)
            .ok_or(PzeudoStorageErr::IndexNotFoundErr(format!(
                "index {:?} not available on storage",
                index,
            )))?
            .take()
            .ok_or(PzeudoStorageErr::RemoveELementErr(format!(
                "element not available, possibly deleted"
            )))?;

        self.get_mut_empty_idx().push(index);

        Ok(())
    }

    fn clear(&mut self) {
        self.get_mut_storage().clear();
        self.get_mut_empty_idx().clear();
    }
}
