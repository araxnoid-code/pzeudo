use std::format;

use crate::prelude::*;

// 1
#[derive(Debug)]
pub enum ContiguousType {
    Arr,
    Grad,
}

// 16
#[derive(Clone, Copy, Debug)]
pub enum ViewStorageType {
    Permanent(usize),
    Storage(usize),
}

#[derive(Clone, Copy, Debug)]
pub enum StorageType {
    Permanent(usize),
    Arr(usize),
    View(usize),
}

impl StorageType {
    pub fn to_view_element_type(&self) -> Result<ViewStorageType, PzeudoErr> {
        match self {
            Self::View(_) => Err(PzeudoErr::CastingStorageTypeToView(format!(""))),
            Self::Arr(arr_idx) => Ok(ViewStorageType::Storage(*arr_idx)),
            Self::Permanent(permanent_idx) => Ok(ViewStorageType::Permanent(*permanent_idx)),
        }
    }
}

pub enum ElementType<F> {
    Arr(Array<F>),
    Grad(Array<F>),
    View(TensorMetadata),
}
