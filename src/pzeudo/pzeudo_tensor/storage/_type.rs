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
    Param(usize),
    Storage(usize, Option<usize>),
}

#[derive(Clone, Copy, Debug)]
pub enum StorageType {
    Param(usize),
    Arr(usize, Option<usize>),
    View(usize),
}

impl StorageType {
    pub fn to_view_element_type(&self) -> Result<ViewStorageType, PzeudoErr> {
        match self {
            Self::View(_) => Err(PzeudoErr::StorageErr(format!(
                "StorageType::to_view_element_type. Cannot cast StorageType::View to ViewStorageType."
            ))),
            Self::Arr(arr_idx, grad_time) => Ok(ViewStorageType::Storage(*arr_idx, *grad_time)),
            Self::Param(permanent_idx) => Ok(ViewStorageType::Param(*permanent_idx)),
        }
    }
}

pub enum ElementType<F> {
    Arr(Array<F>),
    Grad(Array<F>),
    View(TensorMetadata),
}
