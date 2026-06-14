use std::sync::{RwLock, RwLockReadGuard};

use ndarray::{ArrayBase, Dim, IxDynImpl, OwnedRepr, SliceInfo, ViewRepr, s};

pub enum NDArrayF64<'a> {
    OwnRepr(RwLock<ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>, f64>>),
    ViewRepr(RwLock<ArrayBase<ViewRepr<&'a f64>, Dim<IxDynImpl>, f64>>),
}

fn get() -> SliceInfo<[ndarray::SliceInfoElem; 1], Dim<IxDynImpl>, Dim<IxDynImpl>> {
    panic!()
}

impl<'a> NDArrayF64<'a> {
    pub fn test_slice<'b>(&'b self) {
        let slice = match self {
            Self::OwnRepr(arr) => {
                let read: RwLockReadGuard<'b, ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>, f64>> =
                    arr.read().unwrap();
                NDArrayF64::ViewRepr(RwLock::new(read.slice(get())))
            }
            Self::ViewRepr(arr) => {
                let read = arr.read().unwrap();
                NDArrayF64::ViewRepr(RwLock::new(read.slice(get())))
            }
        };
    }

    pub fn add(&'a self, rhs: &Self) -> NDArrayF64<'a> {
        match (self, rhs) {
            (Self::OwnRepr(rhs), Self::OwnRepr(lhs)) => {
                Self::OwnRepr(RwLock::new(&*rhs.read().unwrap() + &*lhs.read().unwrap()))
            }
            (Self::OwnRepr(rhs), Self::ViewRepr(lhs)) => {
                Self::OwnRepr(RwLock::new(&*rhs.read().unwrap() + &*lhs.read().unwrap()))
            }
            (Self::ViewRepr(rhs), Self::OwnRepr(lhs)) => {
                Self::OwnRepr(RwLock::new(&*rhs.read().unwrap() + &*lhs.read().unwrap()))
            }
            (Self::ViewRepr(rhs), Self::ViewRepr(lhs)) => {
                Self::OwnRepr(RwLock::new(&*rhs.read().unwrap() + &*lhs.read().unwrap()))
            }
        }
    }

    pub fn sub(&'a self, rhs: &Self) -> NDArrayF64<'a> {
        match (self, rhs) {
            (Self::OwnRepr(rhs), Self::OwnRepr(lhs)) => {
                Self::OwnRepr(RwLock::new(&*rhs.read().unwrap() - &*lhs.read().unwrap()))
            }
            (Self::OwnRepr(rhs), Self::ViewRepr(lhs)) => {
                Self::OwnRepr(RwLock::new(&*rhs.read().unwrap() - &*lhs.read().unwrap()))
            }
            (Self::ViewRepr(rhs), Self::OwnRepr(lhs)) => {
                Self::OwnRepr(RwLock::new(&*rhs.read().unwrap() - &*lhs.read().unwrap()))
            }
            (Self::ViewRepr(rhs), Self::ViewRepr(lhs)) => {
                Self::OwnRepr(RwLock::new(&*rhs.read().unwrap() - &*lhs.read().unwrap()))
            }
        }
    }

    pub fn mul(&'a self, rhs: &Self) -> NDArrayF64<'a> {
        match (self, rhs) {
            (Self::OwnRepr(rhs), Self::OwnRepr(lhs)) => {
                Self::OwnRepr(RwLock::new(&*rhs.read().unwrap() * &*lhs.read().unwrap()))
            }
            (Self::OwnRepr(rhs), Self::ViewRepr(lhs)) => {
                Self::OwnRepr(RwLock::new(&*rhs.read().unwrap() * &*lhs.read().unwrap()))
            }
            (Self::ViewRepr(rhs), Self::OwnRepr(lhs)) => {
                Self::OwnRepr(RwLock::new(&*rhs.read().unwrap() * &*lhs.read().unwrap()))
            }
            (Self::ViewRepr(rhs), Self::ViewRepr(lhs)) => {
                Self::OwnRepr(RwLock::new(&*rhs.read().unwrap() * &*lhs.read().unwrap()))
            }
        }
    }

    pub fn div(&'a self, rhs: &Self) -> NDArrayF64<'a> {
        match (self, rhs) {
            (Self::OwnRepr(rhs), Self::OwnRepr(lhs)) => {
                Self::OwnRepr(RwLock::new(&*rhs.read().unwrap() / &*lhs.read().unwrap()))
            }
            (Self::OwnRepr(rhs), Self::ViewRepr(lhs)) => {
                Self::OwnRepr(RwLock::new(&*rhs.read().unwrap() / &*lhs.read().unwrap()))
            }
            (Self::ViewRepr(rhs), Self::OwnRepr(lhs)) => {
                Self::OwnRepr(RwLock::new(&*rhs.read().unwrap() / &*lhs.read().unwrap()))
            }
            (Self::ViewRepr(rhs), Self::ViewRepr(lhs)) => {
                Self::OwnRepr(RwLock::new(&*rhs.read().unwrap() / &*lhs.read().unwrap()))
            }
        }
    }
}
