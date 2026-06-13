use std::sync::RwLock;

use ndarray::{ArrayBase, Dim, IxDynImpl, OwnedRepr, ViewRepr};

pub enum NDArrayF64<'a> {
    OwnRepr(RwLock<ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>, f64>>),
    ViewRepr(RwLock<ArrayBase<ViewRepr<&'a f64>, Dim<IxDynImpl>, f64>>),
}

impl<'a> NDArrayF64<'a> {
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
