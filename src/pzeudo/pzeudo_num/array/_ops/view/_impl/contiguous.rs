use crate::prelude::*;

impl<F> OpsSlice<F> for Array<F> where F: Copy {}
impl<F> OpsBroadcast<F> for Array<F> where F: Copy {}
impl<F> OpsPermute<F> for Array<F> where F: Copy {}
impl<F> OpsToShape<F> for Array<F> where F: Copy {}
