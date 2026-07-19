use crate::prelude::*;

impl<F> OpsSlicing<F> for Array<F> where F: Copy {}
impl<F> OpsBroadcast<F> for Array<F> where F: Copy {}
impl<F> OpsPermute<F> for Array<F> where F: Copy {}
