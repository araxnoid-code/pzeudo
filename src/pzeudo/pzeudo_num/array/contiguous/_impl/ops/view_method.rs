use crate::{Array, OpsBroadcast, OpsSlicing};

impl<F> OpsSlicing<F> for Array<F> where F: Copy {}
impl<F> OpsBroadcast<F> for Array<F> where F: Copy {}
