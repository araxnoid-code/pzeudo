use std::{cell::RefCell, rc::Rc};

use crate::{OpsAble, OwnAble, TensorAble};

fn add<Ops>(lhs: &Ops, rhs: &Ops) -> impl OwnAble
where
    Ops: OpsAble,
{
    lhs.add(&rhs)
}

fn add_backward<T>(
    lhs_grad: &Option<Rc<RefCell<T>>>,
    rhs_grad: &Option<Rc<RefCell<T>>>,
    gradient: &Option<Rc<RefCell<T>>>,
) where
    T: TensorAble,
{
}
