use crate::ContiguousType;

pub(super) fn storage_contiguous_type_check(
    contiguous_a: &ContiguousType,
    contiguous_b: &ContiguousType,
) -> Result<(), ()> {
    match (contiguous_a, contiguous_b) {
        (ContiguousType::Arr, ContiguousType::Grad) => return Err(()),
        (ContiguousType::Grad, ContiguousType::Arr) => return Err(()),
        (_, _) => return Ok(()),
    }
}
