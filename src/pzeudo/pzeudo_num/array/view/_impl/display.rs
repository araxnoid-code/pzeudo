use std::fmt::{Debug, Display};

use crate::prelude::*;

impl<F> ArrayView<'_, F>
where
    F: Copy + Debug,
{
    pub fn to_string(&self) -> Result<String, PzeudoErr> {
        let mut string = String::new();
        rec_helper(self, 0, &mut 0, &mut string)?;

        Ok(string)
    }
}

fn rec_helper<F>(
    arr: &ArrayView<'_, F>,
    level: usize,
    count: &mut usize,
    string: &mut String,
) -> Result<(), PzeudoErr>
where
    F: Debug + Copy,
{
    let shape = &arr.shape;

    if shape.len() == 1 {
        let end = shape.iter().product::<usize>();
        string.push_str("[");
        for idx in 0..end {
            let value = ArrayTrait::linear_index(arr, idx)?;
            string.push_str(&format!("{value:?}"));
            if idx < end - 1 {
                string.push_str(", ");
            }
        }
        string.push_str("]");
        return Ok(());
    }

    let space = " ".repeat(level);
    string.push_str(&space);
    string.push_str("[\n");
    for _ in 0..shape[level] {
        if shape.len() - 2 == level {
            let start = *count * *shape.last().unwrap();
            let end = start + shape.last().unwrap();
            string.push_str(&format!(" {}[", space));
            for idx in start..end {
                let value = ArrayTrait::linear_index(arr, idx)?;
                string.push_str(&format!("{value:?}"));
                if idx < end - 1 {
                    string.push_str(", ");
                }
            }
            string.push_str("]\n");
            *count += 1;
        } else {
            rec_helper(arr, level + 1, count, string)?;
        }
    }

    string.push_str(&space);
    string.push_str("]\n");

    Ok(())
}

impl<F> Display for ArrayView<'_, F>
where
    F: Debug + Copy,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&ArrayView::to_string(self).unwrap())
    }
}
