use std::fmt::{Debug, Display};

use crate::Array;

impl<F> Array<F> {
    pub fn _to_string(&self) -> String
    where
        F: Debug,
    {
        let data = &self.data;
        let shape = &self.shape;
        let mut string = String::new();
        rec_helper(data, shape, 0, &mut 0, &mut string);

        string
    }
}

fn rec_helper<F>(data: &[F], shape: &[usize], level: usize, count: &mut usize, string: &mut String)
where
    F: Debug,
{
    if shape.len() == 1 {
        string.push_str(&format!("{data:?}"));
        return;
    }

    let space = " ".repeat(level);
    string.push_str(&space);
    string.push_str("[\n");
    for _ in 0..shape[level] {
        if shape.len() - 2 == level {
            let start = *count * *shape.last().unwrap();
            let end = start + shape.last().unwrap();
            let slice_str = format!(" {}{:?}\n", space, &data[start..end]);
            string.push_str(&slice_str);
            *count += 1;
        } else {
            rec_helper(data, shape, level + 1, count, string);
        }
    }

    string.push_str(&space);
    string.push_str("]\n");
}

impl<F> Display for Array<F>
where
    F: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&Array::_to_string(self))
    }
}
