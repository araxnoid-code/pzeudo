use std::{
    fmt::Debug,
    ops::{Range, RangeFrom, RangeFull, RangeTo},
};

pub struct SliceRange {
    pub(crate) start: Option<usize>,
    pub(crate) end: Option<usize>,
}

impl Debug for SliceRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut start_str = String::new();
        if let Some(start) = self.start {
            start_str.push_str(&format!("{start}"));
        };

        let mut end_str = String::new();
        if let Some(end) = self.end {
            end_str.push_str(&format!("{end}"));
        };

        f.write_str(&format!("{}..{}", start_str, end_str))
    }
}

pub fn r<R>(r: R) -> SliceRange
where
    R: SlicingRangeTrait,
{
    SliceRange {
        start: r.start(),
        end: r.end(),
    }
}

pub trait SlicingRangeTrait {
    fn start(&self) -> Option<usize>;
    fn end(&self) -> Option<usize>;
}

impl SlicingRangeTrait for Range<usize> {
    fn start(&self) -> Option<usize> {
        Some(self.start)
    }

    fn end(&self) -> Option<usize> {
        Some(self.end)
    }
}

impl SlicingRangeTrait for RangeFull {
    fn start(&self) -> Option<usize> {
        None
    }

    fn end(&self) -> Option<usize> {
        None
    }
}

impl SlicingRangeTrait for RangeFrom<usize> {
    fn start(&self) -> Option<usize> {
        Some(self.start)
    }

    fn end(&self) -> Option<usize> {
        None
    }
}

impl SlicingRangeTrait for RangeTo<usize> {
    fn start(&self) -> Option<usize> {
        None
    }

    fn end(&self) -> Option<usize> {
        Some(self.end)
    }
}
