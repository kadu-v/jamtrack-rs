use crate::rect::Rect;
use num::Float;
use std::fmt::Debug;

/*------------------------------------------------------------------------------
Object struct
------------------------------------------------------------------------------*/

#[derive(Debug, Clone)]
pub struct Object<T>
where
    T: Debug + Float,
{
    pub rect: Rect<T>,
    pub label: usize,
    pub prob: f32,
}

impl<T> Object<T>
where
    T: Clone + Debug + Float,
{
    pub fn new(rect: Rect<T>, label: usize, prob: f32) -> Self {
        Self { rect, label, prob }
    }
}
