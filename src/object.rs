use crate::rect::Rect;
use num::Float;
use std::fmt::Debug;

/*------------------------------------------------------------------------------
Object struct
------------------------------------------------------------------------------*/

#[derive(Debug, Clone)]
pub struct Object {
    pub rect: Rect<f32>,
    pub label: usize,
    pub prob: f32,
}

impl Object {
    pub fn new(rect: Rect<f32>, label: usize, prob: f32) -> Self {
        Self { rect, label, prob }
    }
}
