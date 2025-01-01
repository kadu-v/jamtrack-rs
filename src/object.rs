use crate::rect::Rect;
use crate::strack::STrack;
use std::fmt::Debug;

/* ------------------------------------------------------------------------------
 * Object struct
 * ------------------------------------------------------------------------------ */

#[derive(Debug, Clone)]
pub struct Object {
    rect: Rect<f32>,
    label: usize,
    prob: f32,
    track_id: Option<usize>,
}

impl Object {
    pub fn new(
        rect: Rect<f32>,
        label: usize,
        prob: f32,
        track_id: Option<usize>,
    ) -> Self {
        Self {
            rect,
            label,
            prob,
            track_id,
        }
    }

    #[inline(always)]
    pub fn get_rect(&self) -> Rect<f32> {
        self.rect.clone()
    }

    #[inline(always)]
    pub fn get_x(&self) -> f32 {
        self.rect.x()
    }

    #[inline(always)]
    pub fn get_y(&self) -> f32 {
        self.rect.y()
    }

    #[inline(always)]
    pub fn get_width(&self) -> f32 {
        self.rect.width()
    }

    #[inline(always)]
    pub fn get_height(&self) -> f32 {
        self.rect.height()
    }

    #[inline(always)]
    pub fn get_label(&self) -> usize {
        self.label
    }

    #[inline(always)]
    pub fn get_prob(&self) -> f32 {
        self.prob
    }

    #[inline(always)]
    pub fn get_track_id(&self) -> Option<usize> {
        self.track_id
    }
}

impl From<STrack> for Object {
    fn from(strack: STrack) -> Self {
        Object::new(
            strack.get_rect(),
            strack.get_track_id(),
            strack.get_score(),
            Some(strack.get_track_id()),
        )
    }
}

impl From<&STrack> for Object {
    fn from(strack: &STrack) -> Self {
        Object::new(
            strack.get_rect(),
            strack.get_frame_id(),
            strack.get_score(),
            Some(strack.get_track_id()),
        )
    }
}
