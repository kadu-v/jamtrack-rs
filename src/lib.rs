pub mod boost_tracker;
pub mod byte_tracker;
pub mod error;
pub mod lapjv;
pub mod object;

pub use byte_tracker::rect;
pub use byte_tracker::strack;

use crate::error::TrackError;
use crate::object::Object;

pub trait Tracker {
    fn update(
        &mut self,
        detections: Vec<Object>,
    ) -> Result<Vec<Object>, TrackError>;
}
