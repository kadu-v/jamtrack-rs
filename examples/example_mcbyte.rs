use image::{GrayImage, Luma};
use jamtrack_rs::{McByteMask, McByteTracker, Object, Rect};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut tracker = McByteTracker::default().without_cmc();
    let first_frame =
        vec![Object::new(Rect::new(20.0, 20.0, 30.0, 50.0), 0.9, None)];
    let tracks = tracker.update(&first_frame)?;
    let track_id = tracks[0].get_track_id().unwrap();

    // In a real application this mask is propagated to the current frame by
    // an external segmenter such as Cutie. Non-zero pixels belong to track_id.
    let mut mask = GrayImage::new(160, 120);
    for y in 20..70 {
        for x in 22..52 {
            mask.put_pixel(x, y, Luma([255]));
        }
    }
    let masks = vec![McByteMask::new(track_id, mask, 0.95)];
    let detections =
        vec![Object::new(Rect::new(22.0, 20.0, 30.0, 50.0), 0.9, None)];

    for object in tracker.update_with_masks(&detections, &masks)? {
        println!(
            "Track ID: {:?}, Rect: {:?}",
            object.get_track_id(),
            object.get_rect()
        );
    }
    Ok(())
}
