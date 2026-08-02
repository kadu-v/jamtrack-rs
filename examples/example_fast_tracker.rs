use jamtrack_rs::{FastTracker, Object, Rect};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut tracker = FastTracker::new(30, 30, 0.6, 0.7);

    let frames = [
        vec![Object::new(Rect::new(100.0, 100.0, 50.0, 80.0), 0.9, None)],
        vec![Object::new(Rect::new(103.0, 102.0, 50.0, 80.0), 0.88, None)],
    ];

    for detections in frames {
        for track in tracker.update(&detections)? {
            println!(
                "frame={} id={:?} rect={:?}",
                tracker.frame_count(),
                track.get_track_id(),
                track.get_rect()
            );
        }
    }
    Ok(())
}
