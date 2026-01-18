use std::collections::HashMap;

use jamtrack_rs::{
    boost_tracker::BoostTracker, byte_tracker::ByteTracker, object::Object, rect::Rect,
};
use serde::Deserialize;

const DETECTION_JSON_PATH: &str = "data/jsons/detection_results.json";

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct DetectionJson {
    name: String,
    fps: usize,
    track_buffer: usize,
    results: Vec<DetectionResultJson>,
}

#[derive(Debug, Deserialize, Clone)]
struct DetectionResultJson {
    frame_id: String,
    prob: String,
    x: String,
    y: String,
    width: String,
    height: String,
}

impl From<DetectionResultJson> for Object {
    fn from(detection_result: DetectionResultJson) -> Self {
        let x = detection_result.x.parse::<f32>().unwrap();
        let y = detection_result.y.parse::<f32>().unwrap();
        let width = detection_result.width.parse::<f32>().unwrap();
        let height = detection_result.height.parse::<f32>().unwrap();
        let prob = detection_result.prob.parse::<f32>().unwrap();
        let rect = Rect::new(x, y, width, height);
        Object::new(rect, prob, None)
    }
}

fn read_detection_json(path: &str) -> DetectionJson {
    let file = std::fs::File::open(path).unwrap();
    serde_json::from_reader(file).unwrap()
}

fn load_detections() -> Vec<(usize, Vec<Object>)> {
    let detection = read_detection_json(DETECTION_JSON_PATH);
    let mut detections = HashMap::<usize, Vec<Object>>::new();
    for det in detection.results {
        let obj = Object::from(det.clone());
        let frame_id = det.frame_id.parse::<usize>().unwrap();

        if let Some(objs) = detections.get_mut(&frame_id) {
            objs.push(obj);
        } else {
            detections.insert(frame_id, vec![obj]);
        }
    }
    let mut detections = detections
        .into_iter()
        .collect::<Vec<(usize, Vec<Object>)>>();
    detections.sort_by(|a, b| a.0.cmp(&b.0));
    detections
}

fn main() {
    let detections = load_detections();
    let num_frames = detections.len();

    println!("=== Tracker Count Verification ===");
    println!("Total frames: {}", num_frames);
    println!();

    // ByteTracker
    {
        let mut tracker = ByteTracker::new(60, 60, 0.5, 0.6, 0.8);

        for (_, objs) in detections.iter() {
            let _ = tracker.update(objs);
        }
        println!("ByteTracker: (tracker count not exposed)");
    }

    // BoostTrack (basic)
    {
        let mut tracker = BoostTracker::new(0.5, 0.3, 30, 3);
        let mut total_tracker_count = 0usize;
        let mut max_tracker_count = 0usize;

        for (_, objs) in detections.iter() {
            let _ = tracker.update(objs);
            let count = tracker.tracker_count();
            total_tracker_count += count;
            max_tracker_count = max_tracker_count.max(count);
        }

        println!("BoostTrack (basic):");
        println!("  Final tracker count: {}", tracker.tracker_count());
        println!("  Max tracker count:   {}", max_tracker_count);
        println!("  Avg tracker count:   {:.2}", total_tracker_count as f64 / num_frames as f64);
        println!();
    }

    // BoostTrack+
    {
        let mut tracker = BoostTracker::new(0.5, 0.3, 30, 3).with_boost_plus();
        let mut total_tracker_count = 0usize;
        let mut max_tracker_count = 0usize;

        for (_, objs) in detections.iter() {
            let _ = tracker.update(objs);
            let count = tracker.tracker_count();
            total_tracker_count += count;
            max_tracker_count = max_tracker_count.max(count);
        }

        println!("BoostTrack+:");
        println!("  Final tracker count: {}", tracker.tracker_count());
        println!("  Max tracker count:   {}", max_tracker_count);
        println!("  Avg tracker count:   {:.2}", total_tracker_count as f64 / num_frames as f64);
        println!();
    }

    // BoostTrack++
    {
        let mut tracker = BoostTracker::new(0.5, 0.3, 30, 3).with_boost_plus_plus();
        let mut total_tracker_count = 0usize;
        let mut max_tracker_count = 0usize;

        for (_, objs) in detections.iter() {
            let _ = tracker.update(objs);
            let count = tracker.tracker_count();
            total_tracker_count += count;
            max_tracker_count = max_tracker_count.max(count);
        }

        println!("BoostTrack++:");
        println!("  Final tracker count: {}", tracker.tracker_count());
        println!("  Max tracker count:   {}", max_tracker_count);
        println!("  Avg tracker count:   {:.2}", total_tracker_count as f64 / num_frames as f64);
        println!();
    }
}
