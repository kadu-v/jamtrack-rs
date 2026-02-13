use std::collections::HashMap;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use image::{GrayImage, Luma};
use jamtrack_rs::{boost_tracker::BoostTracker, object::Object, rect::Rect};
use serde::Deserialize;

const DETECTION_JSON_PATH: &str = "data/jsons/detection_results.json";

/* ----------------------------------------------------------------------------
 * Json schema for tracking results
 * ---------------------------------------------------------------------------- */

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

/* ----------------------------------------------------------------------------
 * Read json
 * ---------------------------------------------------------------------------- */
fn read_detection_json(path: &str) -> DetectionJson {
    let file = std::fs::File::open(path).unwrap();
    let detection = serde_json::from_reader(file).unwrap();
    detection
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

fn synthetic_frames(
    num_frames: usize,
    width: u32,
    height: u32,
) -> Vec<GrayImage> {
    let mut frames = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let mut frame = GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                // Deterministic pattern with mild frame-to-frame drift.
                let v = (((x + y + (i as u32 % 23)) % 255) as u8).max(8);
                frame.put_pixel(x, y, Luma([v]));
            }
        }
        frames.push(frame);
    }
    frames
}

/* ----------------------------------------------------------------------------
 * Benchmarks
 * ---------------------------------------------------------------------------- */

fn bench_boosttrack_basic(c: &mut Criterion) {
    let detections = load_detections();

    c.bench_function("boosttrack_basic", |b| {
        b.iter(|| {
            let mut tracker = BoostTracker::new(0.5, 0.3, 30, 3);
            for (_, objs) in detections.iter() {
                let _ = tracker.update(objs);
            }
        });
    });
}

fn bench_boosttrack_plus(c: &mut Criterion) {
    let detections = load_detections();

    c.bench_function("boosttrack_plus", |b| {
        b.iter(|| {
            let mut tracker =
                BoostTracker::new(0.5, 0.3, 30, 3).with_boost_plus();
            for (_, objs) in detections.iter() {
                let _ = tracker.update(objs);
            }
        });
    });
}

fn bench_boosttrack_plusplus(c: &mut Criterion) {
    let detections = load_detections();

    c.bench_function("boosttrack_plusplus", |b| {
        b.iter(|| {
            let mut tracker =
                BoostTracker::new(0.5, 0.3, 30, 3).with_boost_plus_plus();
            for (_, objs) in detections.iter() {
                let _ = tracker.update(objs);
            }
        });
    });
}

fn bench_boosttrack_basic_ecc(c: &mut Criterion) {
    let detections = load_detections();
    let frames = synthetic_frames(detections.len(), 640, 384);

    c.bench_function("boosttrack_basic_ecc", |b| {
        b.iter(|| {
            let mut tracker = BoostTracker::new(0.5, 0.3, 30, 3).with_ecc();
            for (i, (_, objs)) in detections.iter().enumerate() {
                let _ = tracker.update_with_frame(objs, &frames[i]);
            }
        });
    });
}

fn bench_boosttrack_plus_ecc(c: &mut Criterion) {
    let detections = load_detections();
    let frames = synthetic_frames(detections.len(), 640, 384);

    c.bench_function("boosttrack_plus_ecc", |b| {
        b.iter(|| {
            let mut tracker = BoostTracker::new(0.5, 0.3, 30, 3)
                .with_boost_plus()
                .with_ecc();
            for (i, (_, objs)) in detections.iter().enumerate() {
                let _ = tracker.update_with_frame(objs, &frames[i]);
            }
        });
    });
}

fn bench_boosttrack_plusplus_ecc(c: &mut Criterion) {
    let detections = load_detections();
    let frames = synthetic_frames(detections.len(), 640, 384);

    c.bench_function("boosttrack_plusplus_ecc", |b| {
        b.iter(|| {
            let mut tracker = BoostTracker::new(0.5, 0.3, 30, 3)
                .with_boost_plus_plus()
                .with_ecc();
            for (i, (_, objs)) in detections.iter().enumerate() {
                let _ = tracker.update_with_frame(objs, &frames[i]);
            }
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(6))
        .warm_up_time(Duration::from_secs(2));
    targets = bench_boosttrack_basic,
              bench_boosttrack_plus,
              bench_boosttrack_plusplus,
              bench_boosttrack_basic_ecc,
              bench_boosttrack_plus_ecc,
              bench_boosttrack_plusplus_ecc
}
criterion_main!(benches);
