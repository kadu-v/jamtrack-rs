use std::collections::HashMap;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
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

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(3));
    targets = bench_boosttrack_basic, bench_boosttrack_plus, bench_boosttrack_plusplus
}
criterion_main!(benches);
