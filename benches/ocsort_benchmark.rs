use std::collections::HashMap;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use jamtrack_rs::{oc_sort_tracker::OCSort, object::Object, rect::Rect};
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

fn bench_ocsort(c: &mut Criterion) {
    let detections = load_detections();

    c.bench_function("ocsort", |b| {
        b.iter(|| {
            let mut tracker = OCSort::new(0.5)
                .with_max_age(30)
                .with_min_hits(3)
                .with_iou_threshold(0.3)
                .with_delta_t(3)
                .with_inertia(0.2);
            for (_, objs) in detections.iter() {
                let _ = tracker.update(objs);
            }
        });
    });
}

fn bench_ocsort_byte(c: &mut Criterion) {
    let detections = load_detections();

    c.bench_function("ocsort_byte", |b| {
        b.iter(|| {
            let mut tracker = OCSort::new(0.5)
                .with_max_age(30)
                .with_min_hits(3)
                .with_iou_threshold(0.3)
                .with_delta_t(3)
                .with_inertia(0.2)
                .with_byte(true);
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
    targets = bench_ocsort,
              bench_ocsort_byte
}
criterion_main!(benches);
