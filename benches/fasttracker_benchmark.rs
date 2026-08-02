use std::collections::BTreeMap;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use jamtrack_rs::{FastTracker, Object, Rect};
use serde::Deserialize;

const DETECTION_JSON_PATH: &str = "data/jsons/detection_results.json";

#[derive(Deserialize)]
struct DetectionJson {
    results: Vec<Detection>,
}

#[derive(Deserialize)]
struct Detection {
    frame_id: String,
    prob: String,
    x: String,
    y: String,
    width: String,
    height: String,
}

fn load_frames() -> Vec<Vec<Object>> {
    let file = std::fs::File::open(DETECTION_JSON_PATH).unwrap();
    let input: DetectionJson = serde_json::from_reader(file).unwrap();
    let mut frames = BTreeMap::<usize, Vec<Object>>::new();
    for detection in input.results {
        let object = Object::new(
            Rect::new(
                detection.x.parse().unwrap(),
                detection.y.parse().unwrap(),
                detection.width.parse().unwrap(),
                detection.height.parse().unwrap(),
            ),
            detection.prob.parse().unwrap(),
            None,
        );
        frames
            .entry(detection.frame_id.parse().unwrap())
            .or_default()
            .push(object);
    }
    frames.into_values().collect()
}

fn bench_fasttracker(c: &mut Criterion) {
    let frames = load_frames();
    c.bench_function("fasttracker", |b| {
        b.iter(|| {
            let mut tracker = FastTracker::new(30, 30, 0.6, 0.7);
            for objects in &frames {
                let _ = tracker.update(objects).unwrap();
            }
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(3));
    targets = bench_fasttracker
}
criterion_main!(benches);
