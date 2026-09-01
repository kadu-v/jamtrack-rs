use std::{collections::BTreeMap, time::Duration};

use criterion::{Criterion, criterion_group, criterion_main};
use image::{GrayImage, Luma};
use jamtrack_rs::{
    McByteMask, McByteTracker, Object, Rect, SparseOptFlowConfig,
};
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
        frames
            .entry(detection.frame_id.parse().unwrap())
            .or_default()
            .push(Object::new(
                Rect::new(
                    detection.x.parse().unwrap(),
                    detection.y.parse().unwrap(),
                    detection.width.parse().unwrap(),
                    detection.height.parse().unwrap(),
                ),
                detection.prob.parse().unwrap(),
                None,
            ));
    }
    frames.into_values().collect()
}

fn crossing_sequence() -> Vec<(Vec<Object>, Vec<McByteMask>, GrayImage)> {
    (0..60)
        .map(|frame| {
            let left_x = 15 + frame;
            let right_x = 125 - frame;
            let objects = vec![
                Object::new(
                    Rect::new(left_x as f32, 45.0, 18.0, 30.0),
                    0.9,
                    None,
                ),
                Object::new(
                    Rect::new(right_x as f32, 45.0, 18.0, 30.0),
                    0.9,
                    None,
                ),
            ];
            let masks = vec![
                rectangular_mask(1, left_x, 45, 18, 30),
                rectangular_mask(2, right_x, 45, 18, 30),
            ];
            let image = textured_frame(160, 120, frame as i32 / 3);
            (objects, masks, image)
        })
        .collect()
}

fn rectangular_mask(
    track_id: usize,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> McByteMask {
    let mut mask = GrayImage::new(160, 120);
    for yy in y..(y + height).min(mask.height()) {
        for xx in x..(x + width).min(mask.width()) {
            mask.put_pixel(xx, yy, Luma([255]));
        }
    }
    McByteMask::new(track_id, mask, 0.95)
}

fn textured_frame(width: u32, height: u32, offset: i32) -> GrayImage {
    let mut image = GrayImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let shifted = x as i32 - offset;
            let value = if shifted >= 0 {
                (((shifted * 17 + y as i32 * 31 + shifted * y as i32) & 255)
                    as u8)
                    .max(4)
            } else {
                0
            };
            image.put_pixel(x, y, Luma([value]));
        }
    }
    image
}

fn bench_mcbyte(c: &mut Criterion) {
    let detection_frames = load_frames();
    c.bench_function("mcbyte_mask_free_1627_frames", |b| {
        b.iter(|| {
            let mut tracker = McByteTracker::default().without_cmc();
            for objects in &detection_frames {
                let _ = tracker.update(objects).unwrap();
            }
        });
    });

    let crossing = crossing_sequence();
    c.bench_function("mcbyte_external_masks_60_frames", |b| {
        b.iter(|| {
            let mut tracker = McByteTracker::default().without_cmc();
            for (objects, masks, _) in &crossing {
                let _ = tracker.update_with_masks(objects, masks).unwrap();
            }
        });
    });

    c.bench_function("mcbyte_masks_sparse_optflow_60_frames", |b| {
        b.iter(|| {
            let mut tracker = McByteTracker::default().with_sparse_opt_flow(
                SparseOptFlowConfig {
                    downscale: 2,
                    max_corners: 100,
                    ..Default::default()
                },
            );
            for (objects, masks, frame) in &crossing {
                let _ = tracker
                    .update_with_frame_and_masks(objects, frame, masks)
                    .unwrap();
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
    targets = bench_mcbyte
}
criterion_main!(benches);
