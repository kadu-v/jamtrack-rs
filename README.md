# JamTrack-rs

<p align="center">
    <img src="./data/logo/jam.jpeg" width="200">
</p>

JamTrack-rs is a Rust crate that provides multi-object tracking algorithms including [ByteTrack](https://arxiv.org/abs/2110.06864) and [BoostTrack](https://arxiv.org/abs/2408.13003).

## Features

- **ByteTracker**: Simple and efficient tracking using IoU-based association
- **BoostTracker**: Advanced tracking with confidence boosting techniques
  - **BoostTrack**: Basic DLO/DUO confidence boost
  - **BoostTrack+**: Rich similarity (Mahalanobis distance + shape + soft BIoU)
  - **BoostTrack++**: Rich similarity + soft boost + varying threshold

## Demo Videos





### ByteTracker

<div align="center">
    <video controls src="https://github.com/user-attachments/assets/dc135e90-4296-408e-8309-bfd921c06700" muted="false" width="500"></video>
</div>

### BoostTracker

<div align="center">
    <video controls src="https://github.com/user-attachments/assets/a3c9c252-cb32-4944-8820-fe981588b90e" muted="false" width="500"></video>
</div>

### BoostTracker+

<div align="center">
    <video controls src="https://github.com/user-attachments/assets/6e05a5ec-337c-4aa9-9202-f635acf56050" muted="false" width="500"></video>
</div>

### BoostTracker++

<div align="center">
    <video controls src="https://github.com/user-attachments/assets/e5c93888-b0b7-42cd-af87-b22dfbe063fe" muted="false" width="500"></video>
</div>

Videos use [NHK Creative Library](https://www2.nhk.or.jp/archives/movies/?id=D0002011239_00000) as the source with YoloX-X as the detector.

## Installation

Add the following to your `Cargo.toml`:

```
[dependencies]
jamtrack-rs = { git = "https://github.com/kadu-v/jamtrack-rs.git" }
```

## Usage

### ByteTracker

```rust
use jamtrack_rs::byte_tracker::ByteTracker;
use jamtrack_rs::object::Object;
use jamtrack_rs::rect::Rect;

// Create tracker: track_thresh, track_buffer, match_thresh
let mut tracker = ByteTracker::new(0.5, 30, 0.8);

// Create detections
let detections = vec![
    Object::new(Rect::new(100.0, 100.0, 50.0, 80.0), 0.9, None),
    Object::new(Rect::new(200.0, 150.0, 60.0, 90.0), 0.85, None),
];

// Update tracker
let tracks = tracker.update(&detections);

for track in tracks {
    println!("Track ID: {:?}, Rect: {:?}", track.get_track_id(), track.get_rect());
}
```

### BoostTracker

```rust
use jamtrack_rs::boost_tracker::BoostTracker;
use jamtrack_rs::object::Object;
use jamtrack_rs::rect::Rect;

// Create tracker: det_thresh, iou_threshold, max_age, min_hits
let mut tracker = BoostTracker::new(0.5, 0.3, 30, 3);

// Create detections
let detections = vec![
    Object::new(Rect::new(100.0, 100.0, 50.0, 80.0), 0.9, None),
    Object::new(Rect::new(200.0, 150.0, 60.0, 90.0), 0.85, None),
];

// Update tracker
let tracks = tracker.update(&detections).unwrap();

for track in tracks {
    println!("Track ID: {:?}, Rect: {:?}", track.get_track_id(), track.get_rect());
}
```

### BoostTracker+ / BoostTracker++

```rust
use jamtrack_rs::boost_tracker::BoostTracker;

// BoostTrack+ (rich similarity)
let mut tracker_plus = BoostTracker::new(0.5, 0.3, 30, 3)
    .with_boost_plus();

// BoostTrack++ (rich similarity + soft boost + varying threshold)
let mut tracker_plus_plus = BoostTracker::new(0.5, 0.3, 30, 3)
    .with_boost_plus_plus();

// Custom configuration
let mut custom_tracker = BoostTracker::new(0.5, 0.3, 30, 3)
    .with_lambdas(0.6, 0.2, 0.2)  // lambda_iou, lambda_mhd, lambda_shape
    .with_boost(true, false)      // use_dlo_boost, use_duo_boost
    .with_boost_plus_plus();
```

## Benchmark

Tested on M3 MacBook Pro with 1627 frames from detection_results.json.

### Performance

| Tracker | Time | Relative |
|---------|------|----------|
| BoostTrack | **63.4 ms** | 1.00x (fastest) |
| BoostTrack++ | 77.9 ms | 1.23x |
| ByteTracker | 80.9 ms | 1.28x |
| BoostTrack+ | 84.1 ms | 1.33x |

### Why is BoostTrack++ faster than BoostTrack+?

BoostTrack++ performs more computation per frame (soft boost + varying threshold), but maintains fewer active tracks due to better matching:

| Tracker | Avg Tracks | Max Tracks |
|---------|------------|------------|
| BoostTrack | 32.12 | 46 |
| BoostTrack+ | 30.94 | 45 |
| BoostTrack++ | **28.97** | 43 |

Fewer tracks = smaller similarity matrices = faster downstream computation.

### Run Benchmarks

```bash
cargo bench
```

### MOT17-train Benchmark (YOLOX-X Detector)

Evaluation results on MOT17 train set using YOLOX-X detector:

| Tracker | HOTA | MOTA | IDF1 | IDSW |
|---------|------|------|------|------|
| **OfficialBoostTrack++ECC (Python)** | **69.71** | 79.92 | **79.82** | **287** |
| OfficialBoostTrackECC (Python) | 69.28 | 79.17 | 79.10 | 308 |
| **ByteTrackerTuned (Rust)** | 68.55 | **80.95** | 78.27 | 450 |
| OfficialByteTrackerTuned (Python) | 67.92 | 80.90 | 77.47 | 453 |
| OfficialBoostTrack++ (Python) | 67.87 | 78.89 | 76.91 | 515 |
| OfficialBoostTrack (Python) | 67.30 | 78.26 | 76.00 | 520 |
| **BoostTrack++ECC (Rust)** | 68.35 | 79.80 | 77.98 | 318 |
| BoostTrackECC (Rust) | 68.39 | 79.06 | 77.94 | 344 |
| OfficialByteTracker (Python) | 67.82 | 80.92 | 77.29 | 458 |
| ByteTracker (Rust) | 68.35 | 80.97 | 77.89 | 454 |
| BoostTrack++ (Rust) | 66.02 | 78.86 | 74.29 | 558 |
| BoostTrack (Rust) | 66.03 | 78.24 | 74.13 | 536 |
| BoostTrack+ (Rust) | 65.93 | 78.57 | 74.11 | 560 |

> [!NOTE]
> - ECC variants show significant improvement in HOTA/IDF1/IDSW due to camera motion compensation
> - Rust BoostTrack supports ECC; embedding (Re-ID features) is still not implemented
> - MOTA is determined by the core algorithm, so Rust and Python versions achieve nearly identical values
> - *Tuned* variants use optimized hyperparameters of a tracker for MOT17 dataset


## Examples

Run the examples with detection data:

```bash
# ByteTracker
cargo run --example example_byte_tracker

# BoostTracker (basic)
cargo run --example example_boost_tracker

# BoostTracker with mode selection
cargo run --example example_boost_tracker_modes basic
cargo run --example example_boost_tracker_modes plus
cargo run --example example_boost_tracker_modes plusplus
```

## Tracker Comparison

| Feature | ByteTracker | BoostTrack | BoostTrack+ | BoostTrack++ |
|---------|-------------|------------|-------------|--------------|
| IoU Association | Yes | Yes | Yes | Yes |
| Mahalanobis Distance | No | Yes | Yes | Yes |
| Shape Similarity | No | No | Yes | Yes |
| DLO Confidence Boost | No | Yes | Yes | Yes |
| DUO Confidence Boost | No | Yes | Yes | Yes |
| Rich Similarity | No | No | Yes | Yes |
| Soft Boost | No | No | No | Yes |
| Varying Threshold | No | No | No | Yes |
| Embedding (Re-ID) | No | No | No | No |
| ECC (Camera Motion Compensation) | No | Yes | Yes | Yes |

## References

- [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)
- [BoostTrack: Boosting the Similarity Measure and Detection Confidence for Improved Multiple Object Tracking](https://arxiv.org/abs/2408.13003)

## License

MIT License
