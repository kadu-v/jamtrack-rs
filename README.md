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

```toml
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

## References

- [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)
- [BoostTrack: Boosting the Similarity Measure and Detection Confidence for Improved Multiple Object Tracking](https://arxiv.org/abs/2408.13003)

## License

MIT License
