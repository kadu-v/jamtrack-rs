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
    <video controls src="https:


//github.com/user-attachments/assets/dc135e90-4296-408e-8309-bfd921c06700" muted="false" width="500"></video>
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

## Benchmark on M3 MacBook Pro
```bash
$ cargo bench
     Running benches/boosttrack_benchmark.rs (target/release/deps/boosttrack_benchmark-9e3103bfc25128ab)
Gnuplot not found, using plotters backend
boosttrack_basic        time:   [63.316 ms 63.424 ms 63.537 ms]
                        change: [-1.0919% -0.8370% -0.6061%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 1 outliers among 50 measurements (2.00%)
  1 (2.00%) high mild

boosttrack_plus         time:   [83.895 ms 84.077 ms 84.343 ms]
                        change: [-0.9095% -0.6342% -0.2771%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 8 outliers among 50 measurements (16.00%)
  5 (10.00%) high mild
  3 (6.00%) high severe

boosttrack_plusplus     time:   [77.748 ms 77.929 ms 78.179 ms]
                        change: [-1.5845% -1.1522% -0.6850%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 9 outliers among 50 measurements (18.00%)
  3 (6.00%) high mild
  6 (12.00%) high severe

     Running benches/bytetrack_benchmark.rs (target/release/deps/bytetrack_benchmark-e0432e8a054012c9)
Gnuplot not found, using plotters backend
bytetrack               time:   [80.770 ms 80.884 ms 81.010 ms]
                        change: [+0.5419% +0.7601% +0.9799%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 6 outliers among 50 measurements (12.00%)
  6 (12.00%) high mild
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
