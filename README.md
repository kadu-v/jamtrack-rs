# JamTrack-rs

<p align="center">
    <img src="./data/logo/jam.jpeg" width="200">
</p>

[![Swift Package CI/CD](https://github.com/kadu-v/jamtrack-rs/actions/workflows/swift.yml/badge.svg)](https://github.com/kadu-v/jamtrack-rs/actions/workflows/swift.yml)

JamTrack-rs is a Rust crate that provides multi-object tracking algorithms including [ByteTrack](https://arxiv.org/abs/2110.06864), [FastTracker](https://arxiv.org/abs/2508.14370), [BoT-SORT](https://arxiv.org/abs/2206.14651), [BoostTrack](https://arxiv.org/abs/2408.13003), and [OC-SORT](https://arxiv.org/abs/2203.14360).

## Features

- **ByteTracker**: Simple and efficient tracking using IoU-based association
- **FastTracker**: Occlusion-aware association with optional road-region and direction-cone constraints
  - Motion reset, bounding-box enlargement, and dampening during occlusion
  - Duplicate-track initialization suppression and recently-occluded lifetime extension
  - Four-point ROI trajectory repair and direction refinement
- **BotSort**: BoT-SORT tracking with BYTE-style association, `xywh` Kalman filter, optional ECC camera compensation, and optional external ReID embeddings
- **BoostTracker**: Advanced tracking with confidence boosting techniques
  - **BoostTrack**: Basic DLO/DUO confidence boost
  - **BoostTrack+**: Rich similarity (Mahalanobis distance + shape + soft BIoU)
  - **BoostTrack++**: Rich similarity + soft boost + varying threshold
- **OC-SORT**: Observation-Centric SORT with online smoothing
  - IoU + VDC (Velocity Direction Consistency) association
  - BYTE association for low-confidence detections
  - OCR (Observation-Centric Re-association) with last observation
  - Kalman filter freeze/unfreeze for online smoothing

## Demo

### FastTracker

<div align="center">
    <video controls src="./data/video/output.mp4" poster="./data/charts/fasttracker_demo.jpg" muted="false" width="640"></video>
    <br>
    <a href="./data/video/output.mp4">Play the FastTracker demo</a>
</div>

The demo uses YOLOX-X detections and the standard FastTracker constructor with
no RoI constraints. The source footage is from the
[NHK Creative Library](https://www2.nhk.or.jp/archives/movies/?id=D0002011239_00000).

**Individual tracker demos:** [ByteTracker](https://github.com/user-attachments/assets/dc135e90-4296-408e-8309-bfd921c06700) | [BoT-SORT](https://github.com/user-attachments/assets/3b4ec6ed-8c74-49f5-9488-5d453c768c66) | [BoostTracker](https://github.com/user-attachments/assets/a3c9c252-cb32-4944-8820-fe981588b90e) | [BoostTracker+](https://github.com/user-attachments/assets/6e05a5ec-337c-4aa9-9202-f635acf56050) | [BoostTracker++](https://github.com/user-attachments/assets/e5c93888-b0b7-42cd-af87-b22dfbe063fe)

## Installation

### Rust

Add the following to your `Cargo.toml`:

```toml
[dependencies]
jamtrack-rs = { git = "https://github.com/kadu-v/jamtrack-rs.git" }
```

### Swift (via Swift Package Manager)

The Swift package is distributed as [JTrackers](https://github.com/kadu-v/JTrackers). Add it in Xcode via **File > Add Package Dependencies** with the URL:

```
https://github.com/kadu-v/JTrackers.git
```

Or add it to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/kadu-v/JTrackers.git", from: "0.5.1"),
]
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

### FastTracker

```rust
use jamtrack_rs::{FastTracker, Object, Rect};

// frame_rate, track_buffer, track_thresh, match_thresh
let mut tracker = FastTracker::new(30, 30, 0.6, 0.7);

let detections = vec![Object::new(
    Rect::new(100.0, 100.0, 50.0, 80.0),
    0.9,
    None,
)];
let tracks = tracker.update(&detections)?;
# Ok::<(), jamtrack_rs::TrackError>(())
```

FastTracker expects detection rectangles in original-image coordinates. Its
occlusion handling is enabled by the standard constructor; RoI constraints are
optional.

An RoI is a camera-specific, four-point polygon. Upstream FastTracker does not
detect it automatically and does not use it to discard detections outside the
polygon. It repairs short trajectory excursions after a track returns to the
RoI, and refines motion that falls outside the direction cone derived from the
four points. Configure it only when fixed pixel coordinates for the scene are
known:

```rust
use jamtrack_rs::{FastTracker, FastTrackerRoi};

// Point order follows upstream FastTracker: (E1, E2, O2, O1).
let road = FastTrackerRoi::new([
    [312.0, 196.0],
    [422.0, 188.0],
    [1399.0, 694.0],
    [152.0, 697.0],
]);

let mut tracker = FastTracker::new(30, 30, 0.6, 0.7)
    .with_rois(vec![road], 15, 10, 2.0);
```

`FastTrackerRoi` is the Rust value that owns these four points. Multiple RoIs
may be supplied; the first polygon containing the current track center is used.
Without `with_rois`, behavior matches upstream with no `ROIs` configuration.

The same tracker is available from Swift:

```swift
let tracker = FastTracker(
    frameRate: 30,
    trackBuffer: 30,
    trackThresh: 0.6,
    matchThresh: 0.7
)

switch tracker.update(detections) {
case .success(let tracks):
    print(tracks)
case .failure(let error):
    print(error)
}
```

### BoT-SORT

```rust
use jamtrack_rs::bot_sort_tracker::BotSort;
use jamtrack_rs::object::Object;
use jamtrack_rs::rect::Rect;

let mut tracker = BotSort::new(30, 30, 0.6, 0.1, 0.7, 0.8);
let detections = vec![
    Object::new(Rect::new(100.0, 100.0, 50.0, 80.0), 0.9, None),
];

let tracks = tracker.update(&detections).unwrap();
```

BoT-SORT can also run ECC camera motion compensation when the caller provides grayscale frames:

```rust
use image::GrayImage;

let mut tracker = BotSort::new(30, 30, 0.6, 0.1, 0.7, 0.8)
    .with_ecc();
let frame = GrayImage::new(640, 480);

let tracks = tracker.update_with_frame(&detections, &frame).unwrap();
```

BoT-SORT ReID matching is optional and expects embeddings from the caller:

```rust
let mut tracker = BotSort::new(30, 30, 0.6, 0.1, 0.7, 0.8)
    .with_reid(true);
let features = vec![vec![1.0, 0.0, 0.0]];

let tracks = tracker.update_with_features(&detections, &features).unwrap();
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

### OC-SORT

```rust
use jamtrack_rs::oc_sort_tracker::OCSort;
use jamtrack_rs::object::Object;
use jamtrack_rs::rect::Rect;

// Create tracker with detection threshold
let mut tracker = OCSort::new(0.5)
    .with_max_age(30)
    .with_min_hits(3)
    .with_iou_threshold(0.3)
    .with_delta_t(3)
    .with_inertia(0.2)
    .with_byte(false); // enable BYTE association for low-score detections

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

## Benchmark

Tested on M3 MacBook Pro with 1627 frames from detection_results.json.

### Performance

Lower is better. Each value is the time required to process all 1627 frames.

<div align="center">
    <img src="./data/charts/performance.png" width="600">
</div>

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

<div align="center">
    <img src="./data/charts/mot17_hota.png" width="700">
    <img src="./data/charts/mot17_mota.png" width="700">
    <img src="./data/charts/mot17_idf1.png" width="700">
    <img src="./data/charts/mot17_idsw.png" width="700">
</div>

> [!NOTE]
> - ECC variants show significant improvement in HOTA/IDF1/IDSW due to camera motion compensation
> - Rust BoostTrack and BoT-SORT support ECC camera motion compensation
> - BoT-SORT ReID matching is implemented, but the MOT17 benchmark above uses the non-ReID path for fair comparison with non-embedding tracker variants
> - MOTA is determined by the core algorithm, so Rust and Python versions achieve nearly identical values
> - *Tuned* variants use optimized hyperparameters of a tracker for MOT17 dataset
> - FastTracker was evaluated with `track_thresh=0.6`, `track_buffer=30`, `match_thresh=0.7`, and no RoI constraints
> - Rust and official Python FastTracker produced identical frame/ID assignments and TrackEval results: HOTA 64.56, MOTA 74.26, IDF1 72.44, and 1,171 ID switches


## Examples

Run the examples with detection data:

```bash
# ByteTracker
cargo run --example example_byte_tracker

# FastTracker
cargo run --example example_fast_tracker

# BoostTracker (basic)
cargo run --example example_boost_tracker

# BoostTracker with mode selection
cargo run --example example_boost_tracker_modes basic
cargo run --example example_boost_tracker_modes plus
cargo run --example example_boost_tracker_modes plusplus

# BoT-SORT
cargo run --example example_bot_sort
```

## Tracker Comparison

| Feature | ByteTracker | FastTracker | BoT-SORT | BoostTrack | BoostTrack+ | BoostTrack++ | OC-SORT |
|---------|-------------|-------------|----------|------------|-------------|--------------|---------|
| IoU Association | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Occlusion Handling | No | Yes | No | No | No | No | No |
| RoI Constraints | No | Yes | No | No | No | No | No |
| Mahalanobis Distance | No | No | No | Yes | Yes | Yes | No |
| Shape Similarity | No | No | No | No | Yes | Yes | No |
| DLO Confidence Boost | No | No | No | Yes | Yes | Yes | No |
| DUO Confidence Boost | No | No | No | Yes | Yes | Yes | No |
| Rich Similarity | No | No | No | No | Yes | Yes | No |
| Soft Boost | No | No | No | No | No | Yes | No |
| Varying Threshold | No | No | No | No | No | Yes | No |
| VDC (Velocity Direction Consistency) | No | No | No | No | No | No | Yes |
| OCR (Re-association) | No | No | No | No | No | No | Yes |
| Online Smoothing (Freeze/Unfreeze) | No | No | No | No | No | No | Yes |
| BYTE Association | Yes | Yes | Yes | No | No | No | Yes |
| Embedding (Re-ID) | No | No | Optional | No | No | No | No |
| ECC (Camera Motion Compensation) | No | No | Yes | Yes | Yes | Yes | No |

## References

- [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)
- [FastTracker: Real-Time and Accurate Visual Tracking](https://arxiv.org/abs/2508.14370)
- [BoT-SORT: Robust Associations Multi-Pedestrian Tracking](https://arxiv.org/abs/2206.14651)
- [BoostTrack: Boosting the Similarity Measure and Detection Confidence for Improved Multiple Object Tracking](https://arxiv.org/abs/2408.13003)
- [OC-SORT: Observation-Centric SORT on video Multi-Object Tracking](https://arxiv.org/abs/2203.14360)

## License

MIT License
