# JTrackers

Multi-object tracking algorithms for Swift, powered by a Rust engine.

## Overview

JTrackers provides Swift bindings for high-performance multi-object tracking
algorithms implemented in Rust. Each tracker takes per-frame detections and
assigns persistent track identifiers across frames.

### Supported Algorithms

- ``ByteTracker`` — ByteTrack: two-stage matching using high and low-confidence detections.
- ``BoostTracker`` — BoostTrack: IoU + Mahalanobis distance + shape similarity with optional BoostTrack+/++ modes.
- ``OCSort`` — OC-SORT: observation-centric online smoothing with velocity direction consistency.

### Quick Start

```swift
let tracker = BoostTracker()
let detections = [
    TrackedObject(x: 10, y: 20, width: 100, height: 200, prob: 0.9),
    TrackedObject(x: 300, y: 400, width: 80, height: 160, prob: 0.8),
]

switch tracker.update(detections) {
case .success(let tracked):
    for obj in tracked {
        print("Track \(obj.trackId ?? -1): (\(obj.x), \(obj.y))")
    }
case .failure(let error):
    print("Error: \(error)")
}
```

## Topics

### Trackers

- ``ByteTracker``
- ``BoostTracker``
- ``OCSort``

### Data Types

- ``TrackedObject``
- ``JamTrackError``
