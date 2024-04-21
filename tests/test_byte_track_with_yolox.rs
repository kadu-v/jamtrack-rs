use std::collections::HashMap;

use bytetrack_rs::byte_tracker::ByteTracker;
use nearly_eq::assert_nearly_eq;
use serde::Deserialize;
use serde_json;

const TRACKING_JSON_PATH: &str =
    "data/YOLOX_ncnn_palace/tracking_results_x.json";
const DETECTION_JSON_PATH: &str =
    "data/YOLOX_ncnn_palace/detection_results.json";

/*----------------------------------------------------------------------------
Json schema for tracking results
----------------------------------------------------------------------------*/

#[derive(Debug, Deserialize)]
struct DetectionJson {
    name: String,
    fps: usize,
    track_buffer: usize,
    results: Vec<DetectionReusltJson>,
}

#[derive(Debug, Deserialize)]
struct DetectionReusltJson {
    frame_id: String,
    prob: String,
    x: String,
    y: String,
    width: String,
    height: String,
}

/*----------------------------------------------------------------------------
Json schema for tracking results
----------------------------------------------------------------------------*/
#[derive(Debug, Deserialize)]
struct TrackingJson {
    name: String,
    fps: usize,
    track_buffer: usize,
    results: Vec<TrackingResultJson>,
}

#[derive(Debug, Deserialize)]
struct TrackingResultJson {
    frame_id: String,
    track_id: String,
    x: String,
    y: String,
    width: String,
    height: String,
}

/*----------------------------------------------------------------------------
DetectionResult struct
----------------------------------------------------------------------------*/
#[derive(Debug, Clone)]
struct Detection {
    name: String,
    fps: usize,
    track_buffer: usize,
    results: HashMap<usize, Vec<DetectionReuslt>>,
}

impl Detection {
    fn new(detection: &DetectionJson) -> Self {
        let mut results: HashMap<usize, Vec<DetectionReuslt>> = HashMap::new();
        for detection in &detection.results {
            let detection = DetectionReuslt::new(detection);
            let frame_id = detection.frame_id;
            if results.contains_key(&frame_id) {
                results.get_mut(&frame_id).unwrap().push(detection);
            } else {
                results.insert(frame_id, vec![detection]);
            }
        }

        Self {
            name: detection.name.clone(),
            fps: detection.fps,
            track_buffer: detection.track_buffer,
            results,
        }
    }
}

#[derive(Debug, Clone)]
struct DetectionReuslt {
    frame_id: usize,
    prob: f32,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

impl DetectionReuslt {
    fn new(detection: &DetectionReusltJson) -> Self {
        Self {
            frame_id: detection.frame_id.parse().unwrap(),
            prob: detection.prob.parse().unwrap(),
            x: detection.x.parse().unwrap(),
            y: detection.y.parse().unwrap(),
            width: detection.width.parse().unwrap(),
            height: detection.height.parse().unwrap(),
        }
    }
}

impl Into<bytetrack_rs::object::Object> for DetectionReuslt {
    fn into(self) -> bytetrack_rs::object::Object {
        bytetrack_rs::object::Object::new(
            bytetrack_rs::rect::Rect::new(
                self.x,
                self.y,
                self.width,
                self.height,
            ),
            0,
            self.prob,
        )
    }
}

/*----------------------------------------------------------------------------
TrackingResult struct
----------------------------------------------------------------------------*/
#[derive(Debug, Clone)]
struct Tracking {
    name: String,
    _fps: usize,
    _track_buffer: usize,
    results: HashMap<usize, HashMap<usize, TrackingResult>>,
}

impl Tracking {
    fn new(tracking: &TrackingJson) -> Self {
        let mut results: HashMap<usize, HashMap<usize, TrackingResult>> =
            HashMap::new();
        for tracking in &tracking.results {
            let tracking = TrackingResult::new(tracking);
            let frame_id = tracking.frame_id;
            let track_id = tracking.track_id;
            if results.contains_key(&frame_id) {
                results
                    .get_mut(&frame_id)
                    .unwrap()
                    .insert(track_id, tracking);
            } else {
                let mut track_results = HashMap::new();
                track_results.insert(track_id, tracking);
                results.insert(frame_id, track_results);
            }
        }

        Self {
            name: tracking.name.clone(),
            _fps: tracking.fps,
            _track_buffer: tracking.track_buffer,
            results,
        }
    }
}

#[derive(Debug, Clone)]
struct TrackingResult {
    frame_id: usize,
    track_id: usize,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

impl TrackingResult {
    fn new(tracking: &TrackingResultJson) -> Self {
        Self {
            frame_id: tracking.frame_id.parse().unwrap(),
            track_id: tracking.track_id.parse().unwrap(),
            x: tracking.x.parse().unwrap(),
            y: tracking.y.parse().unwrap(),
            width: tracking.width.parse().unwrap(),
            height: tracking.height.parse().unwrap(),
        }
    }
}

impl Into<bytetrack_rs::object::Object> for TrackingResult {
    fn into(self) -> bytetrack_rs::object::Object {
        bytetrack_rs::object::Object::new(
            bytetrack_rs::rect::Rect::new(
                self.x,
                self.y,
                self.width,
                self.height,
            ),
            self.track_id,
            0.0,
        )
    }
}
/*----------------------------------------------------------------------------
Read json
----------------------------------------------------------------------------*/
fn read_detection_json(path: &str) -> Detection {
    let file = std::fs::File::open(path).unwrap();
    let detection: DetectionJson = serde_json::from_reader(file).unwrap();
    Detection::new(&detection)
}

fn read_tracking_json(path: &str) -> Tracking {
    let file = std::fs::File::open(path).unwrap();
    let tracking: TrackingJson = serde_json::from_reader(file).unwrap();
    Tracking::new(&tracking)
}

#[test]
fn test_byte_track_with_yolox() {
    let detection = read_detection_json(DETECTION_JSON_PATH);
    let tracking = read_tracking_json(TRACKING_JSON_PATH);

    if detection.name != tracking.name {
        panic!("Detection and tracking names are different");
    }

    let detection_results = detection.results;
    let tracking_results = tracking.results;
    let fps = detection.fps;
    let track_buffer = detection.track_buffer;
    let mut byte_tracker = ByteTracker::new(
        fps,
        track_buffer,
        0.5, /* track thresh */
        0.6, /* high_thresh */
        0.8, /* mathc_thresh */
    );

    for frame_id in 0..detection_results.len() {
        let objects = detection_results
            .get(&frame_id)
            .unwrap()
            .iter()
            .map(|v| <DetectionReuslt>::into(v.clone()))
            .collect();
        let outputs = byte_tracker.update(&objects);

        let expected_outputs = tracking_results.get(&frame_id).unwrap();

        // check that outputs contains all expected_outputs
        for (track_id, _) in expected_outputs.iter() {
            assert!(
                outputs
                    .iter()
                    .any(|output| output.get_track_id() == *track_id),
                "Not found expected track_id: {} in frame_id: {}",
                track_id,
                frame_id
            );
        }

        for output in outputs.iter() {
            #[allow(non_snake_case)]
            let EPS = 1.0e-2;
            let rect = output.get_rect();
            let expected_rect = {
                let obj: bytetrack_rs::object::Object = <TrackingResult>::into(
                    expected_outputs
                        .get(&output.get_track_id())
                        .unwrap()
                        .clone(),
                );
                obj.rect
            };
            assert!(
                expected_outputs.contains_key(&output.get_track_id()),
                "Not found output track_id: {} in frame_id: {}",
                output.get_track_id(),
                frame_id,
            );
            assert_nearly_eq!(rect.x(), expected_rect.x(), EPS);
            assert_nearly_eq!(rect.y(), expected_rect.y(), EPS);
            assert_nearly_eq!(rect.width(), expected_rect.width(), EPS);
            assert_nearly_eq!(rect.height(), expected_rect.height(), EPS);
        }
    }
}
