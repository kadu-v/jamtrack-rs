//! Single track management for BoostTracker
//!
//! This module provides the `KalmanBoxTracker` struct that represents
//! an individual tracked object using a Kalman filter.

use crate::boost_tracker::kalman_filter::DetectBox;

use super::kalman_filter::KalmanFilter;

/// Represents an individual tracked object using Kalman filter state estimation.
///
/// Tracks object state including position, velocity, and maintains tracking metadata
/// like hit streak, age, and time since last update.
pub struct KalmanBoxTracker {
    /// Kalman filter for state estimation
    kf: KalmanFilter,
    /// Unique identifier for this track
    id: usize,
    /// Frames since last successful update
    time_since_update: usize,
    /// Consecutive frames with successful detections
    hit_streak: usize,
    /// Total frames since track initialization
    age: usize,
    // TODO: Add embedding support for BoostTrack++
    // emb: Option<Vec<f32>>,
}

impl KalmanBoxTracker {
    /// Create a new tracker from an initial bounding box.
    ///
    /// # Arguments
    /// * `bbox` - Initial bounding box in [x1, y1, x2, y2] format
    /// * `id` - Unique track identifier
    ///
    /// # Returns
    /// A new `KalmanBoxTracker` instance
    pub fn new(bbox: &[f32; 4], id: usize) -> Self {
        let z = convert_bbox_to_z(bbox);
        let z_mat = DetectBox::from_iterator(z);
        Self {
            kf: KalmanFilter::new(&z_mat),
            id,
            time_since_update: 0,
            hit_streak: 0,
            age: 0,
        }
    }

    /// Get the unique track identifier.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Get frames since last successful update.
    pub fn time_since_update(&self) -> usize {
        self.time_since_update
    }

    /// Get consecutive hit streak count.
    pub fn hit_streak(&self) -> usize {
        self.hit_streak
    }

    /// Get total age of the track in frames.
    #[cfg(test)]
    pub fn age(&self) -> usize {
        self.age
    }

    /// Calculate track confidence based on age and time since update.
    ///
    /// For young tracks (age < 7), confidence decreases exponentially.
    /// For mature tracks, confidence decreases based on time since update.
    ///
    /// # Arguments
    /// * `coef` - Exponential decay coefficient (default: 0.9)
    ///
    /// # Returns
    /// Confidence score in [0, 1]
    pub fn get_confidence(&self, coef: f32) -> f32 {
        let n = 7;
        if self.age < n {
            coef.powi((n as isize - self.age as isize) as i32)
        } else {
            coef.powi((self.time_since_update as isize - 1) as i32)
        }
    }

    /// Update the tracker state with an observed bounding box.
    ///
    /// # Arguments
    /// * `bbox` - Observed bounding box in [x1, y1, x2, y2] format
    /// * `score` - Detection confidence score
    pub fn update(&mut self, bbox: &[f32; 4], score: f32) {
        self.time_since_update = 0;
        self.hit_streak += 1;
        let z = convert_bbox_to_z(bbox);
        self.kf.update(&DetectBox::from_iterator(z), score);
    }

    /// Predict the next state and advance the tracker.
    ///
    /// This advances the Kalman filter state, increments age,
    /// and updates time_since_update.
    ///
    /// # Returns
    /// Predicted bounding box in [x1, y1, x2, y2] format
    pub fn predict(&mut self) -> [f32; 4] {
        self.kf.predict();
        self.age += 1;
        if self.time_since_update > 0 {
            self.hit_streak = 0;
        }
        self.time_since_update += 1;
        self.get_state()
    }

    /// Get the current state estimate as a bounding box.
    ///
    /// # Returns
    /// Current bounding box estimate in [x1, y1, x2, y2] format
    pub fn get_state(&self) -> [f32; 4] {
        let z = self.kf.state();
        convert_z_to_bbox(&[
            z[0], // x
            z[1], // y
            z[2], // h
            z[3], // r
        ])
    }

    /// Get the Kalman filter state vector [x, y, h, r, vx, vy, vh, vr].
    ///
    /// Used for computing Mahalanobis distance.
    pub fn get_kf_state(&self) -> Vec<f32> {
        self.kf.state().iter().copied().collect::<Vec<f32>>()
    }

    /// Get the diagonal of the Kalman filter covariance matrix.
    ///
    /// Used for computing Mahalanobis distance.
    pub fn get_kf_covariance_diag(&self) -> Vec<f32> {
        self.kf
            .covariance()
            .diagonal()
            .iter()
            .copied()
            .collect::<Vec<f32>>()
    }

    /// Apply camera motion compensation transform to the current state.
    ///
    /// The transform is expected to map points from template/previous frame
    /// coordinates to current frame coordinates.
    pub fn camera_update(&mut self, transform: &[[f32; 3]; 3]) {
        let [x1, y1, x2, y2] = self.get_state();
        let x1p = transform[0][0] * x1 + transform[0][1] * y1 + transform[0][2];
        let y1p = transform[1][0] * x1 + transform[1][1] * y1 + transform[1][2];
        let x2p = transform[0][0] * x2 + transform[0][1] * y2 + transform[0][2];
        let y2p = transform[1][0] * x2 + transform[1][1] * y2 + transform[1][2];

        let w = x2p - x1p;
        let h = (y2p - y1p).max(1e-3);
        let cx = x1p + w / 2.0;
        let cy = y1p + h / 2.0;
        let r = w / h;

        let state = self.kf.state_mut();
        let s = state.as_mut_slice();
        s[0] = cx;
        s[1] = cy;
        s[2] = h;
        s[3] = r;
    }

    // TODO: Embedding support for BoostTrack++ (optional)
    // pub fn update_emb(&mut self, emb: &[f32], alpha: f32) {
    //     unimplemented!("KalmanBoxTracker::update_emb")
    // }
    //
    // pub fn get_emb(&self) -> Option<&[f32]> {
    //     unimplemented!("KalmanBoxTracker::get_emb")
    // }
}

impl std::fmt::Debug for KalmanBoxTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KalmanBoxTracker {{ id: {}, age: {}, hit_streak: {}, time_since_update: {} }}",
            self.id, self.age, self.hit_streak, self.time_since_update
        )
    }
}

/// Convert bounding box [x1, y1, x2, y2] to Kalman state [x, y, h, r].
///
/// Where (x, y) is center, h is height, r is aspect ratio (w/h).
pub fn convert_bbox_to_z(bbox: &[f32; 4]) -> [f32; 4] {
    let w = bbox[2] - bbox[0];
    let h = bbox[3] - bbox[1];
    let x = bbox[0] + w / 2.0;
    let y = bbox[1] + h / 2.0;
    let r = w / (h + 1e-6);
    [x, y, h, r]
}

/// Convert Kalman state [x, y, h, r] to bounding box [x1, y1, x2, y2].
pub fn convert_z_to_bbox(z: &[f32; 4]) -> [f32; 4] {
    let h = z[2];
    let r = z[3];
    let w = if r <= 0.0 { 0.0 } else { r * h };
    let x1 = z[0] - w / 2.0;
    let y1 = z[1] - h / 2.0;
    let x2 = z[0] + w / 2.0;
    let y2 = z[1] + h / 2.0;
    [x1, y1, x2, y2]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_xyxy(x1: f32, y1: f32, x2: f32, y2: f32) -> [f32; 4] {
        [x1, y1, x2, y2]
    }

    // ==========================================================================
    // convert_bbox_to_z tests
    // ==========================================================================

    #[test]
    fn test_convert_bbox_to_z_basic() {
        // Box: [0, 0, 100, 100] -> center (50, 50), h=100, r=1.0
        let bbox = box_xyxy(0.0, 0.0, 100.0, 100.0);
        let z = convert_bbox_to_z(&bbox);

        assert!((z[0] - 50.0).abs() < 1e-5, "x center");
        assert!((z[1] - 50.0).abs() < 1e-5, "y center");
        assert!((z[2] - 100.0).abs() < 1e-5, "height");
        assert!((z[3] - 1.0).abs() < 1e-5, "aspect ratio");
    }

    #[test]
    fn test_convert_bbox_to_z_wide_box() {
        // Box: [0, 0, 200, 100] -> center (100, 50), h=100, r=2.0
        let bbox = box_xyxy(0.0, 0.0, 200.0, 100.0);
        let z = convert_bbox_to_z(&bbox);

        assert!((z[0] - 100.0).abs() < 1e-5, "x center");
        assert!((z[1] - 50.0).abs() < 1e-5, "y center");
        assert!((z[2] - 100.0).abs() < 1e-5, "height");
        assert!((z[3] - 2.0).abs() < 1e-5, "aspect ratio");
    }

    #[test]
    fn test_convert_bbox_to_z_tall_box() {
        // Box: [0, 0, 50, 200] -> center (25, 100), h=200, r=0.25
        let bbox = box_xyxy(0.0, 0.0, 50.0, 200.0);
        let z = convert_bbox_to_z(&bbox);

        assert!((z[0] - 25.0).abs() < 1e-5, "x center");
        assert!((z[1] - 100.0).abs() < 1e-5, "y center");
        assert!((z[2] - 200.0).abs() < 1e-5, "height");
        assert!((z[3] - 0.25).abs() < 1e-5, "aspect ratio");
    }

    #[test]
    fn test_convert_bbox_to_z_offset_box() {
        // Box: [100, 200, 300, 400] -> center (200, 300), h=200, r=1.0
        let bbox = box_xyxy(100.0, 200.0, 300.0, 400.0);
        let z = convert_bbox_to_z(&bbox);

        assert!((z[0] - 200.0).abs() < 1e-5, "x center");
        assert!((z[1] - 300.0).abs() < 1e-5, "y center");
        assert!((z[2] - 200.0).abs() < 1e-5, "height");
        assert!((z[3] - 1.0).abs() < 1e-5, "aspect ratio");
    }

    // ==========================================================================
    // convert_z_to_bbox tests
    // ==========================================================================

    #[test]
    fn test_convert_z_to_bbox_basic() {
        // z: [50, 50, 100, 1.0] -> [0, 0, 100, 100]
        let z = [50.0, 50.0, 100.0, 1.0];
        let bbox = convert_z_to_bbox(&z);

        assert!((bbox[0] - 0.0).abs() < 1e-5, "x1");
        assert!((bbox[1] - 0.0).abs() < 1e-5, "y1");
        assert!((bbox[2] - 100.0).abs() < 1e-5, "x2");
        assert!((bbox[3] - 100.0).abs() < 1e-5, "y2");
    }

    #[test]
    fn test_convert_z_to_bbox_wide() {
        // z: [100, 50, 100, 2.0] -> [0, 0, 200, 100]
        let z = [100.0, 50.0, 100.0, 2.0];
        let bbox = convert_z_to_bbox(&z);

        assert!((bbox[0] - 0.0).abs() < 1e-5, "x1");
        assert!((bbox[1] - 0.0).abs() < 1e-5, "y1");
        assert!((bbox[2] - 200.0).abs() < 1e-5, "x2");
        assert!((bbox[3] - 100.0).abs() < 1e-5, "y2");
    }

    #[test]
    fn test_convert_z_to_bbox_roundtrip() {
        // Roundtrip test: bbox -> z -> bbox should be identity
        let original = box_xyxy(50.0, 100.0, 150.0, 300.0);
        let z = convert_bbox_to_z(&original);
        let recovered = convert_z_to_bbox(&z);

        assert!((recovered[0] - original[0]).abs() < 1e-4, "x1 roundtrip");
        assert!((recovered[1] - original[1]).abs() < 1e-4, "y1 roundtrip");
        assert!((recovered[2] - original[2]).abs() < 1e-4, "x2 roundtrip");
        assert!((recovered[3] - original[3]).abs() < 1e-4, "y2 roundtrip");
    }

    // ==========================================================================
    // KalmanBoxTracker::new tests
    // ==========================================================================

    #[test]
    fn test_tracker_new() {
        let bbox = box_xyxy(100.0, 100.0, 200.0, 200.0);
        let tracker = KalmanBoxTracker::new(&bbox, 0);

        assert_eq!(tracker.id(), 0);
        assert_eq!(tracker.time_since_update(), 0);
        assert_eq!(tracker.hit_streak(), 0);
        assert_eq!(tracker.age(), 0);
    }

    #[test]
    fn test_tracker_new_with_id() {
        let bbox = box_xyxy(0.0, 0.0, 50.0, 50.0);
        let tracker = KalmanBoxTracker::new(&bbox, 42);

        assert_eq!(tracker.id(), 42);
    }

    // ==========================================================================
    // KalmanBoxTracker::get_confidence tests
    // ==========================================================================

    #[test]
    fn test_get_confidence_young_track() {
        // Young track (age < 7): confidence = coef^(7 - age)
        let bbox = box_xyxy(100.0, 100.0, 200.0, 200.0);
        let tracker = KalmanBoxTracker::new(&bbox, 0);

        // age = 0, coef = 0.9 -> 0.9^7 ≈ 0.478
        let conf = tracker.get_confidence(0.9);
        assert!((conf - 0.9_f32.powi(7)).abs() < 1e-5);
    }

    #[test]
    fn test_get_confidence_mature_track() {
        // After several updates, track becomes mature
        let bbox = box_xyxy(100.0, 100.0, 200.0, 200.0);
        let mut tracker = KalmanBoxTracker::new(&bbox, 0);

        // Simulate 10 frames of tracking
        for _ in 0..10 {
            tracker.predict();
            tracker.update(&bbox, 0.9);
        }

        // Now time_since_update = 0, age >= 7
        // confidence = coef^(time_since_update - 1) = coef^(-1) = 1/coef
        // Python returns coef^(-1) > 1 for recently updated tracks (bonus)
        let conf = tracker.get_confidence(0.9);
        // 0.9^(-1) = 1/0.9 ≈ 1.111
        assert!((conf - 1.0 / 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_get_confidence_after_miss() {
        let bbox = box_xyxy(100.0, 100.0, 200.0, 200.0);
        let mut tracker = KalmanBoxTracker::new(&bbox, 0);

        // Mature the track
        for _ in 0..10 {
            tracker.predict();
            tracker.update(&bbox, 0.9);
        }

        // Now miss a few frames (predict only, no update)
        tracker.predict();
        tracker.predict();

        // time_since_update = 2 for mature track
        // confidence = 0.9^(2-1) = 0.9
        let conf = tracker.get_confidence(0.9);
        assert!((conf - 0.9).abs() < 1e-5);
    }

    // ==========================================================================
    // KalmanBoxTracker::predict tests
    // ==========================================================================

    #[test]
    fn test_predict_increments_age() {
        let bbox = box_xyxy(100.0, 100.0, 200.0, 200.0);
        let mut tracker = KalmanBoxTracker::new(&bbox, 0);

        assert_eq!(tracker.age(), 0);
        tracker.predict();
        assert_eq!(tracker.age(), 1);
        tracker.predict();
        assert_eq!(tracker.age(), 2);
    }

    #[test]
    fn test_predict_increments_time_since_update() {
        let bbox = box_xyxy(100.0, 100.0, 200.0, 200.0);
        let mut tracker = KalmanBoxTracker::new(&bbox, 0);

        assert_eq!(tracker.time_since_update(), 0);
        tracker.predict();
        assert_eq!(tracker.time_since_update(), 1);
        tracker.predict();
        assert_eq!(tracker.time_since_update(), 2);
    }

    #[test]
    fn test_predict_resets_hit_streak_after_miss() {
        let bbox = box_xyxy(100.0, 100.0, 200.0, 200.0);
        let mut tracker = KalmanBoxTracker::new(&bbox, 0);

        // Build up hit streak
        tracker.predict();
        tracker.update(&bbox, 0.9);
        tracker.predict();
        tracker.update(&bbox, 0.9);

        assert!(tracker.hit_streak() > 0);

        // Miss a frame (predict without update)
        tracker.predict();
        // Hit streak should reset to 0 on next predict if time_since_update > 0
        tracker.predict();

        assert_eq!(tracker.hit_streak(), 0);
    }

    #[test]
    fn test_predict_returns_bbox() {
        let bbox = box_xyxy(100.0, 100.0, 200.0, 200.0);
        let mut tracker = KalmanBoxTracker::new(&bbox, 0);

        let predicted = tracker.predict();

        // Predicted box should be close to initial (no velocity initially)
        assert!((predicted[0] - bbox[0]).abs() < 10.0);
        assert!((predicted[1] - bbox[1]).abs() < 10.0);
        assert!((predicted[2] - bbox[2]).abs() < 10.0);
        assert!((predicted[3] - bbox[3]).abs() < 10.0);
    }

    // ==========================================================================
    // KalmanBoxTracker::update tests
    // ==========================================================================

    #[test]
    fn test_update_resets_time_since_update() {
        let bbox = box_xyxy(100.0, 100.0, 200.0, 200.0);
        let mut tracker = KalmanBoxTracker::new(&bbox, 0);

        tracker.predict();
        tracker.predict();
        assert_eq!(tracker.time_since_update(), 2);

        tracker.update(&bbox, 0.9);
        assert_eq!(tracker.time_since_update(), 0);
    }

    #[test]
    fn test_update_increments_hit_streak() {
        let bbox = box_xyxy(100.0, 100.0, 200.0, 200.0);
        let mut tracker = KalmanBoxTracker::new(&bbox, 0);

        assert_eq!(tracker.hit_streak(), 0);

        tracker.predict();
        tracker.update(&bbox, 0.9);
        assert_eq!(tracker.hit_streak(), 1);

        tracker.predict();
        tracker.update(&bbox, 0.9);
        assert_eq!(tracker.hit_streak(), 2);
    }

    // ==========================================================================
    // KalmanBoxTracker::get_state tests
    // ==========================================================================

    #[test]
    fn test_get_state_initial() {
        let bbox = box_xyxy(100.0, 100.0, 200.0, 200.0);
        let tracker = KalmanBoxTracker::new(&bbox, 0);

        let state = tracker.get_state();

        // Should be close to initial bbox
        assert!((state[0] - bbox[0]).abs() < 1e-3);
        assert!((state[1] - bbox[1]).abs() < 1e-3);
        assert!((state[2] - bbox[2]).abs() < 1e-3);
        assert!((state[3] - bbox[3]).abs() < 1e-3);
    }

    #[test]
    fn test_get_state_after_update() {
        let bbox1 = box_xyxy(100.0, 100.0, 200.0, 200.0);
        let bbox2 = box_xyxy(110.0, 110.0, 210.0, 210.0);
        let mut tracker = KalmanBoxTracker::new(&bbox1, 0);

        tracker.predict();
        tracker.update(&bbox2, 0.9);

        let state = tracker.get_state();

        // State should have moved toward bbox2
        assert!(state[0] > bbox1[0]);
        assert!(state[1] > bbox1[1]);
    }

    // ==========================================================================
    // KalmanBoxTracker Mahalanobis distance support tests
    // ==========================================================================

    #[test]
    fn test_get_kf_state() {
        let bbox = box_xyxy(100.0, 100.0, 200.0, 200.0);
        let tracker = KalmanBoxTracker::new(&bbox, 0);

        let state = tracker.get_kf_state();

        // State should have 8 elements: [x, y, h, r, vx, vy, vh, vr]
        assert_eq!(state.len(), 8);
    }

    #[test]
    fn test_get_kf_covariance_diag() {
        let bbox = box_xyxy(100.0, 100.0, 200.0, 200.0);
        let tracker = KalmanBoxTracker::new(&bbox, 0);

        let cov_diag = tracker.get_kf_covariance_diag();

        // Should have 8 diagonal elements
        assert_eq!(cov_diag.len(), 8);
        // All should be positive (variances)
        for &v in &cov_diag {
            assert!(v > 0.0);
        }
    }

    #[test]
    fn test_camera_update_translation() {
        let bbox = box_xyxy(100.0, 100.0, 200.0, 200.0);
        let mut tracker = KalmanBoxTracker::new(&bbox, 0);
        let t = [[1.0, 0.0, 5.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]];
        tracker.camera_update(&t);
        let state = tracker.get_state();
        assert!((state[0] - 105.0).abs() < 1e-4);
        assert!((state[1] - 97.0).abs() < 1e-4);
        assert!((state[2] - 205.0).abs() < 1e-4);
        assert!((state[3] - 197.0).abs() < 1e-4);
    }
}
