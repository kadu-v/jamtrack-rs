use std::collections::HashMap;

use super::kalman_filter::{DetectBox, KalmanFilter};

/// Individual tracked object for OC-SORT.
pub(crate) struct KalmanBoxTracker {
    kf: KalmanFilter,
    id: usize,
    time_since_update: usize,
    hits: usize,
    hit_streak: usize,
    age: usize,
    /// Last observed bounding box `[x1, y1, x2, y2, score]`, or `None` if never observed.
    last_observation: Option<[f32; 5]>,
    /// Mapping from `age` to the observation `[x1, y1, x2, y2, score]` at that age.
    observations: HashMap<usize, [f32; 5]>,
    /// All observations in order.
    history_observations: Vec<[f32; 5]>,
    /// Normalised velocity direction `[dy, dx]`, or `None`.
    velocity: Option<[f32; 2]>,
    delta_t: usize,
}

impl KalmanBoxTracker {
    pub(crate) fn new(bbox: &[f32; 5], id: usize, delta_t: usize) -> Self {
        let z = convert_bbox_to_z(bbox);
        Self {
            kf: KalmanFilter::new(&z),
            id,
            time_since_update: 0,
            hits: 0,
            hit_streak: 0,
            age: 0,
            last_observation: None,
            observations: HashMap::new(),
            history_observations: Vec::new(),
            velocity: None,
            delta_t,
        }
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    pub(crate) fn id(&self) -> usize {
        self.id
    }

    pub(crate) fn time_since_update(&self) -> usize {
        self.time_since_update
    }

    pub(crate) fn hit_streak(&self) -> usize {
        self.hit_streak
    }

    pub(crate) fn age(&self) -> usize {
        self.age
    }

    pub(crate) fn velocity(&self) -> Option<[f32; 2]> {
        self.velocity
    }

    /// Returns `last_observation` or `[-1,-1,-1,-1,-1]` placeholder.
    pub(crate) fn last_observation_or_placeholder(&self) -> [f32; 5] {
        self.last_observation.unwrap_or([-1.0; 5])
    }

    pub(crate) fn last_observation(&self) -> Option<&[f32; 5]> {
        self.last_observation.as_ref()
    }

    pub(crate) fn observations(&self) -> &HashMap<usize, [f32; 5]> {
        &self.observations
    }

    // ------------------------------------------------------------------
    // Update
    // ------------------------------------------------------------------

    /// Update the tracker with an observed bounding box `[x1,y1,x2,y2,score]`.
    pub(crate) fn update(&mut self, bbox: &[f32; 5]) {
        // Compute velocity from previous observation
        if self.last_observation.is_some() {
            let previous_box = self.find_previous_box();
            if let Some(prev) = previous_box {
                self.velocity = Some(speed_direction(&prev, bbox));
            }
        }

        self.last_observation = Some(*bbox);
        self.observations.insert(self.age, *bbox);
        self.history_observations.push(*bbox);

        self.time_since_update = 0;
        self.hits += 1;
        self.hit_streak += 1;

        let z = convert_bbox_to_z(bbox);
        self.kf.update(Some(&z));
    }

    /// Mark this tracker as unobserved for this frame.
    pub(crate) fn update_none(&mut self) {
        self.kf.update(None);
    }

    // ------------------------------------------------------------------
    // Predict
    // ------------------------------------------------------------------

    /// Predict the next state and return the bounding box `[x1,y1,x2,y2]`.
    pub(crate) fn predict(&mut self) -> [f32; 4] {
        // Prevent negative area
        if self.kf.state()[6] + self.kf.state()[2] <= 0.0 {
            // Zero out velocity of area
            // (self.kf.x[6] *= 0.0 in Python)
            let x = self.kf.state_mut();
            x[6] = 0.0;
        }

        self.kf.predict();
        self.age += 1;
        if self.time_since_update > 0 {
            self.hit_streak = 0;
        }
        self.time_since_update += 1;
        self.get_state()
    }

    /// Get the current bounding box estimate `[x1,y1,x2,y2]`.
    pub(crate) fn get_state(&self) -> [f32; 4] {
        let x = self.kf.state();
        convert_x_to_bbox(x[0], x[1], x[2], x[3])
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// Find the previous observation within `delta_t` frames.
    fn find_previous_box(&self) -> Option<[f32; 5]> {
        for i in 0..self.delta_t {
            let dt = self.delta_t - i;
            if self.age >= dt {
                if let Some(obs) = self.observations.get(&(self.age - dt)) {
                    return Some(*obs);
                }
            }
        }
        // Fall back to last_observation
        self.last_observation
    }
}

impl std::fmt::Debug for KalmanBoxTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KalmanBoxTracker {{ id: {}, age: {}, hit_streak: {}, tsu: {} }}",
            self.id, self.age, self.hit_streak, self.time_since_update
        )
    }
}

// ==========================================================================
// Free functions
// ==========================================================================

/// Convert bounding box `[x1,y1,x2,y2]` to measurement `[cx, cy, s, r]`
/// where `s = w * h` (area) and `r = w / h` (aspect ratio).
pub(crate) fn convert_bbox_to_z(bbox: &[f32]) -> DetectBox {
    let w = bbox[2] - bbox[0];
    let h = bbox[3] - bbox[1];
    let x = bbox[0] + w / 2.0;
    let y = bbox[1] + h / 2.0;
    let s = w * h;
    let r = w / (h + 1e-6);
    DetectBox::new(x, y, s, r)
}

/// Convert state `[cx, cy, s, r]` to bounding box `[x1, y1, x2, y2]`.
pub(crate) fn convert_x_to_bbox(x: f32, y: f32, s: f32, r: f32) -> [f32; 4] {
    let w = (s * r).sqrt();
    let h = s / w;
    [x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0]
}

/// Compute normalised speed direction between two bounding boxes.
///
/// Returns `[dy, dx]` (note: y-component first, matching Python).
pub(crate) fn speed_direction(bbox1: &[f32], bbox2: &[f32]) -> [f32; 2] {
    let cx1 = (bbox1[0] + bbox1[2]) / 2.0;
    let cy1 = (bbox1[1] + bbox1[3]) / 2.0;
    let cx2 = (bbox2[0] + bbox2[2]) / 2.0;
    let cy2 = (bbox2[1] + bbox2[3]) / 2.0;
    let dy = cy2 - cy1;
    let dx = cx2 - cx1;
    let norm = (dy * dy + dx * dx).sqrt() + 1e-6;
    [dy / norm, dx / norm]
}

/// Look back up to `k` frames for a previous observation.
///
/// Returns `None` if `observations` is empty.
pub(crate) fn k_previous_obs(
    observations: &HashMap<usize, [f32; 5]>,
    cur_age: usize,
    k: usize,
) -> Option<[f32; 5]> {
    if observations.is_empty() {
        return None;
    }
    for i in 0..k {
        let dt = k - i;
        if cur_age >= dt {
            if let Some(obs) = observations.get(&(cur_age - dt)) {
                return Some(*obs);
            }
        }
    }
    // Fall back to the most recent observation
    let max_age = observations.keys().max().unwrap();
    Some(observations[max_age])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_bbox_to_z() {
        let z = convert_bbox_to_z(&[100.0, 100.0, 200.0, 200.0]);
        assert!((z[0] - 150.0).abs() < 1e-4); // cx
        assert!((z[1] - 150.0).abs() < 1e-4); // cy
        assert!((z[2] - 10000.0).abs() < 1e-2); // s = 100*100
        assert!((z[3] - 1.0).abs() < 1e-4); // r = 100/100
    }

    #[test]
    fn test_convert_x_to_bbox() {
        let bbox = convert_x_to_bbox(150.0, 150.0, 10000.0, 1.0);
        assert!((bbox[0] - 100.0).abs() < 1e-3);
        assert!((bbox[1] - 100.0).abs() < 1e-3);
        assert!((bbox[2] - 200.0).abs() < 1e-3);
        assert!((bbox[3] - 200.0).abs() < 1e-3);
    }

    #[test]
    fn test_convert_roundtrip() {
        let original = [50.0, 100.0, 200.0, 300.0];
        let z = convert_bbox_to_z(&original);
        let recovered = convert_x_to_bbox(z[0], z[1], z[2], z[3]);
        for i in 0..4 {
            assert!((recovered[i] - original[i]).abs() < 0.1);
        }
    }

    #[test]
    fn test_speed_direction() {
        // Python: speed_direction([100,100,200,200],[110,120,210,220]) = [0.8944, 0.4472]
        let spd = speed_direction(&[100.0, 100.0, 200.0, 200.0], &[110.0, 120.0, 210.0, 220.0]);
        assert!((spd[0] - 0.8944).abs() < 0.001); // dy
        assert!((spd[1] - 0.4472).abs() < 0.001); // dx
    }

    #[test]
    fn test_k_previous_obs_found() {
        let mut obs = HashMap::new();
        obs.insert(0, [100.0, 100.0, 200.0, 200.0, 0.9]);
        obs.insert(2, [120.0, 120.0, 220.0, 220.0, 0.85]);
        // k=3, age=3: look for age-3=0 (found)
        let result = k_previous_obs(&obs, 3, 3);
        assert!(result.is_some());
        assert!((result.unwrap()[0] - 100.0).abs() < 1e-5);
    }

    #[test]
    fn test_k_previous_obs_fallback() {
        let mut obs = HashMap::new();
        obs.insert(0, [100.0, 100.0, 200.0, 200.0, 0.9]);
        obs.insert(2, [120.0, 120.0, 220.0, 220.0, 0.85]);
        // k=3, age=5: look for 2,3,4 → found at 2
        let result = k_previous_obs(&obs, 5, 3);
        assert!(result.is_some());
        assert!((result.unwrap()[0] - 120.0).abs() < 1e-5);
    }

    #[test]
    fn test_k_previous_obs_empty() {
        let obs = HashMap::new();
        assert!(k_previous_obs(&obs, 3, 3).is_none());
    }

    #[test]
    fn test_tracker_new() {
        let bbox = [100.0, 100.0, 200.0, 200.0, 0.9];
        let trk = KalmanBoxTracker::new(&bbox, 0, 3);
        assert_eq!(trk.id(), 0);
        assert_eq!(trk.time_since_update(), 0);
        assert_eq!(trk.hit_streak(), 0);
        assert_eq!(trk.age(), 0);
        assert!(trk.velocity().is_none());
        assert!(trk.last_observation().is_none());
    }

    #[test]
    fn test_tracker_predict_update() {
        let bbox1 = [100.0, 100.0, 200.0, 200.0, 0.9];
        let mut trk = KalmanBoxTracker::new(&bbox1, 0, 3);

        let pred = trk.predict();
        // Should be close to initial
        assert!((pred[0] - 100.0).abs() < 1.0);
        assert!((pred[1] - 100.0).abs() < 1.0);
        assert_eq!(trk.age(), 1);
        assert_eq!(trk.time_since_update(), 1);

        let bbox2 = [110.0, 110.0, 210.0, 210.0, 0.85];
        trk.update(&bbox2);
        assert_eq!(trk.time_since_update(), 0);
        assert_eq!(trk.hit_streak(), 1);

        let state = trk.get_state();
        // Python: [109.999, 109.999, 209.999, 209.999]
        assert!((state[0] - 110.0).abs() < 1.0);
        assert!((state[1] - 110.0).abs() < 1.0);
    }
}
