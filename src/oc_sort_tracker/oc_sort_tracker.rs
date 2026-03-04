use super::assoc::{associate, iou_batch, AssocResult};
use super::strack::{k_previous_obs, KalmanBoxTracker};
use crate::error::TrackError;
use crate::object::Object;
use crate::rect::Rect;

/// OC-SORT: Observation-Centric SORT tracker.
pub struct OCSort {
    det_thresh: f32,
    max_age: usize,
    min_hits: usize,
    iou_threshold: f32,
    delta_t: usize,
    inertia: f32,
    use_byte: bool,

    trackers: Vec<KalmanBoxTracker>,
    frame_count: usize,
    track_id_count: usize,
}

impl OCSort {
    /// Create a new OC-SORT tracker.
    ///
    /// # Arguments
    /// * `det_thresh` — detection confidence threshold
    pub fn new(det_thresh: f32) -> Self {
        Self {
            det_thresh,
            max_age: 30,
            min_hits: 3,
            iou_threshold: 0.3,
            delta_t: 3,
            inertia: 0.2,
            use_byte: false,
            trackers: Vec::new(),
            frame_count: 0,
            track_id_count: 0,
        }
    }

    // ------------------------------------------------------------------
    // Builder methods
    // ------------------------------------------------------------------

    pub fn with_max_age(mut self, max_age: usize) -> Self {
        self.max_age = max_age;
        self
    }

    pub fn with_min_hits(mut self, min_hits: usize) -> Self {
        self.min_hits = min_hits;
        self
    }

    pub fn with_iou_threshold(mut self, iou_threshold: f32) -> Self {
        self.iou_threshold = iou_threshold;
        self
    }

    pub fn with_delta_t(mut self, delta_t: usize) -> Self {
        self.delta_t = delta_t;
        self
    }

    pub fn with_inertia(mut self, inertia: f32) -> Self {
        self.inertia = inertia;
        self
    }

    pub fn with_byte(mut self, use_byte: bool) -> Self {
        self.use_byte = use_byte;
        self
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    pub fn tracker_count(&self) -> usize {
        self.trackers.len()
    }

    // ------------------------------------------------------------------
    // Main update
    // ------------------------------------------------------------------

    /// Process one frame of detections and return active tracks.
    ///
    /// # Arguments
    /// * `objects` — detections as `Object` (rect in TLWH + prob)
    ///
    /// # Returns
    /// Active tracks as `Vec<Object>` with `track_id` set.
    pub fn update(&mut self, objects: &[Object]) -> Result<Vec<Object>, TrackError> {
        self.frame_count += 1;

        // Convert to [x1,y1,x2,y2,score]
        let all_dets: Vec<[f32; 5]> = objects
            .iter()
            .map(|o| {
                let r = o.get_rect().get_xyxy();
                [r[0], r[1], r[2], r[3], o.get_prob()]
            })
            .collect();

        // Split high / low score detections
        let mut dets = Vec::new();
        let mut dets_second = Vec::new();
        for &d in &all_dets {
            if d[4] > self.det_thresh {
                dets.push(d);
            } else if d[4] > 0.1 {
                dets_second.push(d);
            }
        }

        // ------------------------------------------------------------------
        // Predict existing trackers
        // ------------------------------------------------------------------
        let mut trks: Vec<[f32; 4]> = Vec::with_capacity(self.trackers.len());
        let mut to_del = Vec::new();
        for (t, tracker) in self.trackers.iter_mut().enumerate() {
            let pos = tracker.predict();
            if pos.iter().any(|v| v.is_nan()) {
                to_del.push(t);
            }
            trks.push(pos);
        }
        // Remove NaN trackers (reverse order)
        for &t in to_del.iter().rev() {
            self.trackers.remove(t);
            trks.remove(t);
        }

        // ------------------------------------------------------------------
        // Collect tracker metadata for association
        // ------------------------------------------------------------------
        let velocities: Vec<[f32; 2]> = self
            .trackers
            .iter()
            .map(|t| t.velocity().unwrap_or([0.0, 0.0]))
            .collect();

        let last_boxes: Vec<[f32; 5]> = self
            .trackers
            .iter()
            .map(|t| t.last_observation_or_placeholder())
            .collect();

        let k_observations: Vec<[f32; 5]> = self
            .trackers
            .iter()
            .map(|t| {
                k_previous_obs(t.observations(), t.age(), self.delta_t)
                    .unwrap_or([-1.0, -1.0, -1.0, -1.0, -1.0])
            })
            .collect();

        // ==================================================================
        // First round of association (IoU + VDC)
        // ==================================================================
        let AssocResult {
            matches: matched,
            mut unmatched_dets,
            mut unmatched_trks,
        } = associate(
            &dets,
            &trks,
            self.iou_threshold,
            &velocities,
            &k_observations,
            self.inertia,
        )?;

        for &[di, ti] in &matched {
            self.trackers[ti].update(&dets[di]);
        }

        // ==================================================================
        // BYTE association (optional): low-score dets vs unmatched tracks
        // ==================================================================
        if self.use_byte && !dets_second.is_empty() && !unmatched_trks.is_empty() {
            let u_trks: Vec<[f32; 4]> =
                unmatched_trks.iter().map(|&i| trks[i]).collect();
            let det_boxes: Vec<[f32; 4]> = dets_second
                .iter()
                .map(|d| [d[0], d[1], d[2], d[3]])
                .collect();
            let iou_left = iou_batch(&det_boxes, &u_trks);

            if iou_left.max() > self.iou_threshold {
                let byte_matched =
                    simple_iou_match(&iou_left, self.iou_threshold);
                let mut to_remove_trks = Vec::new();
                for (det_ind, rel_trk_ind) in byte_matched {
                    let trk_ind = unmatched_trks[rel_trk_ind];
                    self.trackers[trk_ind].update(&dets_second[det_ind]);
                    to_remove_trks.push(trk_ind);
                }
                unmatched_trks.retain(|t| !to_remove_trks.contains(t));
            }
        }

        // ==================================================================
        // OCR: second round re-association using last_observation
        // ==================================================================
        if !unmatched_dets.is_empty() && !unmatched_trks.is_empty() {
            let left_dets: Vec<[f32; 4]> = unmatched_dets
                .iter()
                .map(|&i| [dets[i][0], dets[i][1], dets[i][2], dets[i][3]])
                .collect();
            let left_trks: Vec<[f32; 4]> = unmatched_trks
                .iter()
                .map(|&i| {
                    let lo = last_boxes[i];
                    [lo[0], lo[1], lo[2], lo[3]]
                })
                .collect();

            let iou_left = iou_batch(&left_dets, &left_trks);

            if iou_left.max() > self.iou_threshold {
                let rematched =
                    simple_iou_match(&iou_left, self.iou_threshold);
                let mut to_remove_dets = Vec::new();
                let mut to_remove_trks_2 = Vec::new();
                for (rel_det, rel_trk) in rematched {
                    let det_ind = unmatched_dets[rel_det];
                    let trk_ind = unmatched_trks[rel_trk];
                    self.trackers[trk_ind].update(&dets[det_ind]);
                    to_remove_dets.push(det_ind);
                    to_remove_trks_2.push(trk_ind);
                }
                unmatched_dets.retain(|d| !to_remove_dets.contains(d));
                unmatched_trks.retain(|t| !to_remove_trks_2.contains(t));
            }
        }

        // ==================================================================
        // Mark unmatched trackers as unobserved
        // ==================================================================
        for &m in &unmatched_trks {
            self.trackers[m].update_none();
        }

        // ==================================================================
        // Create new trackers for unmatched detections
        // ==================================================================
        for &i in &unmatched_dets {
            let trk = KalmanBoxTracker::new(&dets[i], self.track_id_count, self.delta_t);
            self.track_id_count += 1;
            self.trackers.push(trk);
        }

        // ==================================================================
        // Collect output and remove dead tracks
        // ==================================================================
        let mut ret = Vec::new();
        let mut i = self.trackers.len();
        while i > 0 {
            i -= 1;
            let trk = &self.trackers[i];

            // Use last_observation if available, otherwise KF state
            let d = if trk.last_observation().is_some() {
                let lo = trk.last_observation().unwrap();
                [lo[0], lo[1], lo[2], lo[3]]
            } else {
                trk.get_state()
            };

            let output = trk.time_since_update() < 1
                && (trk.hit_streak() >= self.min_hits
                    || self.frame_count <= self.min_hits);

            if output {
                let rect = Rect::from_xyxy(d[0], d[1], d[2], d[3]);
                ret.push(Object::new(
                    rect,
                    1.0,
                    Some(trk.id() + 1), // MOT benchmark requires positive
                ));
            }

            // Remove dead tracklets
            if trk.time_since_update() > self.max_age {
                self.trackers.remove(i);
            }
        }

        Ok(ret)
    }
}

/// Simple IoU-based matching using LAPJV.
/// Returns `Vec<(det_idx, trk_idx)>` pairs above `threshold`.
fn simple_iou_match(
    iou_matrix: &nalgebra::DMatrix<f32>,
    threshold: f32,
) -> Vec<(usize, usize)> {
    let nd = iou_matrix.nrows();
    let nt = iou_matrix.ncols();
    if nd == 0 || nt == 0 {
        return vec![];
    }

    let n = nd.max(nt);
    let mut cost = vec![vec![1e5_f64; n]; n];
    for i in 0..nd {
        for j in 0..nt {
            cost[i][j] = -(iou_matrix[(i, j)] as f64);
        }
    }

    let mut x = vec![-1_isize; n];
    let mut y = vec![-1_isize; n];
    if crate::lapjv::lapjv(&mut cost, &mut x, &mut y).is_err() {
        return vec![];
    }

    let mut result = Vec::new();
    for i in 0..nd {
        let j = x[i];
        if j >= 0 && (j as usize) < nt && iou_matrix[(i, j as usize)] >= threshold {
            result.push((i, j as usize));
        }
    }
    result
}

impl std::fmt::Debug for OCSort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "OCSort {{ frame: {}, trackers: {}, det_thresh: {} }}",
            self.frame_count,
            self.trackers.len(),
            self.det_thresh,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::Object;
    use crate::rect::Rect;

    fn make_det(x1: f32, y1: f32, x2: f32, y2: f32, prob: f32) -> Object {
        Object::new(Rect::from_xyxy(x1, y1, x2, y2), prob, None)
    }

    #[test]
    fn test_four_frame_pipeline() {
        let mut oc = OCSort::new(0.5)
            .with_max_age(30)
            .with_min_hits(3)
            .with_iou_threshold(0.3)
            .with_delta_t(3)
            .with_inertia(0.2);

        let frames = vec![
            vec![
                make_det(100.0, 100.0, 200.0, 200.0, 0.9),
                make_det(300.0, 300.0, 400.0, 400.0, 0.8),
            ],
            vec![
                make_det(105.0, 105.0, 205.0, 205.0, 0.85),
                make_det(305.0, 305.0, 405.0, 405.0, 0.75),
            ],
            vec![
                make_det(110.0, 110.0, 210.0, 210.0, 0.88),
                make_det(310.0, 310.0, 410.0, 410.0, 0.78),
            ],
            vec![
                make_det(115.0, 115.0, 215.0, 215.0, 0.87),
                make_det(315.0, 315.0, 415.0, 415.0, 0.82),
            ],
        ];

        // First 3 frames: output depends on min_hits
        for (i, dets) in frames.iter().enumerate() {
            let ret = oc.update(dets).unwrap();
            if i < 3 {
                // min_hits=3, frame_count <= 3: Python outputs all
                assert_eq!(ret.len(), 2, "frame {} should have 2 outputs", i + 1);
            }
        }

        // Frame 4: tracks have hit_streak >= 3
        let ret = oc.update(&frames[3]).unwrap();
        assert_eq!(ret.len(), 2);
        assert_eq!(oc.tracker_count(), 2);

        // Track IDs should be positive (MOT benchmark)
        for obj in &ret {
            assert!(obj.get_track_id().unwrap() > 0);
        }
    }

    #[test]
    fn test_empty_detections() {
        let mut oc = OCSort::new(0.5);
        let ret = oc.update(&[]).unwrap();
        assert!(ret.is_empty());
    }

    #[test]
    fn test_below_threshold() {
        let mut oc = OCSort::new(0.5);
        // All detections below threshold
        let dets = vec![make_det(100.0, 100.0, 200.0, 200.0, 0.3)];
        let ret = oc.update(&dets).unwrap();
        assert!(ret.is_empty());
    }
}
