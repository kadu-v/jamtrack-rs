//! Main BoostTracker implementation
//!
//! This module provides the `BoostTracker` struct that implements
//! multi-object tracking with confidence boosting techniques.

use super::assoc::{associate, iou_batch, mh_dist_similarity, shape_similarity, soft_biou_batch, AssociateParams};
use super::strack::{convert_bbox_to_z, KalmanBoxTracker};
use crate::error::TrackError;
use crate::object::Object;
use crate::rect::Rect;
use nalgebra::DMatrix;

/// BoostTracker - Multi-object tracker with confidence boosting
///
/// Implements tracking with:
/// - IoU-based association with Mahalanobis distance
/// - Shape similarity for improved matching
/// - DLO (Detection-Level Observation) confidence boost
/// - DUO (Detection-Update Observation) confidence boost
#[derive(Debug)]
pub struct BoostTracker {
    // Required parameters
    det_thresh: f32,
    iou_threshold: f32,
    max_age: usize,
    min_hits: usize,

    // Association weights
    lambda_iou: f32,
    lambda_mhd: f32,
    lambda_shape: f32,

    // Boost settings (BoostTrack basic)
    use_dlo_boost: bool,
    use_duo_boost: bool,
    dlo_boost_coef: f32,

    // BoostTrack+ settings
    use_rich_similarity: bool,

    // BoostTrack++ settings
    use_soft_boost: bool,
    use_varying_threshold: bool,

    // Internal state
    frame_count: usize,
    track_id_count: usize,
    trackers: Vec<KalmanBoxTracker>,
}

impl BoostTracker {
    /// Create a new BoostTracker with default settings.
    ///
    /// # Arguments
    /// * `det_thresh` - Detection confidence threshold
    /// * `iou_threshold` - IoU threshold for matching
    /// * `max_age` - Maximum frames to keep lost track
    /// * `min_hits` - Minimum hits before outputting track
    ///
    /// # Example
    /// ```
    /// use jamtrack_rs::boost_tracker::BoostTracker;
    /// let tracker = BoostTracker::new(0.5, 0.3, 30, 3);
    /// ```
    pub fn new(
        det_thresh: f32,
        iou_threshold: f32,
        max_age: usize,
        min_hits: usize,
    ) -> Self {
        Self {
            det_thresh,
            iou_threshold,
            max_age,
            min_hits,
            lambda_iou: 0.5,
            lambda_mhd: 0.25,
            lambda_shape: 0.25,
            use_dlo_boost: true,
            use_duo_boost: true,
            dlo_boost_coef: 0.65,
            use_rich_similarity: false,
            use_soft_boost: false,
            use_varying_threshold: false,
            frame_count: 0,
            track_id_count: 0,
            trackers: Vec::new(),
        }
    }

    /// Set association lambda weights.
    ///
    /// # Arguments
    /// * `lambda_iou` - Weight for IoU similarity (default: 0.5)
    /// * `lambda_mhd` - Weight for Mahalanobis distance (default: 0.25)
    /// * `lambda_shape` - Weight for shape similarity (default: 0.25)
    pub fn with_lambdas(
        self,
        lambda_iou: f32,
        lambda_mhd: f32,
        lambda_shape: f32,
    ) -> Self {
        Self {
            lambda_iou,
            lambda_mhd,
            lambda_shape,
            ..self
        }
    }

    /// Enable/disable confidence boost methods.
    ///
    /// # Arguments
    /// * `use_dlo_boost` - Enable DLO (Detection-Level Observation) boost
    /// * `use_duo_boost` - Enable DUO (Detection-Update Observation) boost
    pub fn with_boost(self, use_dlo_boost: bool, use_duo_boost: bool) -> Self {
        Self {
            use_dlo_boost,
            use_duo_boost,
            ..self
        }
    }

    /// Enable BoostTrack+ mode.
    ///
    /// BoostTrack+ uses rich similarity (MhDist + shape + soft BIoU) for DLO boost.
    ///
    /// # Example
    /// ```
    /// use jamtrack_rs::boost_tracker::BoostTracker;
    /// let tracker = BoostTracker::new(0.5, 0.3, 30, 3).with_boost_plus();
    /// ```
    pub fn with_boost_plus(self) -> Self {
        Self {
            use_rich_similarity: true,
            ..self
        }
    }

    /// Enable BoostTrack++ mode.
    ///
    /// BoostTrack++ uses:
    /// - Rich similarity (MhDist + shape + soft BIoU)
    /// - Soft boost
    /// - Varying threshold
    ///
    /// # Example
    /// ```
    /// use jamtrack_rs::boost_tracker::BoostTracker;
    /// let tracker = BoostTracker::new(0.5, 0.3, 30, 3).with_boost_plus_plus();
    /// ```
    pub fn with_boost_plus_plus(self) -> Self {
        Self {
            use_rich_similarity: true,
            use_soft_boost: true,
            use_varying_threshold: true,
            ..self
        }
    }

    /// Update tracker with new detections.
    ///
    /// # Arguments
    /// * `objects` - List of detected objects
    ///
    /// # Returns
    /// Vector of tracked objects that meet the min_hits criterion
    pub fn update(&mut self, objects: &Vec<Object>) -> Result<Vec<Object>, TrackError> {
        self.frame_count += 1;

        // Convert objects to mutable detection array [x1, y1, x2, y2, score]
        let mut dets: Vec<[f32; 5]> = objects
            .iter()
            .map(|o| {
                let xyxy = o.get_rect().get_xyxy();
                [xyxy[0], xyxy[1], xyxy[2], xyxy[3], o.get_prob()]
            })
            .collect();

        // Step 1: Predict all existing trackers and get their states
        let mut trks: Vec<[f32; 4]> = Vec::with_capacity(self.trackers.len());
        let mut confs: Vec<f32> = Vec::with_capacity(self.trackers.len());

        for tracker in self.trackers.iter_mut() {
            let pos = tracker.predict();
            let conf = tracker.get_confidence(0.9);
            trks.push(pos);
            confs.push(conf);
        }

        // Step 2: Apply confidence boosts
        if self.use_dlo_boost && !self.trackers.is_empty() {
            self.dlo_confidence_boost(&mut dets, &trks, &confs);
        }

        if self.use_duo_boost && !self.trackers.is_empty() {
            self.duo_confidence_boost(&mut dets);
        }

        // Step 3: Filter detections by threshold
        let remain_inds: Vec<usize> = dets
            .iter()
            .enumerate()
            .filter(|(_, d)| d[4] >= self.det_thresh)
            .map(|(i, _)| i)
            .collect();

        let filtered_dets: Vec<[f32; 5]> = remain_inds.iter().map(|&i| dets[i]).collect();
        let det_bboxes: Vec<[f32; 4]> = filtered_dets
            .iter()
            .map(|d| [d[0], d[1], d[2], d[3]])
            .collect();
        let det_scores: Vec<f32> = filtered_dets.iter().map(|d| d[4]).collect();

        // Step 4: Compute Mahalanobis distance matrix
        let mh_dist = self.get_mh_dist_matrix(&det_bboxes);

        // Step 5: Association
        let params = AssociateParams {
            iou_threshold: self.iou_threshold,
            lambda_iou: self.lambda_iou,
            lambda_mhd: self.lambda_mhd,
            lambda_shape: self.lambda_shape,
        };

        let mh_dist_opt = if mh_dist.nrows() > 0 && mh_dist.ncols() > 0 {
            Some(&mh_dist)
        } else {
            None
        };

        let result = associate(
            &det_bboxes,
            &trks,
            &params,
            mh_dist_opt,
            if confs.is_empty() { None } else { Some(&confs) },
            if det_scores.is_empty() {
                None
            } else {
                Some(&det_scores)
            },
        );

        // Step 6: Update matched trackers
        for (det_idx, trk_idx) in &result.matches {
            let score = det_scores[*det_idx];
            self.trackers[*trk_idx].update(&det_bboxes[*det_idx], score);
        }

        // Step 7: Create new trackers for unmatched detections
        for &det_idx in &result.unmatched_detections {
            if det_scores[det_idx] >= self.det_thresh {
                self.track_id_count += 1;
                let new_tracker =
                    KalmanBoxTracker::new(&det_bboxes[det_idx], self.track_id_count);
                self.trackers.push(new_tracker);
            }
        }

        // Step 8: Build output and remove dead tracklets
        let mut output = Vec::new();

        for tracker in self.trackers.iter() {
            let state = tracker.get_state();

            // Output if recently updated and has enough hits
            if tracker.time_since_update() < 1
                && (tracker.hit_streak() >= self.min_hits || self.frame_count <= self.min_hits)
            {
                // state is [x1, y1, x2, y2] format
                let rect = Rect::from_xyxy(state[0], state[1], state[2], state[3]);
                let obj = Object::new(rect, tracker.get_confidence(0.9), Some(tracker.id()));
                output.push(obj);
            }
        }

        // Remove dead tracklets
        self.trackers
            .retain(|t| t.time_since_update() <= self.max_age);

        Ok(output)
    }

    /// Get current frame count.
    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    /// Get number of active trackers.
    pub fn tracker_count(&self) -> usize {
        self.trackers.len()
    }

    // =========================================================================
    // Internal methods
    // =========================================================================

    /// Compute Mahalanobis distance matrix between detections and tracks.
    fn get_mh_dist_matrix(&self, detections: &[[f32; 4]]) -> DMatrix<f32> {
        let num_dets = detections.len();
        let num_trks = self.trackers.len();
        let n_dims = 4;

        if num_trks == 0 || num_dets == 0 {
            return DMatrix::zeros(num_dets, num_trks);
        }

        // z: detection states [x, y, h, r]
        let mut z = DMatrix::zeros(num_dets, n_dims);
        for (i, det) in detections.iter().enumerate() {
            let zz = convert_bbox_to_z(det);
            for j in 0..n_dims {
                z[(i, j)] = zz[j];
            }
        }

        // x: track states [x, y, h, r]
        let mut x = DMatrix::zeros(num_trks, n_dims);
        let mut sigma_inv = DMatrix::zeros(num_trks, n_dims);

        for (i, tracker) in self.trackers.iter().enumerate() {
            let state = tracker.get_kf_state();
            let cov_diag = tracker.get_kf_covariance_diag();

            for j in 0..n_dims {
                x[(i, j)] = state[j];
                // Reciprocal of covariance diagonal
                if cov_diag[j] > 1e-10 {
                    sigma_inv[(i, j)] = 1.0 / cov_diag[j];
                } else {
                    sigma_inv[(i, j)] = 1e10;
                }
            }
        }

        // Compute Mahalanobis distance: sum((z - x)^2 * sigma_inv)
        let mut mh_dist = DMatrix::zeros(num_dets, num_trks);
        for i in 0..num_dets {
            for j in 0..num_trks {
                let mut dist = 0.0;
                for k in 0..n_dims {
                    let diff = z[(i, k)] - x[(j, k)];
                    dist += diff * diff * sigma_inv[(j, k)];
                }
                mh_dist[(i, j)] = dist;
            }
        }

        mh_dist
    }

    /// Apply DLO (Detection-Level Observation) confidence boost.
    ///
    /// - BoostTrack: Uses soft BIoU with simple coefficient
    /// - BoostTrack+: Uses rich similarity (MhDist + shape + soft BIoU)
    /// - BoostTrack++: Adds soft boost and varying threshold
    fn dlo_confidence_boost(&self, dets: &mut [[f32; 5]], trks: &[[f32; 4]], confs: &[f32]) {
        if trks.is_empty() || dets.is_empty() {
            return;
        }

        let det_bboxes: Vec<[f32; 4]> = dets.iter().map(|d| [d[0], d[1], d[2], d[3]]).collect();

        // Compute similarity matrix
        let similarity = if self.use_rich_similarity {
            // BoostTrack+ / BoostTrack++: Rich similarity
            let sbiou = soft_biou_batch(&det_bboxes, trks, confs);
            let mh_dist = self.get_mh_dist_matrix(&det_bboxes);
            let mhd_sim = mh_dist_similarity(&mh_dist, 1.0);
            let shape_sim = shape_similarity(&det_bboxes, trks);

            // S = (mhd_sim + shape_sim + sbiou) / 3
            let mut s = DMatrix::zeros(dets.len(), trks.len());
            for i in 0..dets.len() {
                for j in 0..trks.len() {
                    s[(i, j)] = (mhd_sim[(i, j)] + shape_sim[(i, j)] + sbiou[(i, j)]) / 3.0;
                }
            }
            s
        } else {
            // BoostTrack basic: Just use IoU
            iou_batch(&det_bboxes, trks)
        };

        // Get time_since_update for varying threshold
        let time_since_updates: Vec<usize> = self
            .trackers
            .iter()
            .map(|t| t.time_since_update())
            .collect();

        // Apply boost based on mode
        if !self.use_soft_boost && !self.use_varying_threshold {
            // BoostTrack / BoostTrack+: Simple coefficient boost
            for i in 0..dets.len() {
                let mut max_s = 0.0f32;
                for j in 0..trks.len() {
                    max_s = max_s.max(similarity[(i, j)]);
                }
                dets[i][4] = dets[i][4].max(max_s * self.dlo_boost_coef);
            }
        } else {
            // BoostTrack++
            if self.use_soft_boost {
                // Soft boost: alpha * score + (1 - alpha) * max_s^1.5
                let alpha = 0.65;
                for i in 0..dets.len() {
                    let mut max_s = 0.0f32;
                    for j in 0..trks.len() {
                        max_s = max_s.max(similarity[(i, j)]);
                    }
                    let boosted = alpha * dets[i][4] + (1.0 - alpha) * max_s.powf(1.5);
                    dets[i][4] = dets[i][4].max(boosted);
                }
            }

            if self.use_varying_threshold {
                // Varying threshold based on time_since_update
                let threshold_s = 0.95;
                let threshold_e = 0.8;
                let n_steps = 20.0;
                let alpha_vt = (threshold_s - threshold_e) / n_steps;

                for i in 0..dets.len() {
                    let mut should_boost = false;
                    for j in 0..trks.len() {
                        let tsu = time_since_updates[j] as f32;
                        let threshold = (threshold_s - tsu * alpha_vt).max(threshold_e);
                        if similarity[(i, j)] > threshold {
                            should_boost = true;
                            break;
                        }
                    }
                    if should_boost {
                        dets[i][4] = dets[i][4].max(self.det_thresh + 1e-5);
                    }
                }
            }
        }
    }

    /// Apply DUO (Detection-Update Observation) confidence boost.
    ///
    /// This matches the official Python implementation which includes IoU-based NMS
    /// to avoid boosting multiple overlapping detections.
    fn duo_confidence_boost(&self, dets: &mut [[f32; 5]]) {
        if self.trackers.is_empty() || dets.is_empty() || self.frame_count <= 1 {
            return;
        }

        let det_bboxes: Vec<[f32; 4]> = dets.iter().map(|d| [d[0], d[1], d[2], d[3]]).collect();
        let mh_dist = self.get_mh_dist_matrix(&det_bboxes);

        if mh_dist.nrows() == 0 || mh_dist.ncols() == 0 {
            return;
        }

        let limit = 13.2767;
        let iou_limit = 0.3;

        // Find minimum Mahalanobis distance for each detection
        let mut min_mh_dists = vec![f32::MAX; dets.len()];
        for i in 0..dets.len() {
            for j in 0..self.trackers.len() {
                min_mh_dists[i] = min_mh_dists[i].min(mh_dist[(i, j)]);
            }
        }

        // Find detections that are candidates for boosting
        // (far from all tracks AND below threshold)
        let boost_candidates: Vec<usize> = (0..dets.len())
            .filter(|&i| min_mh_dists[i] > limit && dets[i][4] < self.det_thresh)
            .collect();

        if boost_candidates.is_empty() {
            return;
        }

        // Compute IoU between boost candidates
        let boost_bboxes: Vec<[f32; 4]> = boost_candidates
            .iter()
            .map(|&i| det_bboxes[i])
            .collect();
        let bdiou = iou_batch(&boost_bboxes, &boost_bboxes);

        // Apply NMS-like filtering
        let mut remaining_boxes: Vec<usize> = Vec::new();

        for (bi, &det_idx) in boost_candidates.iter().enumerate() {
            // Find max IoU with other boost candidates (excluding self)
            let mut max_iou = 0.0f32;
            for bj in 0..boost_candidates.len() {
                if bi != bj {
                    max_iou = max_iou.max(bdiou[(bi, bj)]);
                }
            }

            if max_iou <= iou_limit {
                // No significant overlap, include this detection
                remaining_boxes.push(det_idx);
            } else {
                // Has overlap, only include if it has max confidence among overlapping
                let mut is_max_conf = true;
                for (bj, &other_idx) in boost_candidates.iter().enumerate() {
                    if bi != bj && bdiou[(bi, bj)] > iou_limit {
                        if dets[other_idx][4] > dets[det_idx][4] {
                            is_max_conf = false;
                            break;
                        }
                    }
                }
                if is_max_conf {
                    remaining_boxes.push(det_idx);
                }
            }
        }

        // Boost only the remaining boxes
        for det_idx in remaining_boxes {
            dets[det_idx][4] = self.det_thresh + 1e-4;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(x1: f32, y1: f32, x2: f32, y2: f32, prob: f32) -> Object {
        Object::new(Rect::new(x1, y1, x2, y2), prob, None)
    }

    // =========================================================================
    // BoostTracker::new tests
    // =========================================================================

    #[test]
    fn test_new_creates_tracker() {
        let tracker = BoostTracker::new(0.5, 0.3, 30, 3);

        assert_eq!(tracker.frame_count(), 0);
        assert_eq!(tracker.tracker_count(), 0);
    }

    #[test]
    fn test_new_with_default_lambdas() {
        let tracker = BoostTracker::new(0.5, 0.3, 30, 3);

        // Default lambda values
        assert!((tracker.lambda_iou - 0.5).abs() < 1e-5);
        assert!((tracker.lambda_mhd - 0.25).abs() < 1e-5);
        assert!((tracker.lambda_shape - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_new_with_default_boost() {
        let tracker = BoostTracker::new(0.5, 0.3, 30, 3);

        // Default boost settings
        assert!(tracker.use_dlo_boost);
        assert!(tracker.use_duo_boost);
    }

    // =========================================================================
    // BoostTracker::with_lambdas tests
    // =========================================================================

    #[test]
    fn test_with_lambdas() {
        let tracker =
            BoostTracker::new(0.5, 0.3, 30, 3).with_lambdas(0.6, 0.2, 0.2);

        assert!((tracker.lambda_iou - 0.6).abs() < 1e-5);
        assert!((tracker.lambda_mhd - 0.2).abs() < 1e-5);
        assert!((tracker.lambda_shape - 0.2).abs() < 1e-5);
    }

    // =========================================================================
    // BoostTracker::with_boost tests
    // =========================================================================

    #[test]
    fn test_with_boost_disabled() {
        let tracker =
            BoostTracker::new(0.5, 0.3, 30, 3).with_boost(false, false);

        assert!(!tracker.use_dlo_boost);
        assert!(!tracker.use_duo_boost);
    }

    #[test]
    fn test_with_boost_partial() {
        let tracker =
            BoostTracker::new(0.5, 0.3, 30, 3).with_boost(true, false);

        assert!(tracker.use_dlo_boost);
        assert!(!tracker.use_duo_boost);
    }

    // =========================================================================
    // BoostTracker::with_boost_plus tests
    // =========================================================================

    #[test]
    fn test_with_boost_plus() {
        let tracker = BoostTracker::new(0.5, 0.3, 30, 3).with_boost_plus();

        // BoostTrack+ enables rich similarity
        assert!(tracker.use_rich_similarity);
        // But not soft boost or varying threshold
        assert!(!tracker.use_soft_boost);
        assert!(!tracker.use_varying_threshold);
        // Default boost settings should remain
        assert!(tracker.use_dlo_boost);
        assert!(tracker.use_duo_boost);
    }

    #[test]
    fn test_with_boost_plus_preserves_other_settings() {
        let tracker = BoostTracker::new(0.5, 0.3, 30, 3)
            .with_lambdas(0.6, 0.2, 0.2)
            .with_boost_plus();

        // Lambda settings should be preserved
        assert!((tracker.lambda_iou - 0.6).abs() < 1e-5);
        assert!((tracker.lambda_mhd - 0.2).abs() < 1e-5);
        assert!((tracker.lambda_shape - 0.2).abs() < 1e-5);
        // Rich similarity enabled
        assert!(tracker.use_rich_similarity);
    }

    // =========================================================================
    // BoostTracker::with_boost_plus_plus tests
    // =========================================================================

    #[test]
    fn test_with_boost_plus_plus() {
        let tracker = BoostTracker::new(0.5, 0.3, 30, 3).with_boost_plus_plus();

        // BoostTrack++ enables all three features
        assert!(tracker.use_rich_similarity);
        assert!(tracker.use_soft_boost);
        assert!(tracker.use_varying_threshold);
        // Default boost settings should remain
        assert!(tracker.use_dlo_boost);
        assert!(tracker.use_duo_boost);
    }

    #[test]
    fn test_with_boost_plus_plus_preserves_other_settings() {
        let tracker = BoostTracker::new(0.5, 0.3, 30, 3)
            .with_lambdas(0.7, 0.15, 0.15)
            .with_boost(true, false)
            .with_boost_plus_plus();

        // Lambda settings should be preserved
        assert!((tracker.lambda_iou - 0.7).abs() < 1e-5);
        assert!((tracker.lambda_mhd - 0.15).abs() < 1e-5);
        assert!((tracker.lambda_shape - 0.15).abs() < 1e-5);
        // Boost settings should be preserved
        assert!(tracker.use_dlo_boost);
        assert!(!tracker.use_duo_boost);
        // BoostTrack++ features enabled
        assert!(tracker.use_rich_similarity);
        assert!(tracker.use_soft_boost);
        assert!(tracker.use_varying_threshold);
    }

    #[test]
    fn test_boost_plus_can_be_upgraded_to_plus_plus() {
        let tracker = BoostTracker::new(0.5, 0.3, 30, 3)
            .with_boost_plus()
            .with_boost_plus_plus();

        // All BoostTrack++ features should be enabled
        assert!(tracker.use_rich_similarity);
        assert!(tracker.use_soft_boost);
        assert!(tracker.use_varying_threshold);
    }

    // =========================================================================
    // BoostTracker::update tests - Basic functionality
    // =========================================================================

    #[test]
    fn test_update_empty_detections() {
        let mut tracker = BoostTracker::new(0.5, 0.3, 30, 3);
        let objects: Vec<Object> = vec![];

        let result = tracker.update(&objects).unwrap();

        assert!(result.is_empty());
        assert_eq!(tracker.frame_count(), 1);
    }

    #[test]
    fn test_update_single_detection() {
        let mut tracker = BoostTracker::new(0.5, 0.3, 30, 1);
        let objects = vec![obj(100.0, 100.0, 200.0, 200.0, 0.9)];

        // First frame: new track created but not yet output (min_hits = 1)
        let result = tracker.update(&objects).unwrap();

        assert_eq!(tracker.frame_count(), 1);
        assert_eq!(tracker.tracker_count(), 1);
        // With min_hits=1, should output after first hit
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_update_tracks_across_frames() {
        let mut tracker = BoostTracker::new(0.5, 0.3, 30, 3);

        // Frame 1: Create track
        let obj1 = vec![obj(100.0, 100.0, 200.0, 200.0, 0.9)];
        let _ = tracker.update(&obj1).unwrap();

        // Frame 2: Same position
        let obj2 = vec![obj(105.0, 105.0, 205.0, 205.0, 0.9)];
        let _ = tracker.update(&obj2).unwrap();

        // Frame 3: Same position
        let obj3 = vec![obj(110.0, 110.0, 210.0, 210.0, 0.9)];
        let result = tracker.update(&obj3).unwrap();

        assert_eq!(tracker.frame_count(), 3);
        // After 3 hits, should output (min_hits = 3)
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get_track_id(), Some(1));
    }

    #[test]
    fn test_update_multiple_detections() {
        let mut tracker = BoostTracker::new(0.5, 0.3, 30, 1);
        let objects = vec![
            obj(100.0, 100.0, 200.0, 200.0, 0.9),
            obj(300.0, 300.0, 400.0, 400.0, 0.8),
        ];

        let result = tracker.update(&objects).unwrap();

        assert_eq!(tracker.tracker_count(), 2);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_update_below_threshold_filtered() {
        let mut tracker = BoostTracker::new(0.5, 0.3, 30, 1);
        let objects = vec![
            obj(100.0, 100.0, 200.0, 200.0, 0.9), // above threshold
            obj(300.0, 300.0, 400.0, 400.0, 0.3), // below threshold
        ];

        let result = tracker.update(&objects).unwrap();

        // Only detection above threshold should create track
        assert_eq!(tracker.tracker_count(), 1);
        assert_eq!(result.len(), 1);
    }

    // =========================================================================
    // BoostTracker::update tests - Track lifecycle
    // =========================================================================

    #[test]
    fn test_track_lost_after_max_age() {
        let mut tracker = BoostTracker::new(0.5, 0.3, 3, 1); // max_age = 3

        // Frame 1: Create track
        let obj1 = vec![obj(100.0, 100.0, 200.0, 200.0, 0.9)];
        let _ = tracker.update(&obj1).unwrap();
        assert_eq!(tracker.tracker_count(), 1);

        // Frames 2-5: No detections (track should be lost after max_age)
        for _ in 0..4 {
            let _ = tracker.update(&vec![]).unwrap();
        }

        // Track should be removed after max_age frames without detection
        assert_eq!(tracker.tracker_count(), 0);
    }

    #[test]
    fn test_track_reactivated() {
        let mut tracker = BoostTracker::new(0.5, 0.3, 30, 1);

        // Frame 1: Create track
        let obj1 = vec![obj(100.0, 100.0, 200.0, 200.0, 0.9)];
        let result1 = tracker.update(&obj1).unwrap();
        let track_id = result1[0].get_track_id();

        // Frame 2: No detection (track lost)
        let _ = tracker.update(&vec![]).unwrap();

        // Frame 3: Detection at same position (should reactivate same track)
        let obj3 = vec![obj(100.0, 100.0, 200.0, 200.0, 0.9)];
        let result3 = tracker.update(&obj3).unwrap();

        assert_eq!(result3.len(), 1);
        assert_eq!(result3[0].get_track_id(), track_id);
    }

    // =========================================================================
    // BoostTracker association tests
    // =========================================================================

    #[test]
    fn test_association_by_iou() {
        let mut tracker = BoostTracker::new(0.5, 0.3, 30, 1);

        // Frame 1: Two tracks
        let obj1 = vec![
            obj(100.0, 100.0, 200.0, 200.0, 0.9),
            obj(300.0, 300.0, 400.0, 400.0, 0.9),
        ];
        let _ = tracker.update(&obj1).unwrap();

        // Frame 2: Same positions, should match correctly
        let obj2 = vec![
            obj(105.0, 105.0, 205.0, 205.0, 0.9),
            obj(305.0, 305.0, 405.0, 405.0, 0.9),
        ];
        let result = tracker.update(&obj2).unwrap();

        assert_eq!(result.len(), 2);
        // IDs should be preserved
        let ids: Vec<Option<usize>> =
            result.iter().map(|t| t.get_track_id()).collect();
        assert!(ids.contains(&Some(1)));
        assert!(ids.contains(&Some(2)));
    }
}
