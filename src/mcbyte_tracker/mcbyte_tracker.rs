use std::collections::HashSet;

use image::GrayImage;

use crate::{Object, TrackError};

use super::{
    association::associate,
    sparse_optical_flow::{SparseOptFlow, SparseOptFlowConfig},
    strack::STrack,
};

const DETECTION_DISCARD_THRESHOLD: f32 = 0.1;

#[derive(Debug, Clone)]
/// A propagated binary mask associated with an existing McByte track.
///
/// Every non-zero image pixel is treated as foreground.
pub struct McByteMask {
    track_id: usize,
    mask: GrayImage,
    average_confidence: f32,
}

impl McByteMask {
    /// Creates a mask for `track_id` with its propagation confidence.
    pub fn new(
        track_id: usize,
        mask: GrayImage,
        average_confidence: f32,
    ) -> Self {
        Self {
            track_id,
            mask,
            average_confidence,
        }
    }

    /// Returns the stable track ID associated with this mask.
    pub fn track_id(&self) -> usize {
        self.track_id
    }

    /// Returns the binary mask image.
    pub fn mask(&self) -> &GrayImage {
        &self.mask
    }

    /// Returns the average confidence reported by the mask propagator.
    pub fn average_confidence(&self) -> f32 {
        self.average_confidence
    }

    pub(crate) fn visible_area(&self) -> usize {
        self.mask
            .as_raw()
            .iter()
            .filter(|&&value| value != 0)
            .count()
    }
}

#[derive(Debug, Clone)]
/// McByte multi-object tracker with externally supplied mask conditioning.
///
/// The tracking lifecycle follows ByteTrack while ambiguous associations can
/// be strengthened by propagated masks. Camera-motion compensation is enabled
/// by default and runs when an update method receiving a grayscale frame is
/// used.
pub struct McByteTracker {
    frame_rate: usize,
    lost_track_buffer: usize,
    track_activation_threshold: f32,
    high_conf_det_threshold: f32,
    minimum_consecutive_frames: usize,
    minimum_iou_threshold_first_assoc: f32,
    minimum_iou_threshold_second_assoc: f32,
    minimum_iou_threshold_unconfirmed_assoc: f32,
    minimum_mask_average_confidence: f32,
    minimum_mask_coverage: f32,
    minimum_mask_fill_ratio: f32,
    enable_isolated_mask_matching: bool,
    instant_first_frame_activation: bool,
    cmc: Option<SparseOptFlow>,
    frame_id: usize,
    next_track_id: usize,
    tracks: Vec<STrack>,
}

impl Default for McByteTracker {
    fn default() -> Self {
        Self::new(30, 30, 0.7, 0.6)
    }
}

impl McByteTracker {
    /// Creates a tracker with the Roboflow McByte association defaults.
    pub fn new(
        frame_rate: usize,
        lost_track_buffer: usize,
        track_activation_threshold: f32,
        high_conf_det_threshold: f32,
    ) -> Self {
        Self {
            frame_rate,
            lost_track_buffer,
            track_activation_threshold,
            high_conf_det_threshold,
            minimum_consecutive_frames: 2,
            minimum_iou_threshold_first_assoc: 0.1,
            minimum_iou_threshold_second_assoc: 0.5,
            minimum_iou_threshold_unconfirmed_assoc: 0.3,
            minimum_mask_average_confidence: 0.6,
            minimum_mask_coverage: 0.9,
            minimum_mask_fill_ratio: 0.05,
            enable_isolated_mask_matching: false,
            instant_first_frame_activation: true,
            cmc: Some(SparseOptFlow::new(SparseOptFlowConfig::default())),
            frame_id: 0,
            next_track_id: 0,
            tracks: Vec::new(),
        }
    }

    /// Overrides the first, second, and unconfirmed association thresholds.
    pub fn with_association_thresholds(
        self,
        first: f32,
        second: f32,
        unconfirmed: f32,
    ) -> Self {
        Self {
            minimum_iou_threshold_first_assoc: first,
            minimum_iou_threshold_second_assoc: second,
            minimum_iou_threshold_unconfirmed_assoc: unconfirmed,
            ..self
        }
    }

    /// Configures the required successful updates and first-frame activation.
    pub fn with_confirmation(
        self,
        minimum_consecutive_frames: usize,
        instant_first_frame_activation: bool,
    ) -> Self {
        Self {
            minimum_consecutive_frames,
            instant_first_frame_activation,
            ..self
        }
    }

    /// Configures mask confidence, coverage, and detection fill thresholds.
    pub fn with_mask_thresholds(
        self,
        average_confidence: f32,
        coverage: f32,
        fill_ratio: f32,
    ) -> Self {
        Self {
            minimum_mask_average_confidence: average_confidence,
            minimum_mask_coverage: coverage,
            minimum_mask_fill_ratio: fill_ratio,
            ..self
        }
    }

    /// Enables or disables mask recovery for isolated sub-threshold IoU pairs.
    pub fn with_isolated_mask_matching(self, enabled: bool) -> Self {
        Self {
            enable_isolated_mask_matching: enabled,
            ..self
        }
    }

    /// Enables SparseOptFlow camera-motion compensation with `config`.
    pub fn with_sparse_opt_flow(self, config: SparseOptFlowConfig) -> Self {
        Self {
            cmc: Some(SparseOptFlow::new(config)),
            ..self
        }
    }

    /// Disables camera-motion compensation.
    pub fn without_cmc(self) -> Self {
        Self { cmc: None, ..self }
    }

    /// Returns the number of processed frames.
    pub fn frame_count(&self) -> usize {
        self.frame_id
    }

    /// Returns the number of tracks retained by the lifecycle manager.
    pub fn tracker_count(&self) -> usize {
        self.tracks.len()
    }

    /// Clears all tracks, IDs, frame counters, and optical-flow history.
    pub fn reset(&mut self) {
        self.frame_id = 0;
        self.next_track_id = 0;
        self.tracks.clear();
        if let Some(cmc) = &mut self.cmc {
            cmc.reset();
        }
    }

    /// Updates the tracker without mask input or camera compensation.
    pub fn update(
        &mut self,
        objects: &[Object],
    ) -> Result<Vec<Object>, TrackError> {
        self.update_impl(objects, None, &[])
    }

    /// Updates the tracker with externally propagated masks.
    pub fn update_with_masks(
        &mut self,
        objects: &[Object],
        masks: &[McByteMask],
    ) -> Result<Vec<Object>, TrackError> {
        self.update_impl(objects, None, masks)
    }

    /// Updates the tracker and estimates camera motion from a grayscale frame.
    pub fn update_with_frame(
        &mut self,
        objects: &[Object],
        frame: &GrayImage,
    ) -> Result<Vec<Object>, TrackError> {
        self.update_impl(objects, Some(frame), &[])
    }

    /// Updates the tracker using both camera motion and propagated masks.
    pub fn update_with_frame_and_masks(
        &mut self,
        objects: &[Object],
        frame: &GrayImage,
        masks: &[McByteMask],
    ) -> Result<Vec<Object>, TrackError> {
        self.update_impl(objects, Some(frame), masks)
    }

    fn update_impl(
        &mut self,
        objects: &[Object],
        frame: Option<&GrayImage>,
        masks: &[McByteMask],
    ) -> Result<Vec<Object>, TrackError> {
        self.validate(objects, frame, masks)?;
        self.frame_id += 1;
        for track in &mut self.tracks {
            track.predict();
        }

        if let (Some(cmc), Some(frame)) = (&mut self.cmc, frame) {
            let transform = cmc.estimate(frame);
            for track in &mut self.tracks {
                track.apply_cmc(&transform);
            }
        }

        let mut confirmed = Vec::new();
        let mut unconfirmed = Vec::new();
        let mut lost = Vec::new();
        for (index, track) in self.tracks.iter().enumerate() {
            if track.time_since_update() > 1 {
                lost.push(index);
            } else if track.track_id().is_some()
                || track.successful_updates() >= self.minimum_consecutive_frames
            {
                confirmed.push(index);
            } else {
                unconfirmed.push(index);
            }
        }

        let mut high = Vec::new();
        let mut low = Vec::new();
        for (index, object) in objects.iter().enumerate() {
            if object.get_prob() >= self.high_conf_det_threshold {
                high.push(index);
            } else if object.get_prob() > DETECTION_DISCARD_THRESHOLD {
                low.push(index);
            }
        }

        let mut outputs = Vec::<(usize, Object)>::new();
        let pool = confirmed.into_iter().chain(lost).collect::<Vec<_>>();
        let stage_one = self.associate_indices(
            &pool,
            &high,
            objects,
            masks,
            self.minimum_iou_threshold_first_assoc,
            true,
        )?;
        for &(pool_row, high_col) in &stage_one.matches {
            let track_index = pool[pool_row];
            let detection_index = high[high_col];
            self.update_track(track_index, &objects[detection_index]);
            if let Some(id) = self.tracks[track_index].track_id() {
                outputs.push((
                    detection_index,
                    self.tracks[track_index].to_object(),
                ));
                debug_assert!(id > 0);
            }
        }

        let remaining_tracked = stage_one
            .unmatched_tracks
            .iter()
            .map(|&row| pool[row])
            .filter(|&index| self.tracks[index].time_since_update() == 1)
            .collect::<Vec<_>>();
        let stage_two = self.associate_indices(
            &remaining_tracked,
            &low,
            objects,
            masks,
            self.minimum_iou_threshold_second_assoc,
            false,
        )?;
        for &(track_row, low_col) in &stage_two.matches {
            let track_index = remaining_tracked[track_row];
            let detection_index = low[low_col];
            self.update_track(track_index, &objects[detection_index]);
            if self.tracks[track_index].track_id().is_some() {
                outputs.push((
                    detection_index,
                    self.tracks[track_index].to_object(),
                ));
            }
        }

        let mut unmatched_high = stage_one.unmatched_detections;
        let mut unmatched_unconfirmed =
            (0..unconfirmed.len()).collect::<Vec<_>>();
        if !unconfirmed.is_empty() && !unmatched_high.is_empty() {
            let remaining_high = unmatched_high
                .iter()
                .map(|&col| high[col])
                .collect::<Vec<_>>();
            let stage_three = self.associate_indices(
                &unconfirmed,
                &remaining_high,
                objects,
                masks,
                self.minimum_iou_threshold_unconfirmed_assoc,
                true,
            )?;
            for &(track_row, detection_col) in &stage_three.matches {
                let track_index = unconfirmed[track_row];
                let detection_index = remaining_high[detection_col];
                self.update_track(track_index, &objects[detection_index]);
                if self.tracks[track_index].track_id().is_some() {
                    outputs.push((
                        detection_index,
                        self.tracks[track_index].to_object(),
                    ));
                }
            }
            unmatched_unconfirmed = stage_three.unmatched_tracks;
            unmatched_high = stage_three
                .unmatched_detections
                .into_iter()
                .map(|column| unmatched_high[column])
                .collect();
        }

        let removed_indices = unmatched_unconfirmed
            .into_iter()
            .map(|row| unconfirmed[row])
            .collect::<HashSet<_>>();
        if !removed_indices.is_empty() {
            self.tracks = self
                .tracks
                .drain(..)
                .enumerate()
                .filter_map(|(index, track)| {
                    (!removed_indices.contains(&index)).then_some(track)
                })
                .collect();
        }

        for high_col in unmatched_high {
            let detection_index = high[high_col];
            let object = &objects[detection_index];
            if object.get_prob() < self.track_activation_threshold {
                continue;
            }
            let mut track = STrack::new(object);
            if self.frame_id == 1 && self.instant_first_frame_activation {
                self.next_track_id += 1;
                track.assign_id(self.next_track_id);
                outputs.push((detection_index, track.to_object()));
            }
            self.tracks.push(track);
        }

        // Prune only after association so a track gets its full recovery
        // window. A track that has consumed the entire missed-frame budget can
        // still be matched on this frame; if it remains unmatched, it expires.
        let maximum_frames_without_update =
            self.maximum_frames_without_update();
        self.tracks.retain(|track| {
            track.time_since_update() <= maximum_frames_without_update
        });

        outputs.sort_by_key(|(index, _)| *index);
        Ok(outputs.into_iter().map(|(_, object)| object).collect())
    }

    fn maximum_frames_without_update(&self) -> usize {
        if self.lost_track_buffer == 0 {
            return 0;
        }
        ((self.frame_rate as f64 / 30.0) * self.lost_track_buffer as f64)
            .ceil()
            .max(1.0) as usize
    }

    fn update_track(&mut self, index: usize, object: &Object) {
        self.tracks[index].update(object);
        if self.tracks[index].track_id().is_none()
            && self.tracks[index].successful_updates()
                >= self.minimum_consecutive_frames
        {
            self.next_track_id += 1;
            self.tracks[index].assign_id(self.next_track_id);
        }
    }

    fn associate_indices(
        &self,
        track_indices: &[usize],
        detection_indices: &[usize],
        objects: &[Object],
        masks: &[McByteMask],
        threshold: f32,
        fuse_score: bool,
    ) -> Result<super::association::Association, TrackError> {
        let detections = detection_indices
            .iter()
            .map(|&index| objects[index].clone())
            .collect::<Vec<_>>();
        let raw = track_indices
            .iter()
            .map(|&track_index| {
                detection_indices
                    .iter()
                    .map(|&detection_index| {
                        self.tracks[track_index]
                            .rect()
                            .calc_iou(&objects[detection_index].get_rect())
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let similarity = if fuse_score {
            raw.iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .map(|(column, value)| {
                            *value * detections[column].get_prob()
                        })
                        .collect()
                })
                .collect::<Vec<_>>()
        } else {
            raw.clone()
        };
        let track_ids = track_indices
            .iter()
            .map(|&index| self.tracks[index].track_id())
            .collect::<Vec<_>>();
        associate(
            &similarity,
            &raw,
            &track_ids,
            &detections,
            masks,
            threshold,
            self.minimum_mask_average_confidence,
            self.minimum_mask_coverage,
            self.minimum_mask_fill_ratio,
            self.enable_isolated_mask_matching,
        )
    }

    fn validate(
        &self,
        objects: &[Object],
        frame: Option<&GrayImage>,
        masks: &[McByteMask],
    ) -> Result<(), TrackError> {
        if self.frame_rate == 0 || self.minimum_consecutive_frames == 0 {
            return Err(TrackError::InvalidArgument(
                "frame_rate and minimum_consecutive_frames must be positive"
                    .into(),
            ));
        }
        for (name, value) in [
            (
                "track_activation_threshold",
                self.track_activation_threshold,
            ),
            ("high_conf_det_threshold", self.high_conf_det_threshold),
            (
                "first association threshold",
                self.minimum_iou_threshold_first_assoc,
            ),
            (
                "second association threshold",
                self.minimum_iou_threshold_second_assoc,
            ),
            (
                "unconfirmed association threshold",
                self.minimum_iou_threshold_unconfirmed_assoc,
            ),
            (
                "minimum mask confidence",
                self.minimum_mask_average_confidence,
            ),
            ("minimum mask coverage", self.minimum_mask_coverage),
            ("minimum mask fill ratio", self.minimum_mask_fill_ratio),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(TrackError::InvalidArgument(format!(
                    "{name} must be between 0 and 1"
                )));
            }
        }
        if self.high_conf_det_threshold <= DETECTION_DISCARD_THRESHOLD {
            return Err(TrackError::InvalidArgument(format!(
                "high_conf_det_threshold must be greater than {DETECTION_DISCARD_THRESHOLD}"
            )));
        }
        if objects.iter().any(|object| {
            let rect = object.get_rect();
            !object.get_prob().is_finite()
                || !rect.x().is_finite()
                || !rect.y().is_finite()
                || !rect.width().is_finite()
                || !rect.height().is_finite()
                || rect.width() <= 0.0
                || rect.height() <= 0.0
        }) {
            return Err(TrackError::InvalidArgument(
                "detections must contain finite scores and positive finite rectangles".into(),
            ));
        }
        let mut ids = HashSet::new();
        let expected_dimensions = frame
            .map(GrayImage::dimensions)
            .or_else(|| masks.first().map(|mask| mask.mask.dimensions()));
        for mask in masks {
            if !ids.insert(mask.track_id) {
                return Err(TrackError::InvalidArgument(format!(
                    "duplicate McByte mask for track {}",
                    mask.track_id
                )));
            }
            if !mask.average_confidence.is_finite()
                || !(0.0..=1.0).contains(&mask.average_confidence)
            {
                return Err(TrackError::InvalidArgument(
                    "mask confidence must be between 0 and 1".into(),
                ));
            }
            if Some(mask.mask.dimensions()) != expected_dimensions {
                return Err(TrackError::InvalidArgument(
                    "all McByte masks must match the current frame dimensions"
                        .into(),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use image::{GrayImage, Luma};
    use serde::Deserialize;

    use super::*;
    use crate::Rect;

    fn detection(x: f32, score: f32) -> Object {
        Object::new(Rect::new(x, 10.0, 10.0, 20.0), score, None)
    }

    #[test]
    fn first_frame_activates_and_preserves_id() {
        let mut tracker = McByteTracker::default().without_cmc();
        let first = tracker.update(&[detection(10.0, 0.9)]).unwrap();
        assert_eq!(first[0].get_track_id(), Some(1));
        let second = tracker.update(&[detection(11.0, 0.9)]).unwrap();
        assert_eq!(second[0].get_track_id(), Some(1));
    }

    #[test]
    fn later_track_requires_confirmation() {
        let mut tracker = McByteTracker::default().without_cmc();
        tracker.update(&[]).unwrap();
        assert!(tracker.update(&[detection(10.0, 0.9)]).unwrap().is_empty());
        let confirmed = tracker.update(&[detection(10.0, 0.9)]).unwrap();
        assert_eq!(confirmed[0].get_track_id(), Some(1));
    }

    #[test]
    fn low_confidence_detection_updates_existing_track() {
        let mut tracker = McByteTracker::default().without_cmc();
        tracker.update(&[detection(10.0, 0.9)]).unwrap();
        let output = tracker.update(&[detection(10.5, 0.4)]).unwrap();
        assert_eq!(output[0].get_track_id(), Some(1));
    }

    #[test]
    fn lost_track_can_recover_at_the_end_of_its_scaled_buffer() {
        let mut tracker = McByteTracker::new(15, 3, 0.7, 0.6).without_cmc();
        assert_eq!(tracker.maximum_frames_without_update(), 2);
        tracker.update(&[detection(10.0, 0.9)]).unwrap();
        tracker.update(&[]).unwrap();
        tracker.update(&[]).unwrap();

        let recovered = tracker.update(&[detection(10.0, 0.9)]).unwrap();
        assert_eq!(recovered[0].get_track_id(), Some(1));
    }

    #[test]
    fn zero_lost_buffer_removes_an_unmatched_track_without_grace() {
        let mut tracker = McByteTracker::new(30, 0, 0.7, 0.6).without_cmc();
        assert_eq!(tracker.maximum_frames_without_update(), 0);
        tracker.update(&[detection(10.0, 0.9)]).unwrap();
        tracker.update(&[]).unwrap();
        assert_eq!(tracker.tracker_count(), 0);

        // A later detection starts an unconfirmed track instead of reviving ID 1.
        assert!(tracker.update(&[detection(10.0, 0.9)]).unwrap().is_empty());
    }

    #[test]
    fn invalid_mask_input_is_rejected() {
        let mut tracker = McByteTracker::default().without_cmc();
        let mask = McByteMask::new(1, GrayImage::new(10, 10), f32::NAN);
        assert!(tracker.update_with_masks(&[], &[mask]).is_err());
    }

    #[test]
    fn reset_clears_ids_and_cmc_state() {
        let mut tracker = McByteTracker::default();
        let frame = GrayImage::from_pixel(64, 48, Luma([127]));
        tracker
            .update_with_frame(&[detection(10.0, 0.9)], &frame)
            .unwrap();
        tracker.reset();
        assert_eq!(tracker.frame_count(), 0);
        assert_eq!(tracker.tracker_count(), 0);
        let output = tracker
            .update_with_frame(&[detection(10.0, 0.9)], &frame)
            .unwrap();
        assert_eq!(output[0].get_track_id(), Some(1));
    }

    #[derive(Deserialize)]
    struct Reference {
        source_commit: String,
        tracker_sequence: Vec<ReferenceFrame>,
    }

    #[derive(Deserialize)]
    struct ReferenceFrame {
        detections: Vec<[f32; 5]>,
        tracker_ids: Vec<usize>,
    }

    #[test]
    fn matches_roboflow_reference_sequence() {
        let reference: Reference = serde_json::from_str(include_str!(
            "../../data/jsons/mcbyte_reference.json"
        ))
        .unwrap();
        assert_eq!(
            reference.source_commit,
            "ced34f04886da91dc6bec3dfe02f0a0427231ce8"
        );
        let mut tracker = McByteTracker::default().without_cmc();
        for frame in reference.tracker_sequence {
            let detections = frame
                .detections
                .iter()
                .map(|values| {
                    Object::new(
                        Rect::new(
                            values[0],
                            values[1],
                            values[2] - values[0],
                            values[3] - values[1],
                        ),
                        values[4],
                        None,
                    )
                })
                .collect::<Vec<_>>();
            let output = tracker.update(&detections).unwrap();
            let rust_ids = output
                .iter()
                .map(|object| object.get_track_id().unwrap())
                .collect::<Vec<_>>();
            // Roboflow IDs are zero-based; jamtrack-rs consistently exposes
            // one-based IDs across all tracker implementations.
            let expected = frame
                .tracker_ids
                .iter()
                .map(|id| id + 1)
                .collect::<Vec<_>>();
            assert_eq!(rust_ids, expected);
        }
    }
}
