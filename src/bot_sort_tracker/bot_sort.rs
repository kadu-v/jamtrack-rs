use super::strack::{STrack, STrackState};
use crate::boost_tracker::ecc::{EccAligner, EccConfig};
use crate::byte_tracker::ByteTracker;
use crate::error::TrackError;
use crate::object::Object;
use image::GrayImage;
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
pub struct BotSort {
    track_high_thresh: f32,
    track_low_thresh: f32,
    new_track_thresh: f32,
    match_thresh: f32,
    proximity_thresh: f32,
    appearance_thresh: f32,
    max_time_lost: usize,
    frame_id: usize,
    track_id_count: usize,
    tracked_stracks: Vec<STrack>,
    lost_stracks: Vec<STrack>,
    removed_stracks: Vec<STrack>,
    with_reid: bool,
    mot20: bool,
    use_ecc: bool,
    ecc: Option<EccAligner>,
}

impl BotSort {
    pub fn new(
        frame_rate: usize,
        track_buffer: usize,
        track_high_thresh: f32,
        track_low_thresh: f32,
        new_track_thresh: f32,
        match_thresh: f32,
    ) -> Self {
        Self {
            track_high_thresh,
            track_low_thresh,
            new_track_thresh,
            match_thresh,
            proximity_thresh: 0.5,
            appearance_thresh: 0.25,
            max_time_lost: (track_buffer as f32 * frame_rate as f32 / 30.0)
                as usize,
            frame_id: 0,
            track_id_count: 0,
            tracked_stracks: Vec::new(),
            lost_stracks: Vec::new(),
            removed_stracks: Vec::new(),
            with_reid: false,
            mot20: false,
            use_ecc: false,
            ecc: None,
        }
    }

    pub fn with_reid(self, with_reid: bool) -> Self {
        Self { with_reid, ..self }
    }

    pub fn with_proximity_thresh(self, proximity_thresh: f32) -> Self {
        Self {
            proximity_thresh,
            ..self
        }
    }

    pub fn with_appearance_thresh(self, appearance_thresh: f32) -> Self {
        Self {
            appearance_thresh,
            ..self
        }
    }

    pub fn with_mot20(self, mot20: bool) -> Self {
        Self { mot20, ..self }
    }

    pub fn with_ecc(self) -> Self {
        Self {
            use_ecc: true,
            ecc: Some(EccAligner::new(EccConfig::default())),
            ..self
        }
    }

    pub fn update(
        &mut self,
        objects: &[Object],
    ) -> Result<Vec<Object>, TrackError> {
        self.update_impl(objects, None, None)
    }

    pub fn update_with_features(
        &mut self,
        objects: &[Object],
        features: &[Vec<f32>],
    ) -> Result<Vec<Object>, TrackError> {
        self.update_impl(objects, Some(features), None)
    }

    pub fn update_with_frame(
        &mut self,
        objects: &[Object],
        frame: &GrayImage,
    ) -> Result<Vec<Object>, TrackError> {
        self.update_impl(objects, None, Some(frame))
    }

    pub fn update_with_frame_and_features(
        &mut self,
        objects: &[Object],
        features: &[Vec<f32>],
        frame: &GrayImage,
    ) -> Result<Vec<Object>, TrackError> {
        self.update_impl(objects, Some(features), Some(frame))
    }

    pub fn frame_count(&self) -> usize {
        self.frame_id
    }

    pub fn tracker_count(&self) -> usize {
        self.tracked_stracks.len() + self.lost_stracks.len()
    }

    pub fn uses_reid(&self) -> bool {
        self.with_reid
    }

    fn update_impl(
        &mut self,
        objects: &[Object],
        features: Option<&[Vec<f32>]>,
        frame: Option<&GrayImage>,
    ) -> Result<Vec<Object>, TrackError> {
        self.validate_features(objects, features)?;
        self.frame_id += 1;

        let mut detections = Vec::new();
        let mut detections_second = Vec::new();

        for (idx, obj) in objects.iter().enumerate() {
            let score = obj.get_prob();
            if score <= self.track_low_thresh {
                continue;
            }

            let feature = if self.with_reid && score > self.track_high_thresh {
                features.map(|f| f[idx].clone())
            } else {
                None
            };
            let track = STrack::new(obj.get_rect(), score, feature);
            if score > self.track_high_thresh {
                detections.push(track);
            } else {
                detections_second.push(track);
            }
        }

        let mut activated_stracks = Vec::new();
        let mut refind_stracks = Vec::new();
        let mut lost_stracks = Vec::new();
        let mut removed_stracks = Vec::new();

        let mut unconfirmed = Vec::new();
        let mut tracked_stracks = Vec::new();
        for track in &self.tracked_stracks {
            if !track.is_activated() {
                unconfirmed.push(track.clone());
            } else {
                tracked_stracks.push(track.clone());
            }
        }

        let mut strack_pool =
            Self::joint_stracks(&tracked_stracks, &self.lost_stracks);
        for track in &mut strack_pool {
            track.predict();
        }

        if self.use_ecc {
            if let (Some(frame), Some(ecc)) = (frame, self.ecc.as_mut()) {
                let transform = ecc.estimate(
                    frame.as_raw(),
                    frame.width() as usize,
                    frame.height() as usize,
                );
                for track in &mut strack_pool {
                    track.camera_update(&transform);
                }
                for track in &mut unconfirmed {
                    track.camera_update(&transform);
                }
            }
        }

        let dists = self.first_association_distance(&strack_pool, &detections);
        let (matches, u_track, u_detection) = Self::linear_assignment(
            &dists,
            strack_pool.len(),
            detections.len(),
            self.match_thresh,
        )?;

        for (track_idx, det_idx) in matches {
            let mut track = strack_pool[track_idx].clone();
            let det = &detections[det_idx as usize];
            if track.get_state() == STrackState::Tracked {
                track.update(det, self.frame_id);
                activated_stracks.push(track);
            } else {
                track.re_activate(det, self.frame_id);
                refind_stracks.push(track);
            }
        }

        let mut remaining_tracked = Vec::new();
        let mut remaining_lost = Vec::new();
        for idx in u_track {
            match strack_pool[idx].get_state() {
                STrackState::Tracked => {
                    remaining_tracked.push(strack_pool[idx].clone());
                }
                STrackState::Lost => {
                    remaining_lost.push(strack_pool[idx].clone());
                }
                _ => {}
            }
        }

        let dists = Self::iou_distance(&remaining_tracked, &detections_second);
        let (matches, u_track, _) = Self::linear_assignment(
            &dists,
            remaining_tracked.len(),
            detections_second.len(),
            0.5,
        )?;

        for (track_idx, det_idx) in matches {
            let mut track = remaining_tracked[track_idx].clone();
            let det = &detections_second[det_idx as usize];
            if track.get_state() == STrackState::Tracked {
                track.update(det, self.frame_id);
                activated_stracks.push(track);
            } else {
                track.re_activate(det, self.frame_id);
                refind_stracks.push(track);
            }
        }

        for idx in u_track {
            let mut track = remaining_tracked[idx].clone();
            if track.get_state() != STrackState::Lost {
                track.mark_lost();
                lost_stracks.push(track);
            }
        }

        let remaining_detections: Vec<STrack> = u_detection
            .iter()
            .map(|&idx| detections[idx].clone())
            .collect();
        let dists =
            self.unconfirmed_distance(&unconfirmed, &remaining_detections);
        let (matches, u_unconfirmed, u_detection) = Self::linear_assignment(
            &dists,
            unconfirmed.len(),
            remaining_detections.len(),
            0.7,
        )?;

        for (track_idx, det_idx) in matches {
            let mut track = unconfirmed[track_idx].clone();
            track
                .update(&remaining_detections[det_idx as usize], self.frame_id);
            activated_stracks.push(track);
        }

        for idx in u_unconfirmed {
            let mut track = unconfirmed[idx].clone();
            track.mark_removed();
            removed_stracks.push(track);
        }

        for idx in u_detection {
            let mut track = remaining_detections[idx].clone();
            if track.get_score() < self.new_track_thresh {
                continue;
            }
            self.track_id_count += 1;
            track.activate(self.frame_id, self.track_id_count);
            activated_stracks.push(track);
        }

        let refound_ids: HashSet<usize> = refind_stracks
            .iter()
            .map(STrack::get_track_id)
            .chain(activated_stracks.iter().map(STrack::get_track_id))
            .collect();
        for track in &self.lost_stracks {
            if refound_ids.contains(&track.get_track_id()) {
                continue;
            }
            if self.frame_id - track.get_frame_id() > self.max_time_lost {
                let mut removed = track.clone();
                removed.mark_removed();
                removed_stracks.push(removed);
            }
        }

        // Python mutates track objects through shared references in strack_pool.
        // Rust uses cloned tracks, so rebuilding from updated/refound tracks is
        // required; otherwise stale unmatched clones keep being emitted.
        self.tracked_stracks =
            Self::joint_stracks(&activated_stracks, &refind_stracks);
        self.removed_stracks.extend(removed_stracks);
        self.lost_stracks = Self::sub_stracks(&remaining_lost, &self.tracked_stracks);
        self.lost_stracks.extend(lost_stracks);
        self.lost_stracks =
            Self::sub_stracks(&self.lost_stracks, &self.removed_stracks);

        let (tracked, lost) = Self::remove_duplicate_stracks(
            &self.tracked_stracks,
            &self.lost_stracks,
        );
        self.tracked_stracks = tracked;
        self.lost_stracks = lost;

        Ok(self.tracked_stracks.iter().map(Object::from).collect())
    }

    fn validate_features(
        &self,
        objects: &[Object],
        features: Option<&[Vec<f32>]>,
    ) -> Result<(), TrackError> {
        if !self.with_reid {
            return Ok(());
        }

        let Some(features) = features else {
            return Err(TrackError::InvalidArgument(
                "ReID is enabled but features were not provided".to_string(),
            ));
        };
        if features.len() != objects.len() {
            return Err(TrackError::InvalidArgument(format!(
                "features length {} does not match objects length {}",
                features.len(),
                objects.len()
            )));
        }
        let Some(first) = features.first() else {
            return Ok(());
        };
        if first.is_empty() {
            return Err(TrackError::InvalidArgument(
                "feature dimension must be greater than zero".to_string(),
            ));
        }
        let dim = first.len();
        for feature in features {
            if feature.len() != dim {
                return Err(TrackError::InvalidArgument(
                    "all features must have the same dimension".to_string(),
                ));
            }
            if feature.iter().any(|v| !v.is_finite()) {
                return Err(TrackError::InvalidArgument(
                    "features must contain only finite values".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn first_association_distance(
        &self,
        tracks: &[STrack],
        detections: &[STrack],
    ) -> Vec<Vec<f32>> {
        let mut iou_dists = Self::iou_distance(tracks, detections);
        let iou_mask =
            Self::greater_than_mask(&iou_dists, self.proximity_thresh);

        if !self.mot20 {
            iou_dists = Self::fuse_score(&iou_dists, detections);
        }

        if !self.with_reid {
            return iou_dists;
        }

        let mut emb_dists = Self::embedding_distance(tracks, detections);
        for i in 0..emb_dists.len() {
            for j in 0..emb_dists[i].len() {
                emb_dists[i][j] /= 2.0;
                if emb_dists[i][j] > self.appearance_thresh || iou_mask[i][j] {
                    emb_dists[i][j] = 1.0;
                }
                emb_dists[i][j] = emb_dists[i][j].min(iou_dists[i][j]);
            }
        }
        emb_dists
    }

    fn unconfirmed_distance(
        &self,
        tracks: &[STrack],
        detections: &[STrack],
    ) -> Vec<Vec<f32>> {
        let mut iou_dists = Self::iou_distance(tracks, detections);
        let iou_mask =
            Self::greater_than_mask(&iou_dists, self.proximity_thresh);

        if !self.mot20 {
            iou_dists = Self::fuse_score(&iou_dists, detections);
        }

        if !self.with_reid {
            return iou_dists;
        }

        let mut emb_dists = Self::embedding_distance(tracks, detections);
        for i in 0..emb_dists.len() {
            for j in 0..emb_dists[i].len() {
                emb_dists[i][j] /= 2.0;
                if emb_dists[i][j] > self.appearance_thresh || iou_mask[i][j] {
                    emb_dists[i][j] = 1.0;
                }
                emb_dists[i][j] = emb_dists[i][j].min(iou_dists[i][j]);
            }
        }
        emb_dists
    }

    fn iou_distance(a_tracks: &[STrack], b_tracks: &[STrack]) -> Vec<Vec<f32>> {
        let mut cost_matrix = vec![vec![0.0; b_tracks.len()]; a_tracks.len()];
        for (i, a) in a_tracks.iter().enumerate() {
            for (j, b) in b_tracks.iter().enumerate() {
                cost_matrix[i][j] = 1.0 - a.get_rect().calc_iou(&b.get_rect());
            }
        }
        cost_matrix
    }

    fn fuse_score(
        cost_matrix: &[Vec<f32>],
        detections: &[STrack],
    ) -> Vec<Vec<f32>> {
        cost_matrix
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(j, cost)| {
                        let iou_sim = 1.0 - cost;
                        1.0 - iou_sim * detections[j].get_score()
                    })
                    .collect()
            })
            .collect()
    }

    fn embedding_distance(
        tracks: &[STrack],
        detections: &[STrack],
    ) -> Vec<Vec<f32>> {
        let mut cost_matrix = vec![vec![0.0; detections.len()]; tracks.len()];
        for (i, track) in tracks.iter().enumerate() {
            for (j, det) in detections.iter().enumerate() {
                cost_matrix[i][j] = match (track.smooth_feat(), det.curr_feat())
                {
                    (Some(track_feat), Some(det_feat))
                        if track_feat.len() == det_feat.len() =>
                    {
                        let cosine = track_feat
                            .iter()
                            .zip(det_feat.iter())
                            .map(|(a, b)| a * b)
                            .sum::<f32>();
                        (1.0 - cosine).max(0.0)
                    }
                    _ => 1.0,
                };
            }
        }
        cost_matrix
    }

    fn greater_than_mask(
        cost_matrix: &[Vec<f32>],
        threshold: f32,
    ) -> Vec<Vec<bool>> {
        cost_matrix
            .iter()
            .map(|row| row.iter().map(|v| *v > threshold).collect())
            .collect()
    }

    fn linear_assignment(
        cost_matrix: &[Vec<f32>],
        rows: usize,
        cols: usize,
        thresh: f32,
    ) -> Result<(Vec<(usize, isize)>, Vec<usize>, Vec<usize>), TrackError> {
        if rows == 0 || cols == 0 {
            return Ok((Vec::new(), (0..rows).collect(), (0..cols).collect()));
        }

        let mut rowsol = vec![-1; rows];
        let mut colsol = vec![-1; cols];
        ByteTracker::exec_lapjv(
            &cost_matrix.to_vec(),
            &mut rowsol,
            &mut colsol,
            true,
            thresh as f64,
            true,
        )?;

        let matches = rowsol
            .iter()
            .enumerate()
            .filter_map(|(idx, sol)| (*sol >= 0).then_some((idx, *sol)))
            .collect();
        let unmatched_a = rowsol
            .iter()
            .enumerate()
            .filter_map(|(idx, sol)| (*sol < 0).then_some(idx))
            .collect();
        let unmatched_b = colsol
            .iter()
            .enumerate()
            .filter_map(|(idx, sol)| (*sol < 0).then_some(idx))
            .collect();

        Ok((matches, unmatched_a, unmatched_b))
    }

    fn joint_stracks(a_tracks: &[STrack], b_tracks: &[STrack]) -> Vec<STrack> {
        let mut exists = HashMap::new();
        let mut res = Vec::new();
        for track in a_tracks {
            exists.insert(track.get_track_id(), true);
            res.push(track.clone());
        }
        for track in b_tracks {
            if !exists.contains_key(&track.get_track_id()) {
                exists.insert(track.get_track_id(), true);
                res.push(track.clone());
            }
        }
        res
    }

    fn sub_stracks(a_tracks: &[STrack], b_tracks: &[STrack]) -> Vec<STrack> {
        let removed_ids: HashSet<_> =
            b_tracks.iter().map(STrack::get_track_id).collect();
        a_tracks
            .iter()
            .filter(|track| !removed_ids.contains(&track.get_track_id()))
            .cloned()
            .collect()
    }

    fn remove_duplicate_stracks(
        a_tracks: &[STrack],
        b_tracks: &[STrack],
    ) -> (Vec<STrack>, Vec<STrack>) {
        let dists = Self::iou_distance(a_tracks, b_tracks);
        let mut dup_a = vec![false; a_tracks.len()];
        let mut dup_b = vec![false; b_tracks.len()];

        for i in 0..a_tracks.len() {
            for j in 0..b_tracks.len() {
                if dists[i][j] < 0.15 {
                    let time_a = a_tracks[i].get_frame_id()
                        - a_tracks[i].get_start_frame_id();
                    let time_b = b_tracks[j].get_frame_id()
                        - b_tracks[j].get_start_frame_id();
                    if time_a > time_b {
                        dup_b[j] = true;
                    } else {
                        dup_a[i] = true;
                    }
                }
            }
        }

        let a_res = a_tracks
            .iter()
            .enumerate()
            .filter_map(|(idx, track)| (!dup_a[idx]).then_some(track.clone()))
            .collect();
        let b_res = b_tracks
            .iter()
            .enumerate()
            .filter_map(|(idx, track)| (!dup_b[idx]).then_some(track.clone()))
            .collect();

        (a_res, b_res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rect::Rect;

    fn obj(x: f32, y: f32, w: f32, h: f32, prob: f32) -> Object {
        Object::new(Rect::new(x, y, w, h), prob, None)
    }

    #[test]
    fn update_creates_track() {
        let mut tracker = BotSort::new(30, 30, 0.6, 0.1, 0.7, 0.8);
        let out = tracker.update(&[obj(10.0, 20.0, 30.0, 40.0, 0.9)]).unwrap();

        assert_eq!(tracker.frame_count(), 1);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].get_track_id(), Some(1));
    }

    #[test]
    fn matched_track_updates_position_on_next_frame() {
        let mut tracker = BotSort::new(30, 30, 0.6, 0.1, 0.7, 0.8);
        let first = tracker
            .update(&[obj(10.0, 20.0, 30.0, 40.0, 0.9)])
            .unwrap();
        assert_eq!(first[0].get_track_id(), Some(1));

        let second = tracker
            .update(&[obj(14.0, 22.0, 30.0, 40.0, 0.9)])
            .unwrap();

        assert_eq!(second.len(), 1);
        assert_eq!(second[0].get_track_id(), Some(1));
        assert!(
            second[0].get_x() > first[0].get_x(),
            "matched track should move toward the second detection"
        );
    }

    #[test]
    fn lost_track_keeps_predicted_state_for_reactivation() {
        let mut tracker = BotSort::new(30, 30, 0.6, 0.1, 0.7, 0.8);
        let first = tracker
            .update(&[obj(100.0, 100.0, 100.0, 100.0, 0.9)])
            .unwrap();
        assert_eq!(first[0].get_track_id(), Some(1));

        let second = tracker
            .update(&[obj(130.0, 100.0, 100.0, 100.0, 0.9)])
            .unwrap();
        assert_eq!(second[0].get_track_id(), Some(1));

        assert!(tracker.update(&[]).unwrap().is_empty());
        assert!(tracker.update(&[]).unwrap().is_empty());

        let reactivated = tracker
            .update(&[obj(195.0, 100.0, 100.0, 100.0, 0.9)])
            .unwrap();

        assert_eq!(reactivated.len(), 1);
        assert_eq!(
            reactivated[0].get_track_id(),
            Some(1),
            "lost track should carry prediction forward and reactivate"
        );
    }

    #[test]
    fn refound_lost_track_is_not_marked_removed_by_stale_clone() {
        let mut tracker = BotSort::new(30, 2, 0.6, 0.1, 0.7, 0.8);
        tracker
            .update(&[obj(100.0, 100.0, 100.0, 100.0, 0.9)])
            .unwrap();
        tracker
            .update(&[obj(130.0, 100.0, 100.0, 100.0, 0.9)])
            .unwrap();
        assert!(tracker.update(&[]).unwrap().is_empty());
        assert!(tracker.update(&[]).unwrap().is_empty());

        let refound = tracker
            .update(&[obj(195.0, 100.0, 100.0, 100.0, 0.9)])
            .unwrap();

        assert_eq!(refound.len(), 1);
        assert_eq!(refound[0].get_track_id(), Some(1));
        assert!(
            !tracker
                .removed_stracks
                .iter()
                .any(|track| track.get_track_id() == 1),
            "a refound lost track must not leave a stale removed clone"
        );
    }

    #[test]
    fn reid_requires_features_when_enabled() {
        let mut tracker =
            BotSort::new(30, 30, 0.6, 0.1, 0.7, 0.8).with_reid(true);
        let err = tracker
            .update(&[obj(10.0, 20.0, 30.0, 40.0, 0.9)])
            .unwrap_err();

        assert!(matches!(err, TrackError::InvalidArgument(_)));
    }

    #[test]
    fn reid_accepts_matching_features() {
        let mut tracker =
            BotSort::new(30, 30, 0.6, 0.1, 0.7, 0.8).with_reid(true);
        let out = tracker
            .update_with_features(
                &[obj(10.0, 20.0, 30.0, 40.0, 0.9)],
                &[vec![1.0, 0.0]],
            )
            .unwrap();

        assert_eq!(out.len(), 1);
    }

    #[test]
    fn duplicate_removal_keeps_longer_track() {
        let a = vec![STrack::dummy(1, 10)];
        let b = vec![STrack::dummy(2, 2)];

        let (a_out, b_out) = BotSort::remove_duplicate_stracks(&a, &b);

        assert_eq!(a_out.len(), 1);
        assert!(b_out.is_empty());
    }

    #[test]
    fn sub_stracks_preserves_input_order() {
        let a = vec![
            STrack::dummy(10, 1),
            STrack::dummy(20, 1),
            STrack::dummy(30, 1),
        ];
        let b = vec![STrack::dummy(20, 1)];

        let out = BotSort::sub_stracks(&a, &b);

        assert_eq!(
            out.iter().map(STrack::get_track_id).collect::<Vec<_>>(),
            vec![10, 30]
        );
    }
}
