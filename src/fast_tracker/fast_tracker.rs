use std::collections::HashSet;

use crate::lapjv::lapjv;
use crate::{Object, TrackError};

use super::strack::{STrack, TrackState};

const LOW_SCORE_THRESHOLD: f64 = 0.25;
const LOW_MATCH_THRESHOLD: f64 = 0.5;
const UNCONFIRMED_MATCH_THRESHOLD: f64 = 0.7;
const OCCLUSION_OVERLAP_THRESHOLD: f64 = 0.7;
const RECENT_OCCLUSION_FRAMES: usize = 40;

/// A four-point road region used by FastTracker's trajectory constraints.
///
/// Points must be ordered as `(E1, E2, O2, O1)`, matching the upstream
/// FastTracker implementation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FastTrackerRoi {
    points: [[f64; 2]; 4],
}

impl FastTrackerRoi {
    pub fn new(points: [[f32; 2]; 4]) -> Self {
        Self {
            points: points.map(|point| [point[0] as f64, point[1] as f64]),
        }
    }

    pub fn points(&self) -> [[f32; 2]; 4] {
        self.points.map(|point| [point[0] as f32, point[1] as f32])
    }
}

/// Occlusion-aware multi-object tracker ported from upstream FastTracker.
#[derive(Debug)]
pub struct FastTracker {
    track_thresh: f64,
    match_thresh: f64,
    max_time_lost: usize,
    reset_velocity_offset: usize,
    reset_position_offset: usize,
    enlarge_bbox: f64,
    dampen_motion: f64,
    active_occlusion_to_lost: usize,
    init_iou_suppression: f64,
    rois: Vec<FastTrackerRoi>,
    roi_repair_max_gap: usize,
    direction_window: usize,
    // Kept because it is part of upstream's configuration. Upstream currently
    // does not consume it in the cone calculation either.
    direction_margin_degrees: f64,
    mot20: bool,
    frame_id: usize,
    next_track_id: usize,
    tracked_stracks: Vec<STrack>,
    lost_stracks: Vec<STrack>,
    removed_stracks: Vec<STrack>,
}

impl FastTracker {
    pub fn new(
        frame_rate: usize,
        track_buffer: usize,
        track_thresh: f32,
        match_thresh: f32,
    ) -> Self {
        Self {
            track_thresh: track_thresh as f64,
            match_thresh: match_thresh as f64,
            max_time_lost: ((frame_rate as f64 / 30.0) * track_buffer as f64)
                as usize,
            reset_velocity_offset: 5,
            reset_position_offset: 3,
            enlarge_bbox: 1.2,
            dampen_motion: 0.85,
            active_occlusion_to_lost: 15,
            init_iou_suppression: 0.8,
            rois: Vec::new(),
            roi_repair_max_gap: 15,
            direction_window: 10,
            direction_margin_degrees: 2.0,
            mot20: false,
            frame_id: 0,
            next_track_id: 0,
            tracked_stracks: Vec::new(),
            lost_stracks: Vec::new(),
            removed_stracks: Vec::new(),
        }
    }

    pub fn with_occlusion(
        mut self,
        reset_velocity_offset: usize,
        reset_position_offset: usize,
        enlarge_bbox: f32,
        dampen_motion: f32,
        active_occlusion_to_lost: usize,
    ) -> Self {
        self.reset_velocity_offset = reset_velocity_offset;
        self.reset_position_offset = reset_position_offset;
        self.enlarge_bbox = enlarge_bbox as f64;
        self.dampen_motion = dampen_motion as f64;
        self.active_occlusion_to_lost = active_occlusion_to_lost;
        self
    }

    pub fn with_init_iou_suppression(mut self, threshold: f32) -> Self {
        self.init_iou_suppression = threshold as f64;
        self
    }

    pub fn with_rois(
        mut self,
        rois: Vec<FastTrackerRoi>,
        repair_max_gap: usize,
        direction_window: usize,
        direction_margin_degrees: f32,
    ) -> Self {
        self.rois = rois;
        self.roi_repair_max_gap = repair_max_gap;
        self.direction_window = direction_window;
        self.direction_margin_degrees = direction_margin_degrees as f64;
        self
    }

    pub fn with_mot20(mut self, enabled: bool) -> Self {
        self.mot20 = enabled;
        self
    }

    pub fn frame_count(&self) -> usize {
        self.frame_id
    }

    pub fn tracker_count(&self) -> usize {
        self.tracked_stracks.len()
    }

    pub fn update(
        &mut self,
        objects: &[Object],
    ) -> Result<Vec<Object>, TrackError> {
        self.frame_id += 1;

        let mut detections = Vec::new();
        let mut low_detections = Vec::new();
        for object in objects {
            let score = object.get_prob() as f64;
            if score > self.track_thresh {
                detections.push(STrack::new(object));
            } else if score > LOW_SCORE_THRESHOLD && score < self.track_thresh {
                low_detections.push(STrack::new(object));
            }
        }

        let mut unconfirmed = Vec::new();
        let mut active_tracks = Vec::new();
        for track in &self.tracked_stracks {
            if track.is_activated {
                active_tracks.push(track.clone());
            } else {
                unconfirmed.push(track.clone());
            }
        }

        let mut track_pool =
            Self::joint_stracks(&active_tracks, &self.lost_stracks);
        for track in &mut track_pool {
            track.predict();
            Self::replace_track(&mut self.tracked_stracks, track);
            Self::replace_track(&mut self.lost_stracks, track);
        }

        let first_cost = Self::iou_distance(&track_pool, &detections);
        let first_cost = if self.mot20 {
            first_cost
        } else {
            Self::fuse_score(&first_cost, &detections)
        };
        let (first_matches, unmatched_pool, unmatched_high) =
            Self::linear_assignment(
                &first_cost,
                track_pool.len(),
                detections.len(),
                self.match_thresh,
            )?;

        let mut activated = Vec::new();
        let mut refound = Vec::new();
        for (track_index, detection_index) in first_matches {
            let mut track = track_pool[track_index].clone();
            let detection = &detections[detection_index];
            if track.state == TrackState::Tracked {
                track.update(detection, self.frame_id);
                activated.push(track.clone());
            } else {
                track.re_activate(detection, self.frame_id);
                refound.push(track.clone());
            }
            track.is_occluded = false;
            track.not_matched = 0;
            track.occluded_len = 0;
            track_pool[track_index] = track.clone();
            Self::replace_track(&mut self.tracked_stracks, &track);
            Self::replace_track(&mut self.lost_stracks, &track);
            Self::replace_track(&mut activated, &track);
            Self::replace_track(&mut refound, &track);
        }

        let mut remaining_tracks = unmatched_pool
            .iter()
            .filter_map(|&index| {
                (track_pool[index].state == TrackState::Tracked)
                    .then(|| track_pool[index].clone())
            })
            .collect::<Vec<_>>();

        let second_cost =
            Self::iou_distance(&remaining_tracks, &low_detections);
        let (second_matches, unmatched_remaining, _) = Self::linear_assignment(
            &second_cost,
            remaining_tracks.len(),
            low_detections.len(),
            LOW_MATCH_THRESHOLD,
        )?;
        for (track_index, detection_index) in second_matches {
            let mut track = remaining_tracks[track_index].clone();
            track.update(&low_detections[detection_index], self.frame_id);
            track.is_occluded = false;
            track.not_matched = 0;
            track.occluded_len = 0;
            remaining_tracks[track_index] = track.clone();
            Self::replace_track(&mut self.tracked_stracks, &track);
            activated.push(track);
        }

        let mut newly_lost = Vec::new();
        for index in unmatched_remaining {
            let mut track = remaining_tracks[index].clone();
            track.not_matched += 1;
            if !track.is_occluded && track.state == TrackState::Tracked {
                for other in &activated {
                    if track.track_id == other.track_id
                        || !other.is_activated
                        || other.is_occluded
                    {
                        continue;
                    }
                    if Self::is_occluded_by(
                        track.tlbr(),
                        other.tlbr(),
                        OCCLUSION_OVERLAP_THRESHOLD,
                    ) {
                        track.is_occluded = true;
                        track.occluded_len += 1;
                        track.last_occluded_frame = Some(self.frame_id);
                        track.was_recently_occluded = true;

                        if let Some(old_mean) = Self::history_at_offset(
                            &track,
                            self.reset_velocity_offset,
                        ) {
                            if let Some(mean) = track.mean.as_mut() {
                                mean.fixed_rows_mut::<4>(4)
                                    .copy_from(&old_mean.fixed_rows::<4>(4));
                            }
                        }
                        if let Some(old_mean) = Self::history_at_offset(
                            &track,
                            self.reset_position_offset,
                        ) {
                            if let Some(mean) = track.mean.as_mut() {
                                mean.fixed_rows_mut::<4>(0)
                                    .copy_from(&old_mean.fixed_rows::<4>(0));
                            }
                        }
                        if track.occluded_len == 1 {
                            if let Some(mean) = track.mean.as_mut() {
                                mean[3] *= self.enlarge_bbox;
                            }
                        }
                        if let Some(mean) = track.mean.as_mut() {
                            for value in mean.fixed_rows_mut::<4>(4).iter_mut()
                            {
                                *value *= self.dampen_motion;
                            }
                        }
                        track.refresh_tlwh();
                        break;
                    }
                }
            }

            if !track.is_occluded {
                track.occluded_len = 0;
            } else {
                track.occluded_len += 1;
            }
            if track.was_recently_occluded
                && track.last_occluded_frame.is_some_and(|last| {
                    self.frame_id - last > RECENT_OCCLUSION_FRAMES
                })
            {
                track.was_recently_occluded = false;
            }
            if track.state != TrackState::Lost
                && track.not_matched > 2
                && (!track.is_occluded
                    || track.occluded_len > self.active_occlusion_to_lost)
            {
                track.mark_lost();
                newly_lost.push(track.clone());
            }
            Self::replace_track(&mut self.tracked_stracks, &track);
        }

        let remaining_detections = unmatched_high
            .iter()
            .map(|&index| detections[index].clone())
            .collect::<Vec<_>>();
        let unconfirmed_cost =
            Self::iou_distance(&unconfirmed, &remaining_detections);
        let unconfirmed_cost = if self.mot20 {
            unconfirmed_cost
        } else {
            Self::fuse_score(&unconfirmed_cost, &remaining_detections)
        };
        let (unconfirmed_matches, unmatched_unconfirmed, unmatched_new) =
            Self::linear_assignment(
                &unconfirmed_cost,
                unconfirmed.len(),
                remaining_detections.len(),
                UNCONFIRMED_MATCH_THRESHOLD,
            )?;
        for (track_index, detection_index) in unconfirmed_matches {
            let mut track = unconfirmed[track_index].clone();
            track.update(&remaining_detections[detection_index], self.frame_id);
            Self::replace_track(&mut self.tracked_stracks, &track);
            activated.push(track);
        }
        for track_index in unmatched_unconfirmed {
            let mut track = unconfirmed[track_index].clone();
            track.mark_lost();
            Self::replace_track(&mut self.tracked_stracks, &track);
            newly_lost.push(track);
        }

        for track in &mut activated {
            self.enforce_environment_constraints(track);
            Self::replace_track(&mut self.tracked_stracks, track);
        }
        for track in &mut refound {
            self.enforce_environment_constraints(track);
            Self::replace_track(&mut self.lost_stracks, track);
        }
        let constrained_ids = activated
            .iter()
            .chain(&refound)
            .map(|track| track.track_id)
            .collect::<HashSet<_>>();
        for index in 0..self.tracked_stracks.len() {
            if self.tracked_stracks[index].state == TrackState::Tracked
                && !constrained_ids
                    .contains(&self.tracked_stracks[index].track_id)
            {
                let mut track = self.tracked_stracks[index].clone();
                self.enforce_environment_constraints(&mut track);
                self.tracked_stracks[index] = track;
            }
        }

        let mut active_now = self
            .tracked_stracks
            .iter()
            .filter(|track| track.state == TrackState::Tracked)
            .cloned()
            .collect::<Vec<_>>();
        for track in &activated {
            Self::replace_or_push(&mut active_now, track);
        }
        for new_index in unmatched_new {
            let mut track = remaining_detections[new_index].clone();
            if track.score < self.track_thresh {
                continue;
            }
            let max_iou = active_now
                .iter()
                .map(|active| Self::plain_iou(track.tlbr(), active.tlbr()))
                .fold(0.0, f64::max);
            if max_iou < self.init_iou_suppression {
                self.next_track_id += 1;
                track.activate(self.frame_id, self.next_track_id);
                activated.push(track);
            }
        }

        let mut newly_removed = Vec::new();
        for track in &mut self.lost_stracks {
            let recently_occluded = track.was_recently_occluded
                && track.last_occluded_frame.is_some_and(|last| {
                    self.frame_id - last <= RECENT_OCCLUSION_FRAMES
                });
            if !recently_occluded
                && self.frame_id.saturating_sub(track.frame_id)
                    > self.max_time_lost
            {
                track.mark_removed();
                newly_removed.push(track.clone());
            }
        }

        self.tracked_stracks
            .retain(|track| track.state == TrackState::Tracked);
        self.tracked_stracks =
            Self::joint_stracks(&self.tracked_stracks, &activated);
        self.tracked_stracks =
            Self::joint_stracks(&self.tracked_stracks, &refound);
        self.lost_stracks =
            Self::sub_stracks(&self.lost_stracks, &self.tracked_stracks);
        self.lost_stracks.extend(newly_lost);
        self.lost_stracks =
            Self::sub_stracks(&self.lost_stracks, &self.removed_stracks);
        self.removed_stracks.extend(newly_removed);
        let (tracked, lost) = Self::remove_duplicate_stracks(
            &self.tracked_stracks,
            &self.lost_stracks,
        );
        self.tracked_stracks = tracked;
        self.lost_stracks = lost;

        Ok(self
            .tracked_stracks
            .iter()
            .filter(|track| track.is_activated)
            .cloned()
            .map(STrack::into_object)
            .collect())
    }

    fn history_at_offset(
        track: &STrack,
        offset: usize,
    ) -> Option<super::kalman_filter::StateMean> {
        if track.mean_history.len() < offset {
            return None;
        }
        let index = if offset == 0 {
            0
        } else {
            track.mean_history.len() - offset
        };
        track.mean_history.get(index).copied()
    }

    fn replace_track(tracks: &mut [STrack], replacement: &STrack) {
        if replacement.track_id == 0 {
            return;
        }
        if let Some(track) = tracks
            .iter_mut()
            .find(|track| track.track_id == replacement.track_id)
        {
            *track = replacement.clone();
        }
    }

    fn replace_or_push(tracks: &mut Vec<STrack>, replacement: &STrack) {
        if let Some(track) = tracks
            .iter_mut()
            .find(|track| track.track_id == replacement.track_id)
        {
            *track = replacement.clone();
        } else {
            tracks.push(replacement.clone());
        }
    }

    fn joint_stracks(first: &[STrack], second: &[STrack]) -> Vec<STrack> {
        let mut result = first.to_vec();
        let mut ids = first
            .iter()
            .map(|track| track.track_id)
            .collect::<HashSet<_>>();
        for track in second {
            if ids.insert(track.track_id) {
                result.push(track.clone());
            }
        }
        result
    }

    fn sub_stracks(first: &[STrack], second: &[STrack]) -> Vec<STrack> {
        let removed_ids = second
            .iter()
            .map(|track| track.track_id)
            .collect::<HashSet<_>>();
        first
            .iter()
            .filter(|track| !removed_ids.contains(&track.track_id))
            .cloned()
            .collect()
    }

    fn remove_duplicate_stracks(
        tracked: &[STrack],
        lost: &[STrack],
    ) -> (Vec<STrack>, Vec<STrack>) {
        let distances = Self::iou_distance(tracked, lost);
        let mut duplicate_tracked = HashSet::new();
        let mut duplicate_lost = HashSet::new();
        for (tracked_index, row) in distances.iter().enumerate() {
            for (lost_index, &distance) in row.iter().enumerate() {
                if distance < 0.15 {
                    let tracked_age = tracked[tracked_index]
                        .frame_id
                        .saturating_sub(tracked[tracked_index].start_frame);
                    let lost_age = lost[lost_index]
                        .frame_id
                        .saturating_sub(lost[lost_index].start_frame);
                    if tracked_age > lost_age {
                        duplicate_lost.insert(lost_index);
                    } else {
                        duplicate_tracked.insert(tracked_index);
                    }
                }
            }
        }
        (
            tracked
                .iter()
                .enumerate()
                .filter(|(index, _)| !duplicate_tracked.contains(index))
                .map(|(_, track)| track.clone())
                .collect(),
            lost.iter()
                .enumerate()
                .filter(|(index, _)| !duplicate_lost.contains(index))
                .map(|(_, track)| track.clone())
                .collect(),
        )
    }

    fn iou_distance(first: &[STrack], second: &[STrack]) -> Vec<Vec<f64>> {
        first
            .iter()
            .map(|a| {
                second
                    .iter()
                    .map(|b| 1.0 - Self::matching_iou(a.tlbr(), b.tlbr()))
                    .collect()
            })
            .collect()
    }

    // cython_bbox.bbox_overlaps uses inclusive pixel coordinates.
    fn matching_iou(a: [f64; 4], b: [f64; 4]) -> f64 {
        let width = (a[2].min(b[2]) - a[0].max(b[0]) + 1.0).max(0.0);
        let height = (a[3].min(b[3]) - a[1].max(b[1]) + 1.0).max(0.0);
        if width <= 0.0 || height <= 0.0 {
            return 0.0;
        }
        let intersection = width * height;
        let area_a = (a[2] - a[0] + 1.0) * (a[3] - a[1] + 1.0);
        let area_b = (b[2] - b[0] + 1.0) * (b[3] - b[1] + 1.0);
        intersection / (area_a + area_b - intersection)
    }

    fn plain_iou(a: [f64; 4], b: [f64; 4]) -> f64 {
        let width = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
        let height = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
        let intersection = width * height;
        if intersection == 0.0 {
            return 0.0;
        }
        let area_a = (a[2] - a[0]) * (a[3] - a[1]);
        let area_b = (b[2] - b[0]) * (b[3] - b[1]);
        intersection / (area_a + area_b - intersection + 1e-9)
    }

    fn is_occluded_by(a: [f64; 4], b: [f64; 4], threshold: f64) -> bool {
        let intersection = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0)
            * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
        let area_a = (a[2] - a[0]) * (a[3] - a[1]);
        area_a != 0.0 && intersection / area_a > threshold
    }

    fn fuse_score(cost: &[Vec<f64>], detections: &[STrack]) -> Vec<Vec<f64>> {
        cost.iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(index, value)| {
                        1.0 - (1.0 - value) * detections[index].score
                    })
                    .collect()
            })
            .collect()
    }

    fn linear_assignment(
        cost: &[Vec<f64>],
        rows: usize,
        columns: usize,
        threshold: f64,
    ) -> Result<(Vec<(usize, usize)>, Vec<usize>, Vec<usize>), TrackError> {
        if rows == 0 || columns == 0 {
            return Ok((
                (Vec::new()),
                (0..rows).collect(),
                (0..columns).collect(),
            ));
        }
        let size = rows + columns;
        let mut extended = vec![vec![threshold / 2.0; size]; size];
        for row in rows..size {
            for column in columns..size {
                extended[row][column] = 0.0;
            }
        }
        for row in 0..rows {
            for column in 0..columns {
                extended[row][column] = cost[row][column];
            }
        }
        let mut row_solution = vec![-1; size];
        let mut column_solution = vec![-1; size];
        lapjv(&mut extended, &mut row_solution, &mut column_solution)?;
        let mut matches = Vec::new();
        let mut unmatched_rows = Vec::new();
        for (row, &column) in row_solution.iter().take(rows).enumerate() {
            if column >= 0 && (column as usize) < columns {
                matches.push((row, column as usize));
            } else {
                unmatched_rows.push(row);
            }
        }
        let unmatched_columns = column_solution
            .iter()
            .take(columns)
            .enumerate()
            .filter_map(|(column, &row)| {
                (row < 0 || row as usize >= rows).then_some(column)
            })
            .collect();
        Ok((matches, unmatched_rows, unmatched_columns))
    }

    fn enforce_environment_constraints(&self, track: &mut STrack) {
        if self.rois.is_empty() {
            return;
        }
        let tlwh = track.tlwh();
        let mut current_center =
            [tlwh[0] + tlwh[2] / 2.0, tlwh[1] + tlwh[3] / 2.0];
        track.center_history.push(current_center);

        let Some(roi) = self
            .rois
            .iter()
            .find(|roi| Self::point_in_polygon(current_center, &roi.points))
        else {
            return;
        };

        if track.center_history.len() > 2 {
            let mut last_inside = None;
            let mut last_outside = None;
            for index in (0..track.center_history.len() - 1).rev() {
                let inside = Self::point_in_polygon(
                    track.center_history[index],
                    &roi.points,
                );
                if inside && last_outside.is_some() {
                    last_inside = Some(index);
                    break;
                }
                if !inside && last_outside.is_none() {
                    last_outside = Some(index);
                }
            }
            if let (Some(inside), Some(outside)) = (last_inside, last_outside) {
                let gap = outside - inside;
                if gap > 0 && gap <= self.roi_repair_max_gap {
                    for index in inside + 1..=outside {
                        let clamped = Self::clamp_point_to_polygon(
                            track.center_history[index],
                            &roi.points,
                        );
                        track.center_history[index] = clamped;
                        if let Some(mean) = track.mean_history.get_mut(index) {
                            mean[0] = clamped[0];
                            mean[1] = clamped[1];
                        }
                    }
                    current_center = *track.center_history.last().unwrap();
                    let tlwh = track.tlwh();
                    track.set_mean_position(
                        current_center[0] - 0.5 * tlwh[2],
                        current_center[1] - 0.5 * tlwh[3],
                    );
                }
            }
        }

        let (axis, theta) = Self::cone_axis_and_theta(&roi.points);
        let window = self.direction_window;
        if track.center_history.len() > window {
            let current = *track.center_history.last().unwrap();
            let anchor =
                track.center_history[track.center_history.len() - 1 - window];
            let delta = [current[0] - anchor[0], current[1] - anchor[1]];
            if Self::norm(delta) > 1e-6 {
                let adjusted =
                    Self::clamp_to_cone(anchor, current, axis, theta);
                if (adjusted[0] - current[0]).abs() > 1e-3
                    || (adjusted[1] - current[1]).abs() > 1e-3
                {
                    *track.center_history.last_mut().unwrap() = adjusted;
                    if let Some(mean) = track.mean_history.back_mut() {
                        mean[0] = adjusted[0];
                        mean[1] = adjusted[1];
                    }
                    let tlwh = track.tlwh();
                    track.set_mean_position(
                        adjusted[0] - 0.5 * tlwh[2],
                        adjusted[1] - 0.5 * tlwh[3],
                    );
                }
            }
        }
    }

    #[cfg(test)]
    fn compute_theta(points: &[[f64; 2]; 4]) -> f64 {
        let v1 = [points[2][0] - points[0][0], points[2][1] - points[0][1]];
        let v2 = [points[3][0] - points[1][0], points[3][1] - points[1][1]];
        let cosine = (Self::dot(v1, v2) / (Self::norm(v1) * Self::norm(v2)))
            .clamp(-1.0, 1.0);
        cosine.acos().to_degrees()
    }

    fn point_in_polygon(point: [f64; 2], polygon: &[[f64; 2]; 4]) -> bool {
        let mut inside = false;
        for index in 0..polygon.len() {
            let first = polygon[index];
            let second = polygon[(index + 1) % polygon.len()];
            let crosses = ((first[1] > point[1]) != (second[1] > point[1]))
                && point[0]
                    < (second[0] - first[0]) * (point[1] - first[1])
                        / (second[1] - first[1] + 1e-9)
                        + first[0];
            if crosses {
                inside = !inside;
            }
        }
        inside
    }

    fn closest_point_on_segment(
        point: [f64; 2],
        first: [f64; 2],
        second: [f64; 2],
    ) -> [f64; 2] {
        let ap = [point[0] - first[0], point[1] - first[1]];
        let ab = [second[0] - first[0], second[1] - first[1]];
        let t =
            (Self::dot(ap, ab) / (Self::dot(ab, ab) + 1e-9)).clamp(0.0, 1.0);
        [first[0] + t * ab[0], first[1] + t * ab[1]]
    }

    fn clamp_point_to_polygon(
        point: [f64; 2],
        polygon: &[[f64; 2]; 4],
    ) -> [f64; 2] {
        let mut best = point;
        let mut best_distance = f64::INFINITY;
        for index in 0..polygon.len() {
            let candidate = Self::closest_point_on_segment(
                point,
                polygon[index],
                polygon[(index + 1) % polygon.len()],
            );
            let distance = (candidate[0] - point[0]).powi(2)
                + (candidate[1] - point[1]).powi(2);
            if distance < best_distance {
                best_distance = distance;
                best = candidate;
            }
        }
        best
    }

    fn normalize(vector: [f64; 2]) -> [f64; 2] {
        let norm = Self::norm(vector) + 1e-9;
        [vector[0] / norm, vector[1] / norm]
    }

    fn cone_axis_and_theta(points: &[[f64; 2]; 4]) -> ([f64; 2], f64) {
        let first = Self::normalize([
            points[2][0] - points[0][0],
            points[2][1] - points[0][1],
        ]);
        let second = Self::normalize([
            points[3][0] - points[1][0],
            points[3][1] - points[1][1],
        ]);
        let axis =
            Self::normalize([first[0] + second[0], first[1] + second[1]]);
        let theta = Self::dot(first, second)
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        (axis, theta)
    }

    fn clamp_to_cone(
        anchor: [f64; 2],
        current: [f64; 2],
        axis: [f64; 2],
        theta_degrees: f64,
    ) -> [f64; 2] {
        let delta = [current[0] - anchor[0], current[1] - anchor[1]];
        let magnitude = Self::norm(delta);
        if magnitude < 3.0 {
            return current;
        }
        let delta_unit = [delta[0] / magnitude, delta[1] / magnitude];
        let axis = Self::normalize(axis);
        let angle = Self::dot(delta_unit, axis).clamp(-1.0, 1.0).acos();
        let half = theta_degrees.to_radians() * 0.5;
        if angle <= half {
            return current;
        }
        let cross = axis[0] * delta_unit[1] - axis[1] * delta_unit[0];
        let signed_half = if cross > 0.0 { half } else { -half };
        let (sine, cosine) = signed_half.sin_cos();
        let boundary = [
            axis[0] * cosine - axis[1] * sine,
            axis[0] * sine + axis[1] * cosine,
        ];
        [
            anchor[0] + boundary[0] * magnitude,
            anchor[1] + boundary[1] * magnitude,
        ]
    }

    fn dot(first: [f64; 2], second: [f64; 2]) -> f64 {
        first[0] * second[0] + first[1] * second[1]
    }

    fn norm(vector: [f64; 2]) -> f64 {
        Self::dot(vector, vector).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rect;

    fn object(x: f32, y: f32, width: f32, height: f32, score: f32) -> Object {
        Object::new(Rect::new(x, y, width, height), score, None)
    }

    #[test]
    fn geometry_helpers_compute_expected_values() {
        assert!(
            (FastTracker::matching_iou(
                [0.0, 0.0, 9.0, 9.0],
                [5.0, 5.0, 14.0, 14.0]
            ) - 25.0 / 175.0)
                .abs()
                < 1e-12
        );
        assert!(
            (FastTracker::plain_iou(
                [0.0, 0.0, 10.0, 10.0],
                [5.0, 5.0, 15.0, 15.0]
            ) - 25.0 / 175.000000001)
                .abs()
                < 1e-12
        );
        assert!(FastTracker::is_occluded_by(
            [2.0, 2.0, 8.0, 8.0],
            [0.0, 0.0, 10.0, 10.0],
            0.7
        ));

        let polygon = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        assert!(FastTracker::point_in_polygon([5.0, 5.0], &polygon));
        assert!(!FastTracker::point_in_polygon([15.0, 5.0], &polygon));
        let clamped =
            FastTracker::clamp_point_to_polygon([15.0, 6.0], &polygon);
        assert!((clamped[0] - 10.0).abs() < 1e-9);
        assert!((clamped[1] - 6.0).abs() < 1e-9);
    }

    #[test]
    fn cone_geometry_computes_expected_axis_and_angle() {
        let roi = [[0.0, 0.0], [0.0, 10.0], [20.0, 15.0], [20.0, -5.0]];
        let theta = FastTracker::compute_theta(&roi);
        assert!((theta - 73.73979529168804).abs() < 1e-9);
        let (axis, cone_theta) = FastTracker::cone_axis_and_theta(&roi);
        assert!((axis[0] - 0.999999999375).abs() < 1e-8);
        assert!(axis[1].abs() < 1e-9);
        assert!((cone_theta - theta).abs() < 1e-6);
    }

    #[test]
    fn basic_pipeline_preserves_track_id() {
        let mut tracker = FastTracker::new(30, 30, 0.6, 0.8);
        let first = tracker
            .update(&[object(10.0, 20.0, 30.0, 40.0, 0.9)])
            .unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].get_track_id(), Some(1));
        let second = tracker
            .update(&[object(11.0, 21.0, 30.0, 40.0, 0.9)])
            .unwrap();
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].get_track_id(), Some(1));
    }

    #[test]
    fn low_score_detection_updates_existing_track() {
        let mut tracker = FastTracker::new(30, 30, 0.6, 0.8);
        tracker
            .update(&[object(10.0, 20.0, 30.0, 40.0, 0.9)])
            .unwrap();
        let output = tracker
            .update(&[object(10.5, 20.5, 30.0, 40.0, 0.5)])
            .unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].get_track_id(), Some(1));
        assert!((output[0].get_prob() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn low_score_match_can_occlude_an_unmatched_track() {
        let mut tracker = FastTracker::new(30, 30, 0.6, 0.8);
        tracker
            .update(&[
                object(40.0, 40.0, 10.0, 10.0, 0.9),
                object(0.0, 0.0, 100.0, 100.0, 0.95),
            ])
            .unwrap();

        tracker
            .update(&[object(0.0, 0.0, 100.0, 100.0, 0.5)])
            .unwrap();

        let occluded = tracker
            .tracked_stracks
            .iter()
            .find(|track| track.track_id == 1)
            .unwrap();
        assert!(occluded.is_occluded);
        assert_eq!(occluded.not_matched, 1);
    }

    #[test]
    fn detection_at_track_threshold_is_ignored() {
        let mut tracker = FastTracker::new(30, 30, 0.6, 0.8);
        let output = tracker
            .update(&[object(10.0, 20.0, 30.0, 40.0, 0.6)])
            .unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn builder_settings_are_applied() {
        let roi = FastTrackerRoi::new([
            [0.0, 0.0],
            [0.0, 100.0],
            [200.0, 100.0],
            [200.0, 0.0],
        ]);
        let tracker = FastTracker::new(60, 15, 0.7, 0.85)
            .with_occlusion(10, 4, 1.1, 0.89, 20)
            .with_init_iou_suppression(0.75)
            .with_rois(vec![roi], 8, 4, 3.0)
            .with_mot20(true);
        assert_eq!(tracker.max_time_lost, 30);
        assert_eq!(tracker.reset_velocity_offset, 10);
        assert_eq!(tracker.rois, vec![roi]);
        assert!(tracker.mot20);
    }

    #[test]
    fn occluded_track_keeps_identity_and_updates_state() {
        let mut tracker = FastTracker::new(30, 30, 0.6, 0.8)
            .with_occlusion(1, 1, 1.2, 0.5, 3);
        tracker
            .update(&[
                object(40.0, 40.0, 10.0, 10.0, 0.9),
                object(0.0, 0.0, 100.0, 100.0, 0.95),
            ])
            .unwrap();

        let output = tracker
            .update(&[object(0.0, 0.0, 100.0, 100.0, 0.95)])
            .unwrap();
        assert_eq!(output.len(), 2);
        let occluded = tracker
            .tracked_stracks
            .iter()
            .find(|track| track.track_id == 1)
            .unwrap();
        assert!(occluded.is_occluded);
        assert!(occluded.was_recently_occluded);
        assert_eq!(occluded.not_matched, 1);
        assert_eq!(occluded.occluded_len, 2);
        assert!((occluded.mean.unwrap()[3] - 12.0).abs() < 1e-6);

        tracker
            .update(&[object(0.0, 0.0, 100.0, 100.0, 0.95)])
            .unwrap();
        tracker
            .update(&[object(0.0, 0.0, 100.0, 100.0, 0.95)])
            .unwrap();
        assert!(tracker.lost_stracks.iter().any(|track| track.track_id == 1));

        let refound = tracker
            .update(&[
                object(40.0, 40.0, 10.0, 10.0, 0.9),
                object(0.0, 0.0, 100.0, 100.0, 0.95),
            ])
            .unwrap();
        assert!(refound.iter().any(|track| track.get_track_id() == Some(1)));
    }

    #[test]
    fn unmatched_non_occluded_track_is_lost_after_three_frames() {
        let mut tracker = FastTracker::new(30, 30, 0.6, 0.8);
        tracker
            .update(&[object(10.0, 10.0, 20.0, 20.0, 0.9)])
            .unwrap();
        for _ in 0..2 {
            let output = tracker.update(&[]).unwrap();
            assert_eq!(output.len(), 1);
        }
        assert!(tracker.update(&[]).unwrap().is_empty());
        assert!(tracker.lost_stracks.iter().any(|track| track.track_id == 1));
    }

    #[test]
    fn initialization_iou_suppression_rejects_duplicate_of_active_track() {
        let mut tracker =
            FastTracker::new(30, 30, 0.6, 0.8).with_init_iou_suppression(0.8);
        tracker
            .update(&[object(10.0, 10.0, 40.0, 40.0, 0.9)])
            .unwrap();
        let output = tracker
            .update(&[
                object(10.0, 10.0, 40.0, 40.0, 0.9),
                object(11.0, 11.0, 40.0, 40.0, 0.85),
            ])
            .unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(tracker.next_track_id, 1);
    }

    #[test]
    fn roi_history_repair_updates_history_and_mean() {
        let roi = FastTrackerRoi::new([
            [0.0, 0.0],
            [0.0, 100.0],
            [200.0, 100.0],
            [200.0, 0.0],
        ]);
        let tracker =
            FastTracker::new(30, 30, 0.6, 0.8).with_rois(vec![roi], 3, 10, 2.0);
        let mut track = STrack::from_tlwh([40.0, 40.0, 20.0, 20.0], 0.9);
        track.activate(1, 1);
        track.center_history = vec![[50.0, 50.0], [-5.0, 50.0]];
        track.mean_history.push_back(track.mean.unwrap());
        track.mean_history.push_back(track.mean.unwrap());
        track.set_mean_position(50.0, 50.0);

        tracker.enforce_environment_constraints(&mut track);

        assert_eq!(track.center_history.len(), 3);
        assert!(track.center_history[1][0].abs() < 1e-8);
        // Upstream computes a TLWH top-left and writes it into mean[:2].
        assert!((track.mean.unwrap()[0] - 40.0).abs() < 1e-8);
        assert!((track.mean.unwrap()[1] - 40.0).abs() < 1e-8);
    }

    #[test]
    fn occlusion_sequence_produces_expected_track_state() {
        let mut tracker = FastTracker::new(30, 30, 0.6, 0.8)
            .with_occlusion(1, 1, 1.2, 0.5, 3);
        let frames = [
            vec![
                object(40.0, 40.0, 10.0, 10.0, 0.9),
                object(0.0, 0.0, 100.0, 100.0, 0.95),
            ],
            vec![object(0.0, 0.0, 100.0, 100.0, 0.95)],
            vec![object(0.0, 0.0, 100.0, 100.0, 0.95)],
            vec![object(0.0, 0.0, 100.0, 100.0, 0.95)],
            vec![
                object(40.0, 40.0, 10.0, 10.0, 0.9),
                object(0.0, 0.0, 100.0, 100.0, 0.95),
            ],
        ];
        let expected_ids =
            [vec![1, 2], vec![1, 2], vec![1, 2], vec![2], vec![2, 1]];
        let mut outputs = Vec::new();
        for (frame, expected) in frames.iter().zip(&expected_ids) {
            let output = tracker.update(frame).unwrap();
            assert_eq!(
                output
                    .iter()
                    .map(|track| track.get_track_id().unwrap())
                    .collect::<Vec<_>>(),
                *expected
            );
            outputs.push(output);
        }

        let refound = outputs[4]
            .iter()
            .find(|track| track.get_track_id() == Some(1))
            .unwrap();
        assert!((refound.get_x() - 39.960_014).abs() < 1e-4);
        assert!((refound.get_width() - 10.079_971).abs() < 1e-4);
        let internal = tracker
            .tracked_stracks
            .iter()
            .find(|track| track.track_id == 1)
            .unwrap();
        let expected_mean = [
            45.0,
            45.0,
            1.0,
            10.079970843963139,
            0.0,
            0.0,
            0.0,
            -0.3534475278110411,
        ];
        for (actual, expected) in
            internal.mean.unwrap().iter().zip(expected_mean)
        {
            assert!((actual - expected).abs() < 1e-6);
        }
        assert!(
            (internal.covariance.unwrap()[(0, 0)] - 0.34560524808663295).abs()
                < 1e-6
        );
    }
}
