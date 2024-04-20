use crate::{
    byte_tracker,
    lapjv::lapjv,
    object::Object,
    rect::Rect,
    strack::{STrack, STrackState},
};
use std::{collections::HashMap, sync::mpsc::RecvTimeoutError, vec};

/*-----------------------------------------------------------------------------
ByteTracker
-----------------------------------------------------------------------------*/

#[derive(Debug)]
pub struct ByteTracker {
    track_thresh: f32,
    high_thresh: f32,
    math_thresh: f32,
    max_time_lost: usize,

    frame_id: usize,
    track_id_count: usize,

    tracked_stracks: Vec<STrack>,
    lost_stracks: Vec<STrack>,
    removed_stracks: Vec<STrack>,
}

impl ByteTracker {
    pub fn new(
        track_thresh: f32,
        high_thresh: f32,
        math_thresh: f32,
        max_time_lost: usize,
    ) -> Self {
        Self {
            track_thresh,
            high_thresh,
            math_thresh,
            max_time_lost,

            frame_id: 0,
            track_id_count: 0,

            tracked_stracks: Vec::new(),
            lost_stracks: Vec::new(),
            removed_stracks: Vec::new(),
        }
    }

    pub fn update(&mut self, objects: &Vec<Object>) -> Vec<STrack> {
        /*------------------ Step 1: Get detections -------------------------*/
        self.frame_id += 1;

        // Create new STracks using the result of object detections
        let mut det_stracks = Vec::new();
        let mut det_low_stracks = Vec::new();

        for obj in objects {
            let strack = STrack::new(obj.rect.clone(), obj.prob);
            if obj.prob >= self.track_thresh {
                det_stracks.push(strack);
            } else if obj.prob >= self.high_thresh {
                det_low_stracks.push(strack);
            }
        }

        // Create lists of existing stracks
        let mut active_stracks = Vec::new();
        let mut non_active_stracks = Vec::new();

        for tracked_strack in self.tracked_stracks.iter() {
            if tracked_strack.is_activated() {
                active_stracks.push(tracked_strack.clone());
            } else {
                non_active_stracks.push(tracked_strack.clone());
            }
        }

        let mut strack_pool =
            Self::joint_stracks(&active_stracks, &self.lost_stracks);

        // Predict the current location with KF
        for strack in strack_pool.iter_mut() {
            strack.predict();
        }

        /*------------------ Step 2: First association with IoU -------------------------*/
        let mut current_tracked_stracks = Vec::new();
        let mut remain_tracked_stracks = Vec::new();
        let mut remain_det_stracks = Vec::new();
        let mut refined_stracks = Vec::new();

        {
            let iou_distance =
                Self::calc_iou_distance(&strack_pool, &det_stracks);
            let (matches_idx, unmatched_track_idx, unmatched_detection_idx) =
                self.linear_assignment(
                    &iou_distance,
                    strack_pool.len(),
                    det_stracks.len(),
                    self.track_thresh,
                );
            for (idx, sol) in matches_idx {
                debug_assert!(sol >= 0, "sol is negative {}", sol);
                let mut track = strack_pool[idx].clone();
                let det = &det_stracks[sol as usize];
                if track.get_strack_state() == STrackState::Tracked {
                    track.update(&det, self.frame_id);
                    current_tracked_stracks.push(track.clone());
                    strack_pool[idx] = track; // update the track
                } else {
                    track.reactivate(
                        det,
                        self.frame_id,
                        -1, /* defualt value */
                    );
                    refined_stracks.push(track);
                }
            }

            for &unmatched_idx in unmatched_detection_idx.iter() {
                remain_det_stracks.push(det_stracks[unmatched_idx].clone());
            }

            for &unmatched_idx in unmatched_track_idx.iter() {
                if strack_pool[unmatched_idx].get_strack_state()
                    == STrackState::Tracked
                {
                    remain_tracked_stracks
                        .push(strack_pool[unmatched_idx].clone());
                }
            }
        }

        /*------------------ Step 3: Second association using low score dets -------------------------*/
        let mut current_lost_stracks = Vec::new();
        {
            let iou_distance = Self::calc_iou_distance(
                &remain_tracked_stracks,
                &det_low_stracks,
            );
            let (matches_idx, unmatched_track_idx, unmatched_detection_idx) =
                self.linear_assignment(
                    &iou_distance,
                    remain_tracked_stracks.len(),
                    det_low_stracks.len(),
                    0.5,
                );

            for (idx, sol) in matches_idx {
                debug_assert!(sol >= 0, "sol is negative {}", sol);

                let mut track = remain_tracked_stracks[idx].clone();
                let det = &det_low_stracks[sol as usize];
                if track.get_strack_state() == STrackState::Tracked {
                    track.update(det, self.frame_id);
                    current_tracked_stracks.push(track.clone());
                    remain_tracked_stracks[idx] = track; // update the track
                } else {
                    track.reactivate(
                        det,
                        self.frame_id,
                        -1, /* defulat value */
                    );
                    refined_stracks.push(track);
                }
            }

            for &unmatch_idx in unmatched_detection_idx.iter() {
                let mut track = remain_tracked_stracks[unmatch_idx].clone();
                if track.get_strack_state() != STrackState::Lost {
                    track.mark_as_lost();
                    current_lost_stracks.push(track);
                }
            }
        }

        /*------------------ Step 4: Init new stracks -------------------------*/
        let mut current_removed_stracks = Vec::new();
        {
            let iou_distance = Self::calc_iou_distance(
                &non_active_stracks,
                &remain_det_stracks,
            );

            let (matches_idx, _, unmatched_detection_idx) = self
                .linear_assignment(
                    &iou_distance,
                    non_active_stracks.len(),
                    remain_det_stracks.len(),
                    self.track_thresh,
                );

            for &(idx, sol) in matches_idx.iter() {
                non_active_stracks[idx]
                    .update(&remain_det_stracks[sol as usize], self.frame_id);
                current_tracked_stracks.push(non_active_stracks[idx].clone());
            }

            for &unmatch_idx in unmatched_detection_idx.iter() {
                let mut track = non_active_stracks[unmatch_idx].clone();
                track.mark_as_removed();
                current_removed_stracks.push(track.clone());
                non_active_stracks[unmatch_idx] = track;
            }

            // add new stracks
            for &unmatch_idx in unmatched_detection_idx.iter() {
                let mut track = remain_det_stracks[unmatch_idx].clone();
                if track.get_score() < self.high_thresh {
                    continue;
                }
                self.track_id_count += 1;
                track.activate(self.frame_id, self.track_id_count);
                current_tracked_stracks.push(track.clone());
                remain_det_stracks[unmatch_idx] = track; // update the track
            }
        }

        /*------------------ Step 5: Update state -------------------------*/
        for lost_track in self.lost_stracks.iter_mut() {
            if self.frame_id - lost_track.get_frame_id() > self.max_time_lost {
                lost_track.mark_as_removed();
                current_removed_stracks.push(lost_track.clone());
            }
        }

        let tracked_stracks =
            Self::joint_stracks(&current_tracked_stracks, &self.lost_stracks);
        let lost_stracks =
            Self::sub_stracks(&self.lost_stracks, &current_lost_stracks);
        let removed_stracks = Self::joint_stracks(
            &self.removed_stracks,
            &current_removed_stracks,
        );

        let (tracked_stracks_out, lost_stracks_out) =
            self.remove_duplicate_stracks(&tracked_stracks, &lost_stracks);
        self.tracked_stracks = tracked_stracks_out;
        self.lost_stracks = lost_stracks_out;

        let mut output_stracks = Vec::new();
        for track in self.tracked_stracks.iter() {
            if track.get_strack_state() == STrackState::Tracked {
                output_stracks.push(track.clone());
            }
        }

        output_stracks
    }

    pub fn joint_stracks(
        a_tracks: &Vec<STrack>,
        b_tracks: &Vec<STrack>,
    ) -> Vec<STrack> {
        let mut exists = HashMap::new();
        let mut res = Vec::new();

        for a in a_tracks.iter() {
            exists.insert(a.get_track_id(), 1);
            res.push(a.clone());
        }

        for b in b_tracks.iter() {
            let tid = b.get_track_id();
            // TODO: Check if this is correct
            // The original code is more check
            // if the value corresponding to the key is 0
            // https://github.com/Vertical-Beach/ByteTrack-cpp/blob/d43805d461a714f65da039981bd5f5d21cf5cf59/src/BYTETracker.cpp#L241-L242
            if !exists.contains_key(&tid) {
                exists.insert(tid, 1);
                res.push(b.clone());
            }
        }
        res
    }

    pub fn sub_stracks(
        a_tracks: &Vec<STrack>,
        b_tracks: &Vec<STrack>,
    ) -> Vec<STrack> {
        let mut stracks = HashMap::new();
        for a in a_tracks.iter() {
            stracks.insert(a.get_track_id(), a.clone());
        }

        for b in b_tracks.iter() {
            let tid = b.get_track_id();
            if stracks.contains_key(&tid) {
                stracks.remove(&tid);
            }
        }

        let res = stracks.values().cloned().collect::<Vec<_>>();
        res
    }

    pub fn remove_duplicate_stracks(
        &self,
        a_stracks: &Vec<STrack>,
        b_stracks: &Vec<STrack>,
    ) -> (Vec<STrack>, Vec<STrack>) {
        let mut a_res = Vec::new();
        let mut b_res = Vec::new();

        let ious = Self::calc_iou_distance(a_stracks, b_stracks);
        let mut overlapping_combinations = Vec::new();

        for (i, row) in ious.iter().enumerate() {
            for (j, &iou) in row.iter().enumerate() {
                if iou < 0.15 {
                    overlapping_combinations.push((i, j));
                }
            }

            let mut a_overlapping = vec![false; a_stracks.len()];
            let mut b_overlapping = vec![false; b_stracks.len()];

            for &(i, j) in overlapping_combinations.iter() {
                let timep =
                    a_stracks[i].get_frame_id() - b_stracks[j].get_frame_id();
                let timeq =
                    b_stracks[j].get_frame_id() - a_stracks[i].get_frame_id();
                if timep > timeq {
                    b_overlapping[j] = true;
                } else {
                    a_overlapping[i] = true;
                }
            }

            for (i, a_strack) in a_stracks.iter().enumerate() {
                if !a_overlapping[i] {
                    a_res.push(a_strack.clone());
                }
            }

            for (i, b_strack) in b_stracks.iter().enumerate() {
                if !b_overlapping[i] {
                    b_res.push(b_strack.clone());
                }
            }
        }

        return (a_res, b_res);
    }

    pub fn linear_assignment(
        &self,
        const_matrix: &Vec<Vec<f32>>,
        const_matrix_len: usize,
        const_matrix_row_len: usize,
        thresh: f32,
    ) -> (Vec<(usize, isize)>, Vec<usize>, Vec<usize>) {
        let mut matches = Vec::new();
        let mut a_unmatched = Vec::new();
        let mut b_unmatched = Vec::new();

        if const_matrix.len() == 0 {
            for i in 0..const_matrix_len {
                a_unmatched.push(i);
            }

            for i in 0..const_matrix_row_len {
                b_unmatched.push(i);
            }
        }

        let mut rowsol = Vec::new();
        let mut colsol = Vec::new();

        // The original code is correct?
        // I think the original code is wrong, because the order of arguments is inccorect.
        // https://github.com/Vertical-Beach/ByteTrack-cpp/blob/d43805d461a714f65da039981bd5f5d21cf5cf59/src/BYTETracker.cpp#L353-L354
        let _ = Self::exec_lapjv(
            const_matrix,
            &mut rowsol,
            &mut colsol,
            true,
            thresh as f64,
            true,
        );

        for (i, &sol) in rowsol.iter().enumerate() {
            if sol >= 0 {
                let m = (i, sol);
                matches.push(m);
            } else {
                a_unmatched.push(i);
            }
        }

        for (i, &sol) in colsol.iter().enumerate() {
            if sol < 0 {
                b_unmatched.push(i);
            }
        }

        (matches, a_unmatched, b_unmatched)
    }

    pub fn calc_ious(
        a_rects: &Vec<Rect<f32>>,
        b_rects: &Vec<Rect<f32>>,
    ) -> Vec<Vec<f32>> {
        let mut ious = vec![vec![0.0; b_rects.len()]; a_rects.len()];
        if a_rects.len() * b_rects.len() == 0 {
            return ious;
        }

        for bi in 0..b_rects.len() {
            for ai in 0..a_rects.len() {
                ious[ai][bi] = a_rects[ai].calc_iou(&b_rects[bi])
            }
        }

        ious
    }

    pub fn calc_iou_distance(
        a_tracks: &Vec<STrack>,
        b_tracks: &Vec<STrack>,
    ) -> Vec<Vec<f32>> {
        let mut a_rects = Vec::new();
        let mut b_rects = Vec::new();

        for track in a_tracks.iter() {
            a_rects.push(track.get_rect());
        }

        for track in b_tracks.iter() {
            b_rects.push(track.get_rect());
        }

        let ious = Self::calc_ious(&a_rects, &b_rects);
        let mut cost_matrix = vec![vec![0.0; b_tracks.len()]; a_tracks.len()];
        for ai in 0..a_tracks.len() {
            for bi in 0..b_tracks.len() {
                cost_matrix[ai][bi] = 1.0 - ious[ai][bi];
            }
        }

        cost_matrix
    }

    pub fn exec_lapjv(
        cost: &Vec<Vec<f32>>,
        rowsol: &mut Vec<isize>,
        colsol: &mut Vec<isize>,
        extend_cost: bool,
        cost_limit: f64,
        return_cost: bool,
    ) -> f64 {
        debug_assert!(cost.len() > 0, "cost matrix is empty");
        let mut cost_c = vec![vec![0.0; cost[0].len()]; cost.len()];

        let mut cost_c_extended = vec![vec![0.0f64; cost[0].len()]; cost.len()];
        let n_rows = cost.len();
        let n_cols = cost[0].len();

        debug_assert!(
            rowsol.len() == n_rows,
            "rowsol length {} is not equal to n_rows {}",
            rowsol.len(),
            n_cols
        );
        debug_assert!(
            colsol.len() == n_cols,
            "colsol length {} is not equal to n_cols {}",
            colsol.len(),
            n_rows
        );

        let mut n = 0;

        if n_rows == n_cols {
            n = n_rows;
        } else {
            assert!(
                extend_cost,
                "extend_cost should be true when n_rows != n_cols"
            );
        }

        assert!(
            cost_limit < f64::MAX,
            "cost_limit should be less than f32::MAX"
        );
        if extend_cost || cost_limit < f64::MAX {
            n = n_rows + n_cols;
            cost_c_extended.clear();
            cost_c_extended.resize(n, vec![0.0; n]);

            debug_assert!(
                cost_limit < f64::MAX,
                "cost_limit is not less than f32::MAX"
            );
            if cost_limit < f64::MAX {
                for i in 0..cost_c_extended.len() {
                    for j in 0..cost_c_extended[i].len() {
                        cost_c_extended[i][j] = cost_limit / 2.;
                    }
                }
            } else {
                let mut cost_max = -1.;
                for i in 0..cost_c.len() {
                    for j in 0..cost_c[i].len() {
                        if cost[i][j] > cost_max {
                            cost_max = cost[i][j];
                        }
                    }
                }
                for i in 0..cost_c_extended.len() {
                    for j in 0..cost_c_extended[i].len() {
                        cost_c_extended[i][j] = cost_max as f64 + 1.;
                    }
                }
            }

            for i in n_rows..cost_c_extended.len() {
                for j in n_cols..cost_c_extended[i].len() {
                    cost_c_extended[i][j] = 0.;
                }
            }

            for i in 0..n_rows {
                for j in 0..n_cols {
                    cost_c_extended[i][j] = cost[i][j] as f64;
                }
            }

            // move cost_c_extended to cost_c
            cost_c = cost_c_extended;
        }

        let mut x_c = vec![-1; n];
        let mut y_c = vec![-1; n];

        // TODO: this assertions should be moved to the lapjv function
        debug_assert!(
            cost_c.len() == n,
            "cost_c length is not equal to {}, but got {}",
            n,
            cost_c.len()
        );
        debug_assert!(
            cost_c[0].len() == n,
            "cost_c[0] length is not equal to {}, but got {}",
            n,
            cost_c[0].len()
        );
        debug_assert!(
            x_c.len() == n,
            "x_c length is not equal to {}, but got {}",
            n,
            x_c.len()
        );
        debug_assert!(
            y_c.len() == n,
            "y_c length is not equal to {}, but got {}",
            n,
            y_c.len()
        );
        let ret = lapjv(n, &mut cost_c, &mut x_c, &mut y_c);
        // TODO: should use Result type instead of assert
        assert!(ret == 0, "The return value of lapjv is negative");

        let mut opt = 0.0;
        if n != n_cols {
            for i in 0..n {
                if x_c[i] >= n_cols as isize {
                    x_c[i] = -1;
                }
                if y_c[i] >= n_rows as isize {
                    y_c[i] = -1;
                }
            }

            for i in 0..n_rows {
                rowsol[i] = x_c[i];
            }

            for i in 0..n_cols {
                colsol[i] = y_c[i];
            }

            if return_cost {
                for i in 0..n_rows {
                    debug_assert!(rowsol[i] >= 0, "rowsol[i] is negative");
                    if rowsol[i] >= 0 {
                        opt += cost[i][rowsol[i] as usize] as f64;
                    }
                }
            }
        } else if return_cost {
            for i in 0..rowsol.len() {
                debug_assert!(rowsol[i] >= 0, "rowsol[i] is negative");
                if rowsol[i] >= 0 {
                    opt += cost[i][rowsol[i] as usize] as f64;
                }
            }
        }

        opt
    }
}
