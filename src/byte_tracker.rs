use crate::{lapjv::lapjv, rect::Rect, strack::STrack};
use std::{collections::HashMap, vec};

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

    pub fn remove_duplicate_stracks(&self) -> (Vec<STrack>, Vec<STrack>) {
        unimplemented!("remove_duplicate_stracks")
    }

    pub fn linear_assignment(&self) -> (Vec<usize>, Vec<usize>) {
        unimplemented!("linear_assignment")
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
        cost: &Vec<Vec<f64>>,
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
        for v in cost_c.iter() {
            println!("{:?},", v);
        }
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
