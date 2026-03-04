use crate::error::TrackError;
use crate::lapjv::lapjv;
use nalgebra::DMatrix;

/// Compute IoU between all pairs of detections and tracks.
///
/// Both `detections` and `tracks` use `[x1, y1, x2, y2]` format.
/// Returns a matrix of shape `(num_dets, num_trks)`.
pub(crate) fn iou_batch(detections: &[[f32; 4]], tracks: &[[f32; 4]]) -> DMatrix<f32> {
    let nd = detections.len();
    let nt = tracks.len();
    if nd == 0 || nt == 0 {
        return DMatrix::zeros(nd, nt);
    }

    let mut iou = DMatrix::zeros(nd, nt);
    for (i, d) in detections.iter().enumerate() {
        let d_area = (d[2] - d[0]) * (d[3] - d[1]);
        for (j, t) in tracks.iter().enumerate() {
            let xx1 = d[0].max(t[0]);
            let yy1 = d[1].max(t[1]);
            let xx2 = d[2].min(t[2]);
            let yy2 = d[3].min(t[3]);
            let w = (xx2 - xx1).max(0.0);
            let h = (yy2 - yy1).max(0.0);
            let inter = w * h;
            let t_area = (t[2] - t[0]) * (t[3] - t[1]);
            let union = d_area + t_area - inter;
            iou[(i, j)] = if union > 0.0 { inter / union } else { 0.0 };
        }
    }
    iou
}

/// Batch speed direction from detections to previous observations.
///
/// Returns `(dy, dx)` each of shape `(num_trks, num_dets)`.
/// Note: axis ordering matches Python's `tracks[..., np.newaxis]` broadcast.
pub(crate) fn speed_direction_batch(
    dets: &[[f32; 4]],
    tracks: &[[f32; 4]],
) -> (DMatrix<f32>, DMatrix<f32>) {
    let nd = dets.len();
    let nt = tracks.len();
    let mut dy = DMatrix::zeros(nt, nd);
    let mut dx = DMatrix::zeros(nt, nd);

    for (j, d) in dets.iter().enumerate() {
        let cx1 = (d[0] + d[2]) / 2.0;
        let cy1 = (d[1] + d[3]) / 2.0;
        for (i, t) in tracks.iter().enumerate() {
            let cx2 = (t[0] + t[2]) / 2.0;
            let cy2 = (t[1] + t[3]) / 2.0;
            let raw_dx = cx1 - cx2;
            let raw_dy = cy1 - cy2;
            let norm = (raw_dx * raw_dx + raw_dy * raw_dy).sqrt() + 1e-6;
            dx[(i, j)] = raw_dx / norm;
            dy[(i, j)] = raw_dy / norm;
        }
    }
    (dy, dx)
}

/// Result of the association step.
pub(crate) struct AssocResult {
    /// Matched pairs `(det_idx, trk_idx)`.
    pub matches: Vec<[usize; 2]>,
    /// Indices of unmatched detections.
    pub unmatched_dets: Vec<usize>,
    /// Indices of unmatched trackers.
    pub unmatched_trks: Vec<usize>,
}

/// First-round association with Velocity Direction Consistency (VDC).
///
/// # Arguments
/// * `dets` — detections `[x1,y1,x2,y2,score]`
/// * `trks` — predicted track boxes `[x1,y1,x2,y2]`
/// * `iou_threshold` — minimum IoU to accept a match
/// * `velocities` — per-track velocity `[dy, dx]`
/// * `previous_obs` — per-track previous observation `[x1,y1,x2,y2,score]`
/// * `vdc_weight` — inertia weight for angle cost
pub(crate) fn associate(
    dets: &[[f32; 5]],
    trks: &[[f32; 4]],
    iou_threshold: f32,
    velocities: &[[f32; 2]],
    previous_obs: &[[f32; 5]],
    vdc_weight: f32,
) -> Result<AssocResult, TrackError> {
    let nd = dets.len();
    let nt = trks.len();

    if nt == 0 {
        return Ok(AssocResult {
            matches: vec![],
            unmatched_dets: (0..nd).collect(),
            unmatched_trks: vec![],
        });
    }

    // --- IoU matrix ---
    let det_boxes: Vec<[f32; 4]> = dets.iter().map(|d| [d[0], d[1], d[2], d[3]]).collect();
    let iou_matrix = iou_batch(&det_boxes, trks);

    // --- VDC angle cost ---
    let prev_boxes: Vec<[f32; 4]> = previous_obs
        .iter()
        .map(|p| [p[0], p[1], p[2], p[3]])
        .collect();
    let (y_mat, x_mat) = speed_direction_batch(&det_boxes, &prev_boxes);

    // diff_angle: shape (nt, nd)
    let mut diff_angle = DMatrix::zeros(nt, nd);
    for i in 0..nt {
        let inertia_y = velocities[i][0];
        let inertia_x = velocities[i][1];
        for j in 0..nd {
            let cos_val =
                (inertia_x * x_mat[(i, j)] + inertia_y * y_mat[(i, j)]).clamp(-1.0, 1.0);
            let angle = cos_val.acos();
            diff_angle[(i, j)] =
                (std::f32::consts::FRAC_PI_2 - angle.abs()) / std::f32::consts::PI;
        }
    }

    // valid_mask: 0 if previous_obs has score < 0 (placeholder)
    let mut valid_mask = DMatrix::from_element(nt, nd, 1.0_f32);
    for i in 0..nt {
        if previous_obs[i][4] < 0.0 {
            for j in 0..nd {
                valid_mask[(i, j)] = 0.0;
            }
        }
    }

    // angle_diff_cost: (nt, nd) → transpose to (nd, nt) then multiply by scores
    // Python: angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    //         angle_diff_cost = angle_diff_cost.T  → (nd, nt)
    //         angle_diff_cost = angle_diff_cost * scores  (scores is (nd, nt))
    let mut angle_diff_cost = DMatrix::zeros(nd, nt);
    for i in 0..nt {
        for j in 0..nd {
            let score = dets[j][4];
            angle_diff_cost[(j, i)] = valid_mask[(i, j)] * diff_angle[(i, j)] * vdc_weight * score;
        }
    }

    // --- Combined cost → assignment ---
    let combined = &iou_matrix + &angle_diff_cost;
    let matched_indices = match_or_lapjv(&iou_matrix, &combined, iou_threshold)?;

    // --- Split into matched / unmatched ---
    let mut matched_det_set = vec![false; nd];
    let mut matched_trk_set = vec![false; nt];
    let mut matches = Vec::new();

    for &[di, ti] in &matched_indices {
        if iou_matrix[(di, ti)] < iou_threshold {
            // Below threshold: treat as unmatched
            continue;
        }
        matched_det_set[di] = true;
        matched_trk_set[ti] = true;
        matches.push([di, ti]);
    }

    let unmatched_dets: Vec<usize> = (0..nd).filter(|&i| !matched_det_set[i]).collect();
    let unmatched_trks: Vec<usize> = (0..nt).filter(|&i| !matched_trk_set[i]).collect();

    Ok(AssocResult {
        matches,
        unmatched_dets,
        unmatched_trks,
    })
}

/// Try simple 1-to-1 matching first; fall back to LAPJV.
fn match_or_lapjv(
    iou_matrix: &DMatrix<f32>,
    cost_matrix: &DMatrix<f32>,
    iou_threshold: f32,
) -> Result<Vec<[usize; 2]>, TrackError> {
    let nd = cost_matrix.nrows();
    let nt = cost_matrix.ncols();

    if nd == 0 || nt == 0 {
        return Ok(vec![]);
    }

    // Check if simple assignment is possible (unique max per row and column)
    let mut above = DMatrix::from_element(nd, nt, false);
    for i in 0..nd {
        for j in 0..nt {
            above[(i, j)] = iou_matrix[(i, j)] > iou_threshold;
        }
    }

    let row_max: usize = (0..nd)
        .map(|i| (0..nt).filter(|&j| above[(i, j)]).count())
        .max()
        .unwrap_or(0);
    let col_max: usize = (0..nt)
        .map(|j| (0..nd).filter(|&i| above[(i, j)]).count())
        .max()
        .unwrap_or(0);

    if row_max == 1 && col_max == 1 {
        // Simple case: each row and column has at most one match
        let mut result = Vec::new();
        for i in 0..nd {
            for j in 0..nt {
                if above[(i, j)] {
                    result.push([i, j]);
                }
            }
        }
        return Ok(result);
    }

    // Fall back to LAPJV with negated cost (minimization)
    let n = nd.max(nt);
    let mut cost_for_lapjv = vec![vec![0.0_f64; n]; n];
    for i in 0..nd {
        for j in 0..nt {
            cost_for_lapjv[i][j] = -(cost_matrix[(i, j)] as f64);
        }
    }
    // Pad with large cost
    let large = 1e5;
    for i in 0..n {
        for j in 0..n {
            if i >= nd || j >= nt {
                cost_for_lapjv[i][j] = large;
            }
        }
    }

    let mut x = vec![-1_isize; n];
    let mut y = vec![-1_isize; n];
    lapjv(&mut cost_for_lapjv, &mut x, &mut y)?;

    let mut result = Vec::new();
    for i in 0..nd {
        let j = x[i];
        if j >= 0 && (j as usize) < nt {
            result.push([i, j as usize]);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iou_batch_identical() {
        let dets = vec![[100.0, 100.0, 200.0, 200.0]];
        let trks = vec![[100.0, 100.0, 200.0, 200.0]];
        let iou = iou_batch(&dets, &trks);
        assert!((iou[(0, 0)] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_iou_batch_no_overlap() {
        let dets = vec![[0.0, 0.0, 50.0, 50.0]];
        let trks = vec![[100.0, 100.0, 200.0, 200.0]];
        let iou = iou_batch(&dets, &trks);
        assert!((iou[(0, 0)]).abs() < 1e-5);
    }

    #[test]
    fn test_iou_batch_partial() {
        let dets = vec![[100.0, 100.0, 200.0, 200.0]];
        let trks = vec![[110.0, 120.0, 210.0, 220.0]];
        let iou = iou_batch(&dets, &trks);
        // Expected ~0.528 (from Python)
        assert!(iou[(0, 0)] > 0.5);
        assert!(iou[(0, 0)] < 0.6);
    }

    #[test]
    fn test_speed_direction_batch() {
        let dets = vec![[100.0, 100.0, 200.0, 200.0]];
        let trks = vec![[90.0, 80.0, 190.0, 180.0]];
        let (dy, dx) = speed_direction_batch(&dets, &trks);
        // det center (150,150), trk center (140,130) → dx=10, dy=20
        let norm = (10.0_f32 * 10.0 + 20.0 * 20.0).sqrt();
        assert!((dx[(0, 0)] - 10.0 / norm).abs() < 0.01);
        assert!((dy[(0, 0)] - 20.0 / norm).abs() < 0.01);
    }

    #[test]
    fn test_associate_simple() {
        let dets = vec![
            [100.0, 100.0, 200.0, 200.0, 0.9],
            [300.0, 300.0, 400.0, 400.0, 0.8],
        ];
        let trks = vec![[105.0, 105.0, 205.0, 205.0], [305.0, 305.0, 405.0, 405.0]];
        let velocities = vec![[0.707, 0.707], [0.707, 0.707]];
        let prev_obs = vec![
            [95.0, 95.0, 195.0, 195.0, 0.9],
            [295.0, 295.0, 395.0, 395.0, 0.8],
        ];
        let result = associate(&dets, &trks, 0.3, &velocities, &prev_obs, 0.2).unwrap();
        assert_eq!(result.matches.len(), 2);
        assert!(result.unmatched_dets.is_empty());
        assert!(result.unmatched_trks.is_empty());
    }
}
