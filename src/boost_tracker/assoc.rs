//! Association functions for BoostTracker
//!
//! This module provides functions for computing similarity metrics and
//! performing data association between detections and tracks.

use std::vec;

use crate::lapjv::lapjv;
use nalgebra::DMatrix;

/// Compute IoU (Intersection over Union) between all pairs of detections and tracks.
///
/// # Arguments
/// * `detections` - Slice of detection bounding boxes in [x1, y1, x2, y2] format
/// * `tracks` - Slice of track bounding boxes in [x1, y1, x2, y2] format
///
/// # Returns
/// A matrix of shape (num_detections, num_tracks) containing IoU values
pub fn iou_batch(detections: &[[f32; 4]], tracks: &[[f32; 4]]) -> DMatrix<f32> {
    // D: num_dets, T: num_trks
    let num_dets = detections.len();
    let num_trks = tracks.len();

    if num_dets == 0 || num_trks == 0 {
        return DMatrix::zeros(num_dets, num_trks);
    }

    // xx1: [D, T]
    let mut xx1 = DMatrix::zeros(num_dets, num_trks);
    for (i, det) in detections.iter().enumerate() {
        for (j, trk) in tracks.iter().enumerate() {
            xx1[(i, j)] = det[0].max(trk[0]);
        }
    }

    // yy1: [D, T]
    let mut yy1 = DMatrix::zeros(num_dets, num_trks);
    for (i, det) in detections.iter().enumerate() {
        for (j, trk) in tracks.iter().enumerate() {
            yy1[(i, j)] = det[1].max(trk[1]);
        }
    }

    // xx2: [D, T]
    let mut xx2 = DMatrix::zeros(num_dets, num_trks);
    for (i, det) in detections.iter().enumerate() {
        for (j, trk) in tracks.iter().enumerate() {
            xx2[(i, j)] = det[2].min(trk[2]);
        }
    }

    // yy2: [D, T]
    let mut yy2 = DMatrix::zeros(num_dets, num_trks);
    for (i, det) in detections.iter().enumerate() {
        for (j, trk) in tracks.iter().enumerate() {
            yy2[(i, j)] = det[3].min(trk[3]);
        }
    }
    // w: [D, T], h: [D, T]
    let mut w = DMatrix::zeros(num_dets, num_trks);
    let mut h = DMatrix::zeros(num_dets, num_trks);
    for i in 0..num_dets {
        for j in 0..num_trks {
            w[(i, j)] = (xx2[(i, j)] - xx1[(i, j)]).max(0.0);
            h[(i, j)] = (yy2[(i, j)] - yy1[(i, j)]).max(0.0);
        }
    }

    // wh: [D, T]
    let wh = w.component_mul(&h);

    // union: [D, T]
    let mut union = DMatrix::zeros(num_dets, num_trks);
    for i in 0..num_dets {
        let det_area = (detections[i][2] - detections[i][0])
            * (detections[i][3] - detections[i][1]);
        for j in 0..num_trks {
            let trk_area =
                (tracks[j][2] - tracks[j][0]) * (tracks[j][3] - tracks[j][1]);
            union[(i, j)] = det_area + trk_area - wh[(i, j)];
        }
    }

    // iou: [D, T]
    let mut iou = DMatrix::zeros(num_dets, num_trks);
    for i in 0..num_dets {
        for j in 0..num_trks {
            if union[(i, j)] > 0.0 {
                iou[(i, j)] = wh[(i, j)] / union[(i, j)];
            } else {
                iou[(i, j)] = 0.0;
            }
        }
    }
    iou
}

/// Compute Soft BIoU (Buffered IoU) between all pairs of detections and tracks.
///
/// BIoU expands bounding boxes based on confidence before computing IoU.
/// This is introduced in BoostTrack++ for better association.
///
/// # Arguments
/// * `detections` - Slice of detection bounding boxes in [x1, y1, x2, y2] format
/// * `tracks` - Slice of track bounding boxes in [x1, y1, x2, y2] format
/// * `track_confidences` - Confidence scores for each track
///
/// # Returns
/// A matrix of shape (num_detections, num_tracks) containing Soft BIoU values
pub fn soft_biou_batch(
    detections: &[[f32; 4]],
    tracks: &[[f32; 4]],
    track_confidences: &[f32],
) -> DMatrix<f32> {
    // D: num_dets, T: num_trks
    let num_dets = detections.len();
    let num_trks = tracks.len();

    if num_dets == 0 || num_trks == 0 {
        return DMatrix::zeros(num_dets, num_trks);
    }

    let k1 = 0.25;
    let k2 = 0.5;

    // b1x1: [D, T]
    let mut b1x1 = DMatrix::zeros(num_dets, num_trks);
    for (i, det) in detections.iter().enumerate() {
        for (j, _) in tracks.iter().enumerate() {
            b1x1[(i, j)] =
                det[0] - (det[2] - det[0]) * (1.0 - track_confidences[j]) * k1;
        }
    }
    // b2x1: [D, T]
    let mut b2x1 = DMatrix::zeros(num_dets, num_trks);
    for (i, _) in detections.iter().enumerate() {
        for (j, trk) in tracks.iter().enumerate() {
            b2x1[(i, j)] =
                trk[0] - (trk[2] - trk[0]) * (1.0 - track_confidences[j]) * k2;
        }
    }
    // xx1: [D, T]
    let mut xx1 = DMatrix::zeros(num_dets, num_trks);
    for i in 0..num_dets {
        for j in 0..num_trks {
            xx1[(i, j)] = b1x1[(i, j)].max(b2x1[(i, j)]);
        }
    }

    // b1y1: [D, T]
    let mut b1y1 = DMatrix::zeros(num_dets, num_trks);
    for (i, det) in detections.iter().enumerate() {
        for (j, _) in tracks.iter().enumerate() {
            b1y1[(i, j)] =
                det[1] - (det[3] - det[1]) * (1.0 - track_confidences[j]) * k1;
        }
    }
    // b2y1: [D, T]
    let mut b2y1 = DMatrix::zeros(num_dets, num_trks);
    for (i, _) in detections.iter().enumerate() {
        for (j, trk) in tracks.iter().enumerate() {
            b2y1[(i, j)] =
                trk[1] - (trk[3] - trk[1]) * (1.0 - track_confidences[j]) * k2;
        }
    }
    // yy1: [D, T]
    let mut yy1 = DMatrix::zeros(num_dets, num_trks);
    for i in 0..num_dets {
        for j in 0..num_trks {
            yy1[(i, j)] = b1y1[(i, j)].max(b2y1[(i, j)]);
        }
    }

    // b1x2: [D, T]
    let mut b1x2 = DMatrix::zeros(num_dets, num_trks);
    for (i, det) in detections.iter().enumerate() {
        for (j, _) in tracks.iter().enumerate() {
            b1x2[(i, j)] =
                det[2] + (det[2] - det[0]) * (1.0 - track_confidences[j]) * k1;
        }
    }
    // b2x2: [D, T]
    let mut b2x2 = DMatrix::zeros(num_dets, num_trks);
    for (i, _) in detections.iter().enumerate() {
        for (j, trk) in tracks.iter().enumerate() {
            b2x2[(i, j)] =
                trk[2] + (trk[2] - trk[0]) * (1.0 - track_confidences[j]) * k2;
        }
    }
    // xx2: [D, T]
    let mut xx2 = DMatrix::zeros(num_dets, num_trks);
    for i in 0..num_dets {
        for j in 0..num_trks {
            xx2[(i, j)] = b1x2[(i, j)].min(b2x2[(i, j)]);
        }
    }

    // b1y2: [D, T]
    let mut b1y2 = DMatrix::zeros(num_dets, num_trks);
    for (i, det) in detections.iter().enumerate() {
        for (j, _) in tracks.iter().enumerate() {
            b1y2[(i, j)] =
                det[3] + (det[3] - det[1]) * (1.0 - track_confidences[j]) * k1;
        }
    }
    // b2y2: [D, T]
    let mut b2y2 = DMatrix::zeros(num_dets, num_trks);
    for (i, _) in detections.iter().enumerate() {
        for (j, trk) in tracks.iter().enumerate() {
            b2y2[(i, j)] =
                trk[3] + (trk[3] - trk[1]) * (1.0 - track_confidences[j]) * k2;
        }
    }
    // yy2: [D, T]
    let mut yy2 = DMatrix::zeros(num_dets, num_trks);
    for i in 0..num_dets {
        for j in 0..num_trks {
            yy2[(i, j)] = b1y2[(i, j)].min(b2y2[(i, j)]);
        }
    }

    // w: [D, T], h: [D, T]
    let mut w = DMatrix::zeros(num_dets, num_trks);
    let mut h = DMatrix::zeros(num_dets, num_trks);
    for i in 0..num_dets {
        for j in 0..num_trks {
            w[(i, j)] = (xx2[(i, j)] - xx1[(i, j)]).max(0.0);
            h[(i, j)] = (yy2[(i, j)] - yy1[(i, j)]).max(0.0);
        }
    }
    // wh: [D, T]
    let wh = w.component_mul(&h);

    // union: [D, T]
    let mut union = DMatrix::zeros(num_dets, num_trks);
    for i in 0..num_dets {
        for j in 0..num_trks {
            union[(i, j)] = ((b1x2[(i, j)] - b1x1[(i, j)])
                * (b1y2[(i, j)] - b1y1[(i, j)]))
                + ((b2x2[(i, j)] - b2x1[(i, j)])
                    * (b2y2[(i, j)] - b2y1[(i, j)]))
                - wh[(i, j)];
        }
    }

    // soft_biou: [D, T]
    let mut soft_biou = DMatrix::zeros(num_dets, num_trks);
    for i in 0..num_dets {
        for j in 0..num_trks {
            if union[(i, j)] > 0.0 {
                soft_biou[(i, j)] = wh[(i, j)] / union[(i, j)];
            } else {
                soft_biou[(i, j)] = 0.0;
            }
        }
    }
    soft_biou
}

/// Compute shape similarity between detections and tracks.
///
/// Shape similarity is based on the difference in width and height ratios.
/// Uses the corrected formula (v2): `exp(-(|dw-tw|/max(dw,tw) + |dh-th|/max(dh,th)))`
///
/// # Arguments
/// * `detections` - Slice of detection bounding boxes in [x1, y1, x2, y2] format
/// * `tracks` - Slice of track bounding boxes in [x1, y1, x2, y2] format
///
/// # Returns
/// A matrix of shape (num_detections, num_tracks) containing shape similarity values [0, 1]
pub fn shape_similarity(
    detections: &[[f32; 4]],
    tracks: &[[f32; 4]],
) -> DMatrix<f32> {
    // D: num_dets, T: num_trks
    let num_dets = detections.len();
    let num_trks = tracks.len();

    if num_dets == 0 || num_trks == 0 {
        return DMatrix::zeros(num_dets, num_trks);
    }

    // dw: [D, 1]
    let mut dw = DMatrix::zeros(num_dets, 1);
    for (i, det) in detections.iter().enumerate() {
        dw[(i, 0)] = det[2] - det[0];
    }

    // dh: [T, 1]
    let mut dh = DMatrix::zeros(num_dets, 1);
    for (j, det) in detections.iter().enumerate() {
        dh[(j, 0)] = det[3] - det[1];
    }

    // tw: [1, T]
    let mut tw = DMatrix::zeros(1, num_trks);
    for (j, trk) in tracks.iter().enumerate() {
        tw[(0, j)] = trk[2] - trk[0];
    }

    // th: [1, T]
    let mut th = DMatrix::zeros(1, num_trks);
    for (j, trk) in tracks.iter().enumerate() {
        th[(0, j)] = trk[3] - trk[1];
    }

    let mut shape_sim = DMatrix::zeros(num_dets, num_trks);
    for i in 0..num_dets {
        for j in 0..num_trks {
            shape_sim[(i, j)] = f32::exp(
                -((f32::abs(dw[(i, 0)] - tw[(0, j)])
                    / dw[(i, 0)].max(tw[(0, j)]))
                    + (f32::abs(dh[(i, 0)] - th[(0, j)])
                        / dh[(i, 0)].max(th[(0, j)]))),
            );
        }
    }
    shape_sim
}

/// Convert Mahalanobis distance matrix to similarity scores.
///
/// Uses softmax-like transformation with threshold clipping at chi-square 99% confidence interval.
///
/// # Arguments
/// * `mahalanobis_distance` - Matrix of Mahalanobis distances
/// * `softmax_temp` - Temperature for softmax normalization
///
/// # Returns
/// A matrix of similarity scores, with values above threshold set to 0
pub fn mh_dist_similarity(
    mahalanobis_distance: &DMatrix<f32>,
    softmax_temp: f32,
) -> DMatrix<f32> {
    if mahalanobis_distance.nrows() == 0 || mahalanobis_distance.ncols() == 0 {
        return mahalanobis_distance.clone();
    }

    // R: n_rows, C: n_cols
    let n_rows = mahalanobis_distance.nrows();
    let n_cols = mahalanobis_distance.ncols();
    let limit = 13.2767;

    let mut mask = DMatrix::zeros(n_rows, n_cols);
    for i in 0..n_rows {
        for j in 0..n_cols {
            let v = mahalanobis_distance[(i, j)];
            if v > limit {
                mask[(i, j)] = 0 as u8;
            } else {
                mask[(i, j)] = 1 as u8;
            }
        }
    }

    // mh_dist: [R, C]
    let mut mh_dist = DMatrix::zeros(n_rows, n_cols);
    for i in 0..n_rows {
        for j in 0..n_cols {
            let v = mahalanobis_distance[(i, j)];
            if v > limit {
                mh_dist[(i, j)] = 0.0;
            } else {
                mh_dist[(i, j)] = limit - v;
            }
        }
    }

    // Softmax-like normalization
    let mut normalized = DMatrix::zeros(n_cols, 1);
    for j in 0..n_cols {
        let mut sum_exp = 0.0;
        for i in 0..n_rows {
            sum_exp += f32::exp(mh_dist[(i, j)] / softmax_temp);
        }
        normalized[(j, 0)] = sum_exp;
    }

    let mut ret = DMatrix::zeros(n_rows, n_cols);
    for i in 0..n_rows {
        for j in 0..n_cols {
            if mask[(i, j)] == 1u8 {
                ret[(i, j)] = f32::exp(mh_dist[(i, j)] / softmax_temp)
                    / normalized[(j, 0)];
            } else {
                ret[(i, j)] = 0.0;
            }
        }
    }
    ret
}

/// Match detections to tracks using cost matrix.
///
/// Uses simple one-to-one matching when possible (each row/column has at most one match),
/// otherwise falls back to LAPJV algorithm.
///
/// # Arguments
/// * `cost_matrix` - Cost matrix (higher values = better match)
/// * `threshold` - Minimum cost threshold for valid matches
///
/// # Returns
/// Vector of matched pairs as (detection_index, track_index)
pub fn match_detections(
    cost_matrix: &DMatrix<f32>,
    threshold: f32,
) -> Vec<(usize, usize)> {
    let nrows = cost_matrix.nrows();
    let ncols = cost_matrix.ncols();

    if nrows == 0 || ncols == 0 {
        return Vec::new();
    }

    // Create mask: 1 if cost > threshold, 0 otherwise
    let mut mask = DMatrix::<u8>::zeros(nrows, ncols);
    for i in 0..nrows {
        for j in 0..ncols {
            if cost_matrix[(i, j)] > threshold {
                mask[(i, j)] = 1;
            }
        }
    }

    // Check if simple one-to-one matching is possible
    // (each row has at most 1 match AND each column has at most 1 match)
    let mut row_sums = vec![0u8; nrows];
    let mut col_sums = vec![0u8; ncols];
    for i in 0..nrows {
        for j in 0..ncols {
            row_sums[i] += mask[(i, j)];
            col_sums[j] += mask[(i, j)];
        }
    }

    let max_row_sum = *row_sums.iter().max().unwrap_or(&0);
    let max_col_sum = *col_sums.iter().max().unwrap_or(&0);

    if max_row_sum <= 1 && max_col_sum <= 1 {
        // Simple one-to-one matching
        let mut matches = Vec::new();
        for i in 0..nrows {
            for j in 0..ncols {
                if mask[(i, j)] == 1 {
                    matches.push((i, j));
                }
            }
        }
        return matches;
    }

    // Use LAPJV for complex cases
    // LAPJV requires square matrix and minimizes cost, so we:
    // 1. Create square matrix of size max(nrows, ncols)
    // 2. Negate costs (to convert maximize -> minimize)
    // 3. Use large cost for padding entries
    let n = nrows.max(ncols);
    let large_cost = 1e6;

    let mut cost_vec: Vec<Vec<f64>> = vec![vec![large_cost; n]; n];
    for i in 0..nrows {
        for j in 0..ncols {
            // Negate to convert maximization to minimization
            cost_vec[i][j] = -cost_matrix[(i, j)] as f64;
        }
    }

    let mut x = vec![-1isize; n];
    let mut y = vec![-1isize; n];

    if lapjv(&mut cost_vec, &mut x, &mut y).is_err() {
        return Vec::new();
    }

    // Extract matches: x[i] = j means row i is assigned to column j
    // Filter out padding assignments and those below threshold
    let mut matches = Vec::new();
    for (i, &j) in x.iter().enumerate() {
        if i < nrows && j >= 0 && (j as usize) < ncols {
            matches.push((i, j as usize));
        }
    }
    matches
}

/// Result of linear assignment
#[derive(Debug, Clone, PartialEq)]
pub struct AssignmentResult {
    /// Matched pairs as (detection_index, track_index)
    pub matches: Vec<(usize, usize)>,
    /// Indices of unmatched detections
    pub unmatched_detections: Vec<usize>,
    /// Indices of unmatched tracks
    pub unmatched_tracks: Vec<usize>,
}

/// Perform linear assignment using LAPJV algorithm.
///
/// This function performs two-stage matching:
/// 1. Assignment: Uses `cost_matrix` (combined score) to find optimal matches
/// 2. Validation: Uses `iou_matrix` (pure IoU) to filter out low-quality matches
///
/// # Arguments
/// * `iou_matrix` - Pure IoU matrix for validation
/// * `cost_matrix` - Combined cost matrix for assignment (IoU + Mahalanobis + shape, etc.)
/// * `threshold` - Minimum IoU threshold for valid matches
/// * `emb_cost` - Optional embedding cost matrix for relaxed matching
///
/// # Returns
/// Assignment result containing matches and unmatched indices
pub fn linear_assignment(
    iou_matrix: &DMatrix<f32>,
    cost_matrix: &DMatrix<f32>,
    threshold: f32,
    emb_cost: Option<&DMatrix<f32>>,
) -> AssignmentResult {
    let num_dets = cost_matrix.nrows();
    let num_trks = cost_matrix.ncols();

    if num_dets == 0 {
        return AssignmentResult {
            matches: vec![],
            unmatched_detections: vec![],
            unmatched_tracks: (0..num_trks).collect(),
        };
    }

    if num_trks == 0 {
        return AssignmentResult {
            matches: vec![],
            unmatched_detections: (0..num_dets).collect(),
            unmatched_tracks: vec![],
        };
    }

    // Stage 1: Assignment using cost_matrix
    let matched_indices = match_detections(cost_matrix, threshold);

    // Find unmatched detections and tracks
    let matched_dets: std::collections::HashSet<_> =
        matched_indices.iter().map(|(d, _)| *d).collect();
    let matched_trks: std::collections::HashSet<_> =
        matched_indices.iter().map(|(_, t)| *t).collect();

    let mut unmatched_detections: Vec<usize> = (0..num_dets)
        .filter(|d| !matched_dets.contains(d))
        .collect();
    let mut unmatched_tracks: Vec<usize> = (0..num_trks)
        .filter(|t| !matched_trks.contains(t))
        .collect();

    // Stage 2: Validation using iou_matrix
    let mut matches = Vec::new();
    for (d, t) in matched_indices {
        let iou = iou_matrix[(d, t)];

        // Valid match if:
        // 1. iou >= threshold, OR
        // 2. emb_cost exists AND iou >= threshold/2 AND emb_cost >= 0.75
        let valid_match = iou >= threshold
            || emb_cost
                .map(|emb| iou >= threshold / 2.0 && emb[(d, t)] >= 0.75)
                .unwrap_or(false);

        if valid_match {
            matches.push((d, t));
        } else {
            unmatched_detections.push(d);
            unmatched_tracks.push(t);
        }
    }

    AssignmentResult {
        matches,
        unmatched_detections,
        unmatched_tracks,
    }
}

/// Parameters for association
#[derive(Debug, Clone)]
pub struct AssociateParams {
    pub iou_threshold: f32,
    pub lambda_iou: f32,
    pub lambda_mhd: f32,
    pub lambda_shape: f32,
}

impl Default for AssociateParams {
    fn default() -> Self {
        Self {
            iou_threshold: 0.3,
            lambda_iou: 0.5,
            lambda_mhd: 0.25,
            lambda_shape: 0.25,
        }
    }
}

/// Main association function that combines multiple similarity metrics.
///
/// Computes a combined cost matrix using IoU, Mahalanobis distance, and shape similarity,
/// then performs linear assignment.
///
/// # Arguments
/// * `detections` - Detection bounding boxes [x1, y1, x2, y2]
/// * `tracks` - Track bounding boxes [x1, y1, x2, y2]
/// * `params` - Association parameters (thresholds and weights)
/// * `mahalanobis_distance` - Optional pre-computed Mahalanobis distance matrix
/// * `track_confidence` - Optional confidence scores for tracks
/// * `detection_confidence` - Optional confidence scores for detections
///
/// # Returns
/// Assignment result containing matches and unmatched indices
pub fn associate(
    detections: &[[f32; 4]],
    tracks: &[[f32; 4]],
    params: &AssociateParams,
    mahalanobis_distance: Option<&DMatrix<f32>>,
    track_confidence: Option<&[f32]>,
    detection_confidence: Option<&[f32]>,
) -> AssignmentResult {
    let num_dets = detections.len();
    let num_trks = tracks.len();

    if num_trks == 0 {
        return AssignmentResult {
            matches: vec![],
            unmatched_detections: (0..num_dets).collect(),
            unmatched_tracks: vec![],
        };
    }

    if num_dets == 0 {
        return AssignmentResult {
            matches: Vec::new(),
            unmatched_detections: Vec::new(),
            unmatched_tracks: (0..num_trks).collect(),
        };
    }

    // Step 1: Compute IoU matrix
    let iou_matrix = iou_batch(detections, tracks);

    // Step 2: Initialize cost matrix as copy of IoU
    let mut cost_matrix = iou_matrix.clone();

    // Step 3: Add confidence-weighted IoU term
    let conf = if let (Some(det_conf), Some(trk_conf)) =
        (detection_confidence, track_confidence)
    {
        // conf[i,j] = det_conf[i] * trk_conf[j]
        let mut conf = DMatrix::zeros(num_dets, num_trks);
        for i in 0..num_dets {
            for j in 0..num_trks {
                if iou_matrix[(i, j)] >= params.iou_threshold {
                    conf[(i, j)] = det_conf[i] * trk_conf[j];
                }
            }
        }
        // cost_matrix += lambda_iou * conf * iou_matrix
        for i in 0..num_dets {
            for j in 0..num_trks {
                cost_matrix[(i, j)] +=
                    params.lambda_iou * conf[(i, j)] * iou_matrix[(i, j)];
            }
        }
        Some(conf)
    } else {
        None
    };

    // Step 4: Add Mahalanobis distance term
    if let Some(mh_dist) = mahalanobis_distance {
        if mh_dist.nrows() > 0 && mh_dist.ncols() > 0 {
            let mh_sim = mh_dist_similarity(mh_dist, 1.0);

            // cost_matrix += lambda_mhd * mh_similarity
            for i in 0..num_dets {
                for j in 0..num_trks {
                    cost_matrix[(i, j)] += params.lambda_mhd * mh_sim[(i, j)];
                }
            }

            // Add shape similarity if conf is available
            if let Some(ref conf) = conf {
                let shape_sim = shape_similarity(detections, tracks);
                for i in 0..num_dets {
                    for j in 0..num_trks {
                        cost_matrix[(i, j)] += params.lambda_shape
                            * conf[(i, j)]
                            * shape_sim[(i, j)];
                    }
                }
            }
        }
    }

    // TODO: Add emb_cost (embedding similarity) support for BoostTrack++
    // Step 5: Perform linear assignment
    linear_assignment(&iou_matrix, &cost_matrix, params.iou_threshold, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nearly_eq::assert_nearly_eq;

    // Helper to create detection/track arrays
    fn box_xyxy(x1: f32, y1: f32, x2: f32, y2: f32) -> [f32; 4] {
        [x1, y1, x2, y2]
    }

    // ==========================================================================
    // iou_batch tests
    // ==========================================================================

    #[test]
    fn test_iou_batch_identical_boxes() {
        let dets = [box_xyxy(100.0, 100.0, 200.0, 200.0)];
        let trks = [box_xyxy(100.0, 100.0, 200.0, 200.0)];

        let iou = iou_batch(&dets, &trks);

        assert_eq!(iou.nrows(), 1);
        assert_eq!(iou.ncols(), 1);
        assert_nearly_eq!(iou[(0, 0)], 1.0, 1e-5);
    }

    #[test]
    fn test_iou_batch_no_overlap() {
        let dets = [box_xyxy(0.0, 0.0, 100.0, 100.0)];
        let trks = [box_xyxy(200.0, 200.0, 300.0, 300.0)];

        let iou = iou_batch(&dets, &trks);

        assert_eq!(iou[(0, 0)], 0.0);
    }

    #[test]
    fn test_iou_batch_partial_overlap() {
        // Two 100x100 boxes, shifted by 10 pixels
        let dets = [box_xyxy(100.0, 100.0, 200.0, 200.0)];
        let trks = [box_xyxy(110.0, 110.0, 210.0, 210.0)];

        let iou = iou_batch(&dets, &trks);

        // Intersection: 90x90 = 8100, Union: 2*10000 - 8100 = 11900
        // IoU = 8100/11900 ≈ 0.6807
        assert_nearly_eq!(iou[(0, 0)], 0.6806723, 1e-5);
    }

    #[test]
    fn test_iou_batch_multiple() {
        let dets = [
            box_xyxy(100.0, 100.0, 200.0, 200.0),
            box_xyxy(150.0, 150.0, 250.0, 250.0),
            box_xyxy(300.0, 300.0, 400.0, 400.0),
        ];
        let trks = [
            box_xyxy(100.0, 100.0, 200.0, 200.0),
            box_xyxy(110.0, 110.0, 210.0, 210.0),
        ];

        let iou = iou_batch(&dets, &trks);

        assert_eq!(iou.nrows(), 3);
        assert_eq!(iou.ncols(), 2);

        // Expected values from Python
        assert_nearly_eq!(iou[(0, 0)], 1.0, 1e-5);
        assert_nearly_eq!(iou[(0, 1)], 0.6806723, 1e-5);
        assert_nearly_eq!(iou[(1, 0)], 0.14285715, 1e-5);
        assert_nearly_eq!(iou[(1, 1)], 0.2195122, 1e-5);
        assert_nearly_eq!(iou[(2, 0)], 0.0, 1e-5);
        assert_nearly_eq!(iou[(2, 1)], 0.0, 1e-5);
    }

    #[test]
    fn test_iou_batch_empty_detections() {
        let dets: [[f32; 4]; 0] = [];
        let trks = [box_xyxy(100.0, 100.0, 200.0, 200.0)];

        let iou = iou_batch(&dets, &trks);

        assert_eq!(iou.nrows(), 0);
        assert_eq!(iou.ncols(), 1);
    }

    #[test]
    fn test_iou_batch_empty_tracks() {
        let dets = [box_xyxy(100.0, 100.0, 200.0, 200.0)];
        let trks: [[f32; 4]; 0] = [];

        let iou = iou_batch(&dets, &trks);

        assert_eq!(iou.nrows(), 1);
        assert_eq!(iou.ncols(), 0);
    }

    // ==========================================================================
    // soft_biou_batch tests
    // ==========================================================================

    #[test]
    fn test_soft_biou_batch_basic() {
        let dets = [
            box_xyxy(100.0, 100.0, 200.0, 200.0),
            box_xyxy(150.0, 150.0, 250.0, 250.0),
            box_xyxy(300.0, 300.0, 400.0, 400.0),
        ];
        let trks = [
            box_xyxy(100.0, 100.0, 200.0, 200.0),
            box_xyxy(110.0, 110.0, 210.0, 210.0),
        ];
        let confs = [0.9, 0.8];

        let sbiou = soft_biou_batch(&dets, &trks, &confs);

        assert_eq!(sbiou.nrows(), 3);
        assert_eq!(sbiou.ncols(), 2);

        // Expected values from Python
        assert_nearly_eq!(sbiou[(0, 0)], 0.911157, 1e-4);
        assert_nearly_eq!(sbiou[(0, 1)], 0.7124394, 1e-4);
        assert_nearly_eq!(sbiou[(1, 0)], 0.16682434, 1e-4);
        assert_nearly_eq!(sbiou[(1, 1)], 0.26946107, 1e-4);
        assert_nearly_eq!(sbiou[(2, 0)], 0.0, 1e-5);
        assert_nearly_eq!(sbiou[(2, 1)], 0.0, 1e-5);
    }

    #[test]
    fn test_soft_biou_batch_empty() {
        let dets: [[f32; 4]; 0] = [];
        let trks = [box_xyxy(100.0, 100.0, 200.0, 200.0)];
        let confs = [0.9];

        let sbiou = soft_biou_batch(&dets, &trks, &confs);

        assert_eq!(sbiou.nrows(), 0);
        assert_eq!(sbiou.ncols(), 1);
    }

    // ==========================================================================
    // shape_similarity tests
    // ==========================================================================

    #[test]
    fn test_shape_similarity_identical() {
        let dets = [box_xyxy(0.0, 0.0, 100.0, 100.0)];
        let trks = [box_xyxy(50.0, 50.0, 150.0, 150.0)]; // Same size, different position

        let sim = shape_similarity(&dets, &trks);

        assert_nearly_eq!(sim[(0, 0)], 1.0, 1e-5);
    }

    #[test]
    fn test_shape_similarity_different_sizes() {
        let dets = [
            box_xyxy(0.0, 0.0, 100.0, 100.0), // 100x100
            box_xyxy(0.0, 0.0, 50.0, 100.0),  // 50x100
            box_xyxy(0.0, 0.0, 100.0, 200.0), // 100x200
        ];
        let trks = [
            box_xyxy(0.0, 0.0, 100.0, 100.0), // 100x100
            box_xyxy(0.0, 0.0, 200.0, 100.0), // 200x100
        ];

        let sim = shape_similarity(&dets, &trks);

        assert_eq!(sim.nrows(), 3);
        assert_eq!(sim.ncols(), 2);

        // Expected values from Python (v2 - corrected version)
        // Formula: exp(-(|dw-tw|/max(dw,tw) + |dh-th|/max(dh,th)))
        assert_nearly_eq!(sim[(0, 0)], 1.0, 1e-5);
        assert_nearly_eq!(sim[(0, 1)], 0.60653067, 1e-5);
        assert_nearly_eq!(sim[(1, 0)], 0.60653067, 1e-5);
        assert_nearly_eq!(sim[(1, 1)], 0.47236654, 1e-5);
        assert_nearly_eq!(sim[(2, 0)], 0.60653067, 1e-5); // v2 corrected
        assert_nearly_eq!(sim[(2, 1)], 0.36787945, 1e-5);
    }

    #[test]
    fn test_shape_similarity_width_only_differs() {
        // 100x100 vs 200x100
        let dets = [box_xyxy(0.0, 0.0, 100.0, 100.0)];
        let trks = [box_xyxy(0.0, 0.0, 200.0, 100.0)];

        let sim = shape_similarity(&dets, &trks);

        // |100-200|/max(100,200) = 0.5, |100-100|/max(100,100) = 0
        // exp(-(0.5 + 0)) = exp(-0.5)
        assert_nearly_eq!(sim[(0, 0)], 0.60653067, 1e-5);
    }

    #[test]
    fn test_shape_similarity_height_only_differs() {
        // 100x100 vs 100x200
        let dets = [box_xyxy(0.0, 0.0, 100.0, 100.0)];
        let trks = [box_xyxy(0.0, 0.0, 100.0, 200.0)];

        let sim = shape_similarity(&dets, &trks);

        // |100-100|/max(100,100) = 0, |100-200|/max(100,200) = 0.5
        // exp(-(0 + 0.5)) = exp(-0.5)
        assert_nearly_eq!(sim[(0, 0)], 0.60653067, 1e-5);
    }

    #[test]
    fn test_shape_similarity_both_differ() {
        // 100x100 vs 200x200
        let dets = [box_xyxy(0.0, 0.0, 100.0, 100.0)];
        let trks = [box_xyxy(0.0, 0.0, 200.0, 200.0)];

        let sim = shape_similarity(&dets, &trks);

        // |100-200|/max(100,200) = 0.5, |100-200|/max(100,200) = 0.5
        // exp(-(0.5 + 0.5)) = exp(-1)
        assert_nearly_eq!(sim[(0, 0)], 0.36787945, 1e-5);
    }

    #[test]
    fn test_shape_similarity_half_size() {
        // 100x100 vs 50x50
        let dets = [box_xyxy(0.0, 0.0, 100.0, 100.0)];
        let trks = [box_xyxy(0.0, 0.0, 50.0, 50.0)];

        let sim = shape_similarity(&dets, &trks);

        // |100-50|/max(100,50) = 0.5, |100-50|/max(100,50) = 0.5
        // exp(-(0.5 + 0.5)) = exp(-1)
        assert_nearly_eq!(sim[(0, 0)], 0.36787945, 1e-5);
    }

    #[test]
    fn test_shape_similarity_different_aspect_ratios() {
        // 100x200 (tall) vs 200x100 (wide)
        let dets = [box_xyxy(0.0, 0.0, 100.0, 200.0)];
        let trks = [box_xyxy(0.0, 0.0, 200.0, 100.0)];

        let sim = shape_similarity(&dets, &trks);

        // |100-200|/max(100,200) = 0.5, |200-100|/max(200,100) = 0.5
        // exp(-(0.5 + 0.5)) = exp(-1)
        assert_nearly_eq!(sim[(0, 0)], 0.36787945, 1e-5);
    }

    #[test]
    fn test_shape_similarity_extreme_difference() {
        // 100x100 vs 10x10
        let dets = [box_xyxy(0.0, 0.0, 100.0, 100.0)];
        let trks = [box_xyxy(0.0, 0.0, 10.0, 10.0)];

        let sim = shape_similarity(&dets, &trks);

        // |100-10|/max(100,10) = 0.9, |100-10|/max(100,10) = 0.9
        // exp(-(0.9 + 0.9)) = exp(-1.8)
        assert_nearly_eq!(sim[(0, 0)], 0.16529889, 1e-5);
    }

    #[test]
    fn test_shape_similarity_empty() {
        let dets: [[f32; 4]; 0] = [];
        let trks = [box_xyxy(0.0, 0.0, 100.0, 100.0)];

        let sim = shape_similarity(&dets, &trks);

        assert_eq!(sim.nrows(), 0);
        assert_eq!(sim.ncols(), 1);
    }

    // ==========================================================================
    // mh_dist_similarity tests
    // ==========================================================================

    #[test]
    fn test_mh_dist_similarity_basic() {
        // Mahalanobis distances
        let mh_dist = DMatrix::from_row_slice(
            3,
            2,
            &[
                0.5, 2.0, 5.0, 1.0, 20.0,
                15.0, // Above threshold (13.2767)
            ],
        );

        let sim = mh_dist_similarity(&mh_dist, 1.0);

        assert_eq!(sim.nrows(), 3);
        assert_eq!(sim.ncols(), 2);

        // Expected values from Python
        assert_nearly_eq!(sim[(0, 0)], 0.98901033, 1e-4);
        assert_nearly_eq!(sim[(0, 1)], 0.26894054, 1e-4);
        assert_nearly_eq!(sim[(1, 0)], 0.01098691, 1e-4);
        assert_nearly_eq!(sim[(1, 1)], 0.7310561, 1e-4);
        assert_nearly_eq!(sim[(2, 0)], 0.0, 1e-5);
        assert_nearly_eq!(sim[(2, 1)], 0.0, 1e-5);
    }

    #[test]
    fn test_mh_dist_similarity_empty() {
        let mh_dist = DMatrix::<f32>::zeros(0, 0);
        let sim = mh_dist_similarity(&mh_dist, 1.0);

        assert_eq!(sim.nrows(), 0);
        assert_eq!(sim.ncols(), 0);
    }

    #[test]
    fn test_mh_dist_similarity_above_threshold_is_exactly_zero() {
        // Test that values above threshold are EXACTLY 0, not just close to 0
        let mh_dist = DMatrix::from_row_slice(
            2,
            2,
            &[
                1.0, 1.0, // Below threshold
                20.0, 20.0, // Above threshold (13.2767)
            ],
        );

        let sim = mh_dist_similarity(&mh_dist, 1.0);

        // Above threshold should be exactly 0.0
        assert_eq!(
            sim[(1, 0)],
            0.0,
            "Above threshold values should be exactly 0.0"
        );
        assert_eq!(
            sim[(1, 1)],
            0.0,
            "Above threshold values should be exactly 0.0"
        );
    }

    // ==========================================================================
    // match_detections tests
    // ==========================================================================

    #[test]
    fn test_match_detections_perfect_diagonal() {
        // Perfect diagonal matching - simple one-to-one
        let cost = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);

        let matches = match_detections(&cost, 0.3);

        assert_eq!(matches.len(), 2);
        assert!(matches.contains(&(0, 0)));
        assert!(matches.contains(&(1, 1)));
    }

    #[test]
    fn test_match_detections_3x2_simple() {
        // 3 detections, 2 tracks - simple one-to-one
        let cost = DMatrix::from_row_slice(
            3,
            2,
            &[
                0.9, 0.1, // det 0 matches track 0
                0.1, 0.8, // det 1 matches track 1
                0.2, 0.1, // det 2 unmatched
            ],
        );

        let matches = match_detections(&cost, 0.3);

        assert_eq!(matches.len(), 2);
        assert!(matches.contains(&(0, 0)));
        assert!(matches.contains(&(1, 1)));
    }

    #[test]
    fn test_match_detections_all_below_threshold() {
        // All values below threshold
        let cost = DMatrix::from_row_slice(2, 2, &[0.1, 0.2, 0.2, 0.1]);

        let matches = match_detections(&cost, 0.5);

        assert!(matches.is_empty());
    }

    #[test]
    fn test_match_detections_empty_matrix() {
        let cost = DMatrix::<f32>::zeros(0, 2);

        let matches = match_detections(&cost, 0.3);

        assert!(matches.is_empty());
    }

    #[test]
    fn test_match_detections_empty_tracks() {
        let cost = DMatrix::<f32>::zeros(2, 0);

        let matches = match_detections(&cost, 0.3);

        assert!(matches.is_empty());
    }

    #[test]
    fn test_match_detections_single_element() {
        let cost = DMatrix::from_row_slice(1, 1, &[0.9]);

        let matches = match_detections(&cost, 0.3);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0], (0, 0));
    }

    #[test]
    fn test_match_detections_single_element_below_threshold() {
        let cost = DMatrix::from_row_slice(1, 1, &[0.2]);

        let matches = match_detections(&cost, 0.3);

        assert!(matches.is_empty());
    }

    #[test]
    fn test_match_detections_requires_lapjv() {
        // Multiple candidates above threshold for same row - requires LAPJV
        // Cost matrix with clear optimal assignment: (0,0)=0.9, (1,1)=0.8
        // Total = 1.7 vs alternative (0,1)=0.5, (1,0)=0.6, Total = 1.1
        let cost = DMatrix::from_row_slice(
            2,
            2,
            &[
                0.9, 0.5, // det 0: prefers track 0
                0.6, 0.8, // det 1: prefers track 1
            ],
        );

        let matches = match_detections(&cost, 0.4);

        // LAPJV should find optimal assignment maximizing total cost
        assert_eq!(matches.len(), 2);
        assert!(
            matches.contains(&(0, 0)),
            "Expected (0,0), got {:?}",
            matches
        );
        assert!(
            matches.contains(&(1, 1)),
            "Expected (1,1), got {:?}",
            matches
        );
    }

    #[test]
    fn test_match_detections_lapjv_3x3() {
        // 3x3 matrix requiring LAPJV
        let cost = DMatrix::from_row_slice(
            3,
            3,
            &[
                0.9, 0.2, 0.3, // det 0 -> track 0
                0.1, 0.8, 0.2, // det 1 -> track 1
                0.2, 0.3, 0.7, // det 2 -> track 2
            ],
        );

        let matches = match_detections(&cost, 0.5);

        assert_eq!(matches.len(), 3);
        assert!(matches.contains(&(0, 0)));
        assert!(matches.contains(&(1, 1)));
        assert!(matches.contains(&(2, 2)));
    }

    #[test]
    fn test_match_detections_lapjv_non_square() {
        // 3x2 matrix requiring LAPJV (more detections than tracks)
        let cost = DMatrix::from_row_slice(
            3,
            2,
            &[
                0.9, 0.6, // det 0
                0.6, 0.8, // det 1
                0.5, 0.5, // det 2 - should be unmatched
            ],
        );

        let matches = match_detections(&cost, 0.4);

        // Should match 2 pairs optimally
        assert_eq!(matches.len(), 2);
        assert!(matches.contains(&(0, 0)));
        assert!(matches.contains(&(1, 1)));
    }

    // ==========================================================================
    // linear_assignment tests
    // ==========================================================================

    #[test]
    fn test_linear_assignment_perfect_match() {
        // iou_matrix == cost_matrix (simple case)
        let iou = DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.1, 0.9]);
        let cost = iou.clone();

        let result = linear_assignment(&iou, &cost, 0.3, None);

        assert_eq!(result.matches.len(), 2);
        assert!(result.matches.contains(&(0, 0)));
        assert!(result.matches.contains(&(1, 1)));
        assert!(result.unmatched_detections.is_empty());
        assert!(result.unmatched_tracks.is_empty());
    }

    #[test]
    fn test_linear_assignment_iou_validation() {
        // cost_matrix suggests match but iou_matrix is too low
        // cost_matrix: det 0 -> track 0 (high cost), det 1 -> track 1 (high cost)
        // iou_matrix: det 0 -> track 0 (low iou), det 1 -> track 1 (high iou)
        let iou = DMatrix::from_row_slice(2, 2, &[0.2, 0.1, 0.1, 0.8]);
        let cost = DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.1, 0.9]);

        let result = linear_assignment(&iou, &cost, 0.3, None);

        // (0, 0) should be rejected by iou validation
        assert_eq!(result.matches.len(), 1);
        assert!(result.matches.contains(&(1, 1)));
        assert!(result.unmatched_detections.contains(&0));
        assert!(result.unmatched_tracks.contains(&0));
    }

    #[test]
    fn test_linear_assignment_with_emb_cost() {
        // iou is below threshold but emb_cost saves the match
        // iou >= threshold/2 (0.15) AND emb_cost >= 0.75
        let iou = DMatrix::from_row_slice(2, 2, &[0.2, 0.1, 0.1, 0.2]);
        let cost = DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.1, 0.9]);
        let emb = DMatrix::from_row_slice(2, 2, &[0.8, 0.1, 0.1, 0.8]); // high embedding similarity

        let result = linear_assignment(&iou, &cost, 0.3, Some(&emb));

        // Both should match because:
        // iou (0.2) >= threshold/2 (0.15) AND emb_cost (0.8) >= 0.75
        assert_eq!(result.matches.len(), 2);
        assert!(result.matches.contains(&(0, 0)));
        assert!(result.matches.contains(&(1, 1)));
    }

    #[test]
    fn test_linear_assignment_emb_cost_too_low() {
        // iou is below threshold and emb_cost is also too low
        let iou = DMatrix::from_row_slice(2, 2, &[0.2, 0.1, 0.1, 0.2]);
        let cost = DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.1, 0.9]);
        let emb = DMatrix::from_row_slice(2, 2, &[0.5, 0.1, 0.1, 0.5]); // low embedding similarity

        let result = linear_assignment(&iou, &cost, 0.3, Some(&emb));

        // Both should be rejected: iou < threshold AND emb_cost < 0.75
        assert!(result.matches.is_empty());
        assert_eq!(result.unmatched_detections.len(), 2);
        assert_eq!(result.unmatched_tracks.len(), 2);
    }

    #[test]
    fn test_linear_assignment_partial_match() {
        // 3 detections, 2 tracks
        let iou = DMatrix::from_row_slice(
            3,
            2,
            &[
                0.9, 0.1, // det 0 matches track 0
                0.1, 0.8, // det 1 matches track 1
                0.2, 0.1, // det 2 unmatched
            ],
        );
        let cost = iou.clone();

        let result = linear_assignment(&iou, &cost, 0.3, None);

        assert_eq!(result.matches.len(), 2);
        assert!(result.matches.contains(&(0, 0)));
        assert!(result.matches.contains(&(1, 1)));
        assert_eq!(result.unmatched_detections, vec![2]);
        assert!(result.unmatched_tracks.is_empty());
    }

    #[test]
    fn test_linear_assignment_empty_detections() {
        let iou = DMatrix::<f32>::zeros(0, 2);
        let cost = iou.clone();

        let result = linear_assignment(&iou, &cost, 0.3, None);

        assert!(result.matches.is_empty());
        assert!(result.unmatched_detections.is_empty());
        assert_eq!(result.unmatched_tracks, vec![0, 1]);
    }

    #[test]
    fn test_linear_assignment_empty_tracks() {
        let iou = DMatrix::<f32>::zeros(2, 0);
        let cost = iou.clone();

        let result = linear_assignment(&iou, &cost, 0.3, None);

        assert!(result.matches.is_empty());
        assert_eq!(result.unmatched_detections, vec![0, 1]);
        assert!(result.unmatched_tracks.is_empty());
    }

    // ==========================================================================
    // associate tests
    // ==========================================================================

    #[test]
    fn test_associate_basic() {
        // Two detections close to two tracks
        let dets = [
            box_xyxy(100.0, 100.0, 200.0, 200.0),
            box_xyxy(300.0, 300.0, 400.0, 400.0),
        ];
        let trks = [
            box_xyxy(105.0, 105.0, 205.0, 205.0), // Close to det[0]
            box_xyxy(305.0, 305.0, 405.0, 405.0), // Close to det[1]
        ];
        let params = AssociateParams::default();

        let result = associate(&dets, &trks, &params, None, None, None);

        assert_eq!(result.matches.len(), 2);
        assert!(result.matches.contains(&(0, 0)));
        assert!(result.matches.contains(&(1, 1)));
        assert!(result.unmatched_detections.is_empty());
        assert!(result.unmatched_tracks.is_empty());
    }

    #[test]
    fn test_associate_no_tracks() {
        let dets = [box_xyxy(100.0, 100.0, 200.0, 200.0)];
        let trks: [[f32; 4]; 0] = [];
        let params = AssociateParams::default();

        let result = associate(&dets, &trks, &params, None, None, None);

        assert!(result.matches.is_empty());
        assert_eq!(result.unmatched_detections, vec![0]);
        assert!(result.unmatched_tracks.is_empty());
    }

    #[test]
    fn test_associate_no_detections() {
        let dets: [[f32; 4]; 0] = [];
        let trks = [box_xyxy(100.0, 100.0, 200.0, 200.0)];
        let params = AssociateParams::default();

        let result = associate(&dets, &trks, &params, None, None, None);

        assert!(result.matches.is_empty());
        assert!(result.unmatched_detections.is_empty());
        assert_eq!(result.unmatched_tracks, vec![0]);
    }

    #[test]
    fn test_associate_partial_match() {
        // 3 detections, 2 tracks, one detection unmatched
        let dets = [
            box_xyxy(100.0, 100.0, 200.0, 200.0),
            box_xyxy(300.0, 300.0, 400.0, 400.0),
            box_xyxy(600.0, 600.0, 700.0, 700.0), // Far from any track
        ];
        let trks = [
            box_xyxy(105.0, 105.0, 205.0, 205.0),
            box_xyxy(305.0, 305.0, 405.0, 405.0),
        ];
        let params = AssociateParams::default();

        let result = associate(&dets, &trks, &params, None, None, None);

        assert_eq!(result.matches.len(), 2);
        assert!(result.matches.contains(&(0, 0)));
        assert!(result.matches.contains(&(1, 1)));
        assert_eq!(result.unmatched_detections, vec![2]);
        assert!(result.unmatched_tracks.is_empty());
    }

    #[test]
    fn test_associate_with_confidence() {
        let dets = [
            box_xyxy(100.0, 100.0, 200.0, 200.0),
            box_xyxy(300.0, 300.0, 400.0, 400.0),
        ];
        let trks = [
            box_xyxy(105.0, 105.0, 205.0, 205.0),
            box_xyxy(305.0, 305.0, 405.0, 405.0),
        ];
        let params = AssociateParams::default();
        let det_conf = [0.9, 0.8];
        let trk_conf = [0.85, 0.9];

        let result = associate(
            &dets,
            &trks,
            &params,
            None,
            Some(&trk_conf),
            Some(&det_conf),
        );

        assert_eq!(result.matches.len(), 2);
        assert!(result.matches.contains(&(0, 0)));
        assert!(result.matches.contains(&(1, 1)));
    }

    #[test]
    fn test_associate_with_mahalanobis() {
        let dets = [
            box_xyxy(100.0, 100.0, 200.0, 200.0),
            box_xyxy(300.0, 300.0, 400.0, 400.0),
        ];
        let trks = [
            box_xyxy(105.0, 105.0, 205.0, 205.0),
            box_xyxy(305.0, 305.0, 405.0, 405.0),
        ];
        let params = AssociateParams::default();

        // Low Mahalanobis distance for matching pairs (diagonal)
        let mh_dist = DMatrix::from_row_slice(2, 2, &[0.5, 10.0, 10.0, 0.5]);

        let result =
            associate(&dets, &trks, &params, Some(&mh_dist), None, None);

        assert_eq!(result.matches.len(), 2);
        assert!(result.matches.contains(&(0, 0)));
        assert!(result.matches.contains(&(1, 1)));
    }

    #[test]
    fn test_associate_no_overlap() {
        // Detections and tracks have no overlap
        let dets = [
            box_xyxy(0.0, 0.0, 100.0, 100.0),
            box_xyxy(200.0, 200.0, 300.0, 300.0),
        ];
        let trks = [
            box_xyxy(500.0, 500.0, 600.0, 600.0),
            box_xyxy(700.0, 700.0, 800.0, 800.0),
        ];
        let params = AssociateParams::default();

        let result = associate(&dets, &trks, &params, None, None, None);

        // No matches due to zero IoU
        assert!(result.matches.is_empty());
        assert_eq!(result.unmatched_detections.len(), 2);
        assert_eq!(result.unmatched_tracks.len(), 2);
    }

    #[test]
    fn test_associate_with_all_options() {
        let dets = [
            box_xyxy(100.0, 100.0, 200.0, 200.0),
            box_xyxy(300.0, 300.0, 400.0, 400.0),
        ];
        let trks = [
            box_xyxy(105.0, 105.0, 205.0, 205.0),
            box_xyxy(305.0, 305.0, 405.0, 405.0),
        ];
        let params = AssociateParams {
            iou_threshold: 0.3,
            lambda_iou: 0.5,
            lambda_mhd: 0.25,
            lambda_shape: 0.25,
        };
        let det_conf = [0.9, 0.8];
        let trk_conf = [0.85, 0.9];
        let mh_dist = DMatrix::from_row_slice(2, 2, &[0.5, 10.0, 10.0, 0.5]);

        let result = associate(
            &dets,
            &trks,
            &params,
            Some(&mh_dist),
            Some(&trk_conf),
            Some(&det_conf),
        );

        assert_eq!(result.matches.len(), 2);
        assert!(result.matches.contains(&(0, 0)));
        assert!(result.matches.contains(&(1, 1)));
        assert!(result.unmatched_detections.is_empty());
        assert!(result.unmatched_tracks.is_empty());
    }
}
