use std::collections::{HashMap, HashSet};

use crate::{Object, TrackError, lapjv::lapjv};

use super::mcbyte_tracker::McByteMask;

#[derive(Debug, PartialEq)]
pub(crate) struct Association {
    pub(crate) matches: Vec<(usize, usize)>,
    pub(crate) unmatched_tracks: Vec<usize>,
    pub(crate) unmatched_detections: Vec<usize>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn associate(
    similarity: &[Vec<f32>],
    raw_iou: &[Vec<f32>],
    track_ids: &[Option<usize>],
    detections: &[Object],
    masks: &[McByteMask],
    minimum_similarity: f32,
    minimum_mask_average_confidence: f32,
    minimum_mask_coverage: f32,
    minimum_mask_fill_ratio: f32,
    enable_isolated_mask_matching: bool,
) -> Result<Association, TrackError> {
    validate_matrix(
        similarity,
        track_ids.len(),
        detections.len(),
        "similarity",
    )?;
    validate_matrix(raw_iou, track_ids.len(), detections.len(), "raw_iou")?;

    let rows = track_ids.len();
    let cols = detections.len();
    let eligible = bool_matrix(similarity, |value| value >= minimum_similarity);
    let row_counts = row_counts(&eligible);
    let col_counts = col_counts(&eligible, cols);

    let mut locked = Vec::new();
    for row in 0..rows {
        for col in 0..cols {
            if eligible[row][col]
                && row_counts[row] == 1
                && col_counts[col] == 1
            {
                locked.push((row, col));
            }
        }
    }

    let locked_rows =
        locked.iter().map(|(row, _)| *row).collect::<HashSet<_>>();
    let locked_cols =
        locked.iter().map(|(_, col)| *col).collect::<HashSet<_>>();
    let remaining_rows = (0..rows)
        .filter(|row| !locked_rows.contains(row))
        .collect::<Vec<_>>();
    let remaining_cols = (0..cols)
        .filter(|col| !locked_cols.contains(col))
        .collect::<Vec<_>>();

    let mask_by_track = masks
        .iter()
        .map(|mask| (mask.track_id(), mask))
        .collect::<HashMap<_, _>>();
    let mut reduced =
        vec![vec![0.0; remaining_cols.len()]; remaining_rows.len()];
    let mut cached_areas = HashMap::<usize, usize>::new();

    for (local_row, &row) in remaining_rows.iter().enumerate() {
        for (local_col, &col) in remaining_cols.iter().enumerate() {
            let mut score = similarity[row][col];
            let ambiguous = eligible[row][col]
                && (row_counts[row] > 1 || col_counts[col] > 1);
            let isolated = enable_isolated_mask_matching
                && raw_iou[row][col] > 0.0
                && raw_iou[row][col] < minimum_similarity
                && raw_iou[row].iter().filter(|&&value| value > 0.0).count()
                    == 1
                && (0..rows).filter(|&r| raw_iou[r][col] > 0.0).count() == 1;

            if (ambiguous || isolated)
                && let Some(track_id) = track_ids[row]
                && let Some(mask) = mask_by_track.get(&track_id)
                && mask.average_confidence() >= minimum_mask_average_confidence
            {
                let area = *cached_areas
                    .entry(track_id)
                    .or_insert_with(|| mask.visible_area());
                if let Some((coverage, fill)) =
                    mask_metrics(mask, &detections[col], area)
                    && coverage >= minimum_mask_coverage
                    && fill >= minimum_mask_fill_ratio
                {
                    score += fill;
                }
            }
            reduced[local_row][local_col] = score;
        }
    }

    let reduced_result = linear_assignment(
        &reduced,
        remaining_rows.len(),
        remaining_cols.len(),
        minimum_similarity,
    )?;
    let mut matches = locked;
    matches.extend(
        reduced_result
            .matches
            .into_iter()
            .map(|(row, col)| (remaining_rows[row], remaining_cols[col])),
    );
    matches.sort_unstable();

    Ok(Association {
        matches,
        unmatched_tracks: reduced_result
            .unmatched_tracks
            .into_iter()
            .map(|row| remaining_rows[row])
            .collect(),
        unmatched_detections: reduced_result
            .unmatched_detections
            .into_iter()
            .map(|col| remaining_cols[col])
            .collect(),
    })
}

fn validate_matrix(
    matrix: &[Vec<f32>],
    rows: usize,
    cols: usize,
    name: &str,
) -> Result<(), TrackError> {
    if matrix.len() != rows || matrix.iter().any(|row| row.len() != cols) {
        return Err(TrackError::InvalidArgument(format!(
            "{name} must have shape ({rows}, {cols})"
        )));
    }
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(TrackError::InvalidArgument(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(())
}

fn bool_matrix(
    matrix: &[Vec<f32>],
    predicate: impl Fn(f32) -> bool,
) -> Vec<Vec<bool>> {
    matrix
        .iter()
        .map(|row| row.iter().map(|&value| predicate(value)).collect())
        .collect()
}

fn row_counts(matrix: &[Vec<bool>]) -> Vec<usize> {
    matrix
        .iter()
        .map(|row| row.iter().filter(|&&value| value).count())
        .collect()
}

fn col_counts(matrix: &[Vec<bool>], cols: usize) -> Vec<usize> {
    (0..cols)
        .map(|col| matrix.iter().filter(|row| row[col]).count())
        .collect()
}

fn mask_metrics(
    mask: &McByteMask,
    detection: &Object,
    visible_area: usize,
) -> Option<(f32, f32)> {
    if visible_area == 0 {
        return None;
    }
    let image = mask.mask();
    let rect = detection.get_rect();
    let left = rect.x().floor().clamp(0.0, image.width() as f32) as u32;
    let top = rect.y().floor().clamp(0.0, image.height() as f32) as u32;
    let right = (rect.x() + rect.width())
        .ceil()
        .clamp(0.0, image.width() as f32) as u32;
    let bottom = (rect.y() + rect.height())
        .ceil()
        .clamp(0.0, image.height() as f32) as u32;
    if right <= left || bottom <= top {
        return None;
    }
    let mut inside = 0usize;
    for y in top..bottom {
        for x in left..right {
            inside += usize::from(image.get_pixel(x, y).0[0] != 0);
        }
    }
    let box_area = (right - left) as usize * (bottom - top) as usize;
    Some((
        inside as f32 / visible_area as f32,
        inside as f32 / box_area as f32,
    ))
}

fn linear_assignment(
    similarity: &[Vec<f32>],
    rows: usize,
    cols: usize,
    minimum_similarity: f32,
) -> Result<Association, TrackError> {
    if rows == 0 || cols == 0 {
        return Ok(Association {
            matches: Vec::new(),
            unmatched_tracks: (0..rows).collect(),
            unmatched_detections: (0..cols).collect(),
        });
    }

    let n = rows + cols;
    let cost_limit = 1.0f64 - minimum_similarity as f64;
    let mut cost = vec![vec![cost_limit / 2.0; n]; n];
    for row in cost.iter_mut().skip(rows) {
        for value in row.iter_mut().skip(cols) {
            *value = 0.0;
        }
    }
    for row in 0..rows {
        for col in 0..cols {
            cost[row][col] = 1.0 - similarity[row][col] as f64;
        }
    }

    let mut row_solution = vec![-1isize; n];
    let mut col_solution = vec![-1isize; n];
    lapjv(&mut cost, &mut row_solution, &mut col_solution)?;

    let mut matches = Vec::new();
    let mut matched_cols = HashSet::new();
    let mut unmatched_tracks = Vec::new();
    for row in 0..rows {
        let col = row_solution[row];
        if col >= 0
            && (col as usize) < cols
            && similarity[row][col as usize] >= minimum_similarity
        {
            matches.push((row, col as usize));
            matched_cols.insert(col as usize);
        } else {
            unmatched_tracks.push(row);
        }
    }
    Ok(Association {
        matches,
        unmatched_tracks,
        unmatched_detections: (0..cols)
            .filter(|col| !matched_cols.contains(col))
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use image::{GrayImage, Luma};

    use super::*;
    use crate::Rect;

    fn object(x: f32, y: f32, w: f32, h: f32) -> Object {
        Object::new(Rect::new(x, y, w, h), 0.9, None)
    }

    fn mask(track_id: usize, x: u32, y: u32, w: u32, h: u32) -> McByteMask {
        let mut image = GrayImage::new(20, 20);
        for yy in y..y + h {
            for xx in x..x + w {
                image.put_pixel(xx, yy, Luma([255]));
            }
        }
        McByteMask::new(track_id, image, 0.9)
    }

    #[test]
    fn clear_matches_are_locked_and_indices_remain_stable() {
        let result = associate(
            &[vec![0.9, 0.0], vec![0.0, 0.8]],
            &[vec![0.9, 0.0], vec![0.0, 0.8]],
            &[Some(1), Some(2)],
            &[object(0.0, 0.0, 5.0, 5.0), object(10.0, 10.0, 5.0, 5.0)],
            &[],
            0.5,
            0.6,
            0.9,
            0.05,
            false,
        )
        .unwrap();
        assert_eq!(result.matches, vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn mask_changes_ambiguous_assignment_like_roboflow_reference() {
        let detections =
            [object(0.0, 0.0, 5.0, 5.0), object(10.0, 10.0, 5.0, 5.0)];
        let result = associate(
            &[vec![0.7, 0.8], vec![0.8, 0.7]],
            &[vec![0.7, 0.8], vec![0.8, 0.7]],
            &[Some(10), Some(20)],
            &detections,
            &[mask(10, 0, 0, 5, 5), mask(20, 10, 10, 5, 5)],
            0.5,
            0.6,
            0.9,
            0.05,
            false,
        )
        .unwrap();
        assert_eq!(result.matches, vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn isolated_pair_can_be_rescued() {
        let result = associate(
            &[vec![0.2]],
            &[vec![0.2]],
            &[Some(1)],
            &[object(0.0, 0.0, 5.0, 5.0)],
            &[mask(1, 0, 0, 5, 5)],
            0.5,
            0.6,
            0.9,
            0.05,
            true,
        )
        .unwrap();
        assert_eq!(result.matches, vec![(0, 0)]);
    }
}
