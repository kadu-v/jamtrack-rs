use image::{GrayImage, imageops::FilterType};
use nalgebra::{Matrix4, Vector4};

#[derive(Debug, Clone, Copy, PartialEq)]
/// Configuration for pure Rust sparse optical-flow camera compensation.
pub struct SparseOptFlowConfig {
    /// Integer image downscaling factor used before feature tracking.
    pub downscale: usize,
    /// Maximum number of feature points retained per frame.
    pub max_corners: usize,
    /// Minimum corner response relative to the strongest response.
    pub quality_level: f32,
    /// Minimum distance in pixels between retained corners.
    pub min_distance: usize,
    /// Neighborhood size used to compute corner responses.
    pub block_size: usize,
    /// Uses the Harris response instead of the minimum-eigenvalue response.
    pub use_harris: bool,
    /// Harris detector sensitivity parameter.
    pub harris_k: f32,
}

impl Default for SparseOptFlowConfig {
    fn default() -> Self {
        Self {
            downscale: 6,
            max_corners: 1000,
            quality_level: 0.01,
            min_distance: 1,
            block_size: 3,
            use_harris: false,
            harris_k: 0.04,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SparseOptFlow {
    config: SparseOptFlowConfig,
    previous: Option<GrayImage>,
    previous_points: Vec<[f32; 2]>,
}

impl SparseOptFlow {
    pub(crate) fn new(config: SparseOptFlowConfig) -> Self {
        Self {
            config,
            previous: None,
            previous_points: Vec::new(),
        }
    }

    pub(crate) fn reset(&mut self) {
        self.previous = None;
        self.previous_points.clear();
    }

    pub(crate) fn estimate(&mut self, frame: &GrayImage) -> [[f32; 3]; 3] {
        let current = downscale(frame, self.config.downscale.max(1));
        let scale_x = frame.width() as f32 / current.width() as f32;
        let scale_y = frame.height() as f32 / current.height() as f32;
        let current_points = detect_corners(&current, self.config);
        let transform = match &self.previous {
            None => identity(),
            Some(previous)
                if previous.dimensions() != current.dimensions()
                    || self.previous_points.is_empty()
                    || current_points.is_empty() =>
            {
                identity()
            }
            Some(previous) => {
                let pairs =
                    track_points(previous, &current, &self.previous_points);
                estimate_partial_affine(&pairs).unwrap_or_else(identity)
            }
        };
        self.previous = Some(current);
        self.previous_points = current_points;

        let mut result = transform;
        result[0][2] *= scale_x;
        result[1][2] *= scale_y;
        if finite_transform(&result) {
            result
        } else {
            identity()
        }
    }
}

fn downscale(frame: &GrayImage, factor: usize) -> GrayImage {
    if factor <= 1 {
        return frame.clone();
    }
    let width = (frame.width() / factor as u32).max(1);
    let height = (frame.height() / factor as u32).max(1);
    image::imageops::resize(frame, width, height, FilterType::Triangle)
}

fn detect_corners(
    image: &GrayImage,
    config: SparseOptFlowConfig,
) -> Vec<[f32; 2]> {
    if config.max_corners == 0 {
        return Vec::new();
    }
    let (width, height) = (image.width() as usize, image.height() as usize);
    let radius = config.block_size.max(1) / 2;
    if width <= radius * 2 + 2 || height <= radius * 2 + 2 {
        return Vec::new();
    }
    let pixels = image.as_raw().iter().map(|&v| v as f32).collect::<Vec<_>>();
    let (gx, gy) = gradients(&pixels, width, height);
    let mut response_map = vec![0.0f32; width * height];
    let mut maximum = 0.0f32;
    for y in radius + 1..height - radius - 1 {
        for x in radius + 1..width - radius - 1 {
            let mut xx = 0.0;
            let mut xy = 0.0;
            let mut yy = 0.0;
            for wy in y - radius..=y + radius {
                for wx in x - radius..=x + radius {
                    let i = wy * width + wx;
                    xx += gx[i] * gx[i];
                    xy += gx[i] * gy[i];
                    yy += gy[i] * gy[i];
                }
            }
            let response = if config.use_harris {
                xx * yy - xy * xy - config.harris_k * (xx + yy).powi(2)
            } else {
                0.5 * (xx + yy - ((xx - yy).powi(2) + 4.0 * xy * xy).sqrt())
            };
            if response.is_finite() && response > 0.0 {
                maximum = maximum.max(response);
                response_map[y * width + x] = response;
            }
        }
    }
    if maximum <= 0.0 {
        return Vec::new();
    }
    let threshold = maximum * config.quality_level.clamp(0.0, 1.0);
    let mut responses = Vec::new();
    for y in radius + 1..height - radius - 1 {
        for x in radius + 1..width - radius - 1 {
            let response = response_map[y * width + x];
            if response < threshold {
                continue;
            }
            let mut local_maximum = f32::NEG_INFINITY;
            for neighbor_y in y - 1..=y + 1 {
                for neighbor_x in x - 1..=x + 1 {
                    local_maximum = local_maximum
                        .max(response_map[neighbor_y * width + neighbor_x]);
                }
            }
            if response >= local_maximum {
                responses.push((response, x, y));
            }
        }
    }
    responses.sort_by(|a, b| b.0.total_cmp(&a.0));

    let min_distance_sq = config.min_distance.saturating_pow(2) as f32;
    let mut selected = Vec::<[f32; 2]>::new();
    for (_, x, y) in responses {
        let point = [x as f32, y as f32];
        if selected.iter().all(|other| {
            let dx = other[0] - point[0];
            let dy = other[1] - point[1];
            dx * dx + dy * dy >= min_distance_sq
        }) {
            selected.push(point);
            if selected.len() >= config.max_corners {
                break;
            }
        }
    }
    selected
}

fn gradients(
    image: &[f32],
    width: usize,
    height: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut gx = vec![0.0; image.len()];
    let mut gy = vec![0.0; image.len()];
    if width < 3 || height < 3 {
        return (gx, gy);
    }
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let i = y * width + x;
            gx[i] = (image[i + 1] - image[i - 1]) * 0.5;
            gy[i] = (image[i + width] - image[i - width]) * 0.5;
        }
    }
    (gx, gy)
}

fn track_points(
    previous: &GrayImage,
    current: &GrayImage,
    points: &[[f32; 2]],
) -> Vec<([f32; 2], [f32; 2])> {
    let previous_pyramid = pyramid(previous, 3);
    let current_pyramid =
        pyramid(current, previous_pyramid.len().saturating_sub(1));
    let levels = previous_pyramid.len().min(current_pyramid.len());
    let mut pairs = Vec::new();
    for &point in points {
        if let Some(matched) = track_point(
            &previous_pyramid[..levels],
            &current_pyramid[..levels],
            point,
        ) {
            pairs.push((point, matched));
        }
    }
    pairs
}

fn pyramid(image: &GrayImage, max_level: usize) -> Vec<GrayImage> {
    let mut levels = vec![image.clone()];
    for _ in 0..max_level {
        let previous = levels.last().unwrap();
        if previous.width() < 32 || previous.height() < 32 {
            break;
        }
        levels.push(image::imageops::resize(
            previous,
            (previous.width() / 2).max(1),
            (previous.height() / 2).max(1),
            FilterType::Triangle,
        ));
    }
    levels
}

fn track_point(
    previous: &[GrayImage],
    current: &[GrayImage],
    point: [f32; 2],
) -> Option<[f32; 2]> {
    const RADIUS: i32 = 10;
    const MIN_EIGEN: f32 = 1e-4;
    let mut estimate = [0.0f32; 2];
    for level in (0..previous.len()).rev() {
        let scale = (1usize << level) as f32;
        let source = [point[0] / scale, point[1] / scale];
        if level == previous.len() - 1 {
            estimate = source;
        } else {
            estimate[0] *= 2.0;
            estimate[1] *= 2.0;
        }
        for _ in 0..30 {
            let mut gxx = 0.0;
            let mut gxy = 0.0;
            let mut gyy = 0.0;
            let mut bx = 0.0;
            let mut by = 0.0;
            for wy in -RADIUS..=RADIUS {
                for wx in -RADIUS..=RADIUS {
                    let px = source[0] + wx as f32;
                    let py = source[1] + wy as f32;
                    let qx = estimate[0] + wx as f32;
                    let qy = estimate[1] + wy as f32;
                    let template = bilinear(&previous[level], px, py)?;
                    let observed = bilinear(&current[level], qx, qy)?;
                    let gx = (bilinear(&previous[level], px + 1.0, py)?
                        - bilinear(&previous[level], px - 1.0, py)?)
                        * 0.5;
                    let gy = (bilinear(&previous[level], px, py + 1.0)?
                        - bilinear(&previous[level], px, py - 1.0)?)
                        * 0.5;
                    let error = template - observed;
                    gxx += gx * gx;
                    gxy += gx * gy;
                    gyy += gy * gy;
                    bx += gx * error;
                    by += gy * error;
                }
            }
            let determinant = gxx * gyy - gxy * gxy;
            let trace = gxx + gyy;
            let min_eigen = 0.5
                * (trace - (trace * trace - 4.0 * determinant).max(0.0).sqrt());
            if determinant.abs() < 1e-9 || min_eigen < MIN_EIGEN {
                return None;
            }
            let dx = (gyy * bx - gxy * by) / determinant;
            let dy = (gxx * by - gxy * bx) / determinant;
            estimate[0] += dx;
            estimate[1] += dy;
            if dx * dx + dy * dy <= 0.0001 {
                break;
            }
        }
    }
    estimate
        .iter()
        .all(|value| value.is_finite())
        .then_some(estimate)
}

fn bilinear(image: &GrayImage, x: f32, y: f32) -> Option<f32> {
    if image.width() < 2
        || image.height() < 2
        || !x.is_finite()
        || !y.is_finite()
    {
        return None;
    }
    // OpenCV's pyramidal LK uses reflected image borders. Clamping here gives
    // the same important property: corners near an image edge remain usable
    // at coarse pyramid levels instead of invalidating the whole track.
    let x = x.clamp(0.0, image.width() as f32 - 1.001);
    let y = y.clamp(0.0, image.height() as f32 - 1.001);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let wx = x - x0 as f32;
    let wy = y - y0 as f32;
    let p00 = image.get_pixel(x0, y0).0[0] as f32;
    let p01 = image.get_pixel(x0 + 1, y0).0[0] as f32;
    let p10 = image.get_pixel(x0, y0 + 1).0[0] as f32;
    let p11 = image.get_pixel(x0 + 1, y0 + 1).0[0] as f32;
    Some(
        (1.0 - wy) * ((1.0 - wx) * p00 + wx * p01)
            + wy * ((1.0 - wx) * p10 + wx * p11),
    )
}

fn estimate_partial_affine(
    pairs: &[([f32; 2], [f32; 2])],
) -> Option<[[f32; 3]; 3]> {
    if pairs.len() <= 4 {
        return None;
    }
    let mut best_inliers = Vec::new();
    let mut best_error = f32::INFINITY;
    let pair_count =
        pairs.len().saturating_mul(pairs.len().saturating_sub(1)) / 2;
    let trials = pair_count.min(2000);
    let mut random_state = 0x4d43_4259_u64;
    for trial in 0..trials {
        let (first, second) = if pair_count <= 2000 {
            pair_for_index(pairs.len(), trial)
        } else {
            random_state = xorshift(random_state);
            let first = random_state as usize % pairs.len();
            random_state = xorshift(random_state);
            let mut second = random_state as usize % (pairs.len() - 1);
            if second >= first {
                second += 1;
            }
            (first, second)
        };
        let Some(model) = model_from_pair(pairs[first], pairs[second]) else {
            continue;
        };
        let (inliers, error) = inliers(pairs, &model, 3.0);
        if inliers.len() > best_inliers.len()
            || (inliers.len() == best_inliers.len() && error < best_error)
        {
            best_inliers = inliers;
            best_error = error;
        }
    }
    if best_inliers.len() <= 4 {
        return None;
    }
    let mut model = fit_model(pairs, &best_inliers)?;
    for _ in 0..10 {
        let (next, _) = inliers(pairs, &model, 3.0);
        if next == best_inliers {
            break;
        }
        best_inliers = next;
        if best_inliers.len() <= 4 {
            return None;
        }
        model = fit_model(pairs, &best_inliers)?;
    }
    finite_transform(&model).then_some(model)
}

fn pair_for_index(count: usize, mut index: usize) -> (usize, usize) {
    for first in 0..count - 1 {
        let remaining = count - first - 1;
        if index < remaining {
            return (first, first + index + 1);
        }
        index -= remaining;
    }
    unreachable!("pair index is within the triangular pair count")
}

fn xorshift(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^ (state << 17)
}

fn model_from_pair(
    first: ([f32; 2], [f32; 2]),
    second: ([f32; 2], [f32; 2]),
) -> Option<[[f32; 3]; 3]> {
    let dx = second.0[0] - first.0[0];
    let dy = second.0[1] - first.0[1];
    let du = second.1[0] - first.1[0];
    let dv = second.1[1] - first.1[1];
    let denominator = dx * dx + dy * dy;
    if denominator < 1e-6 {
        return None;
    }
    let a = (du * dx + dv * dy) / denominator;
    let b = (dv * dx - du * dy) / denominator;
    let tx = first.1[0] - a * first.0[0] + b * first.0[1];
    let ty = first.1[1] - b * first.0[0] - a * first.0[1];
    let model = [[a, -b, tx], [b, a, ty], [0.0, 0.0, 1.0]];
    finite_transform(&model).then_some(model)
}

fn fit_model(
    pairs: &[([f32; 2], [f32; 2])],
    selected: &[usize],
) -> Option<[[f32; 3]; 3]> {
    let mut normal = Matrix4::<f32>::zeros();
    let mut rhs = Vector4::<f32>::zeros();
    for &index in selected {
        let ([x, y], [u, v]) = pairs[index];
        let row_u = Vector4::new(x, -y, 1.0, 0.0);
        let row_v = Vector4::new(y, x, 0.0, 1.0);
        normal += row_u * row_u.transpose() + row_v * row_v.transpose();
        rhs += row_u * u + row_v * v;
    }
    let parameters = normal.lu().solve(&rhs)?;
    let model = [
        [parameters[0], -parameters[1], parameters[2]],
        [parameters[1], parameters[0], parameters[3]],
        [0.0, 0.0, 1.0],
    ];
    finite_transform(&model).then_some(model)
}

fn inliers(
    pairs: &[([f32; 2], [f32; 2])],
    model: &[[f32; 3]; 3],
    threshold: f32,
) -> (Vec<usize>, f32) {
    let mut indices = Vec::new();
    let mut total_error = 0.0;
    for (index, &(source, target)) in pairs.iter().enumerate() {
        let x = model[0][0] * source[0] + model[0][1] * source[1] + model[0][2];
        let y = model[1][0] * source[0] + model[1][1] * source[1] + model[1][2];
        let error = ((x - target[0]).powi(2) + (y - target[1]).powi(2)).sqrt();
        if error <= threshold {
            indices.push(index);
            total_error += error;
        }
    }
    (indices, total_error)
}

fn identity() -> [[f32; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn finite_transform(transform: &[[f32; 3]; 3]) -> bool {
    transform.iter().flatten().all(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use image::Luma;

    use super::*;

    fn textured(width: u32, height: u32, dx: i32, dy: i32) -> GrayImage {
        let mut image = GrayImage::new(width, height);
        for y in 0..height as i32 {
            for x in 0..width as i32 {
                let sx = x - dx;
                let sy = y - dy;
                let value = if sx >= 0
                    && sy >= 0
                    && sx < width as i32
                    && sy < height as i32
                {
                    (((sx * 17 + sy * 31 + (sx * sy) % 127) & 255) as u8).max(4)
                } else {
                    0
                };
                image.put_pixel(x as u32, y as u32, Luma([value]));
            }
        }
        image
    }

    #[test]
    fn first_frame_and_resolution_change_return_identity() {
        let mut flow = SparseOptFlow::new(SparseOptFlowConfig {
            downscale: 1,
            max_corners: 100,
            ..Default::default()
        });
        assert_eq!(flow.estimate(&textured(96, 72, 0, 0)), identity());
        assert_eq!(flow.estimate(&textured(80, 60, 0, 0)), identity());
    }

    #[test]
    fn zero_max_corners_detects_no_points() {
        let points = detect_corners(
            &textured(96, 72, 0, 0),
            SparseOptFlowConfig {
                max_corners: 0,
                ..Default::default()
            },
        );
        assert!(points.is_empty());
    }

    #[test]
    fn translation_uses_actual_axis_specific_downscale() {
        let frame = textured(101, 77, 0, 0);
        let current = downscale(&frame, 6);
        let scale_x = frame.width() as f32 / current.width() as f32;
        let scale_y = frame.height() as f32 / current.height() as f32;
        assert!((scale_x - 101.0 / 16.0).abs() < f32::EPSILON);
        assert!((scale_y - 77.0 / 12.0).abs() < f32::EPSILON);
        assert_ne!(scale_x, scale_y);
    }

    #[test]
    fn recovers_translation() {
        let mut flow = SparseOptFlow::new(SparseOptFlowConfig {
            downscale: 1,
            max_corners: 100,
            ..Default::default()
        });
        flow.estimate(&textured(128, 96, 0, 0));
        let transform = flow.estimate(&textured(128, 96, 3, -2));
        assert!((transform[0][2] - 3.0).abs() < 1.0, "{transform:?}");
        assert!((transform[1][2] + 2.0).abs() < 1.0, "{transform:?}");
    }

    #[test]
    fn partial_affine_rejects_outlier() {
        let mut pairs = (0..10)
            .map(|i| {
                let p = [i as f32 * 2.0, i as f32 * 3.0 + (i % 2) as f32];
                (p, [p[0] + 4.0, p[1] - 3.0])
            })
            .collect::<Vec<_>>();
        pairs.push(([10.0, 20.0], [100.0, -50.0]));
        let transform = estimate_partial_affine(&pairs).unwrap();
        assert!((transform[0][2] - 4.0).abs() < 1e-3);
        assert!((transform[1][2] + 3.0).abs() < 1e-3);
    }

    #[test]
    fn partial_affine_recovers_rotation_and_scale() {
        let angle = 0.12f32;
        let scale = 1.08f32;
        let a = scale * angle.cos();
        let b = scale * angle.sin();
        let pairs = (0..4)
            .flat_map(|y| {
                (0..4).map(move |x| {
                    let source = [x as f32 * 8.0, y as f32 * 6.0];
                    let target = [
                        a * source[0] - b * source[1] + 5.0,
                        b * source[0] + a * source[1] - 2.0,
                    ];
                    (source, target)
                })
            })
            .collect::<Vec<_>>();
        let transform = estimate_partial_affine(&pairs).unwrap();
        assert!((transform[0][0] - a).abs() < 1e-4);
        assert!((transform[1][0] - b).abs() < 1e-4);
        assert!((transform[0][2] - 5.0).abs() < 1e-4);
        assert!((transform[1][2] + 2.0).abs() < 1e-4);
    }

    #[test]
    fn affine_fit_rejects_non_finite_parameters() {
        let pairs = vec![
            ([0.0, 0.0], [f32::NAN, 0.0]),
            ([1.0, 0.0], [1.0, 0.0]),
            ([0.0, 1.0], [0.0, 1.0]),
            ([1.0, 1.0], [1.0, 1.0]),
            ([2.0, 1.0], [2.0, 1.0]),
        ];
        assert!(fit_model(&pairs, &[0, 1, 2, 3, 4]).is_none());
    }
}
