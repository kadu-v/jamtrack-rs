//! Pure Rust ECC implementation (Euclidean motion model only).
//!
//! This module is inspired by OpenCV's ECC implementation in
//! `opencv/modules/video/src/ecc.cpp`, adapted to grayscale-only Rust code.

use crate::error::BoostTrackErr;
use nalgebra::{Matrix3, Vector3};

#[derive(Debug, Clone, Copy)]
pub struct EccConfig {
    pub eps: f32,
    pub max_iter: usize,
    pub gauss_filt_size: usize,
    pub resize_long_edge: Option<usize>,
}

impl Default for EccConfig {
    fn default() -> Self {
        Self {
            eps: 1e-4,
            max_iter: 100,
            gauss_filt_size: 1,
            resize_long_edge: Some(350),
        }
    }
}

#[derive(Debug)]
pub struct EccAligner {
    cfg: EccConfig,
    prev_gray: Option<Vec<u8>>,
    prev_w: usize,
    prev_h: usize,
}

impl EccAligner {
    pub fn new(cfg: EccConfig) -> Self {
        Self {
            cfg,
            prev_gray: None,
            prev_w: 0,
            prev_h: 0,
        }
    }

    pub fn estimate(
        &mut self,
        curr_gray: &[u8],
        width: usize,
        height: usize,
    ) -> [[f32; 3]; 3] {
        if curr_gray.len() != width * height {
            return identity3();
        }

        let out = if let Some(prev_gray) = &self.prev_gray {
            if self.prev_w == width && self.prev_h == height {
                find_transform_ecc_euclidean(
                    prev_gray, curr_gray, width, height, self.cfg,
                )
                .unwrap_or_else(|_| identity3())
            } else {
                identity3()
            }
        } else {
            identity3()
        };

        self.prev_gray = Some(curr_gray.to_vec());
        self.prev_w = width;
        self.prev_h = height;
        out
    }
}

pub fn find_transform_ecc_euclidean(
    template_u8: &[u8],
    input_u8: &[u8],
    width: usize,
    height: usize,
    cfg: EccConfig,
) -> Result<[[f32; 3]; 3], BoostTrackErr> {
    if width == 0 || height == 0 {
        return Err(BoostTrackErr::InvalidInput);
    }
    if template_u8.len() != width * height || input_u8.len() != width * height {
        return Err(BoostTrackErr::InvalidInput);
    }
    if cfg.gauss_filt_size == 0 || cfg.gauss_filt_size % 2 == 0 {
        return Err(BoostTrackErr::InvalidInput);
    }

    let mut sx = 1.0f32;
    let mut sy = 1.0f32;
    let (mut tw, mut th) = (width, height);
    if let Some(target_long_edge) = cfg.resize_long_edge {
        if target_long_edge > 0 {
            let current_long = width.max(height);
            if current_long > target_long_edge {
                let scale = target_long_edge as f32 / current_long as f32;
                tw = ((width as f32 * scale).round() as usize).max(8);
                th = ((height as f32 * scale).round() as usize).max(8);
                sx = tw as f32 / width as f32;
                sy = th as f32 / height as f32;
            }
        }
    }

    let template_f32 = gray_u8_to_f32(template_u8);
    let input_f32 = gray_u8_to_f32(input_u8);
    let template_scaled = if tw != width || th != height {
        resize_bilinear(&template_f32, width, height, tw, th)
    } else {
        template_f32
    };
    let input_scaled = if tw != width || th != height {
        resize_bilinear(&input_f32, width, height, tw, th)
    } else {
        input_f32
    };

    let template_blur =
        gaussian_blur_separable(&template_scaled, tw, th, cfg.gauss_filt_size);
    let input_blur =
        gaussian_blur_separable(&input_scaled, tw, th, cfg.gauss_filt_size);
    let grad_x = gradient_x(&input_blur, tw, th);
    let grad_y = gradient_y(&input_blur, tw, th);

    let mut theta = 0.0f32;
    let mut tx = 0.0f32;
    let mut ty = 0.0f32;
    let mut last_rho = f32::NEG_INFINITY;
    let mut rho = -1.0f32;

    let mut xgrid = vec![0.0f32; tw * th];
    let mut ygrid = vec![0.0f32; tw * th];
    for y in 0..th {
        for x in 0..tw {
            let i = y * tw + x;
            xgrid[i] = x as f32;
            ygrid[i] = y as f32;
        }
    }

    for _ in 0..cfg.max_iter {
        let map = affine_from_theta_tx_ty(theta, tx, ty);
        let warped = warp_affine_inverse_map(&input_blur, tw, th, &map);
        let gxw = warp_affine_inverse_map(&grad_x, tw, th, &map);
        let gyw = warp_affine_inverse_map(&grad_y, tw, th, &map);
        let valid = valid_mask_from_map(tw, th, &map);

        let valid_count = valid.iter().filter(|&&v| v).count();
        if valid_count < 16 {
            return Err(BoostTrackErr::NotConverged);
        }

        let (t_mean, i_mean) = means_over_mask(&template_blur, &warped, &valid);
        let (template_zm, image_zm, tmp_norm, img_norm, corr) =
            centered_stats(&template_blur, &warped, &valid, t_mean, i_mean);

        if tmp_norm <= 1e-6 || img_norm <= 1e-6 {
            return Err(BoostTrackErr::NotConverged);
        }

        rho = corr / (img_norm * tmp_norm);
        if !rho.is_finite() {
            return Err(BoostTrackErr::NotConverged);
        }

        if last_rho.is_finite() && (rho - last_rho).abs() < cfg.eps {
            break;
        }
        last_rho = rho;

        let ct = theta.cos();
        let st = theta.sin();

        let mut h = [[0.0f32; 3]; 3];
        let mut image_proj = [0.0f32; 3];
        let mut template_proj = [0.0f32; 3];
        let mut error_proj = [0.0f32; 3];

        for i in 0..(tw * th) {
            if !valid[i] {
                continue;
            }
            let hat_x = -(xgrid[i] * st) - (ygrid[i] * ct);
            let hat_y = (xgrid[i] * ct) - (ygrid[i] * st);
            let j0 = gxw[i] * hat_x + gyw[i] * hat_y;
            let j1 = gxw[i];
            let j2 = gyw[i];
            let j = [j0, j1, j2];

            for r in 0..3 {
                image_proj[r] += j[r] * image_zm[i];
                template_proj[r] += j[r] * template_zm[i];
                for c in 0..3 {
                    h[r][c] += j[r] * j[c];
                }
            }
        }

        let h_mat = Matrix3::new(
            h[0][0], h[0][1], h[0][2], h[1][0], h[1][1], h[1][2], h[2][0],
            h[2][1], h[2][2],
        );
        let Some(h_inv) = h_mat.try_inverse() else {
            return Err(BoostTrackErr::NotConverged);
        };

        let ip = Vector3::new(image_proj[0], image_proj[1], image_proj[2]);
        let tp =
            Vector3::new(template_proj[0], template_proj[1], template_proj[2]);
        let image_proj_hessian = h_inv * ip;
        let lambda_n = img_norm * img_norm - ip.dot(&image_proj_hessian);
        let lambda_d = corr - tp.dot(&image_proj_hessian);

        if lambda_d <= 0.0 || !lambda_d.is_finite() {
            return Err(BoostTrackErr::NotConverged);
        }

        let lambda = lambda_n / lambda_d;
        if !lambda.is_finite() {
            return Err(BoostTrackErr::NotConverged);
        }

        for i in 0..(tw * th) {
            if !valid[i] {
                continue;
            }
            let hat_x = -(xgrid[i] * st) - (ygrid[i] * ct);
            let hat_y = (xgrid[i] * ct) - (ygrid[i] * st);
            let j0 = gxw[i] * hat_x + gyw[i] * hat_y;
            let j1 = gxw[i];
            let j2 = gyw[i];
            let e = lambda * template_zm[i] - image_zm[i];
            error_proj[0] += j0 * e;
            error_proj[1] += j1 * e;
            error_proj[2] += j2 * e;
        }

        let delta =
            h_inv * Vector3::new(error_proj[0], error_proj[1], error_proj[2]);
        theta += delta[0];
        tx += delta[1];
        ty += delta[2];
    }

    if !rho.is_finite() {
        return Err(BoostTrackErr::NotConverged);
    }

    // Translation was estimated in scaled coordinates; map back to original image.
    let tx_unscaled = tx / sx;
    let ty_unscaled = ty / sy;
    let ct = theta.cos();
    let st = theta.sin();

    Ok([
        [ct, -st, tx_unscaled],
        [st, ct, ty_unscaled],
        [0.0, 0.0, 1.0],
    ])
}

fn identity3() -> [[f32; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn affine_from_theta_tx_ty(theta: f32, tx: f32, ty: f32) -> [[f32; 3]; 3] {
    let ct = theta.cos();
    let st = theta.sin();
    [[ct, -st, tx], [st, ct, ty], [0.0, 0.0, 1.0]]
}

fn gray_u8_to_f32(img: &[u8]) -> Vec<f32> {
    img.iter().map(|&v| v as f32).collect()
}

fn resize_bilinear(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; dst_w * dst_h];
    let scale_x = src_w as f32 / dst_w as f32;
    let scale_y = src_h as f32 / dst_h as f32;
    for dy in 0..dst_h {
        let sy = (dy as f32 + 0.5) * scale_y - 0.5;
        let y0 = sy.floor().max(0.0) as isize;
        let y1 = (y0 + 1).min((src_h - 1) as isize);
        let wy = sy - y0 as f32;
        for dx in 0..dst_w {
            let sx = (dx as f32 + 0.5) * scale_x - 0.5;
            let x0 = sx.floor().max(0.0) as isize;
            let x1 = (x0 + 1).min((src_w - 1) as isize);
            let wx = sx - x0 as f32;
            let p00 = src[y0 as usize * src_w + x0 as usize];
            let p01 = src[y0 as usize * src_w + x1 as usize];
            let p10 = src[y1 as usize * src_w + x0 as usize];
            let p11 = src[y1 as usize * src_w + x1 as usize];
            out[dy * dst_w + dx] = (1.0 - wy) * ((1.0 - wx) * p00 + wx * p01)
                + wy * ((1.0 - wx) * p10 + wx * p11);
        }
    }
    out
}

fn gaussian_blur_separable(
    src: &[f32],
    w: usize,
    h: usize,
    ksize: usize,
) -> Vec<f32> {
    if ksize <= 1 {
        return src.to_vec();
    }
    let kernel = gaussian_kernel(ksize);
    let radius = (ksize / 2) as isize;
    let mut tmp = vec![0.0f32; w * h];
    let mut out = vec![0.0f32; w * h];

    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for k in 0..ksize {
                let xx = ((x as isize + k as isize - radius)
                    .clamp(0, (w - 1) as isize))
                    as usize;
                acc += src[y * w + xx] * kernel[k];
            }
            tmp[y * w + x] = acc;
        }
    }
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for k in 0..ksize {
                let yy = ((y as isize + k as isize - radius)
                    .clamp(0, (h - 1) as isize))
                    as usize;
                acc += tmp[yy * w + x] * kernel[k];
            }
            out[y * w + x] = acc;
        }
    }
    out
}

fn gaussian_kernel(ksize: usize) -> Vec<f32> {
    let sigma = 0.3 * ((ksize as f32 - 1.0) * 0.5 - 1.0) + 0.8;
    let radius = (ksize / 2) as isize;
    let mut k = vec![0.0f32; ksize];
    let mut sum = 0.0f32;
    for i in 0..ksize {
        let x = i as isize - radius;
        let v = (-((x * x) as f32) / (2.0 * sigma * sigma)).exp();
        k[i] = v;
        sum += v;
    }
    for v in &mut k {
        *v /= sum;
    }
    k
}

fn gradient_x(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let l = if x == 0 { x } else { x - 1 };
            let r = if x + 1 >= w { x } else { x + 1 };
            out[y * w + x] = 0.5 * (src[y * w + r] - src[y * w + l]);
        }
    }
    out
}

fn gradient_y(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        let u = if y == 0 { y } else { y - 1 };
        let d = if y + 1 >= h { y } else { y + 1 };
        for x in 0..w {
            out[y * w + x] = 0.5 * (src[d * w + x] - src[u * w + x]);
        }
    }
    out
}

fn warp_affine_inverse_map(
    src: &[f32],
    w: usize,
    h: usize,
    map: &[[f32; 3]; 3],
) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let xf = x as f32;
            let yf = y as f32;
            let sx = map[0][0] * xf + map[0][1] * yf + map[0][2];
            let sy = map[1][0] * xf + map[1][1] * yf + map[1][2];
            out[y * w + x] = bilinear_at(src, w, h, sx, sy);
        }
    }
    out
}

fn valid_mask_from_map(w: usize, h: usize, map: &[[f32; 3]; 3]) -> Vec<bool> {
    let mut mask = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let xf = x as f32;
            let yf = y as f32;
            let sx = map[0][0] * xf + map[0][1] * yf + map[0][2];
            let sy = map[1][0] * xf + map[1][1] * yf + map[1][2];
            mask[y * w + x] = sx >= 0.0
                && sy >= 0.0
                && sx < (w - 1) as f32
                && sy < (h - 1) as f32;
        }
    }
    mask
}

fn bilinear_at(src: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    if x < 0.0 || y < 0.0 || x >= (w - 1) as f32 || y >= (h - 1) as f32 {
        return 0.0;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let wx = x - x0 as f32;
    let wy = y - y0 as f32;
    let p00 = src[y0 * w + x0];
    let p01 = src[y0 * w + x1];
    let p10 = src[y1 * w + x0];
    let p11 = src[y1 * w + x1];
    (1.0 - wy) * ((1.0 - wx) * p00 + wx * p01)
        + wy * ((1.0 - wx) * p10 + wx * p11)
}

fn means_over_mask(a: &[f32], b: &[f32], mask: &[bool]) -> (f32, f32) {
    let mut sa = 0.0f32;
    let mut sb = 0.0f32;
    let mut n = 0usize;
    for i in 0..a.len() {
        if mask[i] {
            sa += a[i];
            sb += b[i];
            n += 1;
        }
    }
    let nf = n as f32;
    (sa / nf, sb / nf)
}

fn centered_stats(
    template: &[f32],
    image: &[f32],
    mask: &[bool],
    t_mean: f32,
    i_mean: f32,
) -> (Vec<f32>, Vec<f32>, f32, f32, f32) {
    let mut tz = vec![0.0f32; template.len()];
    let mut iz = vec![0.0f32; image.len()];
    let mut t_ss = 0.0f32;
    let mut i_ss = 0.0f32;
    let mut corr = 0.0f32;
    for i in 0..template.len() {
        if !mask[i] {
            continue;
        }
        let tv = template[i] - t_mean;
        let iv = image[i] - i_mean;
        tz[i] = tv;
        iz[i] = iv;
        t_ss += tv * tv;
        i_ss += iv * iv;
        corr += tv * iv;
    }
    (tz, iz, t_ss.sqrt(), i_ss.sqrt(), corr)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_checkerboard(w: usize, h: usize, step: usize) -> Vec<u8> {
        let mut out = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let v = ((x / step) + (y / step)) % 2;
                out[y * w + x] = if v == 0 { 40 } else { 220 };
            }
        }
        out
    }

    fn warp_u8_with_inverse_map(
        src: &[u8],
        w: usize,
        h: usize,
        map: [[f32; 3]; 3],
    ) -> Vec<u8> {
        let srcf: Vec<f32> = src.iter().map(|&v| v as f32).collect();
        let warped = warp_affine_inverse_map(&srcf, w, h, &map);
        warped
            .iter()
            .map(|v| v.round().clamp(0.0, 255.0) as u8)
            .collect()
    }

    #[test]
    fn test_ecc_euclidean_recovers_transform() {
        let w = 128;
        let h = 96;
        let template = make_checkerboard(w, h, 8);

        let theta = 3.0f32.to_radians();
        let tx = 4.0f32;
        let ty = -3.0f32;
        let forward = affine_from_theta_tx_ty(theta, tx, ty);
        let inv = {
            let c = theta.cos();
            let s = theta.sin();
            [
                [c, s, -(c * tx + s * ty)],
                [-s, c, s * tx - c * ty],
                [0.0, 0.0, 1.0],
            ]
        };
        let input = warp_u8_with_inverse_map(&template, w, h, inv);

        let m = find_transform_ecc_euclidean(
            &template,
            &input,
            w,
            h,
            EccConfig::default(),
        )
        .expect("ecc should converge");

        assert!((m[0][2] - forward[0][2]).abs() < 1.5, "tx");
        assert!((m[1][2] - forward[1][2]).abs() < 1.5, "ty");
        assert!((m[1][0] - forward[1][0]).abs() < 0.05, "sin(theta)");
    }

    #[test]
    fn test_aligner_first_frame_returns_identity() {
        let w = 64;
        let h = 48;
        let frame = make_checkerboard(w, h, 8);
        let mut aligner = EccAligner::new(EccConfig::default());
        let t = aligner.estimate(&frame, w, h);
        assert!((t[0][0] - 1.0).abs() < 1e-6);
        assert!((t[1][1] - 1.0).abs() < 1e-6);
        assert!(t[0][2].abs() < 1e-6);
        assert!(t[1][2].abs() < 1e-6);
    }
}
