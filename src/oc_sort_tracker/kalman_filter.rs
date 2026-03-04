use nalgebra::SMatrix;

/* -----------------------------------------------------------------------------
 * Type aliases  (dim_x = 7, dim_z = 4)
 * ----------------------------------------------------------------------------- */

/// Measurement vector: [x, y, s, r]
pub(crate) type DetectBox = SMatrix<f32, 4, 1>;
/// State vector: [x, y, s, r, vx, vy, vs]
pub(crate) type StateMean = SMatrix<f32, 7, 1>;
/// State covariance (7x7)
pub(crate) type StateCov = SMatrix<f32, 7, 7>;
/// Projected measurement covariance (4x4)
pub(crate) type MeasCov = SMatrix<f32, 4, 4>;
/// State transition matrix (7x7)
type TransMat = SMatrix<f32, 7, 7>;
/// Observation matrix (4x7)
type ObsMat = SMatrix<f32, 4, 7>;

/* -----------------------------------------------------------------------------
 * Saved state for freeze / unfreeze
 * ----------------------------------------------------------------------------- */

#[derive(Clone)]
struct SavedState {
    x: StateMean,
    p: StateCov,
    history_obs: Vec<Option<DetectBox>>,
}

/* -----------------------------------------------------------------------------
 * KalmanFilter  —  faithful port of OC-SORT's KalmanFilterNew
 * ----------------------------------------------------------------------------- */

pub(crate) struct KalmanFilter {
    // Constant matrices
    f: TransMat,
    h: ObsMat,
    r: MeasCov,
    q: StateCov,

    // State
    x: StateMean,
    p: StateCov,

    // freeze / unfreeze
    observed: bool,
    attr_saved: Option<SavedState>,
    history_obs: Vec<Option<DetectBox>>,
}

impl KalmanFilter {
    /// Create a new 7-dim Kalman filter initialised with measurement `z`.
    pub(crate) fn new(z: &DetectBox) -> Self {
        // --- F: constant-velocity transition (7x7) ---
        #[rustfmt::skip]
        let f = TransMat::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);

        // --- H: observation matrix (4x7) ---
        #[rustfmt::skip]
        let h = ObsMat::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ]);

        // --- R: measurement noise ---
        // diag [1, 1, 10, 10]  (R[2:,2:] *= 10)
        let r = MeasCov::from_diagonal(&SMatrix::<f32, 4, 1>::new(1.0, 1.0, 10.0, 10.0));

        // --- P: initial state covariance ---
        // diag [10, 10, 10, 10, 10000, 10000, 10000]
        let p = StateCov::from_diagonal(&SMatrix::<f32, 7, 1>::from_column_slice(&[
            10.0, 10.0, 10.0, 10.0, 10000.0, 10000.0, 10000.0,
        ]));

        // --- Q: process noise ---
        // diag [1, 1, 1, 1, 0.01, 0.01, 0.0001]
        let q = StateCov::from_diagonal(&SMatrix::<f32, 7, 1>::from_column_slice(&[
            1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001,
        ]));

        // --- x: initial state ---
        let mut x = StateMean::zeros();
        x[0] = z[0];
        x[1] = z[1];
        x[2] = z[2];
        x[3] = z[3];

        Self {
            f,
            h,
            r,
            q,
            x,
            p,
            observed: false,
            attr_saved: None,
            history_obs: Vec::new(),
        }
    }

    // ------------------------------------------------------------------
    // Predict
    // ------------------------------------------------------------------

    pub(crate) fn predict(&mut self) {
        // x = F x
        self.x = self.f * self.x;
        // P = F P F^T + Q
        self.p = self.f * self.p * self.f.transpose() + self.q;
    }

    // ------------------------------------------------------------------
    // Update
    // ------------------------------------------------------------------

    /// Update the filter with a measurement.  Pass `None` for a missed frame.
    pub(crate) fn update(&mut self, z: Option<&DetectBox>) {
        self.history_obs.push(z.copied());

        if let Some(z) = z {
            // --- Re-observation after miss: unfreeze ---
            if !self.observed {
                self.unfreeze();
            }
            self.observed = true;
            self.update_inner(z);
        } else {
            // --- No observation ---
            if self.observed {
                self.freeze();
            }
            self.observed = false;
        }
    }

    /// Standard Kalman update step.
    fn update_inner(&mut self, z: &DetectBox) {
        // y = z - H x
        let y = z - self.h * self.x;
        // S = H P H^T + R
        let pht = self.p * self.h.transpose();
        let s = self.h * pht + self.r;
        // K = P H^T S^{-1}
        let s_inv = s.try_inverse().unwrap_or_else(MeasCov::identity);
        let k = pht * s_inv;
        // x = x + K y
        self.x += k * y;
        // P = (I - K H) P (I - K H)^T + K R K^T   (Joseph form)
        let i_kh = StateCov::identity() - k * self.h;
        self.p = i_kh * self.p * i_kh.transpose() + k * self.r * k.transpose();
    }

    // ------------------------------------------------------------------
    // Freeze / Unfreeze  (online smoothing for OC-SORT)
    // ------------------------------------------------------------------

    fn freeze(&mut self) {
        self.attr_saved = Some(SavedState {
            x: self.x,
            p: self.p,
            history_obs: self.history_obs.clone(),
        });
    }

    fn unfreeze(&mut self) {
        let saved = match self.attr_saved.take() {
            Some(s) => s,
            None => return,
        };

        let new_history = self.history_obs.clone();

        // Restore old state
        self.x = saved.x;
        self.p = saved.p;
        self.history_obs = saved.history_obs;
        // Remove the last element (was the first None appended after freeze)
        self.history_obs.pop();

        // Find the last two observed indices in new_history
        let observed_indices: Vec<usize> = new_history
            .iter()
            .enumerate()
            .filter_map(|(i, obs)| obs.as_ref().map(|_| i))
            .collect();

        if observed_indices.len() < 2 {
            return;
        }

        let index1 = observed_indices[observed_indices.len() - 2];
        let index2 = observed_indices[observed_indices.len() - 1];

        let box1 = new_history[index1].unwrap();
        let box2 = new_history[index2].unwrap();

        // box is [x, y, s, r] in measurement space
        let x1 = box1[0];
        let y1 = box1[1];
        let s1 = box1[2];
        let r1 = box1[3];
        let w1 = (s1 * r1).sqrt();
        let h1 = (s1 / r1).sqrt();

        let x2 = box2[0];
        let y2 = box2[1];
        let s2 = box2[2];
        let r2 = box2[3];
        let w2 = (s2 * r2).sqrt();
        let h2 = (s2 / r2).sqrt();

        let time_gap = (index2 - index1) as f32;
        let dx = (x2 - x1) / time_gap;
        let dy = (y2 - y1) / time_gap;
        let dw = (w2 - w1) / time_gap;
        let dh = (h2 - h1) / time_gap;

        for i in 0..(index2 - index1) {
            let t = (i + 1) as f32;
            let x = x1 + t * dx;
            let y = y1 + t * dy;
            let w = w1 + t * dw;
            let h = h1 + t * dh;
            let s = w * h;
            let r = w / h;
            let new_box = DetectBox::new(x, y, s, r);

            self.update_inner(&new_box);
            if i != (index2 - index1 - 1) {
                self.predict();
            }
        }
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    pub(crate) fn state(&self) -> &StateMean {
        &self.x
    }

    pub(crate) fn state_mut(&mut self) -> &mut StateMean {
        &mut self.x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_z(x: f32, y: f32, s: f32, r: f32) -> DetectBox {
        DetectBox::new(x, y, s, r)
    }

    #[test]
    fn test_initial_state() {
        let z = make_z(150.0, 150.0, 10000.0, 1.0);
        let kf = KalmanFilter::new(&z);
        let x = kf.state();
        assert!((x[0] - 150.0).abs() < 1e-5);
        assert!((x[1] - 150.0).abs() < 1e-5);
        assert!((x[2] - 10000.0).abs() < 1e-5);
        assert!((x[3] - 1.0).abs() < 1e-5);
        for i in 4..7 {
            assert!((x[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_predict_no_velocity() {
        let z = make_z(150.0, 150.0, 10000.0, 1.0);
        let mut kf = KalmanFilter::new(&z);
        kf.predict();
        let x = kf.state();
        assert!((x[0] - 150.0).abs() < 1e-5);
        assert!((x[1] - 150.0).abs() < 1e-5);
        assert!((x[2] - 10000.0).abs() < 1e-5);
    }

    #[test]
    fn test_predict_update_cycle() {
        let z_init = make_z(150.0, 150.0, 10000.0, 1.0);
        let mut kf = KalmanFilter::new(&z_init);
        kf.predict();

        let z_update = make_z(160.0, 160.0, 10000.0, 1.0);
        kf.update(Some(&z_update));

        let x = kf.state();
        // Python: x = [159.999, 159.999, 10000, 1.0, 9.988, 9.988, 0.0]
        assert!((x[0] - 159.999).abs() < 0.01);
        assert!((x[1] - 159.999).abs() < 0.01);
        assert!((x[4] - 9.988).abs() < 0.1);
    }

    #[test]
    fn test_p_diagonal() {
        let z = make_z(150.0, 150.0, 10000.0, 1.0);
        let kf = KalmanFilter::new(&z);
        let p = &kf.p;
        assert!((p[(0, 0)] - 10.0).abs() < 1e-5);
        assert!((p[(4, 4)] - 10000.0).abs() < 1e-5);
        assert!((p[(6, 6)] - 10000.0).abs() < 1e-5);
    }

    #[test]
    fn test_update_none_does_not_change_state() {
        let z = make_z(150.0, 150.0, 10000.0, 1.0);
        let mut kf = KalmanFilter::new(&z);
        kf.predict();
        let x_before = *kf.state();
        kf.update(None);
        let x_after = *kf.state();
        assert_eq!(x_before, x_after);
    }
}
