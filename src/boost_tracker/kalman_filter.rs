use nalgebra::SMatrix;

/* -----------------------------------------------------------------------------
 * Type aliases
 * ----------------------------------------------------------------------------- */
// 1x4
pub(crate) type DetectBox = SMatrix<f32, 1, 4>;
// 1x8
pub(crate) type StateMean = SMatrix<f32, 1, 8>;
// 8x8
pub(crate) type StateCov = SMatrix<f32, 8, 8>;
// 1x4
pub(crate) type StateHMean = SMatrix<f32, 1, 4>;
// 4x4
pub(crate) type StateHCov = SMatrix<f32, 4, 4>;

/* -----------------------------------------------------------------------------
 * Covariance policy
 * ----------------------------------------------------------------------------- */
pub(crate) trait CovariancePolicy {
    fn init_state_cov(&self, z: &DetectBox) -> StateCov;
    fn r(&self, x: &StateMean, confidence: f32) -> StateHCov;
    fn q(&self, x: &StateMean) -> StateCov;
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ConstantNoise;

impl CovariancePolicy for ConstantNoise {
    fn init_state_cov(&self, _z: &DetectBox) -> StateCov {
        let mut p = StateCov::identity();
        for i in 4..8 {
            p[(i, i)] *= 1000.0;
        }
        p *= 10.0;
        p
    }

    fn r(&self, _x: &StateMean, _confidence: f32) -> StateHCov {
        StateHCov::from_diagonal(
            &SMatrix::<f32, 1, 4>::from_iterator([1.0, 1.0, 10.0, 0.01])
                .transpose(),
        )
    }

    fn q(&self, _x: &StateMean) -> StateCov {
        let mut q = StateCov::identity();
        for i in 4..8 {
            q[(i, i)] *= 0.01;
        }
        q
    }
}

/* -----------------------------------------------------------------------------
 * Kalman Filter
 * ----------------------------------------------------------------------------- */
pub(crate) struct KalmanFilter {
    motion_mat: StateCov,
    update_mat: SMatrix<f32, 4, 8>,
    x: StateMean,
    covariance: StateCov,
    cov_policy: Box<dyn CovariancePolicy>,
}

impl KalmanFilter {
    pub(crate) fn new(z: &DetectBox) -> Self {
        Self::with_policy(z, Box::new(ConstantNoise::default()), 1.0)
    }

    pub(crate) fn with_policy(
        z: &DetectBox,
        cov_policy: Box<dyn CovariancePolicy>,
        dt: f32,
    ) -> Self {
        let mut motion_mat = StateCov::identity();
        for i in 0..4 {
            motion_mat[(i, i + 4)] = dt;
        }

        let mut update_mat = SMatrix::<f32, 4, 8>::zeros();
        update_mat[(0, 0)] = 1.0;
        update_mat[(1, 1)] = 1.0;
        update_mat[(2, 2)] = 1.0;
        update_mat[(3, 3)] = 1.0;

        let mut x = StateMean::zeros();
        x.as_mut_slice()[0..4].copy_from_slice(z.as_slice());

        let covariance = cov_policy.init_state_cov(z);

        Self {
            motion_mat,
            update_mat,
            x,
            covariance,
            cov_policy,
        }
    }

    pub(crate) fn predict(&mut self) -> (StateMean, StateCov) {
        let motion_cov = self.cov_policy.q(&self.x);
        self.x = (&self.motion_mat * self.x.transpose()).transpose();
        self.covariance =
            self.motion_mat * self.covariance * self.motion_mat.transpose()
                + motion_cov;

        (self.x, self.covariance)
    }

    pub(crate) fn project(&self, confidence: f32) -> (StateHMean, StateHCov) {
        let innovation_cov = self.cov_policy.r(&self.x, confidence);
        let mean = self.x * self.update_mat.transpose();
        let covariance =
            self.update_mat * self.covariance * self.update_mat.transpose();

        (mean, covariance + innovation_cov)
    }

    pub(crate) fn update(
        &mut self,
        measurement: &DetectBox,
        confidence: f32,
    ) -> (StateMean, StateCov) {
        let (projected_mean, projected_covariance) = self.project(confidence);
        let innovation_cov = self.cov_policy.r(&self.x, confidence);

        let b = (self.covariance * self.update_mat.transpose()).transpose();
        let cholesky_factor = projected_covariance.cholesky().unwrap();
        let kalman_gain = cholesky_factor.solve(&b);
        let innovation = measurement - projected_mean;
        self.x += innovation * kalman_gain;
        // Joseph form is numerically more stable than P -= K S K^T in f32.
        let k = kalman_gain.transpose(); // 8x4
        let identity = StateCov::identity();
        let i_minus_kh = identity - k * self.update_mat;
        self.covariance = i_minus_kh * self.covariance * i_minus_kh.transpose()
            + k * innovation_cov * k.transpose();

        (self.x, self.covariance)
    }

    pub(crate) fn state(&self) -> &StateMean {
        &self.x
    }

    pub(crate) fn covariance(&self) -> &StateCov {
        &self.covariance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nearly_eq::assert_nearly_eq;

    fn bbox_to_z(bbox: [f32; 4]) -> SMatrix<f32, 1, 4> {
        let (x1, y1, x2, y2) = (bbox[0], bbox[1], bbox[2], bbox[3]);
        let w = x2 - x1;
        let h = y2 - y1;
        let x = x1 + w / 2.0;
        let y = y1 + h / 2.0;
        let r = w / (h + 1e-6);
        SMatrix::<f32, 1, 4>::from_iterator([x, y, h, r])
    }

    #[test]
    fn test_init_state_cov() {
        let measurement =
            SMatrix::<f32, 1, 4>::from_iterator([1.0, 2.0, 3.0, 4.0]);
        let kf = KalmanFilter::new(&measurement);
        let cov = kf.covariance();

        for i in 0..4 {
            assert_nearly_eq!(cov[(i, i)], 10.0, 1e-6);
        }
        for i in 4..8 {
            assert_nearly_eq!(cov[(i, i)], 10000.0, 1e-6);
        }
    }

    #[test]
    fn test_predict() {
        let measurement = bbox_to_z([1.0, 2.0, 3.0, 4.0]);
        let mut kf = KalmanFilter::new(&measurement);
        let (predicted_mean, predicted_cov) = kf.predict();

        let expected_mean = SMatrix::<f32, 1, 8>::from_iterator([
            2.0, 3.0, 2.0, 0.9999995, 0.0, 0.0, 0.0, 0.0,
        ]);
        for (i, &v) in predicted_mean.iter().enumerate() {
            assert_nearly_eq!(v, expected_mean.iter().nth(i).unwrap(), 1e-6)
        }

        #[rustfmt::skip]
        let expected_cov = SMatrix::<f32, 8, 8>::from_iterator([
            10011.0, 0.0,    0.0,    0.0,    10000.0, 0.0,     0.0,     0.0,
            0.0,    10011.0, 0.0,    0.0,    0.0,     10000.0, 0.0,     0.0,
            0.0,    0.0,    10011.0, 0.0,    0.0,     0.0,     10000.0, 0.0,
            0.0,    0.0,    0.0,    10011.0, 0.0,     0.0,     0.0,     10000.0,
            10000.0, 0.0,    0.0,    0.0,    10000.01, 0.0,     0.0,     0.0,
            0.0,    10000.0, 0.0,    0.0,    0.0,     10000.01, 0.0,     0.0,
            0.0,    0.0,    10000.0, 0.0,    0.0,     0.0,     10000.01, 0.0,
            0.0,    0.0,    0.0,    10000.0, 0.0,     0.0,     0.0,     10000.01,
        ]);
        for (i, &v) in predicted_cov.iter().enumerate() {
            assert_nearly_eq!(v, expected_cov.iter().nth(i).unwrap(), 1e-6)
        }
    }

    #[test]
    fn test_project() {
        let measurement = bbox_to_z([1.0, 2.0, 3.0, 4.0]);
        let mut kf = KalmanFilter::new(&measurement);
        let _ = kf.predict();
        let (projected_mean, projected_cov) = kf.project(0.9);

        let expected_mean =
            SMatrix::<f32, 1, 4>::from_iterator([2.0, 3.0, 2.0, 0.9999995]);
        for (i, &v) in projected_mean.iter().enumerate() {
            assert_nearly_eq!(v, expected_mean.iter().nth(i).unwrap(), 1e-6)
        }

        #[rustfmt::skip]
        let expected_cov = SMatrix::<f32, 4, 4>::from_iterator([
            10012.0, 0.0,     0.0,     0.0,
            0.0,     10012.0, 0.0,     0.0,
            0.0,     0.0,     10021.0, 0.0,
            0.0,     0.0,     0.0,     10011.01,
        ]);
        for (i, &v) in projected_cov.iter().enumerate() {
            assert_nearly_eq!(v, expected_cov.iter().nth(i).unwrap(), 1e-6)
        }
    }

    #[test]
    fn test_update() {
        let measurement = bbox_to_z([1.0, 2.0, 3.0, 4.0]);
        let mut kf = KalmanFilter::new(&measurement);
        let _ = kf.predict();
        let _ = kf.project(0.9);

        // Updated bbox
        let updated_bbox = bbox_to_z([12., 22., 31., 62.]);
        let (updated_mean, updated_cov) = kf.update(&updated_bbox, 0.9);
        let expected_mean = SMatrix::<f32, 1, 8>::from_iterator([
            21.498052337195357,
            41.996104674390715,
            39.96207963277118,
            0.4750005125471235,
            19.47662804634438,
            38.95325609268876,
            37.92036722881948,
            -0.524422123117697,
        ]);
        for (i, &v) in updated_mean.iter().enumerate() {
            assert_nearly_eq!(v, expected_mean.iter().nth(i).unwrap(), 1e-4)
        }

        #[rustfmt::skip]
        let expected_cov = SMatrix::<f32, 8, 8>::from_iterator([
            0.9999001, 0.0,       0.0,       0.0,       0.9988014, 0.0,       0.0,       0.0,
            0.0,       0.9999001, 0.0,       0.0,       0.0,       0.9988014, 0.0,       0.0,
            0.0,       0.0,       9.990021,  0.0,       0.0,       0.0,       9.979044,  0.0,
            0.0,       0.0,       0.0,       0.00999999, 0.0,      0.0,       0.0,       0.009989002,
            0.9988014, 0.0,       0.0,       0.0,       11.995383, 0.0,       0.0,       0.0,
            0.0,       0.9988014, 0.0,       0.0,       0.0,       11.995383, 0.0,       0.0,
            0.0,       0.0,       9.979044,  0.0,       0.0,       0.0,       20.965757, 0.0,
            0.0,       0.0,       0.0,       0.009989002, 0.0,     0.0,       0.0,       11.007657,
        ]);
        for (i, &v) in updated_cov.iter().enumerate() {
            assert_nearly_eq!(v, expected_cov.iter().nth(i).unwrap(), 1e-3);
        }
    }

    #[test]
    fn test_project_shapes() {
        let measurement = bbox_to_z([1.0, 2.0, 3.0, 4.0]);
        let kf = KalmanFilter::new(&measurement);
        let (projected_mean, projected_cov) = kf.project(0.0);
        assert_eq!(projected_mean.nrows(), 1);
        assert_eq!(projected_mean.ncols(), 4);
        assert_eq!(projected_cov.nrows(), 4);
        assert_eq!(projected_cov.ncols(), 4);
    }

    #[test]
    fn test_predict_matches_bbox_to_z() {
        let z = bbox_to_z([1.0, 2.0, 3.0, 4.0]);
        let mut kf = KalmanFilter::new(&z);
        let (mean, _) = kf.predict();

        assert_nearly_eq!(mean[(0, 0)], 2.0, 1e-6);
        assert_nearly_eq!(mean[(0, 1)], 3.0, 1e-6);
        assert_nearly_eq!(mean[(0, 2)], 2.0, 1e-6);
        assert_nearly_eq!(mean[(0, 3)], 0.9999995, 1e-6);
    }
}
