use nalgebra::SMatrix;

pub(crate) type DetectBox = SMatrix<f32, 1, 4>;
pub(crate) type StateMean = SMatrix<f32, 1, 8>;
pub(crate) type StateCov = SMatrix<f32, 8, 8>;
pub(crate) type StateHMean = SMatrix<f32, 1, 4>;
pub(crate) type StateHCov = SMatrix<f32, 4, 4>;

#[derive(Debug, Clone)]
pub(crate) struct KalmanFilter {
    std_weight_position: f32,
    std_weight_velocity: f32,
    motion_mat: SMatrix<f32, 8, 8>,
    update_mat: SMatrix<f32, 4, 8>,
}

impl Default for KalmanFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl KalmanFilter {
    pub(crate) fn new() -> Self {
        let ndim = 4;
        let dt = 1.0;
        let mut motion_mat = SMatrix::<f32, 8, 8>::identity();
        let mut update_mat = SMatrix::<f32, 4, 8>::zeros();

        for i in 0..ndim {
            motion_mat[(i, ndim + i)] = dt;
            update_mat[(i, i)] = 1.0;
        }

        Self {
            std_weight_position: 1.0 / 20.0,
            std_weight_velocity: 1.0 / 160.0,
            motion_mat,
            update_mat,
        }
    }

    pub(crate) fn initiate(
        &self,
        mean: &mut StateMean,
        covariance: &mut StateCov,
        measurement: &DetectBox,
    ) {
        mean.as_mut_slice()[0..4].copy_from_slice(measurement.as_slice());
        mean.as_mut_slice()[4..8].fill(0.0);

        let w = measurement[(0, 2)].max(1e-6);
        let h = measurement[(0, 3)].max(1e-6);
        let std = SMatrix::<f32, 1, 8>::from_iterator([
            2.0 * self.std_weight_position * w,
            2.0 * self.std_weight_position * h,
            2.0 * self.std_weight_position * w,
            2.0 * self.std_weight_position * h,
            10.0 * self.std_weight_velocity * w,
            10.0 * self.std_weight_velocity * h,
            10.0 * self.std_weight_velocity * w,
            10.0 * self.std_weight_velocity * h,
        ]);
        *covariance = SMatrix::<f32, 8, 8>::from_diagonal(
            &std.component_mul(&std).transpose(),
        );
    }

    pub(crate) fn predict(
        &self,
        mean: &mut StateMean,
        covariance: &mut StateCov,
    ) {
        let w = mean[(0, 2)].max(1e-6);
        let h = mean[(0, 3)].max(1e-6);
        let std = SMatrix::<f32, 1, 8>::from_iterator([
            self.std_weight_position * w,
            self.std_weight_position * h,
            self.std_weight_position * w,
            self.std_weight_position * h,
            self.std_weight_velocity * w,
            self.std_weight_velocity * h,
            self.std_weight_velocity * w,
            self.std_weight_velocity * h,
        ]);
        let motion_cov = SMatrix::<f32, 8, 8>::from_diagonal(
            &std.component_mul(&std).transpose(),
        );

        *mean = (self.motion_mat * mean.transpose()).transpose();
        *covariance =
            self.motion_mat * *covariance * self.motion_mat.transpose()
                + motion_cov;
    }

    pub(crate) fn update(
        &self,
        mean: &mut StateMean,
        covariance: &mut StateCov,
        measurement: &DetectBox,
    ) {
        let (projected_mean, projected_covariance) =
            self.project(mean, covariance);
        let b = (*covariance * self.update_mat.transpose()).transpose();
        let Some(cholesky) = projected_covariance.cholesky() else {
            return;
        };
        let kalman_gain = cholesky.solve(&b);
        let innovation = measurement - projected_mean;

        *mean += innovation * kalman_gain;
        *covariance -=
            kalman_gain.transpose() * projected_covariance * kalman_gain;
    }

    pub(crate) fn project(
        &self,
        mean: &StateMean,
        covariance: &StateCov,
    ) -> (StateHMean, StateHCov) {
        let w = mean[(0, 2)].max(1e-6);
        let h = mean[(0, 3)].max(1e-6);
        let std = SMatrix::<f32, 1, 4>::from_iterator([
            self.std_weight_position * w,
            self.std_weight_position * h,
            self.std_weight_position * w,
            self.std_weight_position * h,
        ]);

        let projected_mean = mean * self.update_mat.transpose();
        let innovation_cov = SMatrix::<f32, 4, 4>::from_diagonal(
            &std.component_mul(&std).transpose(),
        );
        let projected_covariance =
            self.update_mat * covariance * self.update_mat.transpose()
                + innovation_cov;

        (projected_mean, projected_covariance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initiate_uses_xywh_state() {
        let kf = KalmanFilter::new();
        let mut mean = StateMean::zeros();
        let mut covariance = StateCov::zeros();
        let measurement = DetectBox::from_iterator([10.0, 20.0, 30.0, 40.0]);

        kf.initiate(&mut mean, &mut covariance, &measurement);

        assert_eq!(mean.as_slice()[0..4], [10.0, 20.0, 30.0, 40.0]);
        assert_eq!(mean.as_slice()[4..8], [0.0, 0.0, 0.0, 0.0]);
        assert!(covariance[(0, 0)] > 0.0);
    }

    #[test]
    fn predict_advances_by_velocity() {
        let kf = KalmanFilter::new();
        let mut mean = StateMean::from_iterator([
            10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0,
        ]);
        let mut covariance = StateCov::identity();

        kf.predict(&mut mean, &mut covariance);

        assert_eq!(mean[(0, 0)], 11.0);
        assert_eq!(mean[(0, 1)], 22.0);
        assert_eq!(mean[(0, 2)], 33.0);
        assert_eq!(mean[(0, 3)], 44.0);
    }
}
