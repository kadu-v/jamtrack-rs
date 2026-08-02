use nalgebra::{SMatrix, SVector};

pub(crate) type Measurement = SVector<f64, 4>;
pub(crate) type StateMean = SVector<f64, 8>;
pub(crate) type StateCov = SMatrix<f64, 8, 8>;
pub(crate) type ProjectedCov = SMatrix<f64, 4, 4>;

#[derive(Debug, Clone)]
pub(crate) struct KalmanFilter {
    motion_mat: SMatrix<f64, 8, 8>,
    update_mat: SMatrix<f64, 4, 8>,
    std_weight_position: f64,
    std_weight_velocity: f64,
}

impl Default for KalmanFilter {
    fn default() -> Self {
        let mut motion_mat = SMatrix::<f64, 8, 8>::identity();
        for i in 0..4 {
            motion_mat[(i, i + 4)] = 1.0;
        }
        let mut update_mat = SMatrix::<f64, 4, 8>::zeros();
        for i in 0..4 {
            update_mat[(i, i)] = 1.0;
        }
        Self {
            motion_mat,
            update_mat,
            std_weight_position: 1.0 / 20.0,
            std_weight_velocity: 1.0 / 160.0,
        }
    }
}

impl KalmanFilter {
    pub(crate) fn initiate(
        &self,
        measurement: &Measurement,
    ) -> (StateMean, StateCov) {
        let mut mean = StateMean::zeros();
        mean.fixed_rows_mut::<4>(0).copy_from(measurement);

        let height = measurement[3];
        let std = StateMean::from_row_slice(&[
            2.0 * self.std_weight_position * height,
            2.0 * self.std_weight_position * height,
            1e-2,
            2.0 * self.std_weight_position * height,
            10.0 * self.std_weight_velocity * height,
            10.0 * self.std_weight_velocity * height,
            1e-5,
            10.0 * self.std_weight_velocity * height,
        ]);
        let covariance = StateCov::from_diagonal(&std.component_mul(&std));
        (mean, covariance)
    }

    pub(crate) fn predict(
        &self,
        mean: &StateMean,
        covariance: &StateCov,
    ) -> (StateMean, StateCov) {
        let height = mean[3];
        let std = StateMean::from_row_slice(&[
            self.std_weight_position * height,
            self.std_weight_position * height,
            1e-2,
            self.std_weight_position * height,
            self.std_weight_velocity * height,
            self.std_weight_velocity * height,
            1e-5,
            self.std_weight_velocity * height,
        ]);
        let motion_cov = StateCov::from_diagonal(&std.component_mul(&std));
        let next_mean = self.motion_mat * mean;
        let next_cov =
            self.motion_mat * covariance * self.motion_mat.transpose()
                + motion_cov;
        (next_mean, next_cov)
    }

    pub(crate) fn project(
        &self,
        mean: &StateMean,
        covariance: &StateCov,
    ) -> (Measurement, ProjectedCov) {
        let height = mean[3];
        let std = Measurement::from_row_slice(&[
            self.std_weight_position * height,
            self.std_weight_position * height,
            1e-1,
            self.std_weight_position * height,
        ]);
        let innovation_cov =
            ProjectedCov::from_diagonal(&std.component_mul(&std));
        let projected_mean = self.update_mat * mean;
        let projected_cov =
            self.update_mat * covariance * self.update_mat.transpose()
                + innovation_cov;
        (projected_mean, projected_cov)
    }

    pub(crate) fn update(
        &self,
        mean: &StateMean,
        covariance: &StateCov,
        measurement: &Measurement,
    ) -> (StateMean, StateCov) {
        let (projected_mean, projected_cov) = self.project(mean, covariance);
        let rhs = (covariance * self.update_mat.transpose()).transpose();
        let kalman_gain_t = projected_cov
            .cholesky()
            .expect("FastTracker covariance must be positive definite")
            .solve(&rhs);
        let kalman_gain = kalman_gain_t.transpose();
        let innovation = measurement - projected_mean;
        let new_mean = mean + kalman_gain * innovation;
        let new_cov =
            covariance - kalman_gain * projected_cov * kalman_gain.transpose();
        (new_mean, new_cov)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= tolerance,
                "index {index}: actual={a}, expected={e}"
            );
        }
    }

    #[test]
    fn initiate_sets_expected_state_and_covariance() {
        let kf = KalmanFilter::default();
        let measurement = Measurement::from_row_slice(&[10.0, 20.0, 0.5, 40.0]);
        let (mean, covariance) = kf.initiate(&measurement);
        assert_eq!(
            mean.as_slice(),
            &[10.0, 20.0, 0.5, 40.0, 0.0, 0.0, 0.0, 0.0]
        );
        assert_close(
            covariance.diagonal().as_slice(),
            &[16.0, 16.0, 0.0001, 16.0, 6.25, 6.25, 1e-10, 6.25],
            1e-12,
        );
    }

    #[test]
    fn predict_project_and_update_produce_expected_values() {
        let kf = KalmanFilter::default();
        let measurement = Measurement::from_row_slice(&[10.0, 20.0, 0.5, 40.0]);
        let (mean, covariance) = kf.initiate(&measurement);
        let (predicted_mean, predicted_cov) = kf.predict(&mean, &covariance);
        let (projected_mean, projected_cov) =
            kf.project(&predicted_mean, &predicted_cov);
        assert_close(projected_mean.as_slice(), measurement.as_slice(), 1e-12);
        assert_close(
            projected_cov.diagonal().as_slice(),
            &[30.25, 30.25, 0.0102000001, 30.25],
            1e-9,
        );

        let next = Measurement::from_row_slice(&[12.0, 23.0, 0.55, 42.0]);
        let (updated_mean, updated_cov) =
            kf.update(&predicted_mean, &predicted_cov, &next);
        assert_close(
            updated_mean.as_slice(),
            &[
                11.735537190082646,
                22.60330578512397,
                0.5009803926374471,
                41.735537190082646,
                0.41322314049586784,
                0.6198347107438018,
                4.901960736255291e-10,
                0.41322314049586784,
            ],
            1e-9,
        );
        assert!(updated_cov.iter().all(|value| value.is_finite()));
    }
}
