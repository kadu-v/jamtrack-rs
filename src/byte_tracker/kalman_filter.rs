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
 * Kalman Filter
 * ----------------------------------------------------------------------------- */
#[derive(Debug, Clone)]
pub(crate) struct KalmanFilter {
    std_weight_position: f32,
    std_weight_velocity: f32,
    motion_mat: SMatrix<f32, 8, 8>, // 8x8
    update_mat: SMatrix<f32, 4, 8>, // 4x8
}

impl KalmanFilter {
    pub(crate) fn new(
        std_weight_position: f32,
        std_weight_velocity: f32,
    ) -> Self {
        let ndim = 4;
        let dt = 1.0;

        let mut motion_mat = SMatrix::<f32, 8, 8>::identity();

        // 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        // 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        // 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        // 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        let mut update_mat = SMatrix::<f32, 4, 8>::zeros();
        update_mat[(0, 0)] = 1.0;
        update_mat[(1, 1)] = 1.0;
        update_mat[(2, 2)] = 1.0;
        update_mat[(3, 3)] = 1.0;

        for i in 0..ndim {
            motion_mat[(i, i + ndim)] = dt;
        }

        return Self {
            std_weight_position,
            std_weight_velocity,
            motion_mat,
            update_mat,
        };
    }

    pub(crate) fn initiate(
        &self,
        mean: &mut StateMean,
        covariance: &mut StateCov,
        measurement: &DetectBox,
    ) {
        let mean_vel = SMatrix::<f32, 1, 4>::zeros();
        let mean_pos = measurement;
        mean.as_mut_slice()[0..4].copy_from_slice(mean_pos.as_slice());
        mean.as_mut_slice()[4..8].copy_from_slice(mean_vel.as_slice());

        let mut std = SMatrix::<f32, 1, 8>::zeros();
        let mesure_val = measurement[(0, 3)];
        std[0] = 2.0 * self.std_weight_position * mesure_val;
        std[1] = 2.0 * self.std_weight_position * mesure_val;
        std[2] = 1e-2;
        std[3] = 2.0 * self.std_weight_position * mesure_val;
        std[4] = 10.0 * self.std_weight_velocity * mesure_val;
        std[5] = 10.0 * self.std_weight_velocity * mesure_val;
        std[6] = 1e-5;
        std[7] = 10.0 * self.std_weight_velocity * mesure_val;

        let tmp = std.component_mul(&std);
        // convert 1-d array to 2-d array that has diagonal values of 1-d array
        *covariance = SMatrix::<f32, 8, 8>::from_diagonal(&tmp.transpose());
    }

    pub(crate) fn predict(
        &mut self,
        mean: &mut StateMean,
        covariance: &mut StateCov,
    ) {
        let mut std = SMatrix::<f32, 1, 8>::zeros();
        std[0] = self.std_weight_position * mean[(0, 3)];
        std[1] = self.std_weight_position * mean[(0, 3)];
        std[2] = 1e-2;
        std[3] = self.std_weight_position * mean[(0, 3)];
        std[4] = self.std_weight_velocity * mean[(0, 3)];
        std[5] = self.std_weight_velocity * mean[(0, 3)];
        std[6] = 1e-5;
        std[7] = self.std_weight_velocity * mean[(0, 3)];

        let tmp = std.component_mul(&std);
        let motion_cov = SMatrix::<f32, 8, 8>::from_diagonal(&tmp.transpose());
        *mean = (&self.motion_mat * mean.transpose()).transpose();

        let tmp = self.motion_mat * *covariance * self.motion_mat.transpose();
        *covariance = tmp + motion_cov;
    }

    pub(crate) fn update(
        &mut self,
        mean: &mut StateMean,      // 1x8
        covariance: &mut StateCov, // 8x8
        measurement: &DetectBox,   // 1x4
    ) {
        let mut projected_mean = SMatrix::<f32, 1, 4>::zeros();
        let mut projected_covariance = SMatrix::<f32, 4, 4>::zeros();
        self.project(
            &mut projected_mean,
            &mut projected_covariance,
            &mean,
            &covariance,
        );

        let b = (*covariance * self.update_mat.transpose()).transpose();
        let choleskey_factor = projected_covariance.cholesky().unwrap();
        // kalman_gain: 8x4
        let kalman_gain = choleskey_factor.solve(&b);
        // innovation: 1x4
        let innovation = measurement - &projected_mean;
        // tmp: 1x8
        let tmp = innovation * &kalman_gain;
        *mean += &tmp;
        *covariance -=
            kalman_gain.transpose() * projected_covariance * kalman_gain;
    }

    pub(crate) fn project(
        &self,
        projected_mean: &mut StateHMean, // 1x4
        projected_covariance: &mut StateHCov, // 4x4
        mean: &StateMean,                // 1x8
        covariance: &StateCov,           // 8x8
    ) {
        let std = SMatrix::<f32, 1, 4>::from_iterator([
            self.std_weight_position * mean[(0, 3)],
            self.std_weight_position * mean[(0, 3)],
            1e-1,
            self.std_weight_position * mean[(0, 3)],
        ]);

        // update_mat: 4x8, mean: 1x8
        // projected_mean: 4x1
        let tmp = mean * self.update_mat.transpose();
        *projected_mean = tmp;

        // 4x4
        let diag = SMatrix::<f32, 4, 4>::from_diagonal(&std.transpose());
        let innovation_cov = diag.component_mul(&diag);
        let cov = self.update_mat * covariance * self.update_mat.transpose();
        *projected_covariance = cov + innovation_cov;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{self, SMatrix};
    use nearly_eq::assert_nearly_eq;

    #[test]
    fn test_initiate() {
        let kalman_filter = KalmanFilter::new(1. / 20., 1. / 160.);
        let mut mean = SMatrix::<f32, 1, 8>::zeros();
        let mut covariance = SMatrix::<f32, 8, 8>::zeros();
        let measurement =
            SMatrix::<f32, 1, 4>::from_iterator(vec![1.0, 2.0, 3.0, 4.0]);

        kalman_filter.initiate(&mut mean, &mut covariance, &measurement);

        // Assert the values of mean and covariance after initiation
        let expected = SMatrix::<f32, 1, 8>::from_iterator(vec![
            1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0,
        ]);
        assert_eq!(mean, expected);
        #[rustfmt::skip]
    let expected = SMatrix::<f32, 8, 8>::from_iterator([
        0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.16, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 6.25e-2, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 6.25e-2, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-10, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.25e-2,
    ]);
        for (i, &v) in covariance.iter().enumerate() {
            assert_nearly_eq!(v, expected.iter().nth(i).unwrap(), 1e-4)
        }
    }

    #[test]
    fn test_predict() {
        let mut kalman_filter = KalmanFilter::new(1. / 20., 1. / 160.);
        let mut mean = SMatrix::<f32, 1, 8>::from_iterator([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        ]);
        #[rustfmt::skip]
    let mut covariance = SMatrix::<f32, 8, 8>::from_iterator([
        0.2, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,      0.0, 
        0.0, 0.2, 0.0,  0.0, 0.0, 0.0, 0.0,      0.0, 
        0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0,      0.0, 
        0.0, 0.0, 0.0,  0.2, 0.0, 0.0, 0.0,      0.0, 
        0.0, 0.0, 0.0,  0.0, 4.0, 0.0, 0.0,      0.0, 
        0.0, 0.0, 0.0,  0.0, 0.0, 4.0, 0.0,      0.0, 
        0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.000001, 0.0, 
        0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,      4.0,
    ]);

        kalman_filter.predict(&mut mean, &mut covariance);

        // Assert the values of mean and covariance after prediction
        assert_eq!(
            mean,
            SMatrix::<f32, 1, 8>::from_iterator([
                6.0, 8.0, 10.0, 12.0, 5.0, 6.0, 7.0, 8.0
            ])
        );
        #[rustfmt::skip]
    let expected = SMatrix::<f32, 8, 8>::from_iterator([
        4.24, 0.0,  0.0,     0.0,  4.0,      0.0,      0.0,   0.0,
        0.0,  4.24, 0.0,     0.0,  0.0,      4.0,      0.0,   0.0,
        0.0,  0.0,  1.01e-2, 0.0,  0.0,      0.0,      1.0e-6, 0.0,
        0.0,  0.0,  0.0,     4.24, 0.0,      0.0,      0.0,    4.0,
        4.0,  0.0,  0.0,     0.0,  4.000625, 0.0,      0.0,    0.0,
        0.0,  4.0,  0.0,     0.0,  0.0,      4.000625, 0.0,    0.0,
        0.0,  0.0,  1.0e-6,  0.0,  0.0,      0.0,      1.0e-6, 0.0,
        0.0,  0.0,  0.0,     4.0,  0.0,      0.0,      0.0,    4.000625,
    ]);
        for (i, &v) in covariance.iter().enumerate() {
            assert_nearly_eq!(v, expected.iter().nth(i).unwrap(), 1e-4)
        }
    }

    #[test]
    fn test_project() {
        let kalman_filter = KalmanFilter::new(1. / 20., 1. / 160.);
        let mean = SMatrix::<f32, 1, 8>::from_iterator([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        ]);
        #[rustfmt::skip]
    let covariance = SMatrix::<f32, 8, 8>::from_iterator([
        4.24, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0      ,
        0.0, 4.24, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0      ,
        0.0, 0.0, 1.01e-2, 0.0, 0.0, 0.0, 1.0e-6, 0.0,
        0.0, 0.0, 0.0, 4.24, 0.0, 0.0, 0.0, 4.0      ,
        4.0, 0.0, 0.0, 0.0, 4.000625, 0.0, 0.0, 0.0  ,
        0.0, 4.0, 0.0, 0.0, 0.0, 4.000625, 0.0, 0.0  ,
        0.0, 0.0, 1.0e-6, 0.0, 0.0, 0.0, 1.0e-6, 0.0 ,
        0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.000625  ,
    ]);
        let mut projected_mean = SMatrix::<f32, 1, 4>::zeros();
        let mut projected_covariance = SMatrix::<f32, 4, 4>::zeros();

        kalman_filter.project(
            &mut projected_mean,
            &mut projected_covariance,
            &mean,
            &covariance,
        );

        assert_eq!(
            projected_mean,
            SMatrix::<f32, 1, 4>::from_iterator([1., 2., 3., 4.])
        );
        #[rustfmt::skip]
    let expected = SMatrix::<f32, 4, 4>::from_iterator([
        4.28,   0.,     0.,     0.    ,
        0.,     4.28,   0.,     0.    ,
        0.,     0.,     0.0201, 0.    ,
        0.,     0.,     0.,     4.28  ]);
        for (i, &v) in projected_covariance.iter().enumerate() {
            assert_nearly_eq!(v, expected.iter().nth(i).unwrap(), 1e-4)
        }
    }

    #[test]
    fn test_update() {
        let mut kalman_filter = KalmanFilter::new(1. / 20., 1. / 160.);
        let mut mean = SMatrix::<f32, 1, 8>::from_iterator([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        ]);
        #[rustfmt::skip]
    let mut covariance = SMatrix::<f32, 8, 8>::from_iterator([
        4.24, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0      ,
        0.0, 4.24, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0      ,
        0.0, 0.0, 1.01e-2, 0.0, 0.0, 0.0, 1.0e-6, 0.0,
        0.0, 0.0, 0.0, 4.24, 0.0, 0.0, 0.0, 4.0      ,
        4.0, 0.0, 0.0, 0.0, 4.000625, 0.0, 0.0, 0.0  ,
        0.0, 4.0, 0.0, 0.0, 0.0, 4.000625, 0.0, 0.0  ,
        0.0, 0.0, 1.0e-6, 0.0, 0.0, 0.0, 1.0e-6, 0.0 ,
        0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.000625  ,
    ]);

        let measurement =
            SMatrix::<f32, 1, 4>::from_iterator([1.0, 2.0, 3.0, 4.0]);
        kalman_filter.update(&mut mean, &mut covariance, &measurement);

        // Assert the values of mean and covariance after update
        assert_eq!(
            mean,
            SMatrix::<f32, 1, 8>::from_iterator([
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0
            ])
        );
        #[rustfmt::skip]
    let expected = SMatrix::<f32, 8, 8>::from_iterator([
       3.96261682e-02, 0.0, 0.0, 0.0,3.73831776e-02, 0.0, 0.0, 0.0 ,
       0.0, 3.96261682e-02, 0.0, 0.0, 0.0, 3.73831776e-02, 0.0, 0.0,
       0.0, 0.0, 5.02487562e-03, 0.0, 0.0, 0.0, 4.97512438e-07, 0.0,
       0.0, 0.0, 0.0, 3.96261682e-02, 0.0, 0.0, 0.0, 3.73831776e-02,
       3.73831776e-02, 0.0, 0.0, 0.0, 2.62307243e-01, 0.0, 0.0, 0.0,
       0.0, 3.73831776e-02, 0.0, 0.0, 0.0, 2.62307243e-01, 0.0, 0.0,
       0.0, 0.0, 4.97512438e-07, 0.0, 0.0, 0.0, 9.99950249e-07, 0.0,
       0.0, 0.0, 0.0, 3.73831776e-02, 0.0, 0.0, 0.0, 2.62307243e-01]);
        for (i, &v) in covariance.iter().enumerate() {
            assert_nearly_eq!(v, expected.iter().nth(i).unwrap(), 1e-4)
        }
    }

    #[test]
    fn test_complex_predict() {
        let mut kalman_filter = KalmanFilter::new(1. / 20., 1. / 160.);
        let expected_mean = SMatrix::<f32, 1, 8>::from_iterator([
            1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0,
        ]);
        #[rustfmt::skip]
    let expected_covariance = SMatrix::<f32, 8, 8>::from_iterator([
        8.4031250e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 7.2187500e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 8.4031250e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 7.2187500e-01, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 1.2000506e-03, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 6.6000000e-09, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 8.4031250e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 7.2187500e-01,
        7.2187500e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 6.9375000e-02, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 7.2187500e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 6.9375000e-02, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 6.6000000e-09, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.2000000e-09, 0.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 7.2187500e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 6.9375000e-02,
        ]);
        let mut mean = SMatrix::<f32, 1, 8>::zeros();
        let mut covariance = SMatrix::<f32, 8, 8>::zeros();
        let measurement =
            SMatrix::<f32, 1, 4>::from_iterator([1.0, 2.0, 3.0, 4.0]);
        kalman_filter.initiate(&mut mean, &mut covariance, &measurement);

        for _ in 0..10 {
            kalman_filter.update(&mut mean, &mut covariance, &measurement);
            kalman_filter.predict(&mut mean, &mut covariance);
        }
        kalman_filter.predict(&mut mean, &mut covariance);

        assert_eq!(mean, expected_mean);
        for (i, &v) in expected_covariance.iter().enumerate() {
            assert_nearly_eq!(
                v,
                expected_covariance.iter().nth(i).unwrap(),
                1e-4
            )
        }
    }
}
