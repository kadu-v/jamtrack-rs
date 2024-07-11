use nalgebra::SMatrix;

/*-----------------------------------------------------------------------------
Type aliases
-------------------------------------------------------------------------------*/
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

/*-----------------------------------------------------------------------------
Kalman Filter
-------------------------------------------------------------------------------*/
#[derive(Debug, Clone)]
pub struct KalmanFilter {
    std_weight_position: f32,
    std_weight_velocity: f32,
    motion_mat: SMatrix<f32, 8, 8>, // 8x8
    update_mat: SMatrix<f32, 4, 8>, // 4x8
}

impl KalmanFilter {
    pub fn new(std_weight_position: f32, std_weight_velocity: f32) -> Self {
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

    pub fn initiate(
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

    pub fn predict(&mut self, mean: &mut StateMean, covariance: &mut StateCov) {
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

    pub fn update(
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

    pub fn project(
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
