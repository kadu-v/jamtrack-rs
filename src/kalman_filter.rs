        
use ndarray::{arr1, s, Array, Array1, Array2};
use ndarray_linalg::{cholesky::*, Diag, Solve, SolveTriangular};

/*-----------------------------------------------------------------------------
Type aliases
-------------------------------------------------------------------------------*/
// 1x4
type DetectBox = Array2<f32>;
// 1x8
type StateMean = Array2<f32>;  
// 8x8
type StateCov = Array2<f32>;  
// 1x4
type StateHMean = Array2<f32>; 
// 4x4
type StateHCov = Array2<f32>;  

/*-----------------------------------------------------------------------------
Kalman Filter
-------------------------------------------------------------------------------*/
pub struct KalmanFilter {
    std_weight_position: f32,
    std_weight_velocity: f32,
    motion_mat: Array2<f32>, // 8x8
    update_mat: Array2<f32>, // 4x8
}

impl KalmanFilter {
    pub fn new(std_weight_position: f32, std_weight_velocity: f32) -> Self {
        let ndim = 4;
        let dt = 1.0;

        let mut motion_mat = Array2::eye(8);
        #[rustfmt::skip]
        let update_mat = Array2::from_shape_vec(
            (4, 8),
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ).unwrap();

        for i in 0..ndim {
            motion_mat[[i, i + ndim]] = dt;
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
        mean.slice_mut(s![0, 0..4]).assign(&measurement.slice(s![0, 0..4]));
        mean.slice_mut(s![0, 4..8]).assign(&Array::zeros(4));

        let mut std = Array1::<f32>::zeros(8);
        std[0] = 2.0 * self.std_weight_position * measurement[[0, 3]];
        std[1] = 2.0 * self.std_weight_position * measurement[[0, 3]];
        std[2] = 1e-2;
        std[3] = 2.0 * self.std_weight_position * measurement[[0, 3]];
        std[4] = 10.0 * self.std_weight_velocity * measurement[[0, 3]];
        std[5] = 10.0 * self.std_weight_velocity * measurement[[0, 3]];
        std[6] = 1e-5;
        std[7] = 10.0 * self.std_weight_velocity * measurement[[0, 3]];
    
        let tmp = std.mapv(|x| x * x);
        // convert 1-d array to 2-d array that has diagonal values of 1-d array
        covariance.assign(&Array::from_diag(&tmp));
    }

    pub fn predict(
        &mut self,
        mean: &mut StateMean,
        covariance: &mut StateCov,
    ) {
        let mut std = Array1::<f32>::zeros(8);
        std[0] = self.std_weight_position * mean[[0, 3]];
        std[1] = self.std_weight_position * mean[[0, 3]];
        std[2] = 1e-2;
        std[3] = self.std_weight_position * mean[[0, 3]];
        std[4] = self.std_weight_velocity * mean[[0, 3]];
        std[5] = self.std_weight_velocity * mean[[0, 3]];
        std[6] = 1e-5;
        std[7] = self.std_weight_velocity * mean[[0, 3]];
    
        let tmp = std.mapv(|x| x * x);
        let motion_cov = Array2::<f32>::from_diag(&tmp);
    
        mean.assign(&self.motion_mat.dot(&mean.t()).t());
        let tmp = self.motion_mat.dot(covariance).dot(&self.motion_mat.t());
        covariance.assign(&(tmp + &motion_cov));
    }

    pub fn update(
        &mut self,
        mean: &mut StateMean, // 1x8
        covariance: &mut StateCov, // 8x8
        measurement: &DetectBox, // 1x4
    ) {
        
        let mut projected_mean = Array2::<f32>::zeros((1, 4));
        let mut projected_covariance = Array2::<f32>::zeros((4, 4));
        self.project(&mut projected_mean, &mut projected_covariance, &mean, &covariance);

        let b = (covariance.dot(&self.update_mat.t())).t().to_owned();
        let cholesky_factor = projected_covariance.cholesky(UPLO::Lower).unwrap();
        dbg!(&cholesky_factor);
        // kalman_gain: 8x4
        let binding = cholesky_factor.solve_triangular(UPLO::Lower, Diag::Unit, &b).unwrap();
        let kalman_gain = binding.t();
        dbg!(kalman_gain);
        // dbg!(kalman_gain);
        // // innovation: 1x4
        // let innovation = measurement - &projected_mean;
        // // tmp: 1x8
        // let tmp = innovation.dot(&kalman_gain.t());
        // *mean += &tmp;
        // *covariance -= &kalman_gain.dot(&projected_covariance).dot(&kalman_gain.t());
    }


    pub fn project(
        &self,
        projected_mean: &mut StateHMean, // 1x4
        projected_covariance: &mut StateHCov, // 4x4
        mean: &StateMean, // 1x8
        covariance: &StateCov, // 8x8
    ) {
        let std = arr1(&[
            self.std_weight_position * mean[[0, 3]],
            self.std_weight_position * mean[[0, 3]],
            1e-1,
            self.std_weight_position * mean[[0, 3]],
        ]);
    
        // update_mat: 4x8, mean: 1x8
        // projected_mean: 4x1
        let tmp = mean.dot(&self.update_mat.t());
        *projected_mean = tmp;


        // 4x4
        let diag =  Array2::<f32>::from_diag(&std);
        let innovation_cov =  &diag.mapv(|x| x * x);
        let cov = self.update_mat.dot(covariance).dot(&self.update_mat.t());
        // projected_covariance: 4x4
        *projected_covariance = cov + innovation_cov;
    }

    // [0.040000003, 0.040000003, 0.010000001, 0.040000003],
}


