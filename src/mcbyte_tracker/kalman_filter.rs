use nalgebra::SMatrix;

pub(crate) type Measurement = SMatrix<f32, 1, 4>;
pub(crate) type StateMean = SMatrix<f32, 1, 8>;
pub(crate) type StateCov = SMatrix<f32, 8, 8>;
type ProjectedMean = SMatrix<f32, 1, 4>;
type ProjectedCov = SMatrix<f32, 4, 4>;

#[derive(Debug, Clone)]
pub(crate) struct KalmanFilter {
    motion: SMatrix<f32, 8, 8>,
    projection: SMatrix<f32, 4, 8>,
}

impl KalmanFilter {
    pub(crate) fn new() -> Self {
        let mut motion = SMatrix::<f32, 8, 8>::identity();
        let mut projection = SMatrix::<f32, 4, 8>::zeros();
        for i in 0..4 {
            motion[(i, i + 4)] = 1.0;
            projection[(i, i)] = 1.0;
        }
        Self { motion, projection }
    }

    pub(crate) fn initiate(
        &self,
        measurement: &Measurement,
    ) -> (StateMean, StateCov) {
        let mut mean = StateMean::zeros();
        mean.as_mut_slice()[..4].copy_from_slice(measurement.as_slice());
        let (w, h) = size(measurement[(0, 2)], measurement[(0, 3)]);
        let std = SMatrix::<f32, 1, 8>::from_iterator([
            0.1 * w,
            0.1 * h,
            0.1 * w,
            0.1 * h,
            0.0625 * w,
            0.0625 * h,
            0.0625 * w,
            0.0625 * h,
        ]);
        let covariance = SMatrix::<f32, 8, 8>::from_diagonal(
            &std.component_mul(&std).transpose(),
        );
        (mean, covariance)
    }

    pub(crate) fn predict(
        &self,
        mean: &mut StateMean,
        covariance: &mut StateCov,
    ) {
        let (w, h) = size(mean[(0, 2)], mean[(0, 3)]);
        let std = SMatrix::<f32, 1, 8>::from_iterator([
            0.05 * w,
            0.05 * h,
            0.05 * w,
            0.05 * h,
            0.00625 * w,
            0.00625 * h,
            0.00625 * w,
            0.00625 * h,
        ]);
        let process_noise = SMatrix::<f32, 8, 8>::from_diagonal(
            &std.component_mul(&std).transpose(),
        );
        *mean = (self.motion * mean.transpose()).transpose();
        *covariance =
            self.motion * *covariance * self.motion.transpose() + process_noise;
        clamp_size(mean);
    }

    pub(crate) fn update(
        &self,
        mean: &mut StateMean,
        covariance: &mut StateCov,
        measurement: &Measurement,
    ) {
        let (projected_mean, projected_covariance) =
            self.project(mean, covariance);
        let rhs = (*covariance * self.projection.transpose()).transpose();
        let Some(cholesky) = projected_covariance.cholesky() else {
            return;
        };
        let gain = cholesky.solve(&rhs);
        let innovation = measurement - projected_mean;
        *mean += innovation * gain;
        *covariance -= gain.transpose() * projected_covariance * gain;
        clamp_size(mean);
    }

    fn project(
        &self,
        mean: &StateMean,
        covariance: &StateCov,
    ) -> (ProjectedMean, ProjectedCov) {
        let (w, h) = size(mean[(0, 2)], mean[(0, 3)]);
        let std = SMatrix::<f32, 1, 4>::from_iterator([
            0.05 * w,
            0.05 * h,
            0.05 * w,
            0.05 * h,
        ]);
        let measurement_noise = SMatrix::<f32, 4, 4>::from_diagonal(
            &std.component_mul(&std).transpose(),
        );
        (
            mean * self.projection.transpose(),
            self.projection * covariance * self.projection.transpose()
                + measurement_noise,
        )
    }
}

fn size(w: f32, h: f32) -> (f32, f32) {
    (w.max(1e-3), h.max(1e-3))
}

fn clamp_size(mean: &mut StateMean) {
    mean[(0, 2)] = mean[(0, 2)].max(1e-3);
    mean[(0, 3)] = mean[(0, 3)].max(1e-3);
}
