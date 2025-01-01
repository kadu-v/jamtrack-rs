/* -----------------------------------------------------------------------------
 * Tests
 * -------------------------------------------------------------------------------*/
use super::kalman_filter::KalmanFilter;
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

    let measurement = SMatrix::<f32, 1, 4>::from_iterator([1.0, 2.0, 3.0, 4.0]);
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
    let measurement = SMatrix::<f32, 1, 4>::from_iterator([1.0, 2.0, 3.0, 4.0]);
    kalman_filter.initiate(&mut mean, &mut covariance, &measurement);

    for _ in 0..10 {
        kalman_filter.update(&mut mean, &mut covariance, &measurement);
        kalman_filter.predict(&mut mean, &mut covariance);
    }
    kalman_filter.predict(&mut mean, &mut covariance);

    assert_eq!(mean, expected_mean);
    for (i, &v) in expected_covariance.iter().enumerate() {
        assert_nearly_eq!(v, expected_covariance.iter().nth(i).unwrap(), 1e-4)
    }
}
