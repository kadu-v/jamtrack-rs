/*-----------------------------------------------------------------------------
Tests
-------------------------------------------------------------------------------*/
use super::kalman_filter::KalmanFilter;
use ndarray::{arr2, Array2};
use nearly_eq::assert_nearly_eq;

#[test]
fn test_initiate() {
    let kalman_filter = KalmanFilter::new(1. / 20., 1. / 160.);
    let mut mean = Array2::<f32>::zeros((1, 8));
    let mut covariance = Array2::<f32>::zeros((8, 8));
    let measurement = arr2(&[[1.0, 2.0, 3.0, 4.0]]);

    kalman_filter.initiate(&mut mean, &mut covariance, &measurement);

    // Assert the values of mean and covariance after initiation
    assert_eq!(mean, arr2(&[[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]]));
    let expected = arr2(&[
        [0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.16, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 6.25e-2, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 6.25e-2, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-10, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.25e-2],
    ]);
    for (i, &v) in covariance.iter().enumerate() {
        assert_nearly_eq!(v, expected.iter().nth(i).unwrap(), 1e-4)
    }
}

#[test]
fn test_predict() {
    let mut kalman_filter = KalmanFilter::new(1. / 20., 1. / 160.);
    let mut mean = arr2(&[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]);
    let mut covariance = arr2(&[
        [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000001, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
    ]);

    kalman_filter.predict(&mut mean, &mut covariance);

    // Assert the values of mean and covariance after prediction
    assert_eq!(mean, arr2(&[[6.0, 8.0, 10.0, 12.0, 5.0, 6.0, 7.0, 8.0]]));
    #[rustfmt::skip]
    let expected = arr2(&[
        [4.24, 0.0,  0.0,     0.0,  4.0,      0.0,      0.0,   0.0],
        [0.0,  4.24, 0.0,     0.0,  0.0,      4.0,      0.0,   0.0],
        [0.0,  0.0,  1.01e-2, 0.0,  0.0,      0.0,      1.0e-6, 0.0],
        [0.0,  0.0,  0.0,     4.24, 0.0,      0.0,      0.0,    4.0],
        [4.0,  0.0,  0.0,     0.0,  4.000625, 0.0,      0.0,    0.0],
        [0.0,  4.0,  0.0,     0.0,  0.0,      4.000625, 0.0,    0.0],
        [0.0,  0.0,  1.0e-6,  0.0,  0.0,      0.0,      1.0e-6, 0.0],
        [0.0,  0.0,  0.0,     4.0,  0.0,      0.0,      0.0,    4.000625],
    ]);
    for (i, &v) in covariance.iter().enumerate() {
        assert_nearly_eq!(v, expected.iter().nth(i).unwrap(), 1e-4)
    }
}

#[test]
fn test_project() {
    let kalman_filter = KalmanFilter::new(1. / 20., 1. / 160.);
    let mean = arr2(&[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]);
    let covariance = arr2(&[
        [4.24, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
        [0.0, 4.24, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
        [0.0, 0.0, 1.01e-2, 0.0, 0.0, 0.0, 1.0e-6, 0.0],
        [0.0, 0.0, 0.0, 4.24, 0.0, 0.0, 0.0, 4.0],
        [4.0, 0.0, 0.0, 0.0, 4.000625, 0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0, 0.0, 0.0, 4.000625, 0.0, 0.0],
        [0.0, 0.0, 1.0e-6, 0.0, 0.0, 0.0, 1.0e-6, 0.0],
        [0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.000625],
    ]);
    let mut projected_mean = Array2::<f32>::zeros((1, 4));
    let mut projected_covariance = Array2::<f32>::zeros((4, 4));

    kalman_filter.project(
        &mut projected_mean,
        &mut projected_covariance,
        &mean,
        &covariance,
    );

    assert_eq!(projected_mean, arr2(&[[1., 2., 3., 4.]]));
    #[rustfmt::skip]
    let expected = arr2(&[
        [4.28,   0.,     0.,     0.    ],
        [0.,     4.28,   0.,     0.    ],
        [0.,     0.,     0.0201, 0.    ],
        [0.,     0.,     0.,     4.28  ]]);
    for (i, &v) in projected_covariance.iter().enumerate() {
        assert_nearly_eq!(v, expected.iter().nth(i).unwrap(), 1e-4)
    }
}

#[test]
fn test_update() {
    let mut kalman_filter = KalmanFilter::new(1. / 20., 1. / 160.);
    let mut mean = arr2(&[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]);
    let mut covariance = arr2(&[
        [4.24, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
        [0.0, 4.24, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
        [0.0, 0.0, 1.01e-2, 0.0, 0.0, 0.0, 1.0e-6, 0.0],
        [0.0, 0.0, 0.0, 4.24, 0.0, 0.0, 0.0, 4.0],
        [4.0, 0.0, 0.0, 0.0, 4.000625, 0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0, 0.0, 0.0, 4.000625, 0.0, 0.0],
        [0.0, 0.0, 1.0e-6, 0.0, 0.0, 0.0, 1.0e-6, 0.0],
        [0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.000625],
    ]);

    let measurement = arr2(&[[1.0, 2.0, 3.0, 4.0]]);
    kalman_filter.update(&mut mean, &mut covariance, &measurement);

    // Assert the values of mean and covariance after update
    assert_eq!(mean, arr2(&[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]));
    #[rustfmt::skip]
    let expected = arr2(&[
       [3.96261682e-02, 0.0, 0.0, 0.0,3.73831776e-02, 0.0, 0.0, 0.0],
       [0.0, 3.96261682e-02, 0.0, 0.0, 0.0, 3.73831776e-02, 0.0, 0.0],
       [0.0, 0.0, 5.02487562e-03, 0.0, 0.0, 0.0, 4.97512438e-07, 0.0],
       [0.0, 0.0, 0.0, 3.96261682e-02, 0.0, 0.0, 0.0, 3.73831776e-02],
       [3.73831776e-02, 0.0, 0.0, 0.0, 2.62307243e-01, 0.0, 0.0, 0.0],
       [0.0, 3.73831776e-02, 0.0, 0.0, 0.0, 2.62307243e-01, 0.0, 0.0],
       [0.0, 0.0, 4.97512438e-07, 0.0, 0.0, 0.0, 9.99950249e-07, 0.0],
       [0.0, 0.0, 0.0, 3.73831776e-02, 0.0, 0.0, 0.0, 2.62307243e-01]]);
    for (i, &v) in covariance.iter().enumerate() {
        assert_nearly_eq!(v, expected.iter().nth(i).unwrap(), 1e-4)
    }
}
