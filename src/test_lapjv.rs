use std::vec;

#[test]
fn test_lapjv_3x3() {
    let n = 3;
    let mut cost = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let mut x = vec![-1; 3];
    let mut y = vec![-1; 3];
    let n_free_rows = crate::lapjv::lapjv(n, &mut cost, &mut x, &mut y);
    assert_eq!(n_free_rows, 0);
    assert_eq!(x, vec![2, 0, 1]);
    assert_eq!(y, vec![1, 2, 0]);
}

#[test]
fn test_lapjv_4x4() {
    let n = 4;
    let mut cost = vec![
        vec![1., 2., 3., 4.],
        vec![5., 6., 7., 8.],
        vec![9., 10., 11., 12.],
        vec![13., 14., 15., 16.],
    ];
    let mut x = vec![-1; 4];
    let mut y = vec![-1; 4];
    let n_free_rows = crate::lapjv::lapjv(n, &mut cost, &mut x, &mut y);
    assert_eq!(n_free_rows, 0);
    assert_eq!(x, vec![3, 0, 1, 2]);
    assert_eq!(y, vec![1, 2, 3, 0]);
}

#[test]
fn test_lapjv_5x5() {
    let n = 5;
    let mut cost = vec![
        vec![1., 2., 3., 4., 1.],
        vec![5., 6., 7., 8., 2.],
        vec![9., 10., 11., 12., 3.],
        vec![13., 14., 15., 16., 4.],
        vec![17., 18., 19., 20., 5.],
    ];
    let mut x = vec![-1; 5];
    let mut y = vec![-1; 5];
    let n_free_rows = crate::lapjv::lapjv(n, &mut cost, &mut x, &mut y);
    assert_eq!(n_free_rows, 0);
    assert_eq!(x, vec![0, 2, 1, 3, 4]);
    assert_eq!(y, vec![0, 2, 1, 3, 4]);
}
