use std::vec;

use quickcheck::{Arbitrary, Gen};
use rand::{self, Rng};

#[test]
fn test_lapjv_3x3() {
    let mut cost = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let mut x = vec![-1; 3];
    let mut y = vec![-1; 3];
    let res = crate::lapjv::lapjv(&mut cost, &mut x, &mut y);
    assert!(res.is_ok(), "expected Ok, got {:?}", res);
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
    let res = crate::lapjv::lapjv(&mut cost, &mut x, &mut y);
    assert!(res.is_ok(), "expected Ok, got {:?}", res);
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
    let res = crate::lapjv::lapjv(&mut cost, &mut x, &mut y);
    assert!(res.is_ok(), "expected Ok, got {:?}", res);
    assert_eq!(x, vec![0, 2, 1, 3, 4]);
    assert_eq!(y, vec![0, 2, 1, 3, 4]);
}

#[test]
fn test_lapjv_10x10_1() {
    let mut cost = vec![
        vec![
            0.84612522, 0.38549337, 0.27955776, 0.76146103, 0.85084611,
            0.02021263, 0.05006527, 0.40961263, 0.19081828, 0.26665063,
        ],
        vec![
            0.09142041, 0.39511703, 0.5287497, 0.43743945, 0.30997663,
            0.76304532, 0.37178294, 0.73159998, 0.59313334, 0.86550584,
        ],
        vec![
            0.03684529, 0.27024986, 0.1672481, 0.14402541, 0.76511803,
            0.94059419, 0.22349045, 0.51600746, 0.61480263, 0.6346781,
        ],
        vec![
            0.68874528, 0.98444085, 0.33925711, 0.83052401, 0.71814185,
            0.62298001, 0.76450538, 0.03825611, 0.50084776, 0.46314705,
        ],
        vec![
            0.05084691, 0.89486244, 0.87147786, 0.64935965, 0.72806465,
            0.05434427, 0.03566491, 0.73072368, 0.94922003, 0.01400043,
        ],
        vec![
            0.20976728, 0.50350434, 0.83373798, 0.15936914, 0.97320944,
            0.00213279, 0.72815469, 0.17278318, 0.87271939, 0.19039888,
        ],
        vec![
            0.24818255, 0.52879636, 0.22082257, 0.69962464, 0.85367808,
            0.0130662, 0.12319754, 0.01034406, 0.44409775, 0.31241999,
        ],
        vec![
            0.2413349, 0.91244109, 0.67805999, 0.84944587, 0.02873252,
            0.60521235, 0.06197102, 0.33353023, 0.01528123, 0.17659061,
        ],
        vec![
            0.84612522, 0.38549337, 0.27955776, 0.76146103, 0.85084611,
            0.02021263, 0.05006527, 0.40961263, 0.19081828, 0.26665063,
        ],
        vec![
            0.09142041, 0.39511703, 0.5287497, 0.43743945, 0.30997663,
            0.76304532, 0.37178294, 0.73159998, 0.59313334, 0.86550584,
        ],
    ];
    let mut x = vec![-1; 10];
    let mut y = vec![-1; 10];
    let res = crate::lapjv::lapjv(&mut cost, &mut x, &mut y);
    assert!(res.is_ok(), "expected Ok, got {:?}", res);
    assert_eq!(x, vec![8, 0, 2, 7, 9, 3, 5, 4, 6, 1]);
    assert_eq!(y, vec![1, 9, 2, 5, 7, 6, 8, 3, 0, 4]);
}

#[test]
fn test_lapjv_10x10_2() {
    let mut cost = vec![
        vec![
            0.84612522, 0.38549337, 0.27955776, 0.76146103, 0.85084611,
            0.02021263, 0.05006527, 0.40961263, 0.19081828, 0.26665063,
        ],
        vec![
            0.09142041, 0.39511703, 0.5287497, 0.43743945, 0.30997663,
            0.76304532, 0.37178294, 0.73159998, 0.59313334, 0.86550584,
        ],
        vec![
            0.03684529, 0.27024986, 0.1672481, 0.14402541, 0.76511803,
            0.94059419, 0.22349045, 0.51600746, 0.61480263, 0.6346781,
        ],
        vec![
            0.68874528, 0.98444085, 0.33925711, 0.83052401, 0.71814185,
            0.62298001, 0.76450538, 0.03825611, 0.50084776, 0.46314705,
        ],
        vec![
            0.05084691, 0.89486244, 0.87147786, 0.64935965, 0.72806465,
            0.05434427, 0.03566491, 0.73072368, 0.94922003, 0.01400043,
        ],
        vec![
            0.20976728, 0.50350434, 0.83373798, 0.15936914, 0.97320944,
            0.00213279, 0.72815469, 0.17278318, 0.87271939, 0.19039888,
        ],
        vec![
            0.24818255, 0.52879636, 0.22082257, 0.69962464, 0.85367808,
            0.0130662, 0.12319754, 0.01034406, 0.44409775, 0.31241999,
        ],
        vec![
            0.2413349, 0.91244109, 0.67805999, 0.84944587, 0.02873252,
            0.60521235, 0.06197102, 0.33353023, 0.01528123, 0.17659061,
        ],
        vec![
            0.2413349, 0.91244109, 0.67805999, 0.84944587, 0.02873252,
            0.60521235, 0.06197102, 0.33353023, 0.01528123, 0.17659061,
        ],
        vec![
            0.2413349, 0.91244109, 0.67805999, 0.84944587, 0.02873252,
            0.60521235, 0.06197102, 0.33353023, 0.01528123, 0.17659061,
        ],
    ];
    let mut x = vec![-1; 10];
    let mut y = vec![-1; 10];
    let res = crate::lapjv::lapjv(&mut cost, &mut x, &mut y);
    assert!(res.is_ok(), "expected Ok, got {:?}", res);
    assert_eq!(x, vec![5, 0, 1, 7, 9, 3, 2, 8, 4, 6]);
    assert_eq!(y, vec![1, 2, 6, 5, 8, 0, 9, 3, 7, 4]);
}

fn gen_cost_matrix(n: usize, gen: &mut Gen) -> Vec<Vec<f64>> {
    let mut cost = vec![];
    for _ in 0..n {
        let row = vec![f64::arbitrary(gen); n];
        cost.push(row);
    }
    cost
}

fn gen_vec_isize(n: usize, gen: &mut Gen) -> Vec<isize> {
    let mut vec = vec![];
    for _ in 0..n {
        vec.push(isize::arbitrary(gen));
    }
    vec
}

#[test]
fn test_quickcheck_lapjv() {
    fn prop(_: usize) -> bool {
        let mut rng = rand::thread_rng();
        let n = rng.gen_range(1..=1000);
        let mut cost = gen_cost_matrix(n, &mut Gen::new(rng.gen()));
        let mut x = gen_vec_isize(n, &mut Gen::new(rng.gen()));
        let mut y = gen_vec_isize(n, &mut Gen::new(rng.gen()));
        let result = crate::lapjv::lapjv(&mut cost, &mut x, &mut y);
        result.is_ok()
    }
    quickcheck::quickcheck(prop as fn(usize) -> bool);
}
