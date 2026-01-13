use crate::error::TrackError::{self, LapjvError};

/* -----------------------------------------------------------------------------
 * Enum
 * ----------------------------------------------------------------------------- */

use std::vec;

const LARGE: isize = 1000000;

/* -----------------------------------------------------------------------------
 * lapjv.rs - Jonker-Volgenant linear assignment algorithm
 * ----------------------------------------------------------------------------- */

fn ccrt_dense(
    n: usize,
    cost: &Vec<Vec<f64>>,
    free_rows: &mut Vec<usize>,
    x: &mut Vec<isize>,
    v: &mut Vec<f64>,
    y: &mut Vec<isize>,
) -> usize {
    debug_assert!(cost.len() == n, "cost.len() must be equal to {}", n);
    debug_assert!(x.len() == n, "x.len() must be equal to {}", n);
    debug_assert!(y.len() == n, "y.len() must be equal to {}", n);
    debug_assert!(v.len() == n, "v.len() must be equal to {}", n);

    // initialize x, y, v
    for i in 0..n {
        x[i] = -1;
        v[i] = LARGE as f64;
        y[i] = 0;
    }
    for i in 0..n {
        for j in 0..n {
            let c = cost[i][j] as f64;
            if c < v[j] {
                v[j] = c;
                y[j] = i as isize;
            }
        }
    }

    let mut unique = vec![true; n];
    let mut j = n;
    debug_assert!(j > 0, "n must be greater than 0");
    {
        while j > 0 {
            j -= 1;
            let i = y[j] as usize;
            if x[i] < 0 {
                x[i] = j as isize;
            } else {
                unique[i] = false;
                y[j] = -1;
            }
        }
    }

    let mut n_free_rows = 0;

    for i in 0..n {
        if x[i] < 0 {
            free_rows[n_free_rows] = i;
            n_free_rows += 1;
        } else if unique[i] {
            let j = x[i] as usize;
            let mut min = LARGE as f64;
            for j2 in 0..n {
                if j2 == j {
                    continue;
                }
                let c = cost[i][j2] as f64 - v[j2];
                if c < min {
                    min = c;
                }
            }
            v[j] -= min;
        }
    }
    return n_free_rows;
}

fn carr_dence(
    n: usize,
    cost: &Vec<Vec<f64>>,
    n_free_rows: usize,
    free_rows: &mut Vec<usize>,
    x: &mut Vec<isize>,
    y: &mut Vec<isize>,
    v: &mut Vec<f64>,
) -> usize {
    let mut current = 0;
    let mut new_free_rows = 0;
    let mut rr_cnt = 0;

    while current < n_free_rows {
        rr_cnt += 1;
        let free_i = free_rows[current];
        current += 1;

        let mut j1 = 0;
        let mut j2 = -1;
        let mut v1 = cost[free_i][0] as f64 - v[0];
        let mut v2 = LARGE as f64;

        for j in 1..n {
            let c = cost[free_i][j] as f64 - v[j];
            if c < v2 {
                if c >= v1 {
                    v2 = c;
                    j2 = j as isize;
                } else {
                    v2 = v1;
                    v1 = c;
                    j2 = j1;
                    j1 = j as isize;
                }
            }
        }

        let mut i0 = y[j1 as usize];
        let v1_new = v[j1 as usize] - (v2 - v1);
        let v1_lowers = v1_new < v[j1 as usize];

        if rr_cnt < current * n {
            if v1_lowers {
                v[j1 as usize] = v1_new;
            } else if i0 >= 0 && j2 >= 0 {
                j1 = j2;
                i0 = y[j2 as usize];
            }

            if i0 >= 0 {
                if v1_lowers {
                    current -= 1;
                    free_rows[current] = i0 as usize;
                } else {
                    free_rows[new_free_rows] = i0 as usize;
                    new_free_rows += 1;
                }
            }
        } else {
            if i0 >= 0 {
                free_rows[new_free_rows] = i0 as usize;
                new_free_rows += 1;
            }
        }
        x[free_i] = j1;
        y[j1 as usize] = free_i as isize;
    }
    return new_free_rows;
}

fn find_dense(
    n: usize,
    lo: usize,
    d: &Vec<f64>,
    cols: &mut Vec<usize>,
) -> usize {
    debug_assert!(d.len() == n, "d.len() must be equal to n");
    debug_assert!(cols.len() == n, "cols.len() must be equal to n");
    let mut hi = lo + 1;
    let mut mind = d[cols[lo]];
    for k in hi..n {
        let j = cols[k];
        debug_assert!(j < d.len(), "j must be less than d.len()");
        if d[j] <= mind {
            if d[j] < mind {
                hi = lo;
                mind = d[j];
            }
            debug_assert!(hi <= cols.len(), "hi must be less than cols.len()");
            debug_assert!(k <= cols.len(), "k must be less than cols.len()");
            cols[k] = cols[hi];
            cols[hi] = j;
            hi += 1;
        }
    }
    return hi;
}

fn scan_dense(
    n: usize,
    cost: &Vec<Vec<f64>>,
    plo: &mut usize,
    phi: &mut usize,
    d: &mut Vec<f64>,
    cols: &mut Vec<usize>,
    pred: &mut Vec<usize>,
    y: &mut Vec<isize>,
    v: &mut Vec<f64>,
) -> isize {
    let mut lo = *plo;
    let mut hi = *phi;
    let mut h: f64;
    let mut cred_ij: f64;

    while lo != hi {
        debug_assert!(lo < cols.len(), "lo must be less than cols.len()");
        let mut j = cols[lo];
        lo += 1;

        debug_assert!(j < y.len(), "j must be less than y.len()");
        debug_assert!(j < d.len(), "j must be less than d.len()");
        debug_assert!(j < v.len(), "j must be less than v.len()");
        let i = y[j] as usize;
        let mind = d[j];

        debug_assert!(y[j] >= 0, "y[j] must be greater than or equal to 0");
        debug_assert!(i < cost.len(), "i must be less than cost.len()");
        h = cost[i][j] - v[j] - mind;
        for k in hi..n {
            j = cols[k];
            cred_ij = cost[i][j] - v[j] - h;
            if cred_ij < d[j] {
                d[j] = cred_ij;
                pred[j] = i;
                if cred_ij == mind {
                    if y[j] < 0 {
                        return j as isize;
                    }
                    cols[k] = cols[hi];
                    cols[hi] = j;
                    hi += 1;
                }
            }
        }
    }
    *plo = lo;
    *phi = hi;
    return -1;
}

fn find_path_dense(
    n: usize,
    cost: &Vec<Vec<f64>>,
    start_i: usize,
    y: &mut Vec<isize>,
    v: &mut Vec<f64>,
    pred: &mut Vec<usize>,
) -> isize {
    let mut lo = 0;
    let mut hi = 0;
    let mut final_j = -1;
    let mut n_ready = 0;
    let mut cols = vec![0; n];
    let mut d = vec![0.0; n];

    for i in 0..n {
        cols[i] = i;
        pred[i] = start_i;
        d[i] = cost[start_i][i] - v[i];
    }

    while final_j == -1 {
        if lo == hi {
            n_ready = lo;
            hi = find_dense(n, lo, &d, &mut cols);
            for k in lo..hi {
                let j = cols[k];
                if y[j] < 0 {
                    final_j = j as isize;
                }
            }
        }
        if final_j == -1 {
            final_j = scan_dense(
                n, cost, &mut lo, &mut hi, &mut d, &mut cols, pred, y, v,
            );
        }
    }

    {
        let mind = d[cols[lo]];
        for k in 0..n_ready {
            let j = cols[k];
            v[j] += d[j] - mind;
        }
    }
    return final_j;
}

fn ca_dense(
    n: usize,
    cost: &Vec<Vec<f64>>,
    n_free_rows: usize,
    free_rows: &mut Vec<usize>,
    x: &mut Vec<isize>,
    y: &mut Vec<isize>,
    v: &mut Vec<f64>,
) -> usize {
    let mut pred = vec![0; n];

    for row_n in 0..n_free_rows {
        let free_row = free_rows[row_n];
        let mut i = -1isize;
        let mut k = 0;

        let mut j = find_path_dense(n, cost, free_row, y, v, &mut pred);
        debug_assert!(j >= 0, "j must be greater than or equal to 0");
        debug_assert!(j < n as isize, "j must be less than n as isize");
        while i != free_row as isize {
            i = pred[j as usize] as isize;
            y[j as usize] = i;

            // swap x[i] and j
            let tmp = j;
            j = x[i as usize];
            x[i as usize] = tmp;

            k += 1;
            debug_assert!(k <= n, "k must be less than or equal to n");
        }
    }
    return 0;
}

pub(crate) fn lapjv(
    cost: &mut Vec<Vec<f64>>,
    x: &mut Vec<isize>,
    y: &mut Vec<isize>,
) -> Result<(), TrackError> {
    let n = cost.len();
    if n == 0 {
        return Err(LapjvError(format!(
            "cost.len() must be greater than 0, but cost.len() = {}",
            n
        )));
    }
    if n != x.len() || n != y.len() {
        return Err(LapjvError(format!(
            "cost.len() must be equal to x.len() and y.len(), but cost.len() = {}, x.len() = {}, y.len() = {}",
            n,
            x.len(),
            y.len()
        )));
    }

    let mut free_rows = vec![0; n];
    let mut v = vec![0.0; n];
    let mut ret = ccrt_dense(n, cost, &mut free_rows, x, &mut v, y);
    let mut i = 0;
    while ret > 0 && i < 2 {
        ret = carr_dence(n, cost, ret, &mut free_rows, x, y, &mut v);
        i += 1;
    }
    if ret > 0 {
        ret = ca_dense(n, cost, ret, &mut free_rows, x, y, &mut v);
    }
    if ret > 0 {
        return Err(LapjvError(format!(
            "ret must be less than or equal to 0, but ret = {}",
            ret
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::{Arbitrary, Gen};
    use rand::{self, Rng};
    use std::vec;

    #[test]
    fn test_lapjv_3x3() {
        let mut cost = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let mut x = vec![-1; 3];
        let mut y = vec![-1; 3];
        let res = lapjv(&mut cost, &mut x, &mut y);
        assert!(res.is_ok(), "expected Ok, got {:?}", res);
        assert_eq!(x, vec![2, 0, 1]);
        assert_eq!(y, vec![1, 2, 0]);
    }

    #[test]
    fn test_lapjv_4x4() {
        let mut cost = vec![
            vec![1., 2., 3., 4.],
            vec![5., 6., 7., 8.],
            vec![9., 10., 11., 12.],
            vec![13., 14., 15., 16.],
        ];
        let mut x = vec![-1; 4];
        let mut y = vec![-1; 4];
        let res = lapjv(&mut cost, &mut x, &mut y);
        assert!(res.is_ok(), "expected Ok, got {:?}", res);
        assert_eq!(x, vec![3, 0, 1, 2]);
        assert_eq!(y, vec![1, 2, 3, 0]);
    }

    #[test]
    fn test_lapjv_5x5() {
        let mut cost = vec![
            vec![1., 2., 3., 4., 1.],
            vec![5., 6., 7., 8., 2.],
            vec![9., 10., 11., 12., 3.],
            vec![13., 14., 15., 16., 4.],
            vec![17., 18., 19., 20., 5.],
        ];
        let mut x = vec![-1; 5];
        let mut y = vec![-1; 5];
        let res = lapjv(&mut cost, &mut x, &mut y);
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
        let res = lapjv(&mut cost, &mut x, &mut y);
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
        let res = lapjv(&mut cost, &mut x, &mut y);
        assert!(res.is_ok(), "expected Ok, got {:?}", res);
        assert_eq!(x, vec![5, 0, 1, 7, 9, 3, 2, 8, 4, 6]);
        assert_eq!(y, vec![1, 2, 6, 5, 8, 0, 9, 3, 7, 4]);
    }

    fn gen_cost_matrix(n: usize, g: &mut Gen) -> Vec<Vec<f64>> {
        let mut cost = vec![];
        for _ in 0..n {
            let row = vec![f64::arbitrary(g); n];
            cost.push(row);
        }
        cost
    }

    fn gen_vec_isize(n: usize, g: &mut Gen) -> Vec<isize> {
        let mut vec = vec![];
        for _ in 0..n {
            vec.push(isize::arbitrary(g));
        }
        vec
    }

    #[test]
    fn test_quickcheck_lapjv() {
        fn prop(_: usize) -> bool {
            let mut rng = rand::thread_rng();
            let n = rng.gen_range(1..=100);
            let mut cost = gen_cost_matrix(n, &mut Gen::new(rng.r#gen()));
            let mut x = gen_vec_isize(n, &mut Gen::new(rng.r#gen()));
            let mut y = gen_vec_isize(n, &mut Gen::new(rng.r#gen()));
            let result = lapjv(&mut cost, &mut x, &mut y);
            result.is_ok()
        }
        quickcheck::quickcheck(prop as fn(usize) -> bool);
    }
}
