/*-----------------------------------------------------------------------------
Enum
-----------------------------------------------------------------------------*/

use std::vec;

const LARGE: isize = 1000000;

#[derive(Clone, Copy, Debug, PartialEq)]
enum FPt {
    FP1,
    FP2,
    FPDynamic,
}

/*-----------------------------------------------------------------------------
lapjv.rs - Jonker-Volgenant linear assignment algorithm
-----------------------------------------------------------------------------*/

pub(crate) fn ccrt_dense(
    n: usize,
    cost: &Vec<Vec<isize>>,
    free_rows: &mut Vec<usize>,
    x: &mut Vec<isize>,
    v: &mut Vec<isize>,
    y: &mut Vec<isize>,
) -> usize {
    // initialize x, y, v
    for i in 0..n {
        x[i] = -1;
        v[i] = LARGE;
        y[i] = 0;
    }
    for i in 0..n {
        for j in 0..n {
            let c = cost[i][j];
            if c < v[j] {
                v[j] = c;
                y[j] = i as isize;
            }
        }
    }

    let mut unique = vec![true; n];
    let mut j = n;
    assert!(j > 0, "n must be greater than 0");
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
            let mut min = LARGE;
            for j2 in 0..n {
                if j2 == j {
                    continue;
                }
                let c = cost[i][j2] - v[j2];
                if c < min {
                    min = c;
                }
            }
            v[j] -= min;
        }
    }
    return n_free_rows;
}

pub(crate) fn carr_dence(
    n: usize,
    cost: &Vec<Vec<isize>>,
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

pub(crate) fn find_dense() {}

pub(crate) fn scan_dense() {}

pub(crate) fn find_path_dense() {}

pub(crate) fn ca_dense() {}

pub fn lapjv(n: usize, cost: &mut Vec<Vec<isize>>) -> Vec<usize> {
    unimplemented!("lapjv.rs - Jonker-Volgenant linear assignment algorithm")
}
