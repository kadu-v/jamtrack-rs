use crate::error::ByteTrackError::{self, LapjvError};

/*-----------------------------------------------------------------------------
Enum
-----------------------------------------------------------------------------*/

use std::{fmt::format, vec};

const LARGE: isize = 1000000;

/*-----------------------------------------------------------------------------
lapjv.rs - Jonker-Volgenant linear assignment algorithm
-----------------------------------------------------------------------------*/

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
) -> Result<(), ByteTrackError> {
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
