use crate::strack::STrack;
use std::{collections::HashMap, hash::Hash};

/*-----------------------------------------------------------------------------
ByteTracker
-----------------------------------------------------------------------------*/

#[derive(Debug)]
pub struct ByteTracker {
    track_thresh: f32,
    high_thresh: f32,
    math_thresh: f32,
    max_time_lost: usize,

    frame_id: usize,
    track_id_count: usize,

    tracked_stracks: Vec<STrack>,
    lost_stracks: Vec<STrack>,
    removed_stracks: Vec<STrack>,
}

impl ByteTracker {
    pub fn new(
        track_thresh: f32,
        high_thresh: f32,
        math_thresh: f32,
        max_time_lost: usize,
    ) -> Self {
        Self {
            track_thresh,
            high_thresh,
            math_thresh,
            max_time_lost,

            frame_id: 0,
            track_id_count: 0,

            tracked_stracks: Vec::new(),
            lost_stracks: Vec::new(),
            removed_stracks: Vec::new(),
        }
    }

    pub fn joint_strack(
        &self,
        a_list: &Vec<STrack>,
        b_list: &Vec<STrack>,
    ) -> Vec<STrack> {
        let mut exists = HashMap::new();
        let mut res = Vec::new();

        for a in a_list.iter() {
            exists.insert(a.get_track_id(), 1);
            res.push(a.clone());
        }

        for b in b_list.iter() {
            let tid = b.get_track_id();
            // TODO: Check if this is correct
            // The original code is more check
            // if the value corresponding to the key is 0
            // https://github.com/Vertical-Beach/ByteTrack-cpp/blob/d43805d461a714f65da039981bd5f5d21cf5cf59/src/BYTETracker.cpp#L241-L242
            if !exists.contains_key(&tid) {
                exists.insert(tid, 1);
                res.push(b.clone());
            }
        }
        res
    }

    pub fn sub_stracks(
        &self,
        a_list: &Vec<STrack>,
        b_list: &Vec<STrack>,
    ) -> Vec<STrack> {
        let mut stracks = HashMap::new();
        for a in a_list.iter() {
            stracks.insert(a.get_track_id(), a.clone());
        }

        for b in b_list.iter() {
            let tid = b.get_track_id();
            if stracks.contains_key(&tid) {
                stracks.remove(&tid);
            }
        }

        let res = stracks.values().cloned().collect::<Vec<_>>();
        res
    }

    pub fn remove_duplicate_stracks(&self) -> Vec<STrack> {
        unimplemented!("remove_duplicate_stracks")
    }

    pub fn linear_assignment(&self) -> (Vec<usize>, Vec<usize>) {
        unimplemented!("linear_assignment")
    }

    pub fn calc_ious(&self) -> Vec<Vec<f32>> {
        unimplemented!("calc_ious")
    }

    pub fn calc_iou_distance(&self) -> Vec<Vec<f32>> {
        unimplemented!("calc_iou_distance")
    }

    pub fn exec_lapj(&self) -> f64 {
        unimplemented!("exec_lapj")
    }
}
