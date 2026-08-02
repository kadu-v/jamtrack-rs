use std::collections::VecDeque;

use super::kalman_filter::{KalmanFilter, Measurement, StateCov, StateMean};
use crate::{Object, Rect};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TrackState {
    New,
    Tracked,
    Lost,
    Removed,
}

#[derive(Debug, Clone)]
pub(crate) struct STrack {
    pub(crate) mean: Option<StateMean>,
    pub(crate) covariance: Option<StateCov>,
    pub(crate) state: TrackState,
    pub(crate) is_activated: bool,
    pub(crate) score: f64,
    pub(crate) track_id: usize,
    pub(crate) frame_id: usize,
    pub(crate) start_frame: usize,
    pub(crate) tracklet_len: usize,
    pub(crate) not_matched: usize,
    pub(crate) is_occluded: bool,
    pub(crate) occluded_len: usize,
    pub(crate) last_occluded_frame: Option<usize>,
    pub(crate) was_recently_occluded: bool,
    pub(crate) mean_history: VecDeque<StateMean>,
    pub(crate) center_history: Vec<[f64; 2]>,
    initial_tlwh: [f64; 4],
    tlwh: [f64; 4],
    kalman_filter: KalmanFilter,
}

impl STrack {
    pub(crate) fn new(object: &Object) -> Self {
        let rect = object.get_rect();
        Self::from_tlwh(
            [
                rect.x() as f64,
                rect.y() as f64,
                rect.width() as f64,
                rect.height() as f64,
            ],
            object.get_prob() as f64,
        )
    }

    pub(crate) fn from_tlwh(tlwh: [f64; 4], score: f64) -> Self {
        Self {
            mean: None,
            covariance: None,
            state: TrackState::New,
            is_activated: false,
            score,
            track_id: 0,
            frame_id: 0,
            start_frame: 0,
            tracklet_len: 0,
            not_matched: 0,
            is_occluded: false,
            occluded_len: 0,
            last_occluded_frame: None,
            was_recently_occluded: false,
            mean_history: VecDeque::new(),
            center_history: Vec::new(),
            initial_tlwh: tlwh,
            tlwh,
            kalman_filter: KalmanFilter::default(),
        }
    }

    pub(crate) fn activate(&mut self, frame_id: usize, track_id: usize) {
        let measurement = Self::tlwh_to_xyah(self.initial_tlwh);
        let (mean, covariance) = self.kalman_filter.initiate(&measurement);
        self.mean = Some(mean);
        self.covariance = Some(covariance);
        self.push_mean_history(mean);
        self.refresh_tlwh();

        self.tracklet_len = 0;
        self.state = TrackState::Tracked;
        if frame_id == 1 {
            self.is_activated = true;
        }
        self.track_id = track_id;
        self.frame_id = frame_id;
        self.start_frame = frame_id;
    }

    pub(crate) fn predict(&mut self) {
        let Some(mut mean) = self.mean else { return };
        let Some(covariance) = self.covariance else {
            return;
        };
        if self.state != TrackState::Tracked {
            mean[7] = 0.0;
        }
        let (mean, covariance) = self.kalman_filter.predict(&mean, &covariance);
        self.mean = Some(mean);
        self.covariance = Some(covariance);
        self.refresh_tlwh();
    }

    pub(crate) fn update(&mut self, detection: &STrack, frame_id: usize) {
        let measurement = Self::tlwh_to_xyah(detection.tlwh());
        let (mean, covariance) = self.kalman_filter.update(
            self.mean.as_ref().expect("activated track has a mean"),
            self.covariance
                .as_ref()
                .expect("activated track has a covariance"),
            &measurement,
        );
        self.mean = Some(mean);
        self.covariance = Some(covariance);
        self.push_mean_history(mean);
        self.refresh_tlwh();
        self.frame_id = frame_id;
        self.tracklet_len += 1;
        self.state = TrackState::Tracked;
        self.is_activated = true;
        self.score = detection.score;
    }

    pub(crate) fn re_activate(&mut self, detection: &STrack, frame_id: usize) {
        let measurement = Self::tlwh_to_xyah(detection.tlwh());
        let (mean, covariance) = self.kalman_filter.update(
            self.mean.as_ref().expect("activated track has a mean"),
            self.covariance
                .as_ref()
                .expect("activated track has a covariance"),
            &measurement,
        );
        self.mean = Some(mean);
        self.covariance = Some(covariance);
        self.push_mean_history(mean);
        self.refresh_tlwh();
        self.tracklet_len = 0;
        self.state = TrackState::Tracked;
        self.is_activated = true;
        self.frame_id = frame_id;
        self.score = detection.score;
    }

    pub(crate) fn mark_lost(&mut self) {
        self.state = TrackState::Lost;
    }

    pub(crate) fn mark_removed(&mut self) {
        self.state = TrackState::Removed;
    }

    pub(crate) fn tlwh(&self) -> [f64; 4] {
        self.tlwh
    }

    pub(crate) fn tlbr(&self) -> [f64; 4] {
        [
            self.tlwh[0],
            self.tlwh[1],
            self.tlwh[0] + self.tlwh[2],
            self.tlwh[1] + self.tlwh[3],
        ]
    }

    pub(crate) fn set_mean_position(&mut self, x: f64, y: f64) {
        if let Some(mean) = self.mean.as_mut() {
            mean[0] = x;
            mean[1] = y;
        }
        self.refresh_tlwh();
    }

    pub(crate) fn refresh_tlwh(&mut self) {
        let Some(mean) = self.mean.as_ref() else {
            self.tlwh = self.initial_tlwh;
            return;
        };
        let width = mean[2] * mean[3];
        let height = mean[3];
        self.tlwh =
            [mean[0] - width / 2.0, mean[1] - height / 2.0, width, height];
    }

    fn push_mean_history(&mut self, mean: StateMean) {
        self.mean_history.push_back(mean);
        if self.mean_history.len() > 100 {
            self.mean_history.pop_front();
        }
    }

    pub(crate) fn tlwh_to_xyah(tlwh: [f64; 4]) -> Measurement {
        Measurement::from_row_slice(&[
            tlwh[0] + tlwh[2] / 2.0,
            tlwh[1] + tlwh[3] / 2.0,
            tlwh[2] / tlwh[3],
            tlwh[3],
        ])
    }

    pub(crate) fn into_object(self) -> Object {
        Object::new(
            Rect::new(
                self.tlwh[0] as f32,
                self.tlwh[1] as f32,
                self.tlwh[2] as f32,
                self.tlwh[3] as f32,
            ),
            self.score as f32,
            Some(self.track_id),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tlwh_to_xyah_converts_coordinates() {
        let xyah = STrack::tlwh_to_xyah([10.0, 20.0, 30.0, 40.0]);
        assert_eq!(xyah.as_slice(), &[25.0, 40.0, 0.75, 40.0]);
    }

    #[test]
    fn activate_and_update_advance_track_lifecycle() {
        let mut track = STrack::from_tlwh([10.0, 20.0, 30.0, 40.0], 0.9);
        track.activate(2, 7);
        assert_eq!(track.track_id, 7);
        assert!(!track.is_activated);
        assert_eq!(track.mean_history.len(), 1);

        let detection = STrack::from_tlwh([12.0, 22.0, 30.0, 40.0], 0.8);
        track.update(&detection, 3);
        assert!(track.is_activated);
        assert_eq!(track.frame_id, 3);
        assert_eq!(track.tracklet_len, 1);
        assert_eq!(track.mean_history.len(), 2);
    }
}
