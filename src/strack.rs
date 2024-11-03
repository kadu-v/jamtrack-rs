use crate::{
    kalman_filter::{KalmanFilter, StateCov, StateMean},
    rect::Rect,
};
use std::fmt::Debug;

/*----------------------------------------------------------------------------
STrack State enums
----------------------------------------------------------------------------*/
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum STrackState {
    New,
    Tracked,
    Lost,
    Removed,
}

/*----------------------------------------------------------------------------
STrack struct
----------------------------------------------------------------------------*/

impl Debug for STrack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "STrack {{ track_id: {}, frame_id: {}, start_frame_id: {}, tracklet_len: {}, state: {:?}, is_activated: {}, score: {}, rect: {:?} }}",
            self.track_id, self.frame_id, self.start_frame_id, self.tracklet_len, self.state, self.is_activated, self.score, self.rect
        )
    }
}

#[derive(Clone)]
pub struct STrack {
    kalman_filter: KalmanFilter,
    pub mean: StateMean,
    pub covariance: StateCov,
    pub rect: Rect<f32>,
    pub state: STrackState,
    pub is_activated: bool,
    pub score: f32,
    pub track_id: usize,
    pub frame_id: usize,
    pub start_frame_id: usize,
    pub tracklet_len: usize,
}

impl STrack {
    pub fn new(rect: Rect<f32>, score: f32) -> Self {
        let kalman_filter = KalmanFilter::new(1.0 / 20., 1.0 / 160.);
        let mean = StateMean::zeros();
        let covariance = StateCov::zeros();
        Self {
            kalman_filter,
            mean,
            covariance,
            rect,
            state: STrackState::New,
            is_activated: false,
            score,
            track_id: 0,
            frame_id: 0,
            start_frame_id: 0,
            tracklet_len: 0,
        }
    }

    // This function is used in the test_joint_strack function in src/test_byte_tracker.rs
    #[cfg(test)]
    pub(crate) fn dummy_strack(track_id: usize) -> Self {
        let kalman_filter = KalmanFilter::new(1.0 / 20., 1.0 / 160.);
        let mean = StateMean::zeros();
        let covariance = StateCov::zeros();
        Self {
            kalman_filter,
            mean,
            covariance,
            rect: Rect::new(0.0, 0.0, 0.0, 0.0),
            state: STrackState::New,
            is_activated: false,
            score: 0.0,
            track_id: track_id,
            frame_id: 0,
            start_frame_id: 0,
            tracklet_len: 0,
        }
    }

    pub fn get_rect(&self) -> Rect<f32> {
        return self.rect.clone();
    }

    pub fn get_strack_state(&self) -> STrackState {
        return self.state;
    }

    pub fn is_activated(&self) -> bool {
        return self.is_activated;
    }

    pub fn get_score(&self) -> f32 {
        return self.score;
    }

    pub fn get_track_id(&self) -> usize {
        return self.track_id;
    }

    pub fn get_frame_id(&self) -> usize {
        return self.frame_id;
    }

    pub fn get_start_frame_id(&self) -> usize {
        return self.start_frame_id;
    }

    pub fn get_tracklet_length(&self) -> usize {
        return self.tracklet_len;
    }

    pub(crate) fn activate(&mut self, frame_id: usize, track_id: usize) {
        self.kalman_filter.initiate(
            &mut self.mean,
            &mut self.covariance,
            &self.rect.get_xyah(),
        );

        self.update_rect();

        self.state = STrackState::Tracked;
        if frame_id == 1 {
            self.is_activated = true;
        }
        self.track_id = track_id;
        self.frame_id = frame_id;
        self.start_frame_id = frame_id;
        self.tracklet_len = 0;
    }

    pub(crate) fn re_activate(
        &mut self,
        new_track: &STrack,
        frame_id: usize,
        new_track_id: isize,
    ) {
        self.kalman_filter.update(
            &mut self.mean,
            &mut self.covariance,
            &new_track.get_rect().get_xyah(),
        );
        self.update_rect();

        self.state = STrackState::Tracked;
        self.is_activated = true;
        self.score = new_track.get_score();

        if 0 <= new_track_id {
            self.track_id = new_track_id as usize;
        }
        self.frame_id = frame_id;
        self.tracklet_len = 0;
    }

    pub(crate) fn predict(&mut self) {
        if self.state != STrackState::Tracked {
            self.mean[(0, 7)] = 0.;
        }
        self.kalman_filter
            .predict(&mut self.mean, &mut self.covariance);
        self.update_rect();
    }

    pub(crate) fn update(&mut self, new_track: &STrack, frame_id: usize) {
        self.kalman_filter.update(
            &mut self.mean,
            &mut self.covariance,
            &new_track.get_rect().get_xyah(),
        );

        self.update_rect();

        self.state = STrackState::Tracked;
        self.is_activated = true;
        self.score = new_track.get_score();
        self.frame_id = frame_id;
        self.tracklet_len += 1;
    }

    pub(crate) fn mark_as_lost(&mut self) {
        self.state = STrackState::Lost;
    }

    pub(crate) fn mark_as_removed(&mut self) {
        self.state = STrackState::Removed;
    }

    pub(crate) fn update_rect(&mut self) {
        self.rect.tlwh[(0, 2)] = self.mean[(0, 2)] * self.mean[(0, 3)];
        self.rect.tlwh[(0, 3)] = self.mean[(0, 3)];
        self.rect.tlwh[(0, 0)] = self.mean[(0, 0)] - self.rect.width() / 2.;
        self.rect.tlwh[(0, 1)] = self.mean[(0, 1)] - self.rect.height() / 2.;
    }
}

impl PartialEq for STrack {
    fn eq(&self, other: &Self) -> bool {
        return self.track_id == other.track_id;
    }
}
