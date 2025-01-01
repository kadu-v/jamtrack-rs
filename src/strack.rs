use crate::{
    kalman_filter::{KalmanFilter, StateCov, StateMean},
    rect::Rect,
};
use std::fmt::Debug;

/* ----------------------------------------------------------------------------
 * STrack State enums
 * ---------------------------------------------------------------------------- */
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum STrackState {
    New,
    Tracked,
    Lost,
    Removed,
}

/* ----------------------------------------------------------------------------
 * STrack struct
 * ---------------------------------------------------------------------------- */

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
pub(crate) struct STrack {
    kalman_filter: KalmanFilter,
    mean: StateMean,
    covariance: StateCov,
    rect: Rect<f32>,
    label: usize,
    state: STrackState,
    is_activated: bool,
    score: f32,
    track_id: usize,
    frame_id: usize,
    start_frame_id: usize,
    tracklet_len: usize,
}

impl STrack {
    pub(crate) fn new(rect: Rect<f32>, score: f32, label: usize) -> Self {
        let kalman_filter = KalmanFilter::new(1.0 / 20., 1.0 / 160.);
        let mean = StateMean::zeros();
        let covariance = StateCov::zeros();
        Self {
            kalman_filter,
            mean,
            covariance,
            rect,
            label,
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
            label: 0,
            state: STrackState::New,
            is_activated: false,
            score: 0.0,
            track_id: track_id,
            frame_id: 0,
            start_frame_id: 0,
            tracklet_len: 0,
        }
    }

    #[inline(always)]
    pub(crate) fn get_rect(&self) -> Rect<f32> {
        return self.rect.clone();
    }

    #[inline(always)]
    pub(crate) fn get_strack_state(&self) -> STrackState {
        return self.state;
    }

    #[inline(always)]
    pub(crate) fn get_label(&self) -> usize {
        return self.label;
    }

    #[inline(always)]
    pub(crate) fn is_activated(&self) -> bool {
        return self.is_activated;
    }

    #[inline(always)]
    pub(crate) fn get_score(&self) -> f32 {
        return self.score;
    }

    #[inline(always)]
    pub(crate) fn get_track_id(&self) -> usize {
        return self.track_id;
    }

    #[inline(always)]
    pub(crate) fn get_frame_id(&self) -> usize {
        return self.frame_id;
    }

    #[inline(always)]
    pub(crate) fn get_start_frame_id(&self) -> usize {
        return self.start_frame_id;
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
        self.rect.set_width(self.mean[(0, 2)] * self.mean[(0, 3)]);
        self.rect.set_height(self.mean[(0, 3)]);
        self.rect.set_x(self.mean[(0, 0)] - self.rect.width() / 2.);
        self.rect.set_y(self.mean[(0, 1)] - self.rect.height() / 2.);
    }
}

impl PartialEq for STrack {
    fn eq(&self, other: &Self) -> bool {
        return self.track_id == other.track_id;
    }
}
