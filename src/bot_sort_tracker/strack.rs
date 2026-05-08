use super::kalman_filter::{DetectBox, KalmanFilter, StateCov, StateMean};
use crate::object::Object;
use crate::rect::Rect;
use std::fmt::Debug;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum STrackState {
    New,
    Tracked,
    Lost,
    Removed,
}

#[derive(Clone)]
pub(crate) struct STrack {
    kalman_filter: KalmanFilter,
    mean: StateMean,
    covariance: StateCov,
    rect: Rect<f32>,
    state: STrackState,
    is_activated: bool,
    score: f32,
    track_id: usize,
    frame_id: usize,
    start_frame_id: usize,
    tracklet_len: usize,
    curr_feat: Option<Vec<f32>>,
    smooth_feat: Option<Vec<f32>>,
    alpha: f32,
}

impl Debug for STrack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "STrack {{ track_id: {}, frame_id: {}, start_frame_id: {}, tracklet_len: {}, state: {:?}, is_activated: {}, score: {}, rect: {:?} }}",
            self.track_id,
            self.frame_id,
            self.start_frame_id,
            self.tracklet_len,
            self.state,
            self.is_activated,
            self.score,
            self.rect
        )
    }
}

impl STrack {
    pub(crate) fn new(
        rect: Rect<f32>,
        score: f32,
        feature: Option<Vec<f32>>,
    ) -> Self {
        let mut track = Self {
            kalman_filter: KalmanFilter::new(),
            mean: StateMean::zeros(),
            covariance: StateCov::zeros(),
            rect,
            state: STrackState::New,
            is_activated: false,
            score,
            track_id: 0,
            frame_id: 0,
            start_frame_id: 0,
            tracklet_len: 0,
            curr_feat: None,
            smooth_feat: None,
            alpha: 0.9,
        };
        if let Some(feature) = feature {
            track.update_features(feature);
        }
        track
    }

    #[cfg(test)]
    pub(crate) fn dummy(track_id: usize, frame_id: usize) -> Self {
        let mut track = Self::new(Rect::new(0.0, 0.0, 10.0, 10.0), 0.9, None);
        track.track_id = track_id;
        track.frame_id = frame_id;
        track.start_frame_id = 1;
        track.state = STrackState::Tracked;
        track.is_activated = true;
        track
    }

    pub(crate) fn update_features(&mut self, mut feature: Vec<f32>) {
        normalize(&mut feature);
        self.curr_feat = Some(feature.clone());

        let mut smooth = if let Some(current) = &self.smooth_feat {
            current
                .iter()
                .zip(feature.iter())
                .map(|(a, b)| self.alpha * *a + (1.0 - self.alpha) * *b)
                .collect()
        } else {
            feature
        };
        normalize(&mut smooth);
        self.smooth_feat = Some(smooth);
    }

    pub(crate) fn get_rect(&self) -> Rect<f32> {
        self.rect.clone()
    }

    pub(crate) fn get_state(&self) -> STrackState {
        self.state
    }

    pub(crate) fn is_activated(&self) -> bool {
        self.is_activated
    }

    pub(crate) fn get_score(&self) -> f32 {
        self.score
    }

    pub(crate) fn get_track_id(&self) -> usize {
        self.track_id
    }

    pub(crate) fn get_frame_id(&self) -> usize {
        self.frame_id
    }

    pub(crate) fn get_start_frame_id(&self) -> usize {
        self.start_frame_id
    }

    pub(crate) fn smooth_feat(&self) -> Option<&[f32]> {
        self.smooth_feat.as_deref()
    }

    pub(crate) fn curr_feat(&self) -> Option<&[f32]> {
        self.curr_feat.as_deref()
    }

    pub(crate) fn activate(&mut self, frame_id: usize, track_id: usize) {
        self.track_id = track_id;
        let measurement = self.rect_to_xywh();
        self.kalman_filter.initiate(
            &mut self.mean,
            &mut self.covariance,
            &measurement,
        );
        self.update_rect();
        self.tracklet_len = 0;
        self.state = STrackState::Tracked;
        if frame_id == 1 {
            self.is_activated = true;
        }
        self.frame_id = frame_id;
        self.start_frame_id = frame_id;
    }

    pub(crate) fn re_activate(&mut self, new_track: &STrack, frame_id: usize) {
        let measurement = new_track.rect_to_xywh();
        self.kalman_filter.update(
            &mut self.mean,
            &mut self.covariance,
            &measurement,
        );
        self.update_rect();
        if let Some(feature) = new_track.curr_feat() {
            self.update_features(feature.to_vec());
        }
        self.tracklet_len = 0;
        self.state = STrackState::Tracked;
        self.is_activated = true;
        self.frame_id = frame_id;
        self.score = new_track.score;
    }

    pub(crate) fn predict(&mut self) {
        if self.state != STrackState::Tracked {
            self.mean[(0, 6)] = 0.0;
            self.mean[(0, 7)] = 0.0;
        }
        self.kalman_filter
            .predict(&mut self.mean, &mut self.covariance);
        self.update_rect();
    }

    pub(crate) fn camera_update(&mut self, transform: &[[f32; 3]; 3]) {
        let r00 = transform[0][0];
        let r01 = transform[0][1];
        let r10 = transform[1][0];
        let r11 = transform[1][1];
        let tx = transform[0][2];
        let ty = transform[1][2];

        for base in [0usize, 2, 4, 6] {
            let x = self.mean[(0, base)];
            let y = self.mean[(0, base + 1)];
            self.mean[(0, base)] = r00 * x + r01 * y;
            self.mean[(0, base + 1)] = r10 * x + r11 * y;
        }
        self.mean[(0, 0)] += tx;
        self.mean[(0, 1)] += ty;

        let mut rotated = StateCov::zeros();
        for block in 0..4 {
            let i = block * 2;
            rotated[(i, i)] = r00;
            rotated[(i, i + 1)] = r01;
            rotated[(i + 1, i)] = r10;
            rotated[(i + 1, i + 1)] = r11;
        }
        self.covariance = rotated * self.covariance * rotated.transpose();
        self.update_rect();
    }

    pub(crate) fn update(&mut self, new_track: &STrack, frame_id: usize) {
        self.frame_id = frame_id;
        self.tracklet_len += 1;
        let measurement = new_track.rect_to_xywh();
        self.kalman_filter.update(
            &mut self.mean,
            &mut self.covariance,
            &measurement,
        );
        self.update_rect();
        if let Some(feature) = new_track.curr_feat() {
            self.update_features(feature.to_vec());
        }
        self.state = STrackState::Tracked;
        self.is_activated = true;
        self.score = new_track.score;
    }

    pub(crate) fn mark_lost(&mut self) {
        self.state = STrackState::Lost;
    }

    pub(crate) fn mark_removed(&mut self) {
        self.state = STrackState::Removed;
    }

    fn rect_to_xywh(&self) -> DetectBox {
        DetectBox::from_iterator([
            self.rect.x() + self.rect.width() / 2.0,
            self.rect.y() + self.rect.height() / 2.0,
            self.rect.width(),
            self.rect.height(),
        ])
    }

    fn update_rect(&mut self) {
        self.rect.set_width(self.mean[(0, 2)]);
        self.rect.set_height(self.mean[(0, 3)]);
        self.rect.set_x(self.mean[(0, 0)] - self.rect.width() / 2.0);
        self.rect
            .set_y(self.mean[(0, 1)] - self.rect.height() / 2.0);
    }
}

impl From<STrack> for Object {
    fn from(track: STrack) -> Self {
        Object::new(
            track.get_rect(),
            track.get_score(),
            Some(track.get_track_id()),
        )
    }
}

impl From<&STrack> for Object {
    fn from(track: &STrack) -> Self {
        Object::new(
            track.get_rect(),
            track.get_score(),
            Some(track.get_track_id()),
        )
    }
}

fn normalize(feature: &mut [f32]) {
    let norm = feature.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 && norm.is_finite() {
        for value in feature {
            *value /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_update_normalizes() {
        let mut track = STrack::new(Rect::new(0.0, 0.0, 10.0, 20.0), 0.9, None);
        track.update_features(vec![3.0, 4.0]);

        let feature = track.smooth_feat().unwrap();
        assert!((feature[0] - 0.6).abs() < 1e-6);
        assert!((feature[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn activate_after_first_frame_starts_unconfirmed() {
        let mut track = STrack::new(Rect::new(0.0, 0.0, 10.0, 20.0), 0.9, None);
        track.activate(2, 1);

        assert_eq!(track.get_state(), STrackState::Tracked);
        assert!(!track.is_activated());
    }
}
