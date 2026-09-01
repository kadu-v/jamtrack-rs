use crate::{Object, Rect};

use super::kalman_filter::{KalmanFilter, Measurement, StateCov, StateMean};

#[derive(Debug, Clone)]
pub(crate) struct STrack {
    kalman: KalmanFilter,
    mean: StateMean,
    covariance: StateCov,
    score: f32,
    track_id: Option<usize>,
    time_since_update: usize,
    successful_updates: usize,
}

impl STrack {
    pub(crate) fn new(object: &Object) -> Self {
        let rect = object.get_rect();
        let measurement = rect_to_xywh(&rect);
        let kalman = KalmanFilter::new();
        let (mean, covariance) = kalman.initiate(&measurement);
        Self {
            kalman,
            mean,
            covariance,
            score: object.get_prob(),
            track_id: None,
            time_since_update: 0,
            successful_updates: 1,
        }
    }

    pub(crate) fn predict(&mut self) {
        self.kalman.predict(&mut self.mean, &mut self.covariance);
        self.time_since_update += 1;
    }

    pub(crate) fn update(&mut self, object: &Object) {
        let measurement = rect_to_xywh(&object.get_rect());
        self.kalman
            .update(&mut self.mean, &mut self.covariance, &measurement);
        self.score = object.get_prob();
        self.time_since_update = 0;
        self.successful_updates += 1;
    }

    pub(crate) fn apply_cmc(&mut self, transform: &[[f32; 3]; 3]) {
        let r00 = transform[0][0];
        let r01 = transform[0][1];
        let r10 = transform[1][0];
        let r11 = transform[1][1];
        for base in [0usize, 4] {
            let x = self.mean[(0, base)];
            let y = self.mean[(0, base + 1)];
            self.mean[(0, base)] = r00 * x + r01 * y;
            self.mean[(0, base + 1)] = r10 * x + r11 * y;
        }
        self.mean[(0, 0)] += transform[0][2];
        self.mean[(0, 1)] += transform[1][2];
        self.mean[(0, 2)] = self.mean[(0, 2)].max(1e-3);
        self.mean[(0, 3)] = self.mean[(0, 3)].max(1e-3);

        let mut rotation = StateCov::identity();
        for i in [0usize, 4] {
            rotation[(i, i)] = r00;
            rotation[(i, i + 1)] = r01;
            rotation[(i + 1, i)] = r10;
            rotation[(i + 1, i + 1)] = r11;
        }
        self.covariance = rotation * self.covariance * rotation.transpose();
    }

    pub(crate) fn rect(&self) -> Rect<f32> {
        Rect::new(
            self.mean[(0, 0)] - self.mean[(0, 2)] * 0.5,
            self.mean[(0, 1)] - self.mean[(0, 3)] * 0.5,
            self.mean[(0, 2)],
            self.mean[(0, 3)],
        )
    }

    pub(crate) fn track_id(&self) -> Option<usize> {
        self.track_id
    }

    pub(crate) fn assign_id(&mut self, track_id: usize) {
        self.track_id = Some(track_id);
    }

    pub(crate) fn time_since_update(&self) -> usize {
        self.time_since_update
    }

    pub(crate) fn successful_updates(&self) -> usize {
        self.successful_updates
    }

    pub(crate) fn to_object(&self) -> Object {
        Object::new(self.rect(), self.score, self.track_id)
    }
}

fn rect_to_xywh(rect: &Rect<f32>) -> Measurement {
    Measurement::from_iterator([
        rect.x() + rect.width() * 0.5,
        rect.y() + rect.height() * 0.5,
        rect.width().max(1e-3),
        rect.height().max(1e-3),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cmc_rotates_center_but_preserves_box_size() {
        let object = Object::new(Rect::new(10.0, 20.0, 30.0, 40.0), 0.9, None);
        let mut track = STrack::new(&object);
        track.apply_cmc(&[[0.0, -1.0, 5.0], [1.0, 0.0, -3.0], [0.0, 0.0, 1.0]]);
        let rect = track.rect();
        assert!((rect.x() + 50.0).abs() < 1e-5);
        assert!((rect.y() - 2.0).abs() < 1e-5);
        assert!((rect.width() - 30.0).abs() < 1e-5);
        assert!((rect.height() - 40.0).abs() < 1e-5);
    }
}
