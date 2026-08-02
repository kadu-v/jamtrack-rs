"""Print parity values from upstream FastTracker commit d35eeda120981c7138b70d13f87bcb4f9fbcbe52.

Run from ``python/`` with ``uv run python fast_tracker/generate_reference_values.py``.
The Rust unit tests contain these values so CI does not require Python.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
KALMAN_PATH = ROOT / "FastTracker/yolox/tracker/kalman_filter.py"
TRACKER_DIR = ROOT / "FastTracker/yolox/tracker"


def load_kalman_filter():
    spec = importlib.util.spec_from_file_location("fasttracker_kalman", KALMAN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.KalmanFilter


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_fasttracker():
    """Load upstream modules without importing YOLOX's detector/torchvision stack."""
    yolox = types.ModuleType("yolox")
    yolox.__path__ = []
    tracker_package = types.ModuleType("yolox.tracker")
    tracker_package.__path__ = [str(TRACKER_DIR)]
    sys.modules["yolox"] = yolox
    sys.modules["yolox.tracker"] = tracker_package

    base = load_module("yolox.tracker.basetrack", TRACKER_DIR / "basetrack.py")
    kalman = load_module(
        "yolox.tracker.kalman_filter", TRACKER_DIR / "kalman_filter.py"
    )
    matching = load_module("yolox.tracker.matching", TRACKER_DIR / "matching.py")
    tracker_package.basetrack = base
    tracker_package.kalman_filter = kalman
    tracker_package.matching = matching
    fasttracker = load_module(
        "yolox.tracker.fasttracker", TRACKER_DIR / "fasttracker.py"
    )
    return fasttracker, base


def plain_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    if intersection == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return intersection / (area_a + area_b - intersection + 1e-9)


def main():
    kalman = load_kalman_filter()()
    measurement = np.array([10.0, 20.0, 0.5, 40.0])
    mean, covariance = kalman.initiate(measurement)
    predicted_mean, predicted_covariance = kalman.predict(mean, covariance)
    projected_mean, projected_covariance = kalman.project(
        predicted_mean, predicted_covariance
    )
    updated_mean, updated_covariance = kalman.update(
        predicted_mean,
        predicted_covariance,
        np.array([12.0, 23.0, 0.55, 42.0]),
    )
    fasttracker, base = load_fasttracker()
    base.BaseTrack._count = 0
    args = types.SimpleNamespace(mot20=False)
    config = {
        "track_thresh": 0.6,
        "track_buffer": 30,
        "match_thresh": 0.8,
        "reset_velocity_offset_occ": 1,
        "reset_pos_offset_occ": 1,
        "enlarge_bbox_occ": 1.2,
        "dampen_motion_occ": 0.5,
        "active_occ_to_lost_thresh": 3,
        "init_iou_suppress": 0.8,
        "roi_repair_max_gap": 15,
        "dir_window_N": 10,
        "dir_margin_deg": 2.0,
    }
    tracker = fasttracker.Fasttracker(args, config, frame_rate=30)
    frames = [
        [[40, 40, 50, 50, 0.9], [0, 0, 100, 100, 0.95]],
        [[0, 0, 100, 100, 0.95]],
        [[0, 0, 100, 100, 0.95]],
        [[0, 0, 100, 100, 0.95]],
        [[40, 40, 50, 50, 0.9], [0, 0, 100, 100, 0.95]],
    ]
    tracking_frames = []
    for frame in frames:
        tracks = tracker.update(np.asarray(frame, dtype=np.float64), [100, 100], [100, 100])
        tracking_frames.append(
            [
                {
                    "track_id": int(track.track_id),
                    "tlwh": track.tlwh.tolist(),
                    "score": float(track.score),
                    "state": int(track.state),
                    "is_occluded": bool(track.is_occluded),
                    "not_matched": int(track.not_matched),
                    "occluded_len": int(track.occluded_len),
                    "mean": track.mean.tolist(),
                    "covariance": track.covariance.tolist(),
                }
                for track in tracks
            ]
        )

    payload = {
        "source_commit": "d35eeda120981c7138b70d13f87bcb4f9fbcbe52",
        "kalman": {
            "initiated_mean": mean.tolist(),
            "initiated_covariance": covariance.tolist(),
            "predicted_mean": predicted_mean.tolist(),
            "predicted_covariance": predicted_covariance.tolist(),
            "projected_mean": projected_mean.tolist(),
            "projected_covariance": projected_covariance.tolist(),
            "updated_mean": updated_mean.tolist(),
            "updated_covariance": updated_covariance.tolist(),
        },
        "geometry": {
            "plain_iou": plain_iou(
                [0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]
            )
        },
        "tracking_frames": tracking_frames,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
