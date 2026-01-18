#!/usr/bin/env python3
import argparse
import json
from typing import Iterable, List, Optional

import numpy as np
from kalmanfilter import KalmanFilter


def bbox_to_z(bbox: Iterable[float]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    x = x1 + w / 2.0
    y = y1 + h / 2.0
    r = w / float(h + 1e-6)
    return np.array([x, y, h, r]).reshape((4, 1))


def parse_vec(arg: str, size: int, name: str) -> List[float]:
    parts = [float(v) for v in arg.split(",") if v != ""]
    if len(parts) != size:
        raise ValueError(f"{name} must have {size} comma-separated values")
    return parts


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print BoostTrack Kalman filter intermediate values."
    )
    parser.add_argument(
        "--bbox",
        help="Init bbox as x1,y1,x2,y2 (uses convert to z).",
    )
    parser.add_argument(
        "--z",
        help="Init measurement z as x,y,h,r.",
    )
    parser.add_argument(
        "--update-bbox",
        help="Update bbox as x1,y1,x2,y2 (uses convert to z).",
    )
    parser.add_argument(
        "--update-z",
        help="Update measurement z as x,y,h,r.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.0,
        help="Confidence score passed to update/project (default: 0).",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
        help="Force numpy dtype for outputs (default: float64).",
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Skip the update step.",
    )

    args = parser.parse_args()

    if (args.bbox is None) == (args.z is None):
        raise ValueError("Provide exactly one of --bbox or --z for init.")

    dtype = np.float32 if args.dtype == "float32" else np.float64

    if args.bbox is not None:
        init_bbox = parse_vec(args.bbox, 4, "--bbox")
        z_init = bbox_to_z(init_bbox).astype(dtype)
    else:
        z_vals = parse_vec(args.z, 4, "--z")
        z_init = np.array(z_vals, dtype=dtype).reshape((4, 1))
    kf = KalmanFilter(z_init)
    kf.x = kf.x.astype(dtype)
    kf.covariance = kf.covariance.astype(dtype)

    out = {
        "init": {
            "x": kf.x.copy(),
            "covariance": kf.covariance.copy(),
        }
    }

    pred_mean, pred_cov = kf.predict()
    pred_mean = pred_mean.astype(dtype)
    pred_cov = pred_cov.astype(dtype)
    kf.x = pred_mean
    kf.covariance = pred_cov
    out["predict"] = {
        "x": pred_mean.copy(),
        "mean": pred_mean.copy(),
        "covariance": pred_cov.copy(),
    }

    proj_mean, proj_cov = kf.project(args.confidence)
    proj_mean = proj_mean.astype(dtype)
    proj_cov = proj_cov.astype(dtype)
    out["project"] = {
        "mean": proj_mean.copy(),
        "covariance": proj_cov.copy(),
    }

    if not args.no_update:
        if args.update_bbox is not None:
            update_bbox = parse_vec(args.update_bbox, 4, "--update-bbox")
            z_update = bbox_to_z(update_bbox).astype(dtype)
        elif args.update_z is not None:
            z_vals = parse_vec(args.update_z, 4, "--update-z")
            z_update = np.array(z_vals, dtype=dtype).reshape((4, 1))
        else:
            z_update = z_init
        upd_mean, upd_cov = kf.update(z_update, args.confidence)
        upd_mean = upd_mean.astype(dtype)
        upd_cov = upd_cov.astype(dtype)
        kf.x = upd_mean
        kf.covariance = upd_cov
        out["update"] = {
            "x": upd_mean.copy(),
            "covariance": upd_cov.copy(),
        }

    print(json.dumps(to_jsonable(out), indent=2))


if __name__ == "__main__":
    main()
