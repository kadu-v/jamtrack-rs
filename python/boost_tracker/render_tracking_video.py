import argparse
import json
import re
from pathlib import Path

from typing import Optional

import cv2
import numpy as np

from boost_track import BoostTracker

FRAME_RE = re.compile(r"frame_(\d+)", re.IGNORECASE)


def parse_frame_id(path: Path) -> Optional[int]:
    match = FRAME_RE.search(path.stem)
    if match is None:
        return None
    return int(match.group(1))


def load_detections(json_path: Path) -> np.ndarray:
    if not json_path.exists():
        return np.empty((0, 6), dtype=np.float32)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        return np.empty((0, 6), dtype=np.float32)

    dets = np.zeros((len(data), 6), dtype=np.float32)
    for idx, det in enumerate(data):
        dets[idx, 0] = float(det["x1"])
        dets[idx, 1] = float(det["y1"])
        dets[idx, 2] = float(det["x2"])
        dets[idx, 3] = float(det["y2"])
        dets[idx, 4] = float(det["score"])
        dets[idx, 5] = float(det.get("class_id", -1))
    return dets


def color_for_track(track_id: int) -> tuple[int, int, int]:
    # Deterministic pseudo-random color from track id.
    r = (track_id * 37) % 255
    g = (track_id * 17) % 255
    b = (track_id * 97) % 255
    base = np.array([b, g, r], dtype=np.float32)
    bright = np.clip(0.5 * base + 128.0, 0, 255)
    return int(bright[0]), int(bright[1]), int(bright[2])


def draw_track(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, track_id: int) -> None:
    color = color_for_track(track_id)
    # Outline for visibility
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"id:{track_id}"
    label_pos = (x1, max(0, y1 - 6))
    cv2.putText(
        frame,
        label,
        label_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        label_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render BoostTrack results to a video."
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("data/onnx_inputs"),
        help="Directory containing input frames (jpg).",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("data/onnx_outputs"),
        help="Directory containing YOLOX JSON outputs.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=Path("data/video/boosttrack_from_onnx.mp4"),
        help="Output video path (mp4).",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--det-thresh", type=float, default=None)
    parser.add_argument("--iou-thresh", type=float, default=None)
    parser.add_argument("--min-hits", type=int, default=None)
    parser.add_argument("--max-age", type=int, default=None)
    args = parser.parse_args()

    frame_paths = sorted(args.frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise SystemExit(f"No frames found in: {args.frames_dir}")

    tracker = BoostTracker(
        frame_rate=int(args.fps),
        det_thresh=args.det_thresh,
        iou_threshold=args.iou_thresh,
        min_hits=args.min_hits,
        max_age=args.max_age,
    )

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise SystemExit(f"Failed to read: {frame_paths[0]}")

    height, width = first_frame.shape[:2]
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        frame_id = parse_frame_id(frame_path)
        if frame_id is None:
            writer.write(frame)
            continue

        det_path = args.outputs_dir / f"{frame_path.stem}.json"
        dets = load_detections(det_path)
        tracks = tracker.update(dets, frame, tag=frame_path.stem)
        if tracks.size == 0:
            writer.write(frame)
            continue

        for x1, y1, x2, y2, track_id, _, _ in tracks.tolist():
            x1_i = int(float(x1))
            y1_i = int(float(y1))
            x2_i = int(float(x2))
            y2_i = int(float(y2))
            track_id_i = int(track_id)
            draw_track(frame, x1_i, y1_i, x2_i, y2_i, track_id_i)

        writer.write(frame)

    writer.release()
    tracker.dump_cache()
    print(f"Saved video to: {args.output_video}")


if __name__ == "__main__":
    main()
