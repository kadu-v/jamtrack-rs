import argparse
import json
import re
from pathlib import Path

from typing import Optional

import numpy as np

from byte_track import BYTETracker

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ByteTrack using YOLOX ONNX JSON outputs."
    )
    parser.add_argument(
        "--inputs-dir",
        type=Path,
        default=Path("data/onnx_inputs"),
        help="Directory containing ONNX input frames (jpg).",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("data/onnx_outputs"),
        help="Directory containing YOLOX JSON outputs.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/jsons/tracking_results_from_onnx.json"),
        help="Path to write ByteTrack results JSON.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--track-thresh", type=float, default=0.45)
    parser.add_argument("--match-thresh", type=float, default=0.8)
    parser.add_argument("--name", type=str, default="ByteTrack")
    args = parser.parse_args()

    input_paths = sorted(args.inputs_dir.glob("*.jpg"))
    if not input_paths:
        raise SystemExit(f"No input frames found in: {args.inputs_dir}")

    tracker = BYTETracker(
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        frame_rate=args.fps,
    )

    results = []
    for input_path in input_paths:
        frame_id = parse_frame_id(input_path)
        if frame_id is None:
            continue
        det_path = args.outputs_dir / f"{input_path.stem}.json"
        dets = load_detections(det_path)

        tracks = tracker.update(dets)

        if tracks.size == 0:
            continue
        for x1, y1, x2, y2, track_id, _, _ in tracks.tolist():
            results.append(
                {
                    "frame_id": str(frame_id),
                    "track_id": str(int(track_id)),
                    "x": str(float(x1)),
                    "y": str(float(y1)),
                    "width": str(float(x2 - x1)),
                    "height": str(float(y2 - y1)),
                }
            )

    output_payload = {
        "name": args.name,
        "fps": args.fps,
        "track_buffer": args.track_buffer,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=True, indent=2)

    print(f"Saved tracking results to: {args.output_json}")


if __name__ == "__main__":
    main()
