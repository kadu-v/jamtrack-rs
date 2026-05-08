import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np

from bot_sort import BoTSORT

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
        description="Run minimal BoT-SORT using YOLOX ONNX JSON outputs."
    )
    parser.add_argument("--inputs-dir", type=Path, default=Path("data/onnx_inputs"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("data/onnx_outputs"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/jsons/bot_sort_python.json"),
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--track-high-thresh", type=float, default=0.6)
    parser.add_argument("--track-low-thresh", type=float, default=0.1)
    parser.add_argument("--new-track-thresh", type=float, default=0.7)
    parser.add_argument("--match-thresh", type=float, default=0.8)
    parser.add_argument("--name", type=str, default="BoT-SORT-Python")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    input_paths = sorted(args.inputs_dir.glob("*.jpg"))
    if not input_paths:
        raise SystemExit(f"No input frames found in: {args.inputs_dir}")
    if args.max_frames > 0:
        input_paths = input_paths[: args.max_frames]

    tracker = BoTSORT(
        frame_rate=args.fps,
        track_buffer=args.track_buffer,
        track_high_thresh=args.track_high_thresh,
        track_low_thresh=args.track_low_thresh,
        new_track_thresh=args.new_track_thresh,
        match_thresh=args.match_thresh,
    )

    results = []
    for input_path in input_paths:
        frame_id = parse_frame_id(input_path)
        if frame_id is None:
            continue
        det_path = args.outputs_dir / f"{input_path.stem}.json"
        tracks = tracker.update(load_detections(det_path))

        for track in tracks:
            tlwh = track.tlwh
            results.append(
                {
                    "frame_id": str(frame_id),
                    "track_id": str(int(track.track_id)),
                    "x": str(float(tlwh[0])),
                    "y": str(float(tlwh[1])),
                    "width": str(float(tlwh[2])),
                    "height": str(float(tlwh[3])),
                    "score": str(float(track.score)),
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
