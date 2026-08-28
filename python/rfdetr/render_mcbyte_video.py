"""Render RF-DETR masks and Rust McByte tracks over the original video."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def records(path: Path, record_type: str) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("type") == record_type:
                yield record


def metadata(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        record = json.loads(stream.readline())
    if record.get("type") != "metadata":
        raise ValueError(f"missing metadata record in {path}")
    return record


def decode_rle(counts: list[int], width: int, height: int) -> np.ndarray:
    flat = np.zeros(width * height, dtype=np.uint8)
    cursor = 0
    foreground = False
    for count in counts:
        end = cursor + int(count)
        if end > flat.size:
            raise ValueError("mask RLE exceeds configured dimensions")
        if foreground:
            flat[cursor:end] = 1
        cursor = end
        foreground = not foreground
    if cursor != flat.size:
        raise ValueError("mask RLE does not fill configured dimensions")
    return flat.reshape(height, width).astype(bool)


def color_for_track(track_id: int) -> np.ndarray:
    return np.array(
        [
            96 + (track_id * 97) % 160,
            96 + (track_id * 57) % 160,
            96 + (track_id * 37) % 160,
        ],
        dtype=np.uint8,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the McByte RF-DETR demo video.")
    parser.add_argument("--video", type=Path, default=Path("data/orig/original.mp4"))
    parser.add_argument(
        "--data", type=Path, default=Path("data/jsons/mcbyte_rfdetr.jsonl")
    )
    parser.add_argument(
        "--tracks",
        type=Path,
        default=Path("data/jsons/mcbyte_rfdetr_tracks.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/video/mcbyte_rfdetr.mp4")
    )
    parser.add_argument("--no-audio", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = metadata(args.data)
    mask_width = int(info["mask_width"])
    mask_height = int(info["mask_height"])
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"failed to open {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    scale_x = width / mask_width
    scale_y = height / mask_height

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mcbyte-rfdetr-") as temp_dir:
        silent_path = Path(temp_dir) / "silent.mp4"
        writer = cv2.VideoWriter(
            str(silent_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("failed to create temporary video writer")

        data_frames = records(args.data, "frame")
        track_frames = records(args.tracks, "frame")
        count = 0
        for data_frame, track_frame in zip(data_frames, track_frames):
            ok, frame = capture.read()
            if not ok:
                break
            if data_frame["frame_id"] != track_frame["frame_id"]:
                raise ValueError("data and tracking frame IDs are not aligned")
            detections = data_frame["detections"]
            for track in track_frame["tracks"]:
                track_id = int(track["track_id"])
                detection_index = int(track["detection_index"])
                detection = detections[detection_index]
                mask = decode_rle(
                    detection["mask_rle"], mask_width, mask_height
                ).astype(np.uint8)
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                color = color_for_track(track_id)
                frame[mask] = (
                    frame[mask].astype(np.float32) * 0.55
                    + color.astype(np.float32) * 0.45
                ).astype(np.uint8)

                x, y, box_width, box_height = track["bbox"]
                x1 = int(round(float(x) * scale_x))
                y1 = int(round(float(y) * scale_y))
                x2 = int(round((float(x) + float(box_width)) * scale_x))
                y2 = int(round((float(y) + float(box_height)) * scale_y))
                bgr = tuple(int(value) for value in color.tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                label = f"#{track_id}"
                label_origin = (x1 + 3, min(height - 6, max(20, y1 + 20)))
                cv2.putText(frame, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2, cv2.LINE_AA)

            cv2.putText(
                frame,
                "RF-DETR-Seg-Nano + McByte (Rust)",
                (24, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 0, 0),
                5,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "RF-DETR-Seg-Nano + McByte (Rust)",
                (24, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            count += 1
            if count % 100 == 0:
                print(f"Rendered {count} frames", flush=True)

        capture.release()
        writer.release()
        if args.no_audio:
            silent_path.replace(args.output)
        else:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(silent_path),
                    "-i",
                    str(args.video),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a?",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "20",
                    "-c:a",
                    "aac",
                    "-af",
                    "apad",
                    "-shortest",
                    str(args.output),
                ],
                check=True,
            )
    print(f"Saved {count} frames to {args.output}")


if __name__ == "__main__":
    main()
