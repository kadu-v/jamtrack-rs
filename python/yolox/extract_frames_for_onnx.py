import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def preprocess_for_yolox(img, input_size):
    resized = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32).transpose(2, 0, 1)
    tensor = np.expand_dims(tensor, axis=0)
    return tensor, resized


def extract_frames(
    video_path: Path,
    frames_dir: Path,
    inputs_dir: Path,
    target_fps: float,
    input_size: int,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        raise RuntimeError("Failed to read FPS from video.")

    frames_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    # inputs_dir stores letterboxed images used for ONNX input.

    frame_idx = 0
    saved_idx = 0
    next_time = 0.0
    interval = 1.0 / target_fps

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_time = frame_idx / src_fps
        if frame_time + 1e-6 < next_time:
            frame_idx += 1
            continue

        frame_name = f"frame_{saved_idx:06d}.jpg"
        frame_path = frames_dir / frame_name
        cv2.imwrite(str(frame_path), frame)

        _, resized = preprocess_for_yolox(frame, input_size)
        input_path = inputs_dir / frame_name
        cv2.imwrite(str(input_path), resized)

        saved_idx += 1
        next_time += interval
        frame_idx += 1

    cap.release()
    return saved_idx, src_fps


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames at target FPS and prepare YOLOX ONNX inputs."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("data/orig/original.mp4"),
        help="Input video path.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("data/frames_30fps"),
        help="Directory for extracted frames.",
    )
    parser.add_argument(
        "--inputs-dir",
        type=Path,
        default=Path("data/onnx_inputs"),
        help="Directory for preprocessed ONNX input images (jpg).",
    )
    parser.add_argument(
        "--fps", type=float, default=30.0, help="Target FPS for frame extraction."
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=640,
        help="YOLOX input size (square).",
    )
    args = parser.parse_args()

    saved, src_fps = extract_frames(
        args.video,
        args.frames_dir,
        args.inputs_dir,
        args.fps,
        args.input_size,
    )
    print(
        f"Extracted {saved} frames at {args.fps}fps (source fps: {src_fps:.2f})."
    )


if __name__ == "__main__":
    main()
