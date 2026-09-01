"""Generate RF-DETR instance masks and detections for the McByte demo."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from rfdetr import RFDETRSegNano


def encode_rle(mask: np.ndarray) -> list[int]:
    """Encode a binary mask as alternating row-major zero/one run lengths."""
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    if flat.size == 0:
        return []
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate(([0], changes, [flat.size]))).tolist()
    if flat[0] != 0:
        counts.insert(0, 0)
    return [int(value) for value in counts]


def video_metadata(path: Path) -> tuple[int, int, float, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return width, height, fps, frame_count


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def existing_frame_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    result: set[int] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") == "frame":
                result.add(int(record["frame_id"]))
    return result


def frame_id_from_path(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def pending_frame_paths(
    frame_paths: list[Path], completed: set[int]
) -> list[Path]:
    return [path for path in frame_paths if frame_id_from_path(path) not in completed]


def detection_record(
    box: np.ndarray,
    score: float,
    class_id: int,
    class_name: str,
    mask: np.ndarray,
    source_width: int,
    source_height: int,
    mask_width: int,
    mask_height: int,
) -> dict[str, object]:
    scaled = cv2.resize(
        mask.astype(np.uint8),
        (mask_width, mask_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    scale_x = mask_width / source_width
    scale_y = mask_height / source_height
    x1, y1, x2, y2 = box.tolist()
    x1 = float(np.clip(x1 * scale_x, 0.0, mask_width))
    y1 = float(np.clip(y1 * scale_y, 0.0, mask_height))
    x2 = float(np.clip(x2 * scale_x, 0.0, mask_width))
    y2 = float(np.clip(y2 * scale_y, 0.0, mask_height))
    return {
        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
        "score": float(score),
        "class_id": int(class_id),
        "class_name": class_name,
        "mask_rle": encode_rle(scaled),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RF-DETR-Seg masks for the Rust McByte demo."
    )
    parser.add_argument("--video", type=Path, default=Path("data/orig/original.mp4"))
    parser.add_argument(
        "--frames-dir", type=Path, default=Path("data/frames_30fps")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/jsons/mcbyte_rfdetr.jsonl")
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=Path("data/jsons/mcbyte_rfdetr_test.json"),
    )
    parser.add_argument("--test-frames", type=int, default=16)
    parser.add_argument("--mask-width", type=int, default=320)
    parser.add_argument("--mask-height", type=int, default=180)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--class-name",
        action="append",
        dest="class_names",
        help="COCO class name to retain; repeat for multiple classes (default: person).",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_names = set(args.class_names or ["person"])
    width, height, fps, video_frame_count = video_metadata(args.video)
    frame_paths = sorted(args.frames_dir.glob("frame_*.jpg"))
    if args.max_frames > 0:
        frame_paths = frame_paths[: args.max_frames]
    if not frame_paths:
        raise SystemExit(f"no input frames found in {args.frames_dir}")
    if args.mask_width <= 0 or args.mask_height <= 0:
        raise SystemExit("mask dimensions must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output.exists():
        args.output.unlink()
    completed = existing_frame_ids(args.output)
    metadata = {
        "type": "metadata",
        "schema_version": 1,
        "generator": "RFDETRSegNano",
        "rfdetr_version": "1.9.4",
        "source_video": str(args.video),
        "source_width": width,
        "source_height": height,
        "mask_width": args.mask_width,
        "mask_height": args.mask_height,
        "fps": fps,
        "frame_count": len(frame_paths),
        "video_frame_count": video_frame_count,
        "threshold": args.threshold,
        "class_names": sorted(class_names),
        "device": resolve_device(args.device),
        "host": platform.platform(),
    }
    if not args.output.exists():
        with args.output.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(metadata, separators=(",", ":")) + "\n")

    pending = pending_frame_paths(frame_paths, completed)
    if not pending:
        print(f"All {len(frame_paths)} frames already exist in {args.output}")
        if args.test_frames > 0 and not args.test_output.exists():
            fixture_frames = []
            with args.output.open("r", encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    if record.get("type") == "frame" and int(record["frame_id"]) < args.test_frames:
                        fixture_frames.append(record)
            fixture = {"metadata": metadata, "frames": fixture_frames[: args.test_frames]}
            args.test_output.parent.mkdir(parents=True, exist_ok=True)
            with args.test_output.open("w", encoding="utf-8") as stream:
                json.dump(fixture, stream, separators=(",", ":"))
                stream.write("\n")
        return

    device = metadata["device"]
    model = RFDETRSegNano(device=device)
    dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
    model.inference(compile=False, dtype=dtype, inplace=True)

    fixture_frames: list[dict[str, object]] = []
    started = time.perf_counter()
    with args.output.open("a", encoding="utf-8") as stream:
        for batch_start in range(0, len(pending), args.batch_size):
            paths = pending[batch_start : batch_start + args.batch_size]
            images_bgr = [cv2.imread(str(path)) for path in paths]
            if any(image is None for image in images_bgr):
                missing = [str(path) for path, image in zip(paths, images_bgr) if image is None]
                raise RuntimeError(f"failed to read frames: {missing}")
            images_rgb = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images_bgr]
            predictions = model.predict(
                images_rgb,
                threshold=args.threshold,
                include_source_image=False,
            )
            if not isinstance(predictions, list):
                predictions = [predictions]

            for path, prediction in zip(paths, predictions):
                frame_id = frame_id_from_path(path)
                names = prediction.data.get("class_name")
                if names is None:
                    names = np.array([str(value) for value in prediction.class_id])
                detections = []
                if prediction.mask is not None:
                    for box, score, class_id, class_name, mask in zip(
                        prediction.xyxy,
                        prediction.confidence,
                        prediction.class_id,
                        names,
                        prediction.mask,
                    ):
                        class_name = str(class_name)
                        if class_name not in class_names:
                            continue
                        detections.append(
                            detection_record(
                                box,
                                float(score),
                                int(class_id),
                                class_name,
                                mask,
                                width,
                                height,
                                args.mask_width,
                                args.mask_height,
                            )
                        )
                record = {
                    "type": "frame",
                    "frame_id": frame_id,
                    "detections": detections,
                }
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                if frame_id < args.test_frames:
                    fixture_frames.append(record)
            stream.flush()
            done = min(batch_start + len(paths), len(pending))
            elapsed = time.perf_counter() - started
            rate = done / elapsed
            print(
                f"RF-DETR: {done}/{len(pending)} frames "
                f"({rate:.1f} fps, {len(pending) - done:.0f} remaining)",
                flush=True,
            )

    # A small checked-in fixture keeps tests fast while the JSONL contains the
    # complete video sequence for demo reproduction.
    if args.test_frames > 0:
        if len(fixture_frames) < args.test_frames:
            fixture_frames = []
            with args.output.open("r", encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    if record.get("type") == "frame" and int(record["frame_id"]) < args.test_frames:
                        fixture_frames.append(record)
        fixture = {"metadata": metadata, "frames": fixture_frames[: args.test_frames]}
        args.test_output.parent.mkdir(parents=True, exist_ok=True)
        with args.test_output.open("w", encoding="utf-8") as stream:
            json.dump(fixture, stream, separators=(",", ":"))
            stream.write("\n")

    elapsed = time.perf_counter() - started
    print(f"Saved {len(frame_paths)} frames to {args.output} in {elapsed:.1f}s")
    print(f"Saved {len(fixture_frames[: args.test_frames])} test frames to {args.test_output}")


if __name__ == "__main__":
    # Keep model downloads inside the project unless the caller selected a cache.
    os.environ.setdefault("RF_HOME", str(Path(__file__).parent / ".model-cache"))
    main()
