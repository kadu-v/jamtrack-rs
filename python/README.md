# Python Utilities

## Render ByteTrack video from ONNX outputs

This command loads YOLOX ONNX output JSON files from `data/onnx_outputs`,
runs ByteTrack on-the-fly, and renders bbox + tracking IDs into a video.

```sh
uv run python python/byte_tracker/render_tracking_video.py \
  --frames-dir data/onnx_inputs \
  --outputs-dir data/onnx_outputs \
  --output-video data/video/tracking_from_onnx.mp4 \
  --fps 30
```

Optional tracker params:

```sh
uv run python python/bytetracker/render_tracking_video.py \
  --frames-dir data/onnx_inputs \
  --outputs-dir data/onnx_outputs \
  --output-video data/video/tracking_from_onnx.mp4 \
  --fps 30 \
  --track-buffer 30 \
  --track-thresh 0.45 \
  --match-thresh 0.8
```

## Render BoostTrack video from ONNX outputs

This command loads YOLOX ONNX output JSON files from `data/onnx_outputs`,
runs BoostTrack on-the-fly, and renders bbox + tracking IDs into a video.

```sh
uv run python python/boost_tracker/render_tracking_video.py \
  --frames-dir data/onnx_inputs \
  --outputs-dir data/onnx_outputs \
  --output-video data/video/boosttrack_from_onnx.mp4 \
  --fps 30
```

Optional tracker params:

```sh
uv run python python/boost_tracker/render_tracking_video.py \
  --frames-dir data/onnx_inputs \
  --outputs-dir data/onnx_outputs \
  --output-video data/video/boosttrack_from_onnx.mp4 \
  --fps 30 \
  --det-thresh 0.5 \
  --iou-thresh 0.3 \
  --min-hits 3 \
  --max-age 30
```
