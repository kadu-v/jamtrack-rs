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

## Generate the RF-DETR + McByte demo

RF-DETR-Seg-Nano produces COCO person detections and instance masks from
`data/orig/original.mp4`. The masks are resized to 320x180 and stored as
row-major RLE so the Rust example can consume the complete sequence without
requiring Python at tracking time.

```sh
# Install the isolated Python 3.10+ RF-DETR environment.
uv sync --project python/rfdetr

# Generate the full mask dataset and the 16-frame test fixture.
uv run --project python/rfdetr python \
  python/rfdetr/generate_mcbyte_data.py --overwrite

# Run the Rust McByte tracker with mask conditioning and SparseOptFlow CMC.
cargo run --release --example example_mcbyte_rfdetr

# Render masks, bounding boxes, IDs, and the original audio.
uv run --project python/rfdetr python \
  python/rfdetr/render_mcbyte_video.py
```

Generated artifacts:

- `data/jsons/mcbyte_rfdetr.jsonl`: all RF-DETR detections and masks
- `data/jsons/mcbyte_rfdetr_test.json`: compact integration-test fixture
- `data/jsons/mcbyte_rfdetr_tracks.jsonl`: Rust McByte output
- `data/video/mcbyte_rfdetr.mp4`: H.264 demo video with AAC audio
