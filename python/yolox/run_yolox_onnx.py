import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def preprocess(img_bgr, input_shape):
    input_h, input_w = input_shape
    h, w = img_bgr.shape[:2]
    r = min(input_h / h, input_w / w)
    resized_h, resized_w = int(h * r), int(w * r)
    resized = cv2.resize(
        img_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
    )

    padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    padded[:resized_h, :resized_w] = resized
    padded = padded.transpose(2, 0, 1)
    padded = np.ascontiguousarray(padded, dtype=np.float32)
    return padded, r


def make_grids(img_size, strides):
    grids = []
    expanded_strides = []
    if isinstance(img_size, int):
        img_h, img_w = img_size, img_size
    else:
        img_h, img_w = img_size

    for stride in strides:
        hsize = img_h // stride
        wsize = img_w // stride
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((*grid.shape[:2], 1), stride))
    return np.concatenate(grids, axis=1), np.concatenate(expanded_strides, axis=1)


def decode_outputs(outputs, img_size, strides):
    grids, expanded_strides = make_grids(img_size, strides)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    return outputs


def xywh_to_xyxy(xywh):
    cx, cy, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def nms(boxes, scores, iou_thresh):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


def demo_postprocess(outputs, img_size, strides=(8, 16, 32)):
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    outputs = decode_outputs(outputs, img_size, strides)
    return outputs


def multiclass_nms(boxes, scores, nms_thr=0.45, score_thr=0.1):
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_mask = cls_scores > score_thr
        if not np.any(valid_mask):
            continue
        valid_scores = cls_scores[valid_mask]
        valid_boxes = boxes[valid_mask]
        keep = nms(valid_boxes, valid_scores, nms_thr)
        for idx in keep:
            final_dets.append(
                [*valid_boxes[idx].tolist(), float(valid_scores[idx]), float(cls_ind)]
            )
    if not final_dets:
        return None
    return np.array(final_dets, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Run YOLOX-X ONNX inference.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("data/onnx/yolox_x.onnx"),
        help="YOLOX-X ONNX model path.",
    )
    parser.add_argument(
        "--inputs-dir",
        type=Path,
        default=Path("data/onnx_inputs"),
        help="Directory with input images (jpg).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/onnx_outputs"),
        help="Directory for output visualizations.",
    )
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Detect all classes.",
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument(
        "--image",
        type=Path,
        help="Run on a single image (jpg).",
    )
    args = parser.parse_args()

    providers = [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    session = ort.InferenceSession(str(args.model), providers=providers)
    input_name = session.get_inputs()[0].name

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image is not None:
        image_paths = [args.image]
    else:
        image_paths = sorted(args.inputs_dir.glob("*.jpg"))
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            continue

        input_shape = (args.input_size, args.input_size)
        img_input, ratio = preprocess(img, input_shape)
        outputs = session.run(None, {input_name: img_input[None, :, :, :]})[0]
        predictions = demo_postprocess(outputs, input_shape)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= ratio

        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=args.iou, score_thr=0.1)
        vis = img.copy()
        json_dets = []
        if dets is not None:
            if not args.all_classes:
                dets = dets[dets[:, 5] == 0]

            for x1, y1, x2, y2, score, cls_id in dets:
                px1 = int(x1)
                py1 = int(y1)
                px2 = int(x2)
                py2 = int(y2)
                if score < args.conf:
                    continue
                json_dets.append(
                    {
                        "x1": px1,
                        "y1": py1,
                        "x2": px2,
                        "y2": py2,
                        "score": float(score),
                        "class_id": int(cls_id),
                    }
                )
                print(
                    f"{image_path.name} det: "
                    f"x1={px1}, y1={py1}, x2={px2}, y2={py2}, "
                    f"score={score:.3f}, cls={int(cls_id)}"
                )
                cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 255, 0), 2)
                cv2.putText(
                    vis,
                    f"{score:.2f}",
                    (px1, max(0, py1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        out_path = output_dir / image_path.name
        cv2.imwrite(str(out_path), vis)
        json_path = output_dir / f"{image_path.stem}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(json_dets, f, ensure_ascii=True)

        if dets is None:
            print(f"{image_path.name}: 0 detections")
        else:
            print(f"{image_path.name}: {len(dets)} detections")

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
