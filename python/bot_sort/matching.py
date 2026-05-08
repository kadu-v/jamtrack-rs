import lap
import numpy as np


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    matches = []
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    return np.asarray(matches), unmatched_a, unmatched_b


def bbox_ious(boxes, query_boxes):
    overlaps = np.zeros((len(boxes), len(query_boxes)), dtype=np.float32)
    for k in range(len(query_boxes)):
        box_area = (query_boxes[k, 2] - query_boxes[k, 0] + 1) * (
            query_boxes[k, 3] - query_boxes[k, 1] + 1
        )
        for n in range(len(boxes)):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2])
                - max(boxes[n, 0], query_boxes[k, 0])
                + 1
            )
            if iw <= 0:
                continue
            ih = (
                min(boxes[n, 3], query_boxes[k, 3])
                - max(boxes[n, 1], query_boxes[k, 1])
                + 1
            )
            if ih <= 0:
                continue
            ua = (
                (boxes[n, 2] - boxes[n, 0] + 1)
                * (boxes[n, 3] - boxes[n, 1] + 1)
                + box_area
                - iw * ih
            )
            overlaps[n, k] = iw * ih / ua
    return overlaps


def iou_distance(atracks, btracks):
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float32),
        np.ascontiguousarray(btlbrs, dtype=np.float32),
    )
    return 1 - ious


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(
        cost_matrix.shape[0], axis=0
    )
    return 1 - iou_sim * det_scores
