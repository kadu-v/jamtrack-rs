import cv2
import numpy as np
from byte_track import BYTETracker

# // 二つのボールが右から左にと、左から右に移動するようにアニメーションを作成
# let mut frames = vec![];
# for i in 0..30 {
#     let mut objects = vec![];

#     // 一つ目のボール
#     let x1 = i as f32 * 10.0;
#     let y1 = 100.0;
#     let prob1 = 1.0;

#     // 二つ目のボール
#     let x2 = 300.0 - i as f32 * 10.0;
#     let y2 = 100.0;
#     let prob2 = if x1 <= x2 && x2 <= x1 + 100. && x1 <= x2 + 100. {
#         let x = 1.0 - (x1 + 100.0 - x2).abs() / 100.0;
#         if x < 0.2 {
#             0.0
#         } else {
#             x
#         }
#     } else if x2 <= x1 && x1 <= x2 + 100. && x2 <= x1 + 100. {
#         let x = 1.0 - (x2 + 100.0 - x1).abs() / 100.0;
#         if x < 0.2 {
#             0.0
#         } else {
#             x
#         }
#     } else {
#         1.0
#     };
#     objects.push(bytetrack_rs::object::Object::new(
#         bytetrack_rs::rect::Rect::new(x1, y1, 100., 80.),
#         0,
#         prob1,
#     ));

#     if prob2 > 0.0 {
#         objects.push(bytetrack_rs::object::Object::new(
#             bytetrack_rs::rect::Rect::new(x2, y2, 100., 100.),
#             1,
#             prob2,
#         ));
#     }

#     frames.push(objects);
# }

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
]


def main():
    fps = 10
    tracker = BYTETracker(frame_rate=fps)

    frames = []

    # 一つ目のボール
    for t in range(30):
        objects = []
        x1 = t * 10
        y1 = 100
        prob1 = 1.0

        # 二つ目のボール
        x2 = 300 - t * 10
        y2 = 100
        if x1 <= x2 and x2 <= x1 + 100 and x1 <= x2 + 100:
            x = 1.0 - abs(x1 + 100 - x2) / 100
            prob2 = 0.0 if x < 0.2 else x
        elif x2 <= x1 and x1 <= x2 + 100 and x2 <= x1 + 100:
            x = 1.0 - abs(x2 + 100 - x1) / 100
            prob2 = 0.0 if x < 0.2 else x
        else:
            prob2 = 1.0

        objects.append(np.array((x1, y1, x1 + 100.0, y1 + 100.0, prob1, -1)))
        if prob2 > 0:
            objects.append(np.array((x2, y2, x2 + 100, y2 + 100, prob2, -1)))
        frames.append(np.array(objects))

    for i, frame in enumerate(frames):
        targets = tracker.update(frame)

        img = np.zeros((400, 400, 3), dtype=np.uint8)
        # draw the bounding boxes
        for target in targets:
            x1, y1, x2, y2, obj_id, _, _ = target
            print(x1, y1, x2, y2, obj_id)
            cv2.rectangle(
                img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                COLORS[int(obj_id) % len(COLORS)],
                2,
            )

            # save the image
        cv2.imwrite(f"output/frame_{i}.jpg", img)


if __name__ == "__main__":
    main()
