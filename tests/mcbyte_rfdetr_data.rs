use image::GrayImage;
use jamtrack_rs::{McByteMask, McByteTracker, Object, Rect};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    metadata: Metadata,
    frames: Vec<Frame>,
}

#[derive(Deserialize)]
struct Metadata {
    schema_version: usize,
    generator: String,
    mask_width: u32,
    mask_height: u32,
    class_names: Vec<String>,
}

#[derive(Deserialize)]
struct Frame {
    frame_id: usize,
    detections: Vec<Detection>,
}

#[derive(Deserialize)]
struct Detection {
    bbox: [f32; 4],
    score: f32,
    class_id: usize,
    class_name: String,
    mask_rle: Vec<u32>,
}

fn decode_rle(counts: &[u32], width: u32, height: u32) -> GrayImage {
    let length = width as usize * height as usize;
    let mut pixels = vec![0u8; length];
    let mut cursor = 0usize;
    let mut foreground = false;
    for &count in counts {
        let end = cursor + count as usize;
        assert!(end <= length);
        if foreground {
            pixels[cursor..end].fill(255);
        }
        cursor = end;
        foreground = !foreground;
    }
    assert_eq!(cursor, length);
    GrayImage::from_raw(width, height, pixels).unwrap()
}

fn masks_for_previous_tracks(
    previous: &[Object],
    current: &[Object],
    masks: &[GrayImage],
    scores: &[f32],
) -> Vec<McByteMask> {
    let mut candidates = previous
        .iter()
        .enumerate()
        .flat_map(|(previous_index, track)| {
            current
                .iter()
                .enumerate()
                .map(move |(current_index, detection)| {
                    (
                        track.get_rect().calc_iou(&detection.get_rect()),
                        previous_index,
                        current_index,
                    )
                })
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| right.0.total_cmp(&left.0));
    let mut used_previous = vec![false; previous.len()];
    let mut used_current = vec![false; current.len()];
    let mut result = Vec::new();
    for (iou, previous_index, current_index) in candidates {
        if iou <= 0.0
            || used_previous[previous_index]
            || used_current[current_index]
        {
            continue;
        }
        used_previous[previous_index] = true;
        used_current[current_index] = true;
        result.push(McByteMask::new(
            previous[previous_index].get_track_id().unwrap(),
            masks[current_index].clone(),
            scores[current_index],
        ));
    }
    result
}

#[test]
fn rfdetr_fixture_is_valid_and_tracks_with_mcbyte() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../data/jsons/mcbyte_rfdetr_test.json"
    ))
    .unwrap();
    assert_eq!(fixture.metadata.schema_version, 1);
    assert_eq!(fixture.metadata.generator, "RFDETRSegNano");
    assert_eq!(fixture.metadata.class_names, ["person"]);
    assert_eq!(fixture.frames.len(), 16);

    let mut tracker = McByteTracker::new(30, 30, 0.5, 0.4)
        .with_isolated_mask_matching(true)
        .without_cmc();
    let mut previous = Vec::<Object>::new();
    for (expected_frame_id, frame) in fixture.frames.iter().enumerate() {
        assert_eq!(frame.frame_id, expected_frame_id);
        assert!(!frame.detections.is_empty());
        let objects = frame
            .detections
            .iter()
            .map(|detection| {
                assert_eq!(detection.class_id, 1);
                assert_eq!(detection.class_name, "person");
                assert!(detection.score >= 0.4);
                assert!(detection.bbox[2] > 0.0 && detection.bbox[3] > 0.0);
                Object::new(
                    Rect::new(
                        detection.bbox[0],
                        detection.bbox[1],
                        detection.bbox[2],
                        detection.bbox[3],
                    ),
                    detection.score,
                    None,
                )
            })
            .collect::<Vec<_>>();
        let masks = frame
            .detections
            .iter()
            .map(|detection| {
                let mask = decode_rle(
                    &detection.mask_rle,
                    fixture.metadata.mask_width,
                    fixture.metadata.mask_height,
                );
                assert!(mask.as_raw().iter().any(|&pixel| pixel != 0));
                mask
            })
            .collect::<Vec<_>>();
        let scores = frame
            .detections
            .iter()
            .map(|detection| detection.score)
            .collect::<Vec<_>>();
        let propagated =
            masks_for_previous_tracks(&previous, &objects, &masks, &scores);
        let output = tracker.update_with_masks(&objects, &propagated).unwrap();
        assert!(!output.is_empty());
        assert!(output.iter().any(|track| track.get_track_id() == Some(1)));
        previous = output;
    }
}
