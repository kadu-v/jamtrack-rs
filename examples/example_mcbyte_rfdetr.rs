use std::{
    env,
    error::Error,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};

use image::{GrayImage, imageops::FilterType};
use jamtrack_rs::{
    McByteMask, McByteTracker, Object, Rect, SparseOptFlowConfig,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
#[serde(tag = "type")]
enum InputRecord {
    #[serde(rename = "metadata")]
    Metadata {
        schema_version: usize,
        source_video: String,
        mask_width: u32,
        mask_height: u32,
        fps: f32,
        frame_count: usize,
    },
    #[serde(rename = "frame")]
    Frame {
        frame_id: usize,
        detections: Vec<InputDetection>,
    },
}

#[derive(Deserialize)]
struct InputDetection {
    bbox: [f32; 4],
    score: f32,
    class_id: usize,
    class_name: String,
    mask_rle: Vec<u32>,
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum OutputRecord<'a> {
    #[serde(rename = "metadata")]
    Metadata {
        schema_version: usize,
        source_data: &'a str,
        source_video: &'a str,
        mask_width: u32,
        mask_height: u32,
        fps: f32,
        frame_count: usize,
        tracker: &'static str,
    },
    #[serde(rename = "frame")]
    Frame {
        frame_id: usize,
        tracks: Vec<OutputTrack>,
    },
}

#[derive(Serialize)]
struct OutputTrack {
    track_id: usize,
    detection_index: usize,
    bbox: [f32; 4],
    score: f32,
    class_id: usize,
    class_name: String,
}

struct PreviousInstance {
    track_id: usize,
    mask: GrayImage,
}

fn decode_rle(
    counts: &[u32],
    width: u32,
    height: u32,
) -> Result<GrayImage, Box<dyn Error>> {
    let length = width as usize * height as usize;
    let mut pixels = vec![0u8; length];
    let mut cursor = 0usize;
    let mut foreground = false;
    for &count in counts {
        let end = cursor
            .checked_add(count as usize)
            .ok_or("mask RLE overflow")?;
        if end > length {
            return Err("mask RLE exceeds configured dimensions".into());
        }
        if foreground {
            pixels[cursor..end].fill(255);
        }
        cursor = end;
        foreground = !foreground;
    }
    if cursor != length {
        return Err("mask RLE does not fill configured dimensions".into());
    }
    GrayImage::from_raw(width, height, pixels)
        .ok_or_else(|| "failed to construct mask image".into())
}

fn mask_iou(first: &GrayImage, second: &GrayImage) -> f32 {
    let mut intersection = 0usize;
    let mut union = 0usize;
    for (&left, &right) in first.as_raw().iter().zip(second.as_raw()) {
        intersection += usize::from(left != 0 && right != 0);
        union += usize::from(left != 0 || right != 0);
    }
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

fn propagated_masks(
    previous: &[PreviousInstance],
    current: &[GrayImage],
    detections: &[InputDetection],
) -> Vec<McByteMask> {
    let mut candidates = previous
        .iter()
        .enumerate()
        .flat_map(|(previous_index, previous)| {
            current.iter().enumerate().filter_map(
                move |(current_index, mask)| {
                    let similarity = mask_iou(&previous.mask, mask);
                    (similarity >= 0.02).then_some((
                        similarity,
                        previous_index,
                        current_index,
                    ))
                },
            )
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| right.0.total_cmp(&left.0));

    let mut used_previous = vec![false; previous.len()];
    let mut used_current = vec![false; current.len()];
    let mut result = Vec::new();
    for (_, previous_index, current_index) in candidates {
        if used_previous[previous_index] || used_current[current_index] {
            continue;
        }
        used_previous[previous_index] = true;
        used_current[current_index] = true;
        result.push(McByteMask::new(
            previous[previous_index].track_id,
            current[current_index].clone(),
            detections[current_index].score,
        ));
    }
    result
}

fn output_detection_matches(
    outputs: &[Object],
    detections: &[Object],
) -> Vec<(usize, usize)> {
    let mut candidates = outputs
        .iter()
        .enumerate()
        .flat_map(|(output_index, output)| {
            detections.iter().enumerate().map(
                move |(detection_index, detection)| {
                    (
                        output.get_rect().calc_iou(&detection.get_rect()),
                        output_index,
                        detection_index,
                    )
                },
            )
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| right.0.total_cmp(&left.0));
    let mut used_outputs = vec![false; outputs.len()];
    let mut used_detections = vec![false; detections.len()];
    let mut matches = Vec::new();
    for (similarity, output_index, detection_index) in candidates {
        if similarity <= 0.0
            || used_outputs[output_index]
            || used_detections[detection_index]
        {
            continue;
        }
        used_outputs[output_index] = true;
        used_detections[detection_index] = true;
        matches.push((output_index, detection_index));
    }
    matches.sort_by_key(|(_, detection_index)| *detection_index);
    matches
}

fn resized_frame(
    frames_dir: &Path,
    frame_id: usize,
    width: u32,
    height: u32,
) -> Result<GrayImage, Box<dyn Error>> {
    let path = frames_dir.join(format!("frame_{frame_id:06}.jpg"));
    let image = image::open(&path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?
        .to_luma8();
    Ok(image::imageops::resize(
        &image,
        width,
        height,
        FilterType::Triangle,
    ))
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = env::args().collect::<Vec<_>>();
    let input_path = PathBuf::from(
        arguments
            .get(1)
            .map(String::as_str)
            .unwrap_or("data/jsons/mcbyte_rfdetr.jsonl"),
    );
    let output_path = PathBuf::from(
        arguments
            .get(2)
            .map(String::as_str)
            .unwrap_or("data/jsons/mcbyte_rfdetr_tracks.jsonl"),
    );
    let frames_dir = PathBuf::from(
        arguments
            .get(3)
            .map(String::as_str)
            .unwrap_or("data/frames_30fps"),
    );
    let maximum_frames = arguments
        .get(4)
        .map(|value| value.parse::<usize>())
        .transpose()?;

    let reader = BufReader::new(File::open(&input_path)?);
    let mut records = reader.lines();
    let metadata_line = records.next().ok_or("input data is empty")??;
    let InputRecord::Metadata {
        schema_version,
        source_video,
        mask_width,
        mask_height,
        fps,
        frame_count,
    } = serde_json::from_str(&metadata_line)?
    else {
        return Err("the first input record must contain metadata".into());
    };

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(&output_path)?);
    let output_metadata = OutputRecord::Metadata {
        schema_version,
        source_data: &input_path.to_string_lossy(),
        source_video: &source_video,
        mask_width,
        mask_height,
        fps,
        frame_count: maximum_frames.unwrap_or(frame_count).min(frame_count),
        tracker: "McByteTracker (Rust)",
    };
    serde_json::to_writer(&mut writer, &output_metadata)?;
    writer.write_all(b"\n")?;

    let mut tracker = McByteTracker::new(30, 30, 0.5, 0.4)
        .with_isolated_mask_matching(true)
        .with_sparse_opt_flow(SparseOptFlowConfig {
            downscale: 3,
            max_corners: 300,
            ..Default::default()
        });
    let mut previous = Vec::<PreviousInstance>::new();
    let mut processed = 0usize;

    for line in records {
        if maximum_frames.is_some_and(|maximum| processed >= maximum) {
            break;
        }
        let InputRecord::Frame {
            frame_id,
            detections,
        } = serde_json::from_str(&line?)?
        else {
            continue;
        };
        let masks = detections
            .iter()
            .map(|detection| {
                decode_rle(&detection.mask_rle, mask_width, mask_height)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let objects = detections
            .iter()
            .map(|detection| {
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
        let propagated = propagated_masks(&previous, &masks, &detections);
        let frame =
            resized_frame(&frames_dir, frame_id, mask_width, mask_height)?;
        let outputs = tracker.update_with_frame_and_masks(
            &objects,
            &frame,
            &propagated,
        )?;
        let matches = output_detection_matches(&outputs, &objects);

        let mut tracks = Vec::new();
        let mut next_previous = Vec::new();
        for (output_index, detection_index) in matches {
            let output = &outputs[output_index];
            let Some(track_id) = output.get_track_id() else {
                continue;
            };
            let detection = &detections[detection_index];
            let rect = output.get_rect();
            tracks.push(OutputTrack {
                track_id,
                detection_index,
                bbox: [rect.x(), rect.y(), rect.width(), rect.height()],
                score: detection.score,
                class_id: detection.class_id,
                class_name: detection.class_name.clone(),
            });
            next_previous.push(PreviousInstance {
                track_id,
                mask: masks[detection_index].clone(),
            });
        }
        previous = next_previous;

        serde_json::to_writer(
            &mut writer,
            &OutputRecord::Frame { frame_id, tracks },
        )?;
        writer.write_all(b"\n")?;
        processed += 1;
        if processed.is_multiple_of(100) {
            eprintln!("McByte: {processed} frames");
        }
    }
    writer.flush()?;
    eprintln!(
        "Saved {processed} McByte frames to {}",
        output_path.display()
    );
    Ok(())
}
