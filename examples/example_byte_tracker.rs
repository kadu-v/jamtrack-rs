use indicatif::{ProgressBar, ProgressStyle};
use jamtrack_rs::{byte_tracker::ByteTracker, object::Object, rect::Rect};
use serde::Deserialize;
use std::{
    env,
    error::Error,
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Detection {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    score: f32,
    class_id: i32,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print_usage();
        return Ok(());
    }

    let inputs_dir = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/onnx_inputs"));
    let outputs_dir = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/onnx_outputs"));
    let output_video = args.get(3).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("data/video/byte_tracker_from_onnx.mp4")
    });
    let output_frames_dir = args
        .get(4)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/video/byte_tracker_frames"));
    let max_frames = args
        .get(5)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);

    let mut tracker = ByteTracker::new(30, 25, 0.45, 0.55, 0.8);
    let json_files = list_json_files(&outputs_dir)?;

    if json_files.is_empty() {
        return Err(format!(
            "No json files found in {}",
            outputs_dir.display()
        )
        .into());
    }

    fs::create_dir_all(&output_frames_dir)?;

    let total_frames = if max_frames > 0 && max_frames < json_files.len() {
        max_frames
    } else {
        json_files.len()
    };
    let progress = ProgressBar::new(total_frames as u64);
    let style = ProgressStyle::with_template(
        "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}",
    )?
    .progress_chars("=>-");
    progress.set_style(style);
    progress.set_message("tracking");

    let mut processed = 0usize;
    for json_path in json_files {
        if max_frames > 0 && processed >= max_frames {
            break;
        }

        let stem = json_path
            .file_stem()
            .and_then(OsStr::to_str)
            .ok_or("Invalid json filename")?;
        let image_path = inputs_dir.join(format!("{stem}.jpg"));
        if !image_path.exists() {
            return Err(format!(
                "Missing image for {stem}: {}",
                image_path.display()
            )
            .into());
        }

        let detections = load_detections(&json_path)?;
        let objects = detections_to_objects(detections);
        let tracked = tracker.update(&objects)?;

        let mut frame = image::open(&image_path)?.into_rgb8();
        draw_tracks(&mut frame, &tracked);

        let output_frame = output_frames_dir.join(format!("{stem}.png"));
        frame.save(&output_frame)?;
        processed += 1;
        progress.inc(1);
    }

    progress.finish_with_message("encoding video");
    encode_video(&output_frames_dir, &output_video)?;
    progress.finish_with_message("done");
    println!("Saved video to {}", output_video.display());

    Ok(())
}

fn print_usage() {
    println!(
        "Usage: cargo run --example example_byte_tracker [inputs_dir] [outputs_dir] [output_video] [output_frames_dir] [max_frames]\n\
Defaults:\n\
  inputs_dir: data/onnx_inputs\n\
  outputs_dir: data/onnx_outputs\n\
  output_video: data/video/byte_tracker_from_onnx.mp4\n\
  output_frames_dir: data/video/byte_tracker_frames\n\
  max_frames: 0 (all)\n\
Requires ffmpeg available on PATH."
    );
}

fn list_json_files(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut jsons = Vec::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(OsStr::to_str) == Some("json") {
            jsons.push(path);
        }
    }
    jsons.sort();
    Ok(jsons)
}

fn load_detections(path: &Path) -> Result<Vec<Detection>, Box<dyn Error>> {
    let data = fs::read_to_string(path)?;
    let detections: Vec<Detection> = serde_json::from_str(&data)?;
    Ok(detections)
}

fn detections_to_objects(detections: Vec<Detection>) -> Vec<Object> {
    let mut objects = Vec::new();
    for det in detections {
        if det.score < 0.1 {
            continue;
        }
        let width = det.x2 - det.x1;
        let height = det.y2 - det.y1;
        if width <= 0.0 || height <= 0.0 {
            continue;
        }
        let rect = Rect::new(det.x1, det.y1, width, height);
        objects.push(Object::new(rect, det.score, None));
    }
    objects
}

fn draw_tracks(frame: &mut image::RgbImage, tracks: &[Object]) {
    for track in tracks {
        let rect = track.get_rect();
        let x1 = rect.x().round() as i32;
        let y1 = rect.y().round() as i32;
        let x2 = (rect.x() + rect.width()).round() as i32;
        let y2 = (rect.y() + rect.height()).round() as i32;
        let track_id = track.get_track_id().unwrap_or(0);
        let color = color_for_id(track_id);

        draw_rect(frame, x1, y1, x2, y2, color, 2);
        if track_id > 0 {
            draw_number(
                frame,
                x1 + 2,
                y1 + 2,
                track_id as u32,
                [255, 255, 255],
            );
        }
    }
}

fn color_for_id(track_id: usize) -> [u8; 3] {
    let r = 50 + ((track_id * 37) % 206) as u8;
    let g = 50 + ((track_id * 17) % 206) as u8;
    let b = 50 + ((track_id * 29) % 206) as u8;
    [r, g, b]
}

fn draw_rect(
    frame: &mut image::RgbImage,
    mut x1: i32,
    mut y1: i32,
    mut x2: i32,
    mut y2: i32,
    color: [u8; 3],
    thickness: i32,
) {
    let (width, height) = frame.dimensions();
    let max_x = width.saturating_sub(1) as i32;
    let max_y = height.saturating_sub(1) as i32;

    x1 = x1.clamp(0, max_x);
    y1 = y1.clamp(0, max_y);
    x2 = x2.clamp(0, max_x);
    y2 = y2.clamp(0, max_y);

    if x2 <= x1 || y2 <= y1 {
        return;
    }

    for offset in 0..thickness {
        let top = (y1 + offset).clamp(0, max_y);
        let bottom = (y2 - offset).clamp(0, max_y);
        let left = (x1 + offset).clamp(0, max_x);
        let right = (x2 - offset).clamp(0, max_x);

        for x in left..=right {
            set_pixel(frame, x, top, color);
            set_pixel(frame, x, bottom, color);
        }
        for y in top..=bottom {
            set_pixel(frame, left, y, color);
            set_pixel(frame, right, y, color);
        }
    }
}

fn set_pixel(frame: &mut image::RgbImage, x: i32, y: i32, color: [u8; 3]) {
    if x < 0 || y < 0 {
        return;
    }
    let (width, height) = frame.dimensions();
    if x >= width as i32 || y >= height as i32 {
        return;
    }
    frame.put_pixel(x as u32, y as u32, image::Rgb(color));
}

fn draw_number(
    frame: &mut image::RgbImage,
    x: i32,
    y: i32,
    value: u32,
    color: [u8; 3],
) {
    let digits: Vec<u32> = value
        .to_string()
        .chars()
        .filter_map(|c| c.to_digit(10))
        .collect();
    let mut cursor_x = x;
    for digit in digits {
        draw_digit(frame, cursor_x, y, digit, color, 2);
        cursor_x += 8;
    }
}

fn draw_digit(
    frame: &mut image::RgbImage,
    x: i32,
    y: i32,
    digit: u32,
    color: [u8; 3],
    scale: i32,
) {
    const DIGITS: [[u8; 15]; 10] = [
        [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
        [0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    ];

    let digit = digit.min(9) as usize;
    let pattern = DIGITS[digit];
    for row in 0..5 {
        for col in 0..3 {
            if pattern[row * 3 + col] == 0 {
                continue;
            }
            let base_x = x + col as i32 * scale;
            let base_y = y + row as i32 * scale;
            for sy in 0..scale {
                for sx in 0..scale {
                    set_pixel(frame, base_x + sx, base_y + sy, color);
                }
            }
        }
    }
}

fn encode_video(
    frames_dir: &Path,
    output_video: &Path,
) -> Result<(), Box<dyn Error>> {
    let input_pattern = frames_dir.join("frame_%06d.png");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-framerate",
            "30",
            "-i",
            input_pattern.to_str().ok_or("Invalid frames path")?,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            output_video.to_str().ok_or("Invalid output path")?,
        ])
        .status();

    match status {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => Err(format!("ffmpeg failed with status {status}").into()),
        Err(err) => Err(format!("Failed to run ffmpeg: {err}").into()),
    }
}
