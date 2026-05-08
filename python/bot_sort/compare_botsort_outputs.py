import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path: Path):
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    frames = defaultdict(list)
    for item in payload["results"]:
        frames[int(item["frame_id"])].append(
            (
                int(item["track_id"]),
                float(item["x"]),
                float(item["y"]),
                float(item["width"]),
                float(item["height"]),
                float(item.get("score", 0.0)),
            )
        )
    for frame_tracks in frames.values():
        frame_tracks.sort(key=lambda row: row[0])
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("python_json", type=Path)
    parser.add_argument("rust_json", type=Path)
    parser.add_argument("--max-diffs", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-2)
    args = parser.parse_args()

    py = load(args.python_json)
    rs = load(args.rust_json)
    frames = sorted(set(py.keys()) | set(rs.keys()))
    diff_count = 0
    first_diff = None

    for frame_id in frames:
        py_tracks = py.get(frame_id, [])
        rs_tracks = rs.get(frame_id, [])
        if len(py_tracks) != len(rs_tracks):
            diff_count += 1
            first_diff = first_diff or frame_id
            print(
                f"frame {frame_id}: count differs python={len(py_tracks)} rust={len(rs_tracks)}"
            )
        else:
            for p, r in zip(py_tracks, rs_tracks):
                if p[0] != r[0] or any(abs(a - b) > args.tol for a, b in zip(p[1:], r[1:])):
                    diff_count += 1
                    first_diff = first_diff or frame_id
                    print(f"frame {frame_id}: python={p} rust={r}")
                    break
        if diff_count >= args.max_diffs:
            break

    if diff_count == 0:
        print("BoT-SORT outputs match")
    else:
        raise SystemExit(f"Found {diff_count} differing frames; first={first_diff}")


if __name__ == "__main__":
    main()
