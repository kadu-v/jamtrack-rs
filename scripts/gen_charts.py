import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

matplotlib.rcParams["font.family"] = "sans-serif"

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "benchmarks"
OUTPUT_DIR = ROOT_DIR / "data" / "charts"

COLOR_RUST = "#CD853F"
COLOR_PYTHON = "#4C72B0"
COLOR_PERFORMANCE = "#4C72B0"
COLOR_HIGHLIGHT = "#2CA02C"
MOT_METRICS = {
    "HOTA": ("hota", "mot17_hota.png"),
    "MOTA": ("mota", "mot17_mota.png"),
    "IDF1": ("idf1", "mot17_idf1.png"),
    "IDSW": ("idsw", "mot17_idsw.png"),
}


def load_tracker_files():
    paths = sorted(DATA_DIR.glob("*.json"))
    if not paths:
        raise RuntimeError(f"No tracker benchmark JSON files found in {DATA_DIR}")

    trackers = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            tracker = json.load(stream)
        if tracker.get("schema_version") != 1:
            raise ValueError(f"{path}: unsupported or missing schema_version")
        if not tracker.get("tracker"):
            raise ValueError(f"{path}: missing tracker name")
        tracker["_source"] = path.name
        trackers.append(tracker)
    return trackers


def collect_performance_results(trackers):
    results = []
    contexts = set()
    for tracker in trackers:
        performance = tracker.get("performance")
        if performance is None:
            continue
        context = (
            performance["device"],
            performance["frames"],
            performance["unit"],
            performance["input"],
        )
        contexts.add(context)
        for result in performance["results"]:
            for key in ("label", "implementation", "variant", "elapsed_ms"):
                if key not in result:
                    raise ValueError(
                        f"{tracker['_source']}: performance result missing {key}"
                    )
            results.append(result)

    if len(contexts) != 1:
        raise ValueError(f"Performance benchmark contexts do not match: {contexts}")
    if not results:
        raise ValueError("No performance benchmark results found")
    return results, contexts.pop()


def collect_mot_results(trackers):
    results = []
    contexts = set()
    for tracker in trackers:
        benchmark = tracker.get("mot17_train")
        if benchmark is None:
            continue
        contexts.add((benchmark["detector"], benchmark["evaluator"]))
        for result in benchmark["results"]:
            for key in ("label", "implementation", "variant", "metrics"):
                if key not in result:
                    raise ValueError(
                        f"{tracker['_source']}: MOT17 result missing {key}"
                    )
            missing_metrics = {
                metric_key for metric_key, _ in MOT_METRICS.values()
                if metric_key not in result["metrics"]
            }
            if missing_metrics:
                raise ValueError(
                    f"{tracker['_source']}: missing metrics {sorted(missing_metrics)}"
                )
            if result["implementation"] == "python" and not result.get("official"):
                raise ValueError(
                    f"{tracker['_source']}: Python result must declare official=true"
                )
            results.append(result)

    if len(contexts) != 1:
        raise ValueError(f"MOT17 benchmark contexts do not match: {contexts}")
    if not results:
        raise ValueError("No MOT17 benchmark results found")
    return results, contexts.pop()


def make_performance_chart(results, context):
    device, frames, unit, _ = context
    results = sorted(results, key=lambda result: result["elapsed_ms"], reverse=True)
    names = [result["label"] for result in results]
    times = [result["elapsed_ms"] for result in results]
    colors = [
        COLOR_HIGHLIGHT if result.get("highlight") else COLOR_PERFORMANCE
        for result in results
    ]

    fig, ax = plt.subplots(figsize=(8, 4.0))
    bars = ax.barh(names, times, color=colors, edgecolor="white", height=0.6)
    ax.set_xlabel(f"Time ({unit})", fontsize=11)
    ax.set_title(
        f"Performance ({device}, {frames} frames)", fontsize=13, fontweight="bold"
    )
    ax.invert_yaxis()
    for bar, elapsed_ms in zip(bars, times):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{elapsed_ms} {unit}",
            va="center",
            fontsize=10,
        )
    ax.set_xlim(0, max(times) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "performance.png", dpi=150, bbox_inches="tight")
    plt.close()


def make_mot_chart(results, context, metric, metric_key, filename):
    detector, _ = context
    lower_is_better = metric == "IDSW"
    results = sorted(
        results,
        key=lambda result: result["metrics"][metric_key],
        reverse=not lower_is_better,
    )
    names = [result["label"] for result in results]
    values = [result["metrics"][metric_key] for result in results]
    colors = [
        COLOR_PYTHON if result["implementation"] == "python" else COLOR_RUST
        for result in results
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(names, values, color=colors, edgecolor="white", height=0.65)
    ax.set_xlabel(metric, fontsize=11)
    ax.set_title(
        f"MOT17-train {metric} ({detector} Detector)",
        fontsize=13,
        fontweight="bold",
    )

    value_format = "d" if lower_is_better else ".2f"
    if lower_is_better:
        margin = max(values) * 0.008
        ax.set_xlim(0, max(values) * 1.15)
    else:
        value_range = max(values) - min(values)
        margin = value_range * 0.03
        ax.set_xlim(min(values) - value_range * 0.05, max(values) + value_range * 0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + margin,
            bar.get_y() + bar.get_height() / 2,
            f"{value:{value_format}}",
            va="center",
            fontsize=9,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_elements = [
        Patch(facecolor=COLOR_RUST, label="Rust"),
        Patch(facecolor=COLOR_PYTHON, label="Python (Official)"),
    ]
    legend_location = "lower right" if lower_is_better else "upper right"
    ax.legend(
        handles=legend_elements,
        loc=legend_location,
        fontsize=10,
        framealpha=0.9,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trackers = load_tracker_files()
    performance_results, performance_context = collect_performance_results(trackers)
    mot_results, mot_context = collect_mot_results(trackers)

    make_performance_chart(performance_results, performance_context)
    for metric, (metric_key, filename) in MOT_METRICS.items():
        make_mot_chart(mot_results, mot_context, metric, metric_key, filename)

    print(
        f"Loaded {len(trackers)} tracker files, "
        f"{len(performance_results)} performance results, and "
        f"{len(mot_results)} MOT17 results"
    )
    print(f"Charts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
