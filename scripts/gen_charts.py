import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib.lines import Line2D

matplotlib.rcParams["font.family"] = "sans-serif"

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "benchmarks"
OUTPUT_DIR = ROOT_DIR / "data" / "charts"

COLOR_RUST = "#CD853F"
COLOR_PYTHON = "#4C72B0"
COLOR_PERFORMANCE = "#4C72B0"
COLOR_HIGHLIGHT = "#2CA02C"
COLOR_FRONTIER = "#666666"
VARIANT_MARKERS = {
    "Default": "o",
    "Tuned": "^",
    "Enhanced": "D",
    "ECC": "s",
}
MOT_METRIC_KEYS = ("hota", "mota", "idf1", "idsw")


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
            for key in (
                "label",
                "chart_label",
                "implementation",
                "variant",
                "metrics",
            ):
                if key not in result:
                    raise ValueError(f"{tracker['_source']}: MOT17 result missing {key}")
            missing_metrics = {
                metric_key
                for metric_key in MOT_METRIC_KEYS
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


def variant_group(variant):
    if "ecc" in variant:
        return "ECC"
    if variant == "tuned":
        return "Tuned"
    if variant in {"plus", "plusplus"}:
        return "Enhanced"
    return "Default"


def pareto_frontier(results, x_key, y_key, maximize_x=True, maximize_y=True):
    frontier = []
    for candidate in results:
        candidate_x = candidate["metrics"][x_key]
        candidate_y = candidate["metrics"][y_key]
        dominated = False
        for other in results:
            if other is candidate:
                continue
            other_x = other["metrics"][x_key]
            other_y = other["metrics"][y_key]
            x_at_least_as_good = (
                other_x >= candidate_x if maximize_x else other_x <= candidate_x
            )
            y_at_least_as_good = (
                other_y >= candidate_y if maximize_y else other_y <= candidate_y
            )
            x_strictly_better = (
                other_x > candidate_x if maximize_x else other_x < candidate_x
            )
            y_strictly_better = (
                other_y > candidate_y if maximize_y else other_y < candidate_y
            )
            if (
                x_at_least_as_good
                and y_at_least_as_good
                and (x_strictly_better or y_strictly_better)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier


def selected_mot_labels(results):
    official_frontier_labels = {
        result["label"]
        for result in pareto_frontier(
            results, "hota", "idsw", maximize_x=True, maximize_y=False
        )
        + pareto_frontier(
            results, "mota", "idf1", maximize_x=True, maximize_y=True
        )
        if result["implementation"] == "python"
    }
    return {
        result["label"]
        for result in results
        if result["implementation"] == "rust"
        or result["label"] in official_frontier_labels
    }


def make_performance_chart(results, context):
    device, frames, unit, _ = context
    results = sorted(results, key=lambda result: result["elapsed_ms"])
    labels = [result["label"] for result in results]
    times = [result["elapsed_ms"] for result in results]
    colors = [
        COLOR_HIGHLIGHT if result.get("highlight") else COLOR_PERFORMANCE
        for result in results
    ]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars = ax.bar(labels, times, color=colors, edgecolor="white", width=0.68)
    for bar, elapsed_ms in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(times) * 0.018,
            f"{elapsed_ms:g} {unit}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylim(0, max(times) * 1.14)
    ax.set_ylabel(f"Elapsed time ({unit}) — lower is better", fontsize=11)
    ax.set_title(
        f"Performance ({device}, {frames} frames)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelrotation=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "performance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_pareto_line(ax, frontier, x_key, y_key):
    if len(frontier) < 2:
        return
    points = sorted(
        (
            result["metrics"][x_key],
            result["metrics"][y_key],
        )
        for result in frontier
    )
    ax.plot(
        [point[0] for point in points],
        [point[1] for point in points],
        color=COLOR_FRONTIER,
        linestyle="--",
        linewidth=1.1,
        alpha=0.75,
        zorder=2,
    )


def draw_mot_panel(
    ax,
    results,
    labels_to_show,
    x_key,
    y_key,
    x_label,
    y_label,
    title,
    maximize_y,
):
    texts = []
    x_values = []
    y_values = []
    for result in results:
        x_value = result["metrics"][x_key]
        y_value = result["metrics"][y_key]
        x_values.append(x_value)
        y_values.append(y_value)
        color = (
            COLOR_PYTHON
            if result["implementation"] == "python"
            else COLOR_RUST
        )
        marker = VARIANT_MARKERS[variant_group(result["variant"])]
        ax.scatter(
            x_value,
            y_value,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.8,
            s=80,
            alpha=0.9,
            zorder=3,
        )
        if result["label"] in labels_to_show:
            texts.append(
                ax.text(
                    x_value,
                    y_value,
                    result["chart_label"],
                    fontsize=8,
                    color="#222222",
                    zorder=4,
                )
            )

    frontier = pareto_frontier(
        results,
        x_key,
        y_key,
        maximize_x=True,
        maximize_y=maximize_y,
    )
    draw_pareto_line(ax, frontier, x_key, y_key)
    ax.margins(x=0.1, y=0.13)
    if not maximize_y:
        ax.invert_yaxis()
    adjust_text(
        texts,
        ax=ax,
        x=x_values,
        y=y_values,
        expand=(1.08, 1.18),
        force_text=(0.2, 0.25),
        force_static=(0.08, 0.12),
        arrowprops={"arrowstyle": "-", "color": "#999999", "linewidth": 0.6},
    )

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(color="#DDDDDD", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.annotate(
        "better ↗",
        xy=(0.98, 0.97),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
        color="#666666",
    )


def mot_legend_handles():
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLOR_RUST,
            markeredgecolor="white",
            markersize=8,
            label="Rust",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLOR_PYTHON,
            markeredgecolor="white",
            markersize=8,
            label="Python (Official)",
        ),
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor="#777777",
            markeredgecolor="white",
            markersize=8,
            label=label,
        )
        for label, marker in VARIANT_MARKERS.items()
    )
    handles.append(
        Line2D(
            [0],
            [0],
            color=COLOR_FRONTIER,
            linestyle="--",
            linewidth=1.1,
            label="Pareto frontier",
        )
    )
    return handles


def make_mot_tradeoff_chart(results, context):
    detector, _ = context
    labels_to_show = selected_mot_labels(results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    draw_mot_panel(
        axes[0],
        results,
        labels_to_show,
        "hota",
        "idsw",
        "HOTA — higher is better",
        "ID switches — lower is better",
        "Tracking quality vs. identity switches",
        maximize_y=False,
    )
    draw_mot_panel(
        axes[1],
        results,
        labels_to_show,
        "mota",
        "idf1",
        "MOTA — higher is better",
        "IDF1 — higher is better",
        "Detection accuracy vs. identity preservation",
        maximize_y=True,
    )

    fig.suptitle(
        f"MOT17-train trade-offs ({detector} Detector)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.legend(
        handles=mot_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=7,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    fig.savefig(OUTPUT_DIR / "mot17_tradeoffs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trackers = load_tracker_files()
    performance_results, performance_context = collect_performance_results(trackers)
    mot_results, mot_context = collect_mot_results(trackers)

    make_performance_chart(performance_results, performance_context)
    make_mot_tradeoff_chart(mot_results, mot_context)

    print(
        f"Loaded {len(trackers)} tracker files, "
        f"{len(performance_results)} performance results, and "
        f"{len(mot_results)} MOT17 results"
    )
    print(f"Charts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
