import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch

matplotlib.rcParams['font.family'] = 'sans-serif'

OUTPUT_DIR = "data/charts"

# --- Performance chart ---
trackers_perf = ["BoostTrack+", "ByteTracker", "BoostTrack++", "BoostTrack", "OC-SORT + BYTE", "OC-SORT"]
times = [84.1, 80.9, 77.9, 63.4, 56.8, 33.2]

fig, ax = plt.subplots(figsize=(8, 3.5))
colors = ["#4C72B0"] * len(trackers_perf)
colors[-1] = "#2CA02C"
bars = ax.barh(trackers_perf, times, color=colors, edgecolor="white", height=0.6)
ax.set_xlabel("Time (ms)", fontsize=11)
ax.set_title("Performance (M3 MacBook Pro, 1627 frames)", fontsize=13, fontweight="bold")
ax.invert_yaxis()
for bar, t in zip(bars, times):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{t} ms", va="center", fontsize=10)
ax.set_xlim(0, max(times) * 1.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/performance.png", dpi=150, bbox_inches="tight")
plt.close()

# --- MOT17 Benchmark data ---
trackers_mot = [
    "BoostTrack+ (Rust)",
    "BoostTrack (Rust)",
    "BoostTrack++ (Rust)",
    "OfficialBoostTrack (Python)",
    "OCSortTracker (Rust)",
    "OfficialByteTracker (Python)",
    "OfficialBoostTrack++ (Python)",
    "OfficialByteTrackerTuned (Python)",
    "ByteTracker (Rust)",
    "BoostTrackECC (Rust)",
    "BoostTrack++ECC (Rust)",
    "ByteTrackerTuned (Rust)",
    "BotSort (Rust)",
    "OfficialBoostTrackECC (Python)",
    "OfficialBoostTrack++ECC (Python)",
    "BotSortECC (Rust)",
]

metrics = {
    "HOTA": [65.93, 66.03, 66.02, 67.30, 67.73, 67.82, 67.87, 67.92, 68.35, 68.39, 68.35, 68.55, 68.97, 69.28, 69.71, 70.94],
    "MOTA": [78.57, 78.24, 78.86, 78.26, 78.55, 80.92, 78.89, 80.90, 80.97, 79.06, 79.80, 80.95, 81.26, 79.17, 79.92, 82.11],
    "IDF1": [74.11, 74.13, 74.29, 76.00, 76.67, 77.29, 76.91, 77.47, 77.89, 77.94, 77.98, 78.27, 77.68, 79.10, 79.82, 80.83],
    "IDSW": [560, 536, 558, 520, 484, 458, 515, 453, 454, 344, 318, 450, 784, 308, 287, 347],
}

best_tracker = {
    "HOTA": "BotSortECC (Rust)",
    "MOTA": "BotSortECC (Rust)",
    "IDF1": "BotSortECC (Rust)",
    "IDSW": "OfficialBoostTrack++ECC (Python)",
}

COLOR_RUST = "#CD853F"
COLOR_PYTHON = "#4C72B0"

def make_mot_chart(metric, values, filename):
    lower_is_better = metric == "IDSW"
    sorted_pairs = sorted(zip(trackers_mot, values), key=lambda x: x[1], reverse=not lower_is_better)
    names = [p[0] for p in sorted_pairs]
    vals = [p[1] for p in sorted_pairs]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors_mot = [COLOR_PYTHON if "Python" in t else COLOR_RUST for t in names]

    bars = ax.barh(names, vals, color=colors_mot, edgecolor="white", height=0.65)
    ax.set_xlabel(metric, fontsize=11)
    ax.set_title(f"MOT17-train {metric} (YOLOX-X Detector)", fontsize=13, fontweight="bold")

    fmt = "d" if metric == "IDSW" else ".2f"
    if lower_is_better:
        margin = max(vals) * 0.008
        ax.set_xlim(0, max(vals) * 1.15)
    else:
        margin = (max(vals) - min(vals)) * 0.03
        ax.set_xlim(min(vals) - (max(vals) - min(vals)) * 0.05,
                     max(vals) + (max(vals) - min(vals)) * 0.25)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + margin, bar.get_y() + bar.get_height() / 2,
                f"{v:{fmt}}", va="center", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_elements = [
        Patch(facecolor=COLOR_RUST, label="Rust"),
        Patch(facecolor=COLOR_PYTHON, label="Python (Official)"),
    ]
    loc = "lower right" if lower_is_better else "upper right"
    ax.legend(handles=legend_elements, loc=loc, fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=150, bbox_inches="tight")
    plt.close()

make_mot_chart("HOTA", metrics["HOTA"], "mot17_hota.png")
make_mot_chart("MOTA", metrics["MOTA"], "mot17_mota.png")
make_mot_chart("IDF1", metrics["IDF1"], "mot17_idf1.png")
make_mot_chart("IDSW", metrics["IDSW"], "mot17_idsw.png")

print("Charts saved to data/charts/")
